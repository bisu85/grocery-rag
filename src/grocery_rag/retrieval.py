from qdrant_client import models
import asyncio
from functools import lru_cache

from grocery_rag.clients import co, qdrant, ollama_client
from grocery_rag.config import (
    CANDIDATES, COLLECTION, DEFAULT_K, EMBED_MODEL, EMBED_DIM, 
    RERANK_MODEL, RERANK_PROVIDER, LOCAL_RERANK_MODEL,
    EMBED_PROVIDER, LOCAL_EMBED_MODEL
)

@lru_cache(maxsize=1)
def _local_reranker():
    """Load the cross-encoder once, on first use (lazy — avoids slow import at startup)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(LOCAL_RERANK_MODEL, max_length=512)

async def _rerank(query: str, docs: list[str], k: int) -> list[dict]:
    if RERANK_PROVIDER == "local":
        model = _local_reranker()
        pairs = [(query, d) for d in docs]
        # predict is blocking/CPU-bound → run off the event loop so the server stays responsive
        scores = await asyncio.to_thread(model.predict, pairs)
        ranked = sorted(zip(docs, scores), key=lambda ds: ds[1], reverse=True)[:k]
        return [{"text": d, "score": float(s)} for d, s in ranked]
    else:  # cohere
        rr = await co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=k)
        return [{"text": docs[r.index], "score": r.relevance_score} for r in rr.results]


async def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed a batch. input_type ('search_query'/'search_document') is Cohere-only;
    bge-m3 doesn't use it, so we just ignore it in the local branch."""
    if EMBED_PROVIDER == "local":
        # Ollama's async embed; returns {'embeddings': [[...], ...]}
        res = await ollama_client.embed(model=LOCAL_EMBED_MODEL, input=texts)
        return res["embeddings"]
    res = await co.embed(
        texts=texts, model=EMBED_MODEL, input_type=input_type,
        output_dimension=EMBED_DIM, embedding_types=["float"],
    )
    return getattr(res.embeddings, "float", None) or res.embeddings.float_


# # --- Old - Cohere embed model ---
# async def embed_query(q: str) -> list[float]:
#     res = await co.embed(
#         texts=[q], model=EMBED_MODEL, input_type="search_query",
#         output_dimension=EMBED_DIM, embedding_types=["float"],
#     )
#     return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]
async def embed_query(q: str) -> list[float]:
    return (await embed_texts([q], "search_query"))[0]

async def hybrid_candidates(query: str, n: int = CANDIDATES):
    qvec = await embed_query(query)
    return (await qdrant.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=qvec, using="dense", limit=n),
            models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"),
                            using="bm25", limit=n),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=n, with_payload=True,
    )).points


## --- Retrieve from Cohere + Qdrant: hybrid candidates -> rerank -> top-k as plain dicts ---
# async def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
#     """Full retrieval: hybrid candidates -> rerank -> top-k as plain dicts."""
#     cands = await hybrid_candidates(query)
#     if not cands:
#         return []
#     docs = [c.payload["text"] for c in cands]
#     rr = await co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=k)
#     return [{"text": docs[r.index], "score": r.relevance_score} for r in rr.results]

# --- Local reranker version of retrieve.
async def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
    cands = await hybrid_candidates(query)
    if not cands:
        return []
    docs = [c.payload["text"] for c in cands]
    return await _rerank(query, docs, k)