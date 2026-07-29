from qdrant_client import models

from grocery_rag.clients import co, qdrant
from grocery_rag.config import (
    CANDIDATES, COLLECTION, DEFAULT_K, EMBED_MODEL, EMBED_DIM, RERANK_MODEL,
)


async def embed_query(q: str) -> list[float]:
    res = await co.embed(
        texts=[q], model=EMBED_MODEL, input_type="search_query",
        output_dimension=EMBED_DIM, embedding_types=["float"],
    )
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]


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


async def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Full retrieval: hybrid candidates -> rerank -> top-k as plain dicts."""
    cands = await hybrid_candidates(query)
    if not cands:
        return []
    docs = [c.payload["text"] for c in cands]
    rr = await co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=k)
    return [{"text": docs[r.index], "score": r.relevance_score} for r in rr.results]
