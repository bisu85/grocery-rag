import os

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_hybrid"
RERANK_MODEL = "rerank-v4.0-fast"   # v3.5 deprecated Jul 2026; v4-fast is the successor
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")


def embed_query(q: str) -> list[float]:
    res = co.embed(texts=[q], model="embed-v4.0", input_type="search_query",
                   output_dimension=1024, embedding_types=["float"])
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]


def hybrid_candidates(query: str, n: int = 20):
    """STAGE 1 — cast a wide, cheap net (optimise for recall)."""
    return client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=embed_query(query), using="dense", limit=n),
            models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"),
                            using="bm25", limit=n),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=n, with_payload=True,
    ).points


def rerank(query: str, candidates, top_n: int = 5):
    """STAGE 2 — cross-encoder re-scores query+doc TOGETHER (optimise for precision)."""
    if not candidates:
        return []
    docs = [c.payload["text"] for c in candidates]
    resp = co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=top_n)
    # each result.index points back into `candidates`; carry the score along
    return [(candidates[r.index], r.relevance_score) for r in resp.results]


def show(query: str):
    cands = hybrid_candidates(query, n=20)
    print(f"\n{'=' * 70}\nQUERY: {query!r}")

    print("  -- HYBRID top 5 (before rerank) --")
    for i, h in enumerate(cands[:5], 1):
        print(f"    {i}. {h.payload['text'][:60]}")

    print("  -- RERANKED top 5 (cross-encoder) --")
    for i, (h, score) in enumerate(rerank(query, cands, top_n=5), 1):
        print(f"    {i}. [{score:.3f}] {h.payload['text'][:60]}")


if __name__ == "__main__":
    show("what do I need to cook a mild creamy curry")
    show("clarified butter for frying spices")
    show("cheese I can cube into a spinach dish")
    show("Tilda basmati rice")