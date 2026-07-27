import os

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_hybrid"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")


def embed_query(q: str) -> list[float]:
    res = co.embed(texts=[q], model="embed-v4.0", input_type="search_query",
                   output_dimension=1024, embedding_types=["float"])
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]


def semantic_only(query: str, k: int = 5):
    return client.query_points(
        COLLECTION, query=embed_query(query), using="dense",
        limit=k, with_payload=True,
    ).points


def hybrid(query: str, k: int = 5):
    return client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=embed_query(query), using="dense", limit=20),
            models.Prefetch(
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="bm25", limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k, with_payload=True,
    ).points


def show(query: str):
    print(f"\n{'=' * 70}\nQUERY: {query!r}")
    print("  -- SEMANTIC ONLY --")
    for i, h in enumerate(semantic_only(query), 1):
        print(f"    {i}. {h.payload['text'][:60]}")
    print("  -- HYBRID (dense + BM25, RRF) --")
    for i, h in enumerate(hybrid(query), 1):
        print(f"    {i}. {h.payload['text'][:60]}")


if __name__ == "__main__":
    # exact-term queries: where BM25 should visibly help
    show("Amul ghee")
    show("Tilda basmati rice")
    show("Verstegen garam masala")
    # a conceptual query: semantic should still carry its weight
    show("something creamy for a mild curry")