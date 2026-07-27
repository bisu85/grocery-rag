import os

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION = "grocery_v2"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")


def embed_query(q: str) -> list[float]:
    """Cohere's only job at search time: turn the question text into a query vector."""
    res = co.embed(
        texts=[q],
        model="embed-v4.0",
        input_type="search_query",
        output_dimension=1024,
        embedding_types=["float"],
    )
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]


def search(qvec: list[float], flt: models.Filter | None = None, label: str = "") -> None:
    """Run PHASE 1 (filter) and PHASE 2 (rank) using a PRE-COMPUTED query vector.

    The query vector is passed in (not re-embedded here) so that the same item
    scores identically across different filters — proving the filter only changes
    WHICH items are eligible, never their similarity score.
    """
    print(f"\n{'=' * 70}\nQuery filter: {label or 'none'}")

    # ---- PHASE 1: FILTERING ONLY (no vectors involved) ----
    total = client.count(COLLECTION).count
    eligible = client.count(COLLECTION, count_filter=flt).count
    print(
        f"  PHASE 1 (filter):  {total} total items  ->  {eligible} eligible "
        f"({total - eligible} removed before any vector math)"
    )

    # ---- PHASE 2: SEMANTIC RANKING over the eligible survivors ----
    hits = client.query_points(
        COLLECTION,
        query=qvec,
        query_filter=flt,
        limit=eligible or 1,
        with_payload=True,
        # search_params=models.SearchParams(exact=True),  # uncomment to force brute-force cosine
    ).points

    print(f"  PHASE 2 (rank):    cosine similarity over those {eligible} survivors:")
    for rank, h in enumerate(hits, 1):
        p = h.payload
        print(
            f"    #{rank}  score={h.score:.4f}  [{p['doc_type']:7}] "
            f"{p['text'][:45]:45} €{p['price']}  {p['dietary_tags']}"
        )


if __name__ == "__main__":
    query = "something healthy low calories for dinner"

    # Embed the query ONCE, then reuse the identical vector for every search below.
    qvec = embed_query(query)
    print(f"Query: {query!r}")

    # No filter — the whole collection competes.
    search(qvec, None, "none")

    # Only vegetarian items are eligible.
    veg = models.Filter(
        must=[
            models.FieldCondition(
                key="dietary_tags", match=models.MatchValue(value="vegetarian")
            )
        ]
    )
    search(qvec, veg, "vegetarian")

    # Vegetarian AND price <= 5 (recipes drop out here: their price is null).
    veg_cheap = models.Filter(
        must=[
            models.FieldCondition(
                key="dietary_tags", match=models.MatchValue(value="vegetarian")
            ),
            models.FieldCondition(key="price", range=models.Range(lte=5.0)),
        ]
    )
    search(qvec, veg_cheap, "vegetarian AND price <= 5")
