import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_v2"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")

def embed_query(q):
    res = co.embed(texts=[q], model="embed-v4.0", input_type="search_query",
                   output_dimension=1024, embedding_types=["float"])
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]

def search(q, flt=None, label=""):
    print(f"\n{'='*70}\nQuery: {q!r}   Filter: {label or 'none'}")

    # ---- PHASE 1: FILTERING ONLY (no vectors involved) ----
    total = client.count(COLLECTION).count                       # whole collection
    eligible = client.count(COLLECTION, count_filter=flt).count  # survivors of the filter
    print(f"  PHASE 1 (filter):  {total} total items  ->  {eligible} eligible "
          f"({total - eligible} removed before any vector math)")

    # ---- PHASE 2: SEMANTIC RANKING over the eligible survivors ----
    qvec = embed_query(q)   # Cohere's only job: text -> query vector
    hits = client.query_points(COLLECTION, query=qvec,
                               query_filter=flt, limit=eligible or 1,
                               with_payload=True,
                               search_params=models.SearchParams(exact=True)).points
    print(f"  PHASE 2 (rank):    cosine similarity computed over those {eligible} survivors:")
    for rank, h in enumerate(hits, 1):
        p = h.payload
        print(f"    #{rank}  score={h.score:.3f}  [{p['doc_type']:7}] {p['text'][:45]:45} "
              f"€{p['price']}  {p['dietary_tags']}")

q = "something healthy low calories for dinner"

search(q, None, "none")

veg = models.Filter(must=[models.FieldCondition(
    key="dietary_tags", match=models.MatchValue(value="vegetarian"))])
search(q, veg, "vegetarian")

veg_cheap = models.Filter(must=[
    models.FieldCondition(key="dietary_tags", match=models.MatchValue(value="vegetarian")),
    models.FieldCondition(key="price", range=models.Range(lte=5.0)),
])
search(q, veg_cheap, "vegetarian AND price <= 5")