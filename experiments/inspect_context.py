# experiments/inspect_context.py
import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_hybrid"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")


def embed_query(q):
    res = co.embed(texts=[q], model="embed-v4.0", input_type="search_query",
                   output_dimension=1024, embedding_types=["float"])
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]


query = "What can I use to make palak paneer?"

# hybrid candidates (top 15), then rerank to top 4 — exactly what eval_generation used
cands = client.query_points(
    COLLECTION,
    prefetch=[
        models.Prefetch(query=embed_query(query), using="dense", limit=15),
        models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"),
                        using="bm25", limit=15),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=15, with_payload=True,
).points

docs = [c.payload["text"] for c in cands]
rr = co.rerank(model="rerank-v4.0-fast", query=query, documents=docs, top_n=4)

print("=== TOP 15 HYBRID CANDIDATES (what reranker saw) ===")
for i, d in enumerate(docs):
    spinach = "  <-- SPINACH" if "spinach" in d.lower() else ""
    print(f"  {i:2}. {d[:55]}{spinach}")

print("\n=== TOP 4 AFTER RERANK (what the LLM actually got) ===")
for r in rr.results:
    print(f"  [{r.relevance_score:.3f}] {docs[r.index][:55]}")