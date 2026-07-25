import os
import cohere
from qdrant_client import QdrantClient, models

COLLECTION = "grocery_cohere"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")

def embed(texts: list[str], input_type: str) -> list[list[float]]:
    """Cohere embedder. input_type is 'search_document' (for stored items)
    or 'search_query' (for the user's question)."""
    res = co.embed(
        texts=texts,
        model="embed-v4.0",
        input_type=input_type,
        output_dimension=1024,          # v4 supports 256/512/1024/1536
        embedding_types=["float"],
    )
    # SDK versions expose the vectors as .float or .float_ — handle both.
    return getattr(res.embeddings, "float", None) or res.embeddings.float_

# --- same tiny catalogue as before ---
items = [
    "Paneer 200g block, fresh Indian cheese",
    "Basmati rice 5kg, long grain",
    "Amul ghee 1L, clarified butter",
    "Toor dal 1kg, split pigeon peas",
    "Palak paneer recipe: creamy spinach curry with cubes of Indian cheese",
    "Coca-Cola 1.5L bottle",
]

# stored items → "search_document"
vectors = embed(items, "search_document")
dim = len(vectors[0])                    # 1024

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

client.upsert(
    collection_name=COLLECTION,
    points=[
        models.PointStruct(id=i, vector=vectors[i], payload={"text": items[i]})
        for i in range(len(items))
    ],
)

# the user's question → "search_query"
query = "soft kaas for making curry"
query_vector = embed([query], "search_query")[0]

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=3,
).points

print(f"Query: {query!r}\n")
for r in results:
    print(f"score={r.score:.4f}  {r.payload['text']}")