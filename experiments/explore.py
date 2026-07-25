from qdrant_client import QdrantClient, models

# Connect to the Qdrant you just started
client = QdrantClient(url="http://localhost:6333")

# Create a collection that stores 4-dimensional vectors, compared by cosine similarity.
# (Real embeddings are ~1024 dims; we use 4 so you can read the numbers.)
client.recreate_collection(
    collection_name="toy",
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
)

# Insert 3 "products" as points. Each has an id, a hand-made vector, and a payload (the human-readable data).
client.upsert(
    collection_name="toy",
    points=[
        models.PointStruct(id=1, vector=[0.9, 0.1, 0.0, 0.0], payload={"name": "Paneer"}),
        models.PointStruct(id=2, vector=[0.85, 0.15, 0.0, 0.0], payload={"name": "Indian cottage cheese"}),
        models.PointStruct(id=3, vector=[0.0, 0.0, 0.9, 0.1], payload={"name": "Basmati rice"}),
    ],
)

# Now "search": find the 2 stored vectors closest to a query vector.
# Our query vector points in roughly the "paneer" direction.
results = client.query_points(
    collection_name="toy",
    query=[0.88, 0.12, 0.0, 0.0],
    limit=2,
).points

for r in results:
    print(f"{r.payload['name']:30} score={r.score:.4f}")