import json, os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_v2"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")

def embed(texts, input_type):
    res = co.embed(texts=texts, model="embed-v4.0", input_type=input_type,
                   output_dimension=1024, embedding_types=["float"])
    return getattr(res.embeddings, "float", None) or res.embeddings.float_

# load the dataset
with open("data/catalogue.json") as f:
    records = json.load(f)

# embed ONLY the text field (one batched call)
vectors = embed([r["text"] for r in records], "search_document")

# fresh collection
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)

# payload indexes = the fields we'll filter on (makes filtering fast & valid)
for field, schema in [
    ("doc_type",     models.PayloadSchemaType.KEYWORD),
    ("store",        models.PayloadSchemaType.KEYWORD),
    ("price",        models.PayloadSchemaType.FLOAT),
    ("dietary_tags", models.PayloadSchemaType.KEYWORD),
    ("cuisine",      models.PayloadSchemaType.KEYWORD),
]:
    client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)

# upsert: vector + the FULL record as payload
client.upsert(COLLECTION, points=[
    models.PointStruct(id=r["id"], vector=vectors[i], payload=r)
    for i, r in enumerate(records)
])
print(f"Ingested {len(records)} records into '{COLLECTION}'.")