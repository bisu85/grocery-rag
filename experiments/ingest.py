import glob
import json
import os

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_v2"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")

SCHEMA_FIELDS = ("doc_type", "text", "store", "price", "dietary_tags", "cuisine")


def embed_all(texts: list[str], input_type: str, batch: int = 90) -> list[list[float]]:
    """Embed in chunks — Cohere allows up to 96 texts per call; 90 is a safe margin."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        res = co.embed(
            texts=texts[i : i + batch],
            model="embed-v4.0",
            input_type=input_type,
            output_dimension=1024,
            embedding_types=["float"],
        )
        out.extend(getattr(res.embeddings, "float", None) or res.embeddings.float_)
    return out


all_json = glob.glob("data/*.json")
clean_paths = {p for p in all_json if p.endswith(".clean.json")}
# for each raw file that has a .clean.json twin, use the clean one; keep files with no twin
use_paths = []
for p in sorted(all_json):
    if p.endswith(".clean.json"):
        use_paths.append(p)
    else:
        twin = p.replace(".json", ".clean.json")
        if twin not in clean_paths:      # no cleaned version -> use raw (e.g. catalogue.json)
            use_paths.append(p)

records = []
for path in use_paths:
    with open(path) as f:
        items = json.load(f)
    print(f"  loaded {len(items):3} records from {path}")
    records.extend(items)

clean = []
for i, r in enumerate(records):
    rec = {k: r.get(k) for k in SCHEMA_FIELDS}  # keep only schema fields
    rec["id"] = i
    clean.append(rec)
print(f"Total records to ingest: {len(clean)}")

# --- embed the text field (batched) ---
vectors = embed_all([r["text"] for r in clean], "search_document")

# --- fresh collection + filterable payload indexes ---
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)
for field, schema in [
    ("doc_type", models.PayloadSchemaType.KEYWORD),
    ("store", models.PayloadSchemaType.KEYWORD),
    ("price", models.PayloadSchemaType.FLOAT),
    ("dietary_tags", models.PayloadSchemaType.KEYWORD),
    ("cuisine", models.PayloadSchemaType.KEYWORD),
]:
    client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)

client.upsert(
    COLLECTION,
    points=[
        models.PointStruct(id=r["id"], vector=vectors[i], payload=r)
        for i, r in enumerate(clean)
    ],
)
print(f"Ingested {len(clean)} records into '{COLLECTION}'.")