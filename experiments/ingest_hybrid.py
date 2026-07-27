import glob
import json
import os
import re

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_hybrid"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")

SCHEMA_FIELDS = ("doc_type", "text", "store", "price", "dietary_tags", "cuisine")
JUNK_MARKERS = ("nutri-score", "http", "www.")
MIN_LEN = 8


def is_valid(text: str) -> bool:
    """Drop gibberish / junk crowdsourced records (folds in the cleaning step)."""
    text = (text or "").strip()
    if len(text) < MIN_LEN:
        return False
    if any(m in text.lower() for m in JUNK_MARKERS):
        return False
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return False  # mostly SHOUTING/garbled
    return True


def embed_dense(texts: list[str], input_type: str, batch: int = 90) -> list[list[float]]:
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


# --- load, merge, clean, de-dupe ---
raw = []
for path in sorted(glob.glob("data/*.json")):
    if path.endswith(".clean.json"):
        continue  # ignore any leftovers from the earlier clean.py experiment
    with open(path) as f:
        raw.extend(json.load(f))

clean, seen = [], set()
dropped = 0
for r in raw:
    text = r.get("text", "")
    if not is_valid(text):
        dropped += 1
        continue
    key = re.sub(r"\s+", " ", text.lower()).strip()
    if key in seen:
        dropped += 1
        continue
    seen.add(key)
    rec = {k: r.get(k) for k in SCHEMA_FIELDS}
    rec["id"] = len(clean)
    clean.append(rec)

print(f"Kept {len(clean)} records, dropped {dropped} (junk/duplicates).")

# --- dense vectors via Cohere ---
dense_vectors = embed_dense([r["text"] for r in clean], "search_document")

# --- fresh collection with TWO named vectors: dense + sparse bm25 ---
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
for field, schema in [
    ("doc_type", models.PayloadSchemaType.KEYWORD),
    ("store", models.PayloadSchemaType.KEYWORD),
    ("price", models.PayloadSchemaType.FLOAT),
    ("dietary_tags", models.PayloadSchemaType.KEYWORD),
    ("cuisine", models.PayloadSchemaType.KEYWORD),
]:
    client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)

# --- upsert: dense = Cohere list, bm25 = local Document (computed by the client) ---
client.upsert(
    COLLECTION,
    points=[
        models.PointStruct(
            id=r["id"],
            vector={
                "dense": dense_vectors[i],
                "bm25": models.Document(text=r["text"], model="Qdrant/bm25"),
            },
            payload=r,
        )
        for i, r in enumerate(clean)
    ],
)
print(f"Ingested {len(clean)} records into '{COLLECTION}' (dense + bm25).")