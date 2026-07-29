import asyncio
import glob
import json
import re

from qdrant_client import models

from grocery_rag.clients import qdrant
from grocery_rag.config import COLLECTION, EMBED_DIM
from grocery_rag.retrieval import embed_texts

SCHEMA_FIELDS = ("doc_type", "text", "store", "price", "dietary_tags", "cuisine")
JUNK_MARKERS = ("nutri-score", "http", "www.")
MIN_LEN = 8


def is_valid(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < MIN_LEN:
        return False
    if any(m in text.lower() for m in JUNK_MARKERS):
        return False
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return False
    return True


async def main() -> None:
    raw = []
    for path in sorted(glob.glob("data/*.json")):
        if path.endswith(".clean.json"):
            continue
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
    print(f"Kept {len(clean)}, dropped {dropped}. Embedding into '{COLLECTION}'...")

    # local (or cohere) document embeddings via the package
    vectors = await embed_texts([r["text"] for r in clean], "search_document")

    if await qdrant.collection_exists(COLLECTION):
        await qdrant.delete_collection(COLLECTION)
    await qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field, schema in [
        ("doc_type", models.PayloadSchemaType.KEYWORD),
        ("store", models.PayloadSchemaType.KEYWORD),
        ("price", models.PayloadSchemaType.FLOAT),
        ("dietary_tags", models.PayloadSchemaType.KEYWORD),
        ("cuisine", models.PayloadSchemaType.KEYWORD),
    ]:
        await qdrant.create_payload_index(COLLECTION, field_name=field, field_schema=schema)

    await qdrant.upsert(COLLECTION, points=[
        models.PointStruct(
            id=r["id"],
            vector={"dense": vectors[i], "bm25": models.Document(text=r["text"], model="Qdrant/bm25")},
            payload=r,
        )
        for i, r in enumerate(clean)
    ])
    print(f"Ingested {len(clean)} into '{COLLECTION}' (local dense + bm25).")


if __name__ == "__main__":
    asyncio.run(main())