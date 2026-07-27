import os
import cohere
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()   # reads .env into the environment
COLLECTION = "grocery_cohere"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")

items = [
    "Paneer 200g block, fresh Indian cheese",
    "Basmati rice 5kg, long grain",
    "Amul ghee 1L, clarified butter",
    "Toor dal 1kg, split pigeon peas",
    "Palak paneer recipe: creamy spinach curry with cubes of Indian cheese",
    "Coca-Cola 1.5L bottle",
]

def embed(texts, input_type):
    res = co.embed(texts=texts, model="embed-v4.0", input_type=input_type,
                   output_dimension=1024, embedding_types=["float"])
    return getattr(res.embeddings, "float", None) or res.embeddings.float_

# --- make sure the collection exists and is populated (only embeds if empty) ---
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
if client.count(COLLECTION).count == 0:
    vecs = embed(items, "search_document")
    client.upsert(COLLECTION, points=[
        models.PointStruct(id=i, vector=vecs[i], payload={"text": items[i]})
        for i in range(len(items))
    ])

# ========== THE RAG LOOP ==========
def answer(question: str, k: int = 3) -> str:
    # 1. RETRIEVE
    qvec = embed([question], "search_query")[0]
    hits = client.query_points(COLLECTION, query=qvec, limit=k, with_payload=True).points

    # 2. AUGMENT — assemble the retrieved text into a context block
    context = "\n".join(f"- {h.payload['text']}" for h in hits)

    # 3. GENERATE — grounded prompt
    system = (
        "You are a helpful grocery assistant for an Indian foodie living in the "
        "Netherlands. Answer ONLY using the provided context. If the answer is not "
        "in the context, say you don't have that information. Be concise."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    resp = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=300,
    )
    return resp.message.content[0].text

if __name__ == "__main__":
    q = "What curry can I make tonight?"
    print("Q:", q, "\n")
    print("A:", answer(q))