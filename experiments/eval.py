import os

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_hybrid"
K = 5
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")

# Golden set: each query + substrings that mark a retrieved item as "relevant".
# (Edit / add your own real questions — you're the foodie; this is the real skill.)
GOLDEN = [
    {"query": "clarified butter for indian cooking", "relevant": ["ghee"]},
    {"query": "Amul ghee",                           "relevant": ["amul ghee"]},
    {"query": "fresh cheese for palak paneer",       "relevant": ["paneer 200g", "paneer, kle"]},
    {"query": "long grain rice for biryani",         "relevant": ["basmati"]},
    {"query": "Tilda basmati",                       "relevant": ["tilda"]},
    {"query": "red lentils for dal",                 "relevant": ["lentil", "linzen", "dal", "toor"]},
    {"query": "coconut milk for curry",              "relevant": ["coconut milk", "kokos"]},
    {"query": "garam masala spice blend",            "relevant": ["garam masala"]},
    {"query": "creamy mild curry recipe",            "relevant": ["korma", "palak paneer:", "coconut vegetable curry"]},
]


def embed_query(q: str) -> list[float]:
    res = co.embed(texts=[q], model="embed-v4.0", input_type="search_query",
                   output_dimension=1024, embedding_types=["float"])
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]


def is_relevant(text: str, substrings: list[str]) -> bool:
    t = text.lower()
    return any(s.lower() in t for s in substrings)


def semantic_only(query: str, k: int):
    return client.query_points(COLLECTION, query=embed_query(query),
                               using="dense", limit=k, with_payload=True).points


def hybrid(query: str, k: int):
    return client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=embed_query(query), using="dense", limit=20),
            models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"),
                            using="bm25", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k, with_payload=True,
    ).points

RERANK_MODEL = "rerank-v4.0-fast"

def rerank_method(query: str, k: int):
    # retrieve wide, then rerank down to k
    cands = client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=embed_query(query), using="dense", limit=20),
            models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"),
                            using="bm25", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=20, with_payload=True,
    ).points
    if not cands:
        return []
    docs = [c.payload["text"] for c in cands]
    resp = co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=k)
    return [cands[r.index] for r in resp.results]


def load_all_texts() -> list[str]:
    """Pull every item's text so we can count total relevant per query."""
    texts, offset = [], None
    while True:
        pts, offset = client.scroll(COLLECTION, limit=100, offset=offset, with_payload=True)
        texts.extend(p.payload["text"] for p in pts)
        if offset is None:
            break
    return texts


def evaluate(search_fn, all_texts: list[str], k: int):
    recalls, rrs, hits = [], [], []
    for case in GOLDEN:
        total_relevant = sum(is_relevant(t, case["relevant"]) for t in all_texts)
        results = search_fn(case["query"], k)
        flags = [is_relevant(h.payload["text"], case["relevant"]) for h in results]

        found = sum(flags)
        recalls.append(found / total_relevant if total_relevant else 0.0)
        hits.append(1.0 if any(flags) else 0.0)
        rr = next((1.0 / i for i, f in enumerate(flags, 1) if f), 0.0)
        rrs.append(rr)

    n = len(GOLDEN)
    return sum(recalls) / n, sum(rrs) / n, sum(hits) / n


if __name__ == "__main__":
    all_texts = load_all_texts()
    print(f"Evaluating over {len(GOLDEN)} golden queries, k={K}, corpus={len(all_texts)} items\n")
    print(f"{'method':<16}{'recall@k':>10}{'MRR':>8}{'hit@k':>8}")
    print("-" * 42)
    for name, fn in [
        ("semantic-only", semantic_only),
        ("hybrid (RRF)", hybrid),
        ("hybrid+rerank", rerank_method),
    ]:
        r, m, h = evaluate(fn, all_texts, K)
        print(f"{name:<16}{r:>10.3f}{m:>8.3f}{h:>8.3f}")