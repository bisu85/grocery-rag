import os

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_hybrid"
RERANK_MODEL = "rerank-v4.0-fast"
K = 5
RELEVANCE_THRESHOLD = 0.30   # rerank scores below this are treated as "not relevant"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")

# Hardened golden set. `relevant`: substrings marking a hit. Empty list = NEGATIVE
# query (nothing in the corpus should match; correct behaviour is to return nothing).
GOLDEN = [
    # -- soft (kept, as a baseline) --
    {"query": "Amul ghee", "relevant": ["amul ghee"]},
    {"query": "garam masala spice blend", "relevant": ["garam masala"]},
    # -- paraphrase / indirect (no shared keywords) --
    {"query": "what can I fry my spices in", "relevant": ["ghee", "clarified butter"]},
    {"query": "soft white cheese to cube into curry", "relevant": ["paneer"]},
    {"query": "something to thicken a curry and make it creamy", "relevant": ["coconut milk", "cream", "yogurt"]},
    {"query": "long grain rice for biryani", "relevant": ["basmati"]},
    # -- exact token / brand+size (BM25's turf) --
    {"query": "Verstegen 36g", "relevant": ["verstegen"]},
    {"query": "Tilda wholegrain", "relevant": ["tilda"]},
    # -- multilingual (Dutch) --
    {"query": "kaas voor curry", "relevant": ["paneer"]},
    {"query": "rode linzen", "relevant": ["linzen", "lentil", "dal"]},
    # -- NEGATIVE: nothing relevant exists; correct answer is to return nothing --
    {"query": "what red wine pairs with steak", "relevant": []},
    {"query": "gaming laptop under 1000 euro", "relevant": []},
]


def embed_query(q: str) -> list[float]:
    res = co.embed(texts=[q], model="embed-v4.0", input_type="search_query",
                   output_dimension=1024, embedding_types=["float"])
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]


def is_relevant(text: str, subs: list[str]) -> bool:
    t = text.lower()
    return any(s.lower() in t for s in subs)


def semantic_only(query: str, k: int):
    hits = client.query_points(COLLECTION, query=embed_query(query),
                               using="dense", limit=k, with_payload=True).points
    return [h.payload["text"] for h in hits]


def hybrid(query: str, k: int):
    hits = client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=embed_query(query), using="dense", limit=20),
            models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"),
                            using="bm25", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k, with_payload=True,
    ).points
    return [h.payload["text"] for h in hits]


def hybrid_rerank(query: str, k: int):
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
    rr = co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=k)
    return [docs[r.index] for r in rr.results]


def hybrid_rerank_threshold(query: str, k: int):
    """Like hybrid_rerank, but DROP results below the relevance threshold.
    This lets the system correctly return NOTHING for unanswerable queries."""
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
    rr = co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=k)
    return [docs[r.index] for r in rr.results if r.relevance_score >= RELEVANCE_THRESHOLD]


def load_all_texts() -> list[str]:
    texts, offset = [], None
    while True:
        pts, offset = client.scroll(COLLECTION, limit=100, offset=offset, with_payload=True)
        texts.extend(p.payload["text"] for p in pts)
        if offset is None:
            break
    return texts


def evaluate(search_fn, all_texts, k):
    recalls, precisions, rrs, hits, neg_correct = [], [], [], [], []
    n_neg = 0
    for case in GOLDEN:
        subs = case["relevant"]
        results = search_fn(case["query"], k)
        flags = [is_relevant(t, subs) for t in results]

        if not subs:  # NEGATIVE query: correct behaviour is to return nothing relevant
            n_neg += 1
            neg_correct.append(1.0 if len(results) == 0 else 0.0)
            precisions.append(1.0 if len(results) == 0 else 0.0)
            continue

        total_relevant = sum(is_relevant(t, subs) for t in all_texts)
        found = sum(flags)
        recalls.append(found / total_relevant if total_relevant else 0.0)
        precisions.append(found / len(results) if results else 0.0)
        hits.append(1.0 if any(flags) else 0.0)
        rrs.append(next((1.0 / i for i, f in enumerate(flags, 1) if f), 0.0))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "recall@k": avg(recalls), "precision@k": avg(precisions),
        "MRR": avg(rrs), "hit@k": avg(hits),
        "neg_correct": avg(neg_correct),
    }


if __name__ == "__main__":
    all_texts = load_all_texts()
    n_neg = sum(1 for c in GOLDEN if not c["relevant"])
    print(f"Golden queries: {len(GOLDEN)} ({n_neg} negative), k={K}, "
          f"threshold={RELEVANCE_THRESHOLD}, corpus={len(all_texts)}\n")
    header = f"{'method':<24}{'recall@k':>10}{'precision':>11}{'MRR':>7}{'hit@k':>7}{'neg_ok':>8}"
    print(header)
    print("-" * len(header))
    for name, fn in [
        ("semantic-only", semantic_only),
        ("hybrid (RRF)", hybrid),
        ("hybrid+rerank", hybrid_rerank),
        ("hybrid+rerank+thresh", hybrid_rerank_threshold),
    ]:
        m = evaluate(fn, all_texts, K)
        print(f"{name:<24}{m['recall@k']:>10.3f}{m['precision@k']:>11.3f}"
              f"{m['MRR']:>7.3f}{m['hit@k']:>7.3f}{m['neg_correct']:>8.3f}")