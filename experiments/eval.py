"""Hardened retrieval eval — now imports from the grocery_rag package.

Because retrieval/rerank live in the package, this script automatically uses
whatever RERANK_PROVIDER is set in grocery_rag/config.py (local vs cohere).
No duplicated pipeline code = no drift between the app and the eval.
"""

import asyncio

from qdrant_client import models

# --- one source of truth: import the SAME pipeline the app uses ---
from grocery_rag.clients import co, qdrant
from grocery_rag.config import (
    COLLECTION,
    EMBED_DIM,
    EMBED_MODEL,
    RELEVANCE_THRESHOLD,
)
from grocery_rag.retrieval import embed_query, hybrid_candidates, retrieve

K = 5

# Hardened golden set. `relevant`: substrings marking a hit. Empty list = NEGATIVE
# query (nothing in the corpus should match; correct behaviour is to return nothing).
GOLDEN = [
    # -- soft baseline --
    {"query": "Amul ghee", "relevant": ["amul ghee"]},
    {"query": "garam masala spice blend", "relevant": ["garam masala"]},
    # -- paraphrase / indirect (no shared keywords) --
    {"query": "what can I fry my spices in", "relevant": ["ghee", "clarified butter"]},
    {"query": "soft white cheese to cube into curry", "relevant": ["paneer"]},
    {"query": "something to thicken a curry and make it creamy",
     "relevant": ["coconut milk", "cream", "yogurt"]},
    {"query": "long grain rice for biryani", "relevant": ["basmati"]},
    # -- exact token / brand+size (BM25's turf) --
    {"query": "Verstegen 36g", "relevant": ["verstegen"]},
    {"query": "Tilda wholegrain", "relevant": ["tilda"]},
    # -- multilingual (Dutch) --
    {"query": "kaas voor curry", "relevant": ["paneer"]},
    {"query": "rode linzen", "relevant": ["linzen", "lentil", "dal"]},
    # -- NEGATIVE: nothing relevant exists --
    {"query": "what red wine pairs with steak", "relevant": []},
    {"query": "gaming laptop under 1000 euro", "relevant": []},
]


def is_relevant(text: str, subs: list[str]) -> bool:
    t = text.lower()
    return any(s.lower() in t for s in subs)


# ---- the methods to compare (all async now, since the package is async) ----
async def semantic_only(query: str, k: int) -> list[str]:
    hits = (await qdrant.query_points(
        COLLECTION, query=await embed_query(query),
        using="dense", limit=k, with_payload=True,
    )).points
    return [h.payload["text"] for h in hits]


async def hybrid(query: str, k: int) -> list[str]:
    hits = (await qdrant.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=await embed_query(query), using="dense", limit=20),
            models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"),
                            using="bm25", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k, with_payload=True,
    )).points
    return [h.payload["text"] for h in hits]


async def hybrid_rerank(query: str, k: int) -> list[str]:
    # retrieve() already does hybrid -> rerank via the package (honors RERANK_PROVIDER)
    results = await retrieve(query, k)
    return [r["text"] for r in results]


async def hybrid_rerank_threshold(query: str, k: int) -> list[str]:
    # same as above, but drop anything below the calibrated relevance threshold
    results = await retrieve(query, k)
    return [r["text"] for r in results if r["score"] >= RELEVANCE_THRESHOLD]


async def load_all_texts() -> list[str]:
    texts: list[str] = []
    offset = None
    while True:
        pts, offset = await qdrant.scroll(
            COLLECTION, limit=100, offset=offset, with_payload=True
        )
        texts.extend(p.payload["text"] for p in pts)
        if offset is None:
            break
    return texts


async def evaluate(search_fn, all_texts: list[str], k: int) -> dict:
    recalls, precisions, rrs, hits, neg_correct = [], [], [], [], []
    for case in GOLDEN:
        subs = case["relevant"]
        results = await search_fn(case["query"], k)
        flags = [is_relevant(t, subs) for t in results]

        if not subs:  # NEGATIVE query: correct behaviour is to return nothing relevant
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
        "MRR": avg(rrs), "hit@k": avg(hits), "neg_correct": avg(neg_correct),
    }


async def main() -> None:
    all_texts = await load_all_texts()
    n_neg = sum(1 for c in GOLDEN if not c["relevant"])
    print(f"Golden queries: {len(GOLDEN)} ({n_neg} negative), k={K}, "
          f"threshold={RELEVANCE_THRESHOLD}, corpus={len(all_texts)}\n")
    header = (f"{'method':<24}{'recall@k':>10}{'precision':>11}"
              f"{'MRR':>7}{'hit@k':>7}{'neg_ok':>8}")
    print(header)
    print("-" * len(header))
    for name, fn in [
        ("semantic-only", semantic_only),
        ("hybrid (RRF)", hybrid),
        ("hybrid+rerank", hybrid_rerank),
        ("hybrid+rerank+thresh", hybrid_rerank_threshold),
    ]:
        m = await evaluate(fn, all_texts, K)
        print(f"{name:<24}{m['recall@k']:>10.3f}{m['precision@k']:>11.3f}"
              f"{m['MRR']:>7.3f}{m['hit@k']:>7.3f}{m['neg_correct']:>8.3f}")


if __name__ == "__main__":
    asyncio.run(main())