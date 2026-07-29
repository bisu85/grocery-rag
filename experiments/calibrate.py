import asyncio
from grocery_rag.retrieval import hybrid_candidates, _rerank

# a clearly-relevant query and a clearly-irrelevant (negative) one
PROBES = [
    ("what can I fry my spices in", "RELEVANT (expect ghee near top)"),
    ("soft white cheese to cube into curry", "RELEVANT (expect paneer)"),
    ("what red wine pairs with steak", "NEGATIVE (expect all low)"),
    ("gaming laptop under 1000 euro", "NEGATIVE (expect all low)"),
]

async def main():
    for query, label in PROBES:
        cands = await hybrid_candidates(query)
        docs = [c.payload["text"] for c in cands]
        ranked = await _rerank(query, docs, k=5)   # returns [{text, score}, ...]
        print(f"\n{label}\n  Q: {query!r}")
        for r in ranked:
            print(f"    {r['score']:.3f}  {r['text'][:55]}")

if __name__ == "__main__":
    asyncio.run(main())