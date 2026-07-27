import json
import os

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION = "grocery_hybrid"
GEN_MODEL = "command-a-03-2025"
JUDGE_MODEL = "command-a-03-2025"
RERANK_MODEL = "rerank-v4.0-fast"
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
client = QdrantClient(url="http://localhost:6333")


def embed_query(q: str) -> list[float]:
    res = co.embed(texts=[q], model="embed-v4.0", input_type="search_query",
                   output_dimension=1024, embedding_types=["float"])
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]


def retrieve(query: str, k: int = 4) -> list[str]:
    """Best pipeline: hybrid retrieve wide, rerank down to k."""
    cands = client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=embed_query(query), using="dense", limit=15),
            models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"),
                            using="bm25", limit=15),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=15, with_payload=True,
    ).points
    docs = [c.payload["text"] for c in cands]
    if not docs:
        return []
    rr = co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=k)
    return [docs[r.index] for r in rr.results]


def generate(query: str, context: list[str]) -> str:
    system = (
        "You are a grocery assistant for an Indian foodie in the Netherlands. "
        "Answer ONLY using the provided context. If the answer isn't in the context, "
        "say you don't have that information. Be concise."
    )
    ctx = "\n".join(f"- {c}" for c in context)
    resp = co.chat(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}],
        max_tokens=300,
    )
    return resp.message.content[0].text


def parse_json(text: str) -> dict:
    """LLMs sometimes wrap JSON in ```fences; strip them before parsing."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def judge(question: str, context: list[str], answer: str) -> dict:
    ctx = "\n".join(f"- {c}" for c in context)
    prompt = f"""You are an evaluator of a RAG system's answer. Be strict and objective.

QUESTION:
{question}

CONTEXT PROVIDED TO THE SYSTEM:
{ctx}

THE SYSTEM'S ANSWER:
{answer}

Do two things:
1. FAITHFULNESS — break the answer into individual factual claims. For each claim,
   decide if it is SUPPORTED by the context above (true/false). A refusal like
   "I don't have that information" counts as zero claims.
2. RELEVANCE — rate 0.0 to 1.0 how well the answer addresses the QUESTION.

Respond with ONLY this JSON, no prose:
{{
  "claims": [{{"claim": "...", "supported": true}}],
  "relevance": 0.0,
  "relevance_reason": "..."
}}"""
    resp = co.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0,  # judging should be deterministic
    )
    data = parse_json(resp.message.content[0].text)
    claims = data.get("claims", [])
    supported = sum(1 for c in claims if c.get("supported"))
    # faithfulness = supported / total; a refusal (no claims) is trivially faithful -> 1.0
    data["faithfulness"] = supported / len(claims) if claims else 1.0
    data["n_claims"] = len(claims)
    return data


QUESTIONS = [
    "What can I use to make palak paneer?",
    "Suggest a vegan curry I can cook tonight.",
    "What clarified butter do you stock?",
    "What red wine pairs with steak?",   # trap: not in the corpus
]

if __name__ == "__main__":
    faiths, rels = [], []
    for q in QUESTIONS:
        ctx = retrieve(q, k=4)
        ans = generate(q, ctx)
        v = judge(q, ctx, ans)
        faiths.append(v["faithfulness"])
        rels.append(v["relevance"])
        print(f"\n{'=' * 70}\nQ: {q}")
        print(f"A: {ans}")
        print(f"   faithfulness={v['faithfulness']:.2f} ({v['n_claims']} claims)  "
              f"relevance={v['relevance']:.2f}")
        print(f"   why: {v.get('relevance_reason', '')}")

    n = len(QUESTIONS)
    print(f"\n{'=' * 70}\nAVERAGES over {n} questions:")
    print(f"   faithfulness = {sum(faiths) / n:.3f}")
    print(f"   relevance    = {sum(rels) / n:.3f}")