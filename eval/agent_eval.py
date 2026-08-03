import asyncio, json, re
import httpx
from grocery_rag.clients import anthropic_client
from grocery_rag.config import CLAUDE_MODEL

API = "http://localhost:8000/ask_agent"

# ---- golden set: the four labelled failures ----
CASES = [
    {"name": "ingredients_clean",
     "turns": [{"question": "What's in palak paneer?"}],
     "expect_tools": ["get_recipe_ingredients"],
     "criteria": ["The answer does not apologize, hedge, or dump raw tool JSON."]},
    {"name": "korma_total_faithful",
     "turns": [{"question": "What does chicken korma cost in total ingredients?"}],
     "expect_tools": ["get_recipe_ingredients", "check_prices"],
     "criteria": ["The answer states only prices returned by the tools and, at most, their arithmetic sum. "
                  "It fabricates NO specific numeric quantities, weights, package sizes, or per-unit/per-kg "
                  "figures. It need NOT mention quantities, and need NOT disclaim that prices are per catalogue item."]},
    {"name": "memory_trust",
     "turns": [{"question": "What's in palak paneer?", "session_id": "eval-mem"},
               {"question": "What was the very first dish I asked about?", "session_id": "eval-mem"}],
     "criteria": ["The answer names the first dish directly, without disclaiming or refusing to trust its own memory."]},
]

# ---- deterministic checks (cheap, no LLM) ----
APOLOGY = re.compile(r"i apologi[sz]e|you'?re absolutely right|corrected final answer", re.I)

def det_checks(result: dict, case: dict) -> dict:
    a = result["answer"]
    out = {"no_apology": not APOLOGY.search(a), "no_raw_json": '{"' not in a,
           "clean_stop": result["stopped_on"] in ("completed", "budget_reached")}
    if "expect_tools" in case:
        out["tools_ok"] = set(result["tools_used"]) == set(case["expect_tools"])
    return out

# ---- LLM-as-judge: binary, over a natural-language criterion (== a RAGAS AspectCritic) ----
JUDGE = 'Evaluate whether the ANSWER satisfies the CRITERION. Reply ONLY JSON: {"pass": true|false, "reason": "<short>"}.'

async def judge(question: str, answer: str, criterion: str) -> dict:
    res = await anthropic_client.messages.create(
        model=CLAUDE_MODEL, max_tokens=300, system=JUDGE,
        messages=[{"role": "user", "content": f"CRITERION: {criterion}\nQUESTION: {question}\nANSWER:\n{answer}"}])
    t = "".join(b.text for b in res.content if b.type == "text")
    try:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in judge output")
        return json.loads(t[start:end + 1])
    except (json.JSONDecodeError, ValueError) as e:
        return {"pass": False, "reason": f"JUDGE PARSE ERROR: {e} | raw: {t[:200]!r}"}

# ---- runner ----
async def run_case(client, case):
    result = None
    for turn in case["turns"]:
        result = (await client.post(API, json=turn)).json()
    scores = det_checks(result, case)
    judgements = []
    for crit in case.get("criteria", []):
        v = await judge(case["turns"][-1]["question"], result["answer"], crit)
        scores[f"judge[{len(judgements)}]"] = bool(v.get("pass"))
        judgements.append((crit, v))
    return scores, judgements, result["answer"]

async def main():
    async with httpx.AsyncClient(timeout=90) as client:
        for case in CASES:
            scores, judgements, answer = await run_case(client, case)
            print(f"\n{case['name']}: {sum(bool(v) for v in scores.values())}/{len(scores)}")
            for k, v in scores.items():
                print(f"  [{'PASS' if v else 'FAIL'}] {k}")
            for crit, v in judgements:
                if not v.get("pass"):
                    print(f"    ↳ criterion: {crit}")
                    print(f"    ↳ reason:    {v['reason']}")
                    print(f"    ↳ answer:    {answer}…")

if __name__ == "__main__":
    asyncio.run(main())