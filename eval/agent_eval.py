import asyncio, json, re
import httpx
from grocery_rag.clients import anthropic_client
from grocery_rag.config import CLAUDE_MODEL

API = "http://localhost:8000/ask_agent"
#API = "http://localhost:8000/ask_lg"

# ---- golden set: the four labelled failures ----
CASES = [
    # {"name": "ingredients_clean",
    #  "turns": [{"question": "What's in palak paneer?"}],
    #  "expect_tools": ["get_recipe_ingredients"],
    #  "criteria": ["The answer does not apologize, hedge, or dump raw tool JSON."]},
    {"name": "korma_total_faithful",
     "turns": [{"question": "What does chicken korma cost in total ingredients?"}],
     "expect_tools": ["get_recipe_ingredients", "check_prices"],
     "faithfulness": True},
    # {"name": "memory_trust",
    #  "turns": [{"question": "What's in palak paneer?", "session_id": "eval-mem"},
    #            {"question": "What was the very first dish I asked about?", "session_id": "eval-mem"}],
    #  "criteria": ["The answer names the first dish directly, without disclaiming or refusing to trust its own memory."]},
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


FAITH_JUDGE = (
    "You compare an ANSWER to EVIDENCE (tool outputs). Decide if EVERY factual claim in the answer "
    "(prices, stores, ingredients, totals) is supported by the evidence; a total is supported if it "
    "equals the sum of the evidence prices. "
    "First reason through each claim and compute the sum, then decide. "
    "Reply with ONE JSON object and NOTHING else — no fences, no text around it. "
    "Put your reasoning in reason, and set pass LAST, consistent with your reasoning: "
    '{"reason": "<claim-by-claim check + the sum>", "pass": true}'
)

def _extract_json(text: str) -> dict:
    """Pull the first complete JSON object out of a model reply, tolerating fences/prose around it."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    obj, _ = decoder.raw_decode(text[start:])   # parses the first object, ignores any trailing prose
    return obj

async def faithfulness_check(answer: str, tool_evidence: list[dict]) -> dict:
    ev = json.dumps(tool_evidence) or "(none)"
    res = await anthropic_client.messages.create(
        model=CLAUDE_MODEL, max_tokens=500, system=FAITH_JUDGE,
        messages=[{"role": "user", "content": f"EVIDENCE:\n{ev}\n\nANSWER:\n{answer}"}])
    t = "".join(b.text for b in res.content if b.type == "text")
    try:
        verdict = _extract_json(t)
        if not isinstance(verdict.get("pass"), bool):
            return {"pass": False, "reason": f"NO CLEAN VERDICT | raw: {t[:200]!r}"}
        # guard: flag reason/verdict disagreement instead of trusting a contradicted boolean
        r = verdict["reason"].lower()
        if verdict["pass"] is False and ("is correct" in r or "actually correct" in r or "all factual claims" in r and "supported" in r):
            verdict["reason"] += "  [NOTE: reason contradicts pass=false — treat as suspect]"
        return verdict
    except (json.JSONDecodeError, ValueError) as e:
        return {"pass": False, "reason": f"PARSE ERROR: {e} | raw: {t[:200]!r}"}


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
    if case.get("faithfulness"):
        v = await faithfulness_check(result["answer"], result.get("tool_evidence", []))
        scores["faithfulness"] = bool(v.get("pass"))
        judgements.append(("faithfulness (evidence-grounded)", v))
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