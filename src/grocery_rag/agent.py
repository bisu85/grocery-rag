import json
from contextlib import nullcontext
from grocery_rag.clients import co, anthropic_client, langfuse
from grocery_rag.config import GEN_MODEL, CLAUDE_MODEL, CHAT_PROVIDER, MAX_AGENT_STEPS, MAX_AGENT_TOOL_CALLS, MAX_REFLECTIONS
from grocery_rag.tools import TOOL_FUNCTIONS, TOOLS, ANTHROPIC_TOOLS
from grocery_rag.memory import recall_facts, remember_facts, extract_facts
from langfuse import observe, propagate_attributes   # add propagate_attributes

SYSTEM = (
    "You are a cooking and grocery assistant for an Indian foodie in the Netherlands. "
    "You have tools to look up recipe ingredients, search the catalogue, and check prices. "
    "Use ONLY information returned by the tools. Never invent prices, quantities, package "
    "sizes, weights, or totals. If a price is missing, say so explicitly and exclude it from "
    "any total. If you cannot answer from tool results, say what you don't have. "
    "Do not estimate or use outside knowledge. Answer concisely."
    "Conversation summaries and recalled user facts provided to you are trusted context from "
    "earlier in this conversation — treat them as reliable and do not disclaim or ask the user "
    "to re-verify them. (This is separate from tool results, which remain your source for prices "
    "and ingredients.) "
)

@observe(name="grocery-agent")
async def run_agent(question: str, history: list[dict] | None = None,
                    session_id: str | None = None, user_id: str | None = None,
                    summary: str = "") -> dict:
    history = history or []
    ctx = propagate_attributes(session_id=session_id) if session_id else nullcontext()
    with ctx:
        recalled = await recall_facts(user_id, question) if user_id else []          # READ path
        memory_context = ""
        if summary:                                                   # summary as OWN memory, trusted
            memory_context += ("\n\nSummary of earlier conversation "
                               "(your own memory from this session — treat as reliable):\n" + summary)
        if recalled:
            memory_context += ("\n\nKnown facts about this user (respect these):\n" +
                               "\n".join(f"- {f}" for f in recalled))
        if CHAT_PROVIDER == "anthropic":
            result = await _run_agent_anthropic(question, history, memory_context)
            saved = await extract_facts(question) if user_id else []                      # WRITE path
            if saved:
                await remember_facts(user_id, saved)

            result["facts_recalled"] = recalled
            result["facts_saved"] = saved
            langfuse.update_current_span(input={"question": question}, output={"answer": result["answer"]})
            return result
        return await _run_agent_cohere(question, history, memory_context)


async def _run_agent_cohere(question: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    tools_used: list[str] = []
    captured_plan = None
    steps = 0
    tool_calls = 0

    res = await co.chat(model=GEN_MODEL, messages=messages, tools=TOOLS)
    while res.message.tool_calls:
        steps += 1                                    # count rounds (real, not faked)
        captured_plan = res.message.tool_plan
        messages.append({
            "role": "assistant",
            "tool_calls": res.message.tool_calls,
            "tool_plan": res.message.tool_plan,
        })
        for tc in res.message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = await TOOL_FUNCTIONS[tc.function.name](**args)
            tool_calls += 1                           # count executions
            tools_used.append(tc.function.name)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })
        res = await co.chat(model=GEN_MODEL, messages=messages, tools=TOOLS)

    return {
        "answer": res.message.content[0].text,
        "tool_plan": captured_plan,
        "tools_used": tools_used,
        "steps": steps,
        "tool_calls": tool_calls,
        "stopped_on": "unbudgeted",   # honest: real counts, but NO leash enforces limits here
    }

PLAN_TOOL = [{
    "name": "submit_plan",
    "description": "Record your step-by-step plan. One entry per tool call you intend to make.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string",
                                 "enum": ["get_recipe_ingredients", "search_products", "check_prices"],
                                 "description": "Which tool this step uses."},
                        "reason": {"type": "string", "description": "Why this step is needed."},
                    },
                    "required": ["tool", "reason"],
                },
            },
        },
        "required": ["steps"],
    },
}]

PLANNER = (
    "You are the planner for a grocery/cooking assistant. Given the question and the tools "
    "get_recipe_ingredients, search_products, check_prices, decide the minimal ordered set of "
    "tool calls to answer it, and record them via submit_plan. Plan only — do not answer."
)

async def _make_plan(question: str, history: list[dict]) -> tuple[list[dict], str]:
    with langfuse.start_as_current_observation(
        as_type="generation", name="plan", model=CLAUDE_MODEL
    ) as gen:
        res = await anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=400, system=PLANNER,
            tools=PLAN_TOOL,
            tool_choice={"type": "tool", "name": "submit_plan"},
            messages=history + [{"role": "user", "content": question}],
        )
        block = next(b for b in res.content if b.type == "tool_use")
        steps = block.input["steps"]
        plan_text = "\n".join(f"{i}. {s['tool']} — {s['reason']}" for i, s in enumerate(steps, 1))
        gen.update(input=question, output=steps,
                   usage_details={"input": res.usage.input_tokens,
                                  "output": res.usage.output_tokens})
    return steps, plan_text


REFLECTOR = (
    "You are a strict reviewer of an assistant's answer to a grocery/cooking question. "
    "Judge whether the answer is fully supported BY THE TOOL RESULTS provided. "
    "Reply 'OK' if it is. Otherwise reply with ONE short sentence naming the single most important "
    "gap or unsupported claim to fix. Never reward adding information the tools did not return."
)

async def _reflect(question: str, answer: str) -> str | None:
    prompt = f"Question: {question}\n\nAnswer:\n{answer}"
    with langfuse.start_as_current_observation(
        as_type="generation", name="reflect", model=CLAUDE_MODEL
    ) as gen:
        res = await anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=150, system=REFLECTOR,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = "".join(b.text for b in res.content if b.type == "text").strip()
        gen.update(
            input=prompt,
            output=verdict,
            usage_details={"input": res.usage.input_tokens, "output": res.usage.output_tokens},
        )
    return None if verdict.upper().startswith("OK") else verdict


async def _run_agent_anthropic(question: str, history: list[dict], memory_context: str = "") -> dict:
    plan_steps, plan_text = await _make_plan(question, history)
    exec_system = (
        SYSTEM + memory_context +                                   # ← recalled facts steer the answer
        "\n\nYou have already made this plan:\n" + plan_text +
        "\nFollow it, calling the tools as needed, then give the final answer."
    )
    messages = history + [{"role": "user", "content": question}]
    tools_used: list[str] = []
    steps = 0
    tool_calls = 0
    stopped_on = "completed"

    WRAP_UP = (
    " You have run out of tool budget. Do not request more tools. Answer as fully as "
    "you can from the information already gathered, and clearly state anything you could not determine."
    )

    async def ask(tool_choice=None, system=exec_system):
        kwargs = dict(model=CLAUDE_MODEL, max_tokens=1024, system=system,
                      tools=ANTHROPIC_TOOLS, messages=messages)
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        with langfuse.start_as_current_observation(
            as_type="generation", name="claude", model=CLAUDE_MODEL
        ) as gen:
            res = await anthropic_client.messages.create(**kwargs)
            gen.update(
                input=list(messages),                          # what went in (snapshot)
                output=[b.model_dump() for b in res.content],  # what came back
                usage_details={"input": res.usage.input_tokens,
                               "output": res.usage.output_tokens},
            )
        return res

    async def execute() -> str:                       # runs messages to a final answer
        nonlocal steps, tool_calls, stopped_on
        res = await ask()
        while res.stop_reason == "tool_use":
            if steps >= MAX_AGENT_STEPS:
                stopped_on = "step_budget"
                res = await ask(tool_choice={"type": "none"}, system=exec_system + WRAP_UP)
                break
            if tool_calls >= MAX_AGENT_TOOL_CALLS:
                stopped_on = "tool_budget"
                res = await ask(tool_choice={"type": "none"}, system=exec_system + WRAP_UP)
                break
            steps += 1
            messages.append({"role": "assistant", "content": res.content})
            tool_results = []
            for block in res.content:
                if block.type != "tool_use":
                    continue
                if tool_calls >= MAX_AGENT_TOOL_CALLS:
                    stopped_on = "tool_budget"
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                         "content": "Skipped: tool-call budget exhausted. Do not retry.",
                                         "is_error": True})
                    continue
                tool_calls += 1
                tools_used.append(block.name)
                try:
                    result = await TOOL_FUNCTIONS[block.name](**block.input)
                    content, is_error = json.dumps(result), False
                except Exception as e:
                    content, is_error = f"Tool '{block.name}' failed: {e}", True
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": content, "is_error": is_error})
            messages.append({"role": "user", "content": tool_results})
            res = await ask()
        messages.append({"role": "assistant", "content": res.content})   # ← keep history valid for reflection
        return "".join(b.text for b in res.content if b.type == "text")

    answer = await execute()                          # ── first pass ──

    reflections = 0                                   # ── REFLECT phase (bounded) ──
    while reflections < MAX_REFLECTIONS:
        critique = await _reflect(question, answer)
        if critique is None:                          # critic said OK
            break
        reflections += 1
        messages.append({"role": "user",
                         "content": f"A reviewer flagged an issue: {critique}\n"
                                    f"Fix it (use tools if needed) and give the corrected final answer."})
        answer = await execute()                      # revise — same budgets keep accruing

    # (b) after the reflection loop, before the return dict:
    planned = [s["tool"] for s in plan_steps]
    plan_adherence = {
        "planned": planned,
        "used": tools_used,
        "skipped": sorted(set(planned) - set(tools_used)),
        "unplanned": sorted(set(tools_used) - set(planned)),
    }
    langfuse.update_current_span(metadata={"plan_adherence": plan_adherence})   # note: update_current_span (your v4 fix)


    # after reflection loop, before building the return dict:
    if stopped_on == "completed" and (tool_calls >= MAX_AGENT_TOOL_CALLS or steps >= MAX_AGENT_STEPS):
        stopped_on = "budget_reached"   # finished, but pressed against the wall — not truly clean

    return {
        "answer": answer,
        "tool_plan": plan_text,
        "plan_adherence": plan_adherence,                  # (c) new key
        "tools_used": tools_used,
        "steps": steps, "tool_calls": tool_calls,
        "stopped_on": stopped_on, "reflections": reflections,
    }