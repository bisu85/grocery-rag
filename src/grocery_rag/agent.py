import json

from grocery_rag.clients import co, anthropic_client
from grocery_rag.config import GEN_MODEL, CLAUDE_MODEL, CHAT_PROVIDER, MAX_AGENT_STEPS, MAX_AGENT_TOOL_CALLS
from grocery_rag.tools import TOOL_FUNCTIONS, TOOLS, ANTHROPIC_TOOLS

SYSTEM = (
    "You are a cooking and grocery assistant for an Indian foodie in the Netherlands. "
    "You have tools to look up recipe ingredients, search the product catalogue, and "
    "check prices. Use whatever tools help — you may use several to fully answer. "
    "Answer concisely."
)


async def run_agent(question: str) -> dict:
    if CHAT_PROVIDER == "anthropic":
        return await _run_agent_anthropic(question)
    return await _run_agent_cohere(question)


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


async def _run_agent_anthropic(question: str) -> dict:
    messages = [{"role": "user", "content": question}]
    tools_used: list[str] = []
    captured_plan = None
    steps = 0          # model round-trips (serial depth)
    tool_calls = 0     # total tool executions, incl. parallel ones inside a round (fan-out)
    stopped_on = "completed"

    WRAP_UP = (
    " You have run out of tool budget. Do not request more tools. Answer as fully as "
    "you can from the information already gathered, and clearly state anything you could not determine."
    )

    async def ask(tool_choice=None, system=SYSTEM):          # ← add system param
        kwargs = dict(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system,
            tools=ANTHROPIC_TOOLS, 
            messages=messages,
            )
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return await anthropic_client.messages.create(**kwargs)

    res = await ask()
    while res.stop_reason == "tool_use":
        # ── the leashes: check BEFORE another round ──
        if steps >= MAX_AGENT_STEPS:
            stopped_on = "step_budget"
            res = await ask(tool_choice={"type": "none"}, system=SYSTEM + WRAP_UP)   # ← nudge
            break
        if tool_calls >= MAX_AGENT_TOOL_CALLS:
            stopped_on = "tool_budget"
            res = await ask(tool_choice={"type": "none"}, system=SYSTEM + WRAP_UP)   # ← nudge
            break
        steps += 1

        plan = next((b.text for b in res.content if b.type == "text"), None)
        if plan:
            captured_plan = plan

        messages.append({"role": "assistant", "content": res.content})

        tool_results = []                                   # ALL results in ONE user message
        for block in res.content:
            if block.type != "tool_use":
                continue

            if tool_calls >= MAX_AGENT_TOOL_CALLS:
                # over budget mid-round: the API still requires a result for THIS block,
                # so fabricate an error result instead of executing the tool.
                stopped_on = "tool_budget"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Skipped: tool-call budget (MAX_AGENT_TOOL_CALLS) exhausted. Do not retry.",
                    "is_error": True,
                })
                continue

            tool_calls += 1
            tools_used.append(block.name)
            try:
                result = await TOOL_FUNCTIONS[block.name](**block.input)
                content, is_error = json.dumps(result), False
            except Exception as e:
                content, is_error = f"Tool '{block.name}' failed: {e}", True
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": tool_results})
        res = await ask()

    answer = "".join(b.text for b in res.content if b.type == "text")
    return {
        "answer": answer,
        "tool_plan": captured_plan,
        "tools_used": tools_used,
        "steps": steps,
        "tool_calls": tool_calls,
        "stopped_on": stopped_on,
    }