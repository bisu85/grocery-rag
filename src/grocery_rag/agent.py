import json

from grocery_rag.clients import co
from grocery_rag.config import GEN_MODEL
from grocery_rag.tools import TOOL_FUNCTIONS, TOOLS

SYSTEM = (
    "You are a cooking and grocery assistant for an Indian foodie in the Netherlands. "
    "You have tools to look up recipe ingredients, search the product catalogue, and "
    "check prices. Use whatever tools help — you may use several to fully answer. "
    "Answer concisely."
)


async def run_agent(question: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    tools_used: list[str] = []
    captured_plan = None

    res = await co.chat(model=GEN_MODEL, messages=messages, tools=TOOLS)
    while res.message.tool_calls:
        captured_plan = res.message.tool_plan
        messages.append({
            "role": "assistant",
            "tool_calls": res.message.tool_calls,
            "tool_plan": res.message.tool_plan,
        })
        for tc in res.message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = await TOOL_FUNCTIONS[tc.function.name](**args)
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
    }