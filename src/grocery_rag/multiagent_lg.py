"""6i — a minimal MANUAL supervisor multi-agent (LangGraph), side-by-side with the single agent.

A supervisor routes each turn to ONE specialist (cook / shopper) or FINISH.
Workers are prebuilt create_react_agent instances with a FOCUSED tool subset + prompt.
(A prebuilt shortcut exists — langgraph_supervisor.create_supervisor — but the manual
Command-routing pattern is the docs-recommended one for control, and needs no new dependency.)

Honest note (see 6k): for the grocery task this is arguably OVERKILL vs the single agent —
multi-agent is a token multiplier (supervisor routing call + each worker's own loop). Built to
learn the coordination pattern; use it for real only when specialists genuinely diverge.
"""
from typing import Annotated, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.errors import GraphRecursionError

from grocery_rag.config import CLAUDE_MODEL
from grocery_rag.tools import (
    get_recipe_ingredients as _recipe,
    search_products as _search,
    check_prices as _prices,
)

_model = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024)

def _text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))

# ---- specialist workers: focused tool subsets + focused prompts ----
_cook_tools = [
    StructuredTool.from_function(coroutine=_recipe, name="get_recipe_ingredients",
        description="Get the ingredient list to cook a named Indian dish."),
    StructuredTool.from_function(coroutine=_search, name="search_products",
        description="Search the grocery catalogue for products."),
]
_shop_tools = [
    StructuredTool.from_function(coroutine=_prices, name="check_prices",
        description="Get current prices and stores for one or more products (pass a list)."),
]

cook_agent = create_react_agent(_model, tools=_cook_tools,
    prompt="You are the COOKING specialist. Answer recipe/ingredient questions using ONLY tool "
           "results. Never invent. Be concise.")
shopper_agent = create_react_agent(_model, tools=_shop_tools,
    prompt="You are the SHOPPING specialist. Answer price/cost questions using ONLY tool results. "
           "Never invent prices. Be concise.")


# ---- supervisor: route to a specialist or finish ----
class Route(BaseModel):
    next: Literal["cook", "shopper", "FINISH"]

_router = _model.with_structured_output(Route)

SUPERVISOR = (
    "You are a supervisor coordinating two specialists: 'cook' (recipes/ingredients) and 'shopper' "
    "(prices/costs). Given the conversation so far, choose which specialist should act NEXT, or "
    "'FINISH' if the user's question is already fully answered. Route to ONE at a time. A question "
    "about both ingredients AND their cost needs cook first, then shopper."
)


class MAState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    notes: str
    used: Annotated[list, lambda a, b: a + b]        # which specialists have run (accumulates)

async def _run_worker(agent, state, label: str) -> Command:
    prior = f"\n\nFindings so far from other specialists:{state['notes']}" if state.get("notes") else ""
    task = HumanMessage(content=state["question"] + prior)      # self-contained instruction
    result = await agent.ainvoke({"messages": [task]})
    answer = _text(result["messages"][-1])
    return Command(goto="supervisor", update={
        "messages": [AIMessage(content=answer)],               # so the supervisor sees progress
        "notes": (state.get("notes", "") + f"\n[{label}] {answer}").strip(),
        "used": [label],                             # ← record which specialist ran
    })


async def supervisor_node(state: MAState) -> Command[Literal["cook", "shopper", "__end__"]]:
    used = state.get("used", [])
    remaining = [w for w in ("cook", "shopper") if w not in used]
    if not remaining:                                # every specialist has had a turn → stop
        return Command(goto=END)
    decision = await _router.ainvoke(
        [SystemMessage(content=SUPERVISOR +
            f"\nSpecialists already done: {used or 'none'}. Only route to one of {remaining}, "
            "or FINISH if the question is fully answered.")]
        + state["messages"])
    nxt = decision.next
    if nxt == "FINISH" or nxt not in remaining:      # guard against re-routing a used specialist
        return Command(goto=END if nxt == "FINISH" else remaining[0])
    return Command(goto=nxt)

async def cook_node(state: MAState) -> Command[Literal["supervisor"]]:
    return await _run_worker(cook_agent, state, "cook")

async def shopper_node(state: MAState) -> Command[Literal["supervisor"]]:
    return await _run_worker(shopper_agent, state, "shopper")


_g = StateGraph(MAState)
_g.add_node("supervisor", supervisor_node)
_g.add_node("cook", cook_node)
_g.add_node("shopper", shopper_node)
_g.add_edge(START, "supervisor")            # workers return Command(goto="supervisor"); no static edges needed
multiagent = _g.compile()


async def run_multiagent(question: str) -> dict:
    try:
        result = await multiagent.ainvoke(
            {"messages": [HumanMessage(content=question)], "question": question, "notes": ""},
            config={"recursion_limit": 12})
    except GraphRecursionError:
        return {"answer": "(supervisor did not converge)"}
    return {"answer": result.get("notes", "").strip() or _text(result["messages"][-1])}
