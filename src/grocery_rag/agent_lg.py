"""LangGraph port of the grocery agent — FULL PARITY (plan -> loop -> reflection -> memory),
running the SAME Claude model, tools, and long-term memory as the hand-built agent.py.
Lives beside agent.py so you can compare the imperative loop vs the declarative graph.

Construct mapping (hand-built  ->  LangGraph):
  while stop_reason=='tool_use'      ->  conditional edge  agent -> tools -> agent
  MAX_AGENT_STEPS leash              ->  config recursion_limit (super-step cap; RAISES on overflow)
  try/except tool-error observation  ->  ToolNode (catches exceptions -> error ToolMessages)
  forced submit_plan tool            ->  llm.with_structured_output(Plan)
  short-term SQLite session store     ->  checkpointer (MemorySaver) keyed by thread_id
  long-term Qdrant facts              ->  reused unchanged as recall/remember nodes
  reflection loop                     ->  reflect node + conditional edge back to agent
"""
from typing import Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import interrupt, Command #for Governance

from grocery_rag.config import CLAUDE_MODEL, MAX_REFLECTIONS
from grocery_rag.agent import SYSTEM, PLANNER, _reflect              # reuse prompts + critic
from grocery_rag.memory import recall_facts, remember_facts, extract_facts
from grocery_rag.tools import (
    get_recipe_ingredients as _recipe,
    search_products as _search,
    check_prices as _prices,
    place_order as _order
)

# ---- tools: wrap the EXISTING functions as LangChain tools (same logic, LC interface) ----
LC_TOOLS = [
    StructuredTool.from_function(coroutine=_recipe, name="get_recipe_ingredients",
        description="Get the ingredient list to cook a named Indian dish."),
    StructuredTool.from_function(coroutine=_search, name="search_products",
        description="Search the grocery catalogue for products matching a query."),
    StructuredTool.from_function(coroutine=_prices, name="check_prices",
        description="Get current prices and stores for one or more products (pass a list)."),
    StructuredTool.from_function(coroutine=_order, name="place_order",
        description="Place a grocery order for a list of items. A real, consequential action."),
]

_llm_tools = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024).bind_tools(LC_TOOLS)


# ---- structured plan (replaces the forced submit_plan tool) ----
class PlanStep(BaseModel):
    tool: str = Field(description="one of: get_recipe_ingredients, search_products, check_prices")
    reason: str

class Plan(BaseModel):
    steps: list[PlanStep]

_planner = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=400).with_structured_output(Plan)


# ---- graph state (the shared object every node reads/writes) ----
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # add_messages reducer = append, not overwrite
    question: str
    user_id: str | None
    memory_context: str
    plan: str
    reflections: int
    needs_revision: bool


def _text(msg) -> str:
    """Extract plain text from an AIMessage (content may be a string or a list of blocks)."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))


# ---- nodes ----
async def recall_node(state: AgentState) -> dict:                    # long-term memory READ
    uid = state.get("user_id")
    facts = await recall_facts(uid, state["question"]) if uid else []
    mc = ("\n\nKnown facts about this user (respect these):\n" +
          "\n".join(f"- {f}" for f in facts)) if facts else ""
    return {"memory_context": mc}

async def plan_node(state: AgentState) -> dict:                      # structured plan
    plan: Plan = await _planner.ainvoke(
        [SystemMessage(content=PLANNER), HumanMessage(content=state["question"])])
    text = "\n".join(f"{i}. {s.tool} — {s.reason}" for i, s in enumerate(plan.steps, 1))
    return {"plan": text}

async def agent_node(state: AgentState) -> dict:                     # call Claude with tools
    sys = (SYSTEM + state.get("memory_context", "") +
           "\n\nYou have already made this plan:\n" + state.get("plan", "") +
           "\nFollow it, calling tools as needed, then give the final answer.")
    resp = await _llm_tools.ainvoke([SystemMessage(content=sys)] + state["messages"])
    return {"messages": [resp]}

async def reflect_node(state: AgentState) -> dict:                   # bounded self-critique
    if state.get("reflections", 0) >= MAX_REFLECTIONS:
        return {"needs_revision": False}
    critique = await _reflect(state["question"], _text(state["messages"][-1]))
    if critique is None:
        return {"needs_revision": False}
    return {"messages": [HumanMessage(content=(
                f"A reviewer flagged an issue: {critique} "
                "Give ONLY the corrected final answer in natural language."))],
            "reflections": state.get("reflections", 0) + 1,
            "needs_revision": True}

async def remember_node(state: AgentState) -> dict:                  # long-term memory WRITE
    uid = state.get("user_id")
    if uid:
        facts = await extract_facts(state["question"])
        if facts:
            await remember_facts(uid, facts)
    return {}

# the governance gate
SENSITIVE_TOOLS = {"place_order"}

def review_node(state: AgentState):
    calls = state["messages"][-1].tool_calls or []
    if not any(c["name"] in SENSITIVE_TOOLS for c in calls):
        return Command(goto="tools")                       # nothing sensitive → just run tools
    decision = interrupt({                                 # PAUSE — surface to the human, wait
        "type": "approval_required",
        "pending": [{"name": c["name"], "args": c["args"]}
                    for c in calls if c["name"] in SENSITIVE_TOOLS],
    })
    if str(decision).lower() in ("approve", "yes", "y"):
        return Command(goto="tools")                       # approved → execute
    rejections = [ToolMessage(                             # rejected → answer every call, go back
        content="Human reviewer REJECTED this action; it was NOT performed. "
                "Tell the user it was cancelled; do not retry.",
        tool_call_id=c["id"]) for c in calls]
    return Command(goto="agent", update={"messages": rejections})


# ---- routers (the loop + reflection, expressed declaratively) ----
def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    #return "tools" if getattr(last, "tool_calls", None) else "reflect"
    return "review" if getattr(last, "tool_calls", None) else "reflect" # routing: send tool calls through review first

def route_after_reflect(state: AgentState) -> str:
    return "agent" if state.get("needs_revision") else "remember"


# ---- build the graph ----
_g = StateGraph(AgentState)
_g.add_node("recall", recall_node)
_g.add_node("plan", plan_node)
_g.add_node("agent", agent_node)
_g.add_node("tools", ToolNode(LC_TOOLS))
_g.add_node("reflect", reflect_node)
_g.add_node("remember", remember_node)
_g.add_node("review", review_node)

_g.add_edge(START, "recall")
_g.add_edge("recall", "plan")
_g.add_edge("plan", "agent")
_g.add_conditional_edges("agent", route_after_agent, {"review": "review", "reflect": "reflect"})
_g.add_edge("tools", "agent")
_g.add_conditional_edges("reflect", route_after_reflect, {"agent": "agent", "remember": "remember"})
_g.add_edge("remember", END)

# checkpointer = short-term memory (thread_id = session_id). MemorySaver = in-RAM (lost on restart);
# swap for langgraph.checkpoint.sqlite.SqliteSaver to match your durable SQLite store.
builder = _g                                            # expose the uncompiled builder
lg_agent = _g.compile(checkpointer=MemorySaver())       # default; overridden at startup for durability

def use_checkpointer(saver) -> None:
    """Recompile the module-level graph with a durable checkpointer (called from the app lifespan)."""
    global lg_agent
    lg_agent = builder.compile(checkpointer=saver)


# ---- convenience wrapper for the API ----
async def run_agent_lg(question: str, session_id: str | None = None,
                       user_id: str | None = None) -> dict:
    config = {"configurable": {"thread_id": session_id or "default"}}  # recursion_limit defaults to 25 (the leash)
    try:
        state = await lg_agent.ainvoke(
            {"messages": [HumanMessage(content=question)], "question": question,
             "user_id": user_id, "reflections": 0},
            config=config)
    except GraphRecursionError:
        return {"answer": "(stopped: recursion/step budget reached)",
                "tool_plan": None, "tools_used": [], "stopped_on": "step_budget"}

    if isinstance(state, dict) and state.get("__interrupt__"):
        return {"answer": "(paused for human approval — use /order + /approve)",
                "tool_plan": state.get("plan"), "tools_used": []}

    msgs = state["messages"]
    tools_used = [tc["name"] for m in msgs if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
    return {"answer": _text(msgs[-1]), "tool_plan": state.get("plan"), "tools_used": tools_used}
