from fastapi import FastAPI
from pydantic import BaseModel

from grocery_rag.agent import run_agent
from grocery_rag.config import DEFAULT_K
from grocery_rag.generation import generate
from grocery_rag.retrieval import retrieve

from contextlib import asynccontextmanager
from grocery_rag.clients import langfuse
from grocery_rag.memory import build_context, append_turn

from grocery_rag.agent_lg import run_agent_lg

from langgraph.types import Command
from langchain_core.messages import HumanMessage
from grocery_rag.agent_lg import lg_agent, _text

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    langfuse.shutdown()

app = FastAPI(title="Grocery RAG Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    k: int = DEFAULT_K
    session_id: str | None = None            # ← optional; omit it = old stateless behaviour
    user_id: str | None = None


class Source(BaseModel):
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class AgentResponse(BaseModel):
    answer: str
    tool_plan: str | None = None
    tools_used: list[str] = []
    tool_evidence: list[dict] = []
    steps: int = 0
    tool_calls: int = 0
    stopped_on: str = "completed"   # "completed" | "step_budget" | "tool_budget"
    plan_adherence: dict | None = None
    facts_recalled: list[str] = []
    facts_saved: list[str] = []

class OrderReq(BaseModel):
    question: str
    session_id: str

class ApproveReq(BaseModel):
    session_id: str
    decision: str            # "approve" | "reject"



@app.get("/")
async def root() -> dict:
    return {"service": "Grocery RAG Assistant", "docs": "/docs"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    sources = await retrieve(req.question, req.k)
    answer = await generate(req.question, sources)
    return AskResponse(answer=answer, sources=[Source(**s) for s in sources])


@app.post("/ask_agent", response_model=AgentResponse)
async def ask_agent(req: AskRequest) -> AgentResponse:
    if req.session_id:
        summary, history = await build_context(req.session_id)   # now returns (summary, recent_turns)
    else:
        summary, history = "", []
    result = await run_agent(req.question, history, req.session_id, req.user_id, summary)
    if req.session_id:
        await append_turn(req.session_id, req.question, result["answer"])
    return AgentResponse(**result)

@app.post("/ask_lg", response_model=AgentResponse)
async def ask_lg(req: AskRequest) -> AgentResponse:
    return AgentResponse(**await run_agent_lg(req.question, req.session_id, req.user_id))


@app.post("/order")
async def order(req: OrderReq) -> dict:
    config = {"configurable": {"thread_id": req.session_id}}
    result = await lg_agent.ainvoke(
        {"messages": [HumanMessage(content=req.question)], "question": req.question,
         "user_id": None, "reflections": 0}, config=config)
    if result.get("__interrupt__"):
        return {"status": "pending_approval", "session_id": req.session_id,
                "review": result["__interrupt__"][0].value}
    return {"status": "completed", "answer": _text(result["messages"][-1])}

@app.post("/approve")
async def approve(req: ApproveReq) -> dict:
    config = {"configurable": {"thread_id": req.session_id}}
    result = await lg_agent.ainvoke(Command(resume=req.decision), config=config)
    if result.get("__interrupt__"):
        return {"status": "pending_approval", "review": result["__interrupt__"][0].value}
    return {"status": "completed", "answer": _text(result["messages"][-1])}