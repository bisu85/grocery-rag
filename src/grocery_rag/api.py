from fastapi import FastAPI
from pydantic import BaseModel

from grocery_rag.agent import run_agent
from grocery_rag.config import DEFAULT_K
from grocery_rag.generation import generate
from grocery_rag.retrieval import retrieve

from contextlib import asynccontextmanager
from grocery_rag.clients import langfuse
from grocery_rag.memory import get_history, append_turn


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
    steps: int = 0
    tool_calls: int = 0
    stopped_on: str = "completed"   # "completed" | "step_budget" | "tool_budget"
    plan_adherence: dict | None = None
    facts_recalled: list[str] = []
    facts_saved: list[str] = []


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
    history = await get_history(req.session_id) if req.session_id else []
    result = await run_agent(req.question, history, req.session_id, req.user_id)
    if req.session_id:
        await append_turn(req.session_id, req.question, result["answer"])
    return AgentResponse(**result)