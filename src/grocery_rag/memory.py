"""Memory: durable short-term session history (SQLite) + long-term semantic fact memory (Qdrant).
Short-term = this conversation's turns (durable across restarts).
Long-term  = user facts/preferences, retrieved by similarity — RAG pointed inward."""
import sqlite3
import asyncio
import uuid

from qdrant_client import models

from grocery_rag.clients import qdrant, anthropic_client, langfuse
from grocery_rag.config import (
    SESSION_DB, MEMORY_COLLECTION, MEMORY_TOP_K, EMBED_DIM, CLAUDE_MODEL,
)
from grocery_rag.retrieval import embed_query, embed_texts


# ---------- short-term: durable session history (SQLite) ----------
def _init_db() -> None:
    con = sqlite3.connect(SESSION_DB)
    con.execute("CREATE TABLE IF NOT EXISTS turns (session_id TEXT, role TEXT, content TEXT)")
    con.commit(); con.close()

_init_db()

def _get_history_sync(session_id: str) -> list[dict]:
    con = sqlite3.connect(SESSION_DB)
    rows = con.execute("SELECT role, content FROM turns WHERE session_id=? ORDER BY rowid",
                       (session_id,)).fetchall()
    con.close()
    return [{"role": r, "content": c} for r, c in rows]

def _append_turn_sync(session_id: str, question: str, answer: str) -> None:
    con = sqlite3.connect(SESSION_DB)
    con.executemany("INSERT INTO turns (session_id, role, content) VALUES (?, ?, ?)",
                    [(session_id, "user", question), (session_id, "assistant", answer)])
    con.commit(); con.close()

async def get_history(session_id: str) -> list[dict]:          # now async (was a dict lookup)
    return await asyncio.to_thread(_get_history_sync, session_id)

async def append_turn(session_id: str, question: str, answer: str) -> None:
    await asyncio.to_thread(_append_turn_sync, session_id, question, answer)


# ---------- long-term: semantic fact memory (Qdrant) ----------
async def _ensure_memory_collection() -> None:
    if not await qdrant.collection_exists(MEMORY_COLLECTION):
        await qdrant.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )

async def remember_facts(user_id: str, facts: list[str]) -> None:
    if not facts:
        return
    await _ensure_memory_collection()
    vecs = await embed_texts(facts, "search_document")
    await qdrant.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=v,
                                   payload={"user_id": user_id, "fact": f})
                for v, f in zip(vecs, facts)],
    )

async def recall_facts(user_id: str, query: str, k: int = MEMORY_TOP_K) -> list[str]:
    await _ensure_memory_collection()
    with langfuse.start_as_current_observation(as_type="span", name="recall_facts") as span:
        qvec = await embed_query(query)
        res = await qdrant.query_points(
            collection_name=MEMORY_COLLECTION, query=qvec,
            query_filter=models.Filter(must=[                       # only THIS user's facts
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]),
            with_payload=True, limit=k,
        )
        facts = [p.payload["fact"] for p in res.points]
        span.update(input=query, output=facts)
    return facts


# ---------- write path: fact extraction (LLM, structured via forced tool) ----------
EXTRACTOR = (
    "Extract durable personal facts or preferences about the user from their message worth remembering "
    "for future grocery/cooking help — diet, allergies, budget, preferred stores, dislikes, household "
    "size. ONLY lasting facts, never one-off requests. If none, return an empty list."
)
MEMORY_TOOL = [{
    "name": "save_facts",
    "description": "Save durable user facts/preferences worth remembering across sessions.",
    "input_schema": {"type": "object",
        "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
        "required": ["facts"]},
}]

async def extract_facts(question: str) -> list[str]:
    with langfuse.start_as_current_observation(
        as_type="generation", name="extract_facts", model=CLAUDE_MODEL) as gen:
        res = await anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=200, system=EXTRACTOR,
            tools=MEMORY_TOOL, tool_choice={"type": "tool", "name": "save_facts"},
            messages=[{"role": "user", "content": question}],
        )
        block = next(b for b in res.content if b.type == "tool_use")
        facts = block.input.get("facts", [])
        gen.update(input=question, output=facts,
                   usage_details={"input": res.usage.input_tokens, "output": res.usage.output_tokens})
    return facts