# grocery_rag/memory.py
"""Short-term conversation memory: per-session history of clean user/assistant turns.
In-process dict for now — swap the backend (Redis / DB / Qdrant) to get durable, shared,
and long-term memory later. Interface stays the same."""

_SESSIONS: dict[str, list[dict]] = {}

def get_history(session_id: str) -> list[dict]:
    return _SESSIONS.get(session_id, [])

def append_turn(session_id: str, question: str, answer: str) -> None:
    # store ONLY the clean turn — not the tool-call scratchpad
    _SESSIONS.setdefault(session_id, []).extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])