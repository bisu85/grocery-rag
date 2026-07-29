from grocery_rag.clients import co
from grocery_rag.config import GEN_MODEL

SYSTEM = (
    "You are a grocery assistant for an Indian foodie in the Netherlands. "
    "Answer ONLY using the provided context. If the answer isn't in the context, "
    "say you don't have that information. Be concise."
)


async def generate(query: str, sources: list[dict]) -> str:
    ctx = "\n".join(f"- {s['text']}" for s in sources)
    resp = await co.chat(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}],
        max_tokens=300,
    )
    return resp.message.content[0].text
