import asyncio
from grocery_rag.clients import anthropic_client
from grocery_rag.config import CLAUDE_MODEL

async def main():
    msg = await anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=100,                     # REQUIRED on every Anthropic call
        messages=[{"role": "user", "content": "Reply with exactly: Claude is wired in."}],
    )
    # msg.content is a LIST of typed blocks, not a string — grab the text block
    text = next(b.text for b in msg.content if b.type == "text")
    print(f"stop_reason={msg.stop_reason} | {text}")

asyncio.run(main())