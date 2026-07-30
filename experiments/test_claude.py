import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment automatically

resp = client.messages.create(
    model="claude-haiku-4-5",           # cheapest model — a few hundredths of a cent
    max_tokens=20,
    messages=[{"role": "user", "content": "Reply with exactly: key works"}],
)
print(resp.content[0].text)