import os
from contextlib import asynccontextmanager

import cohere
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
import json

load_dotenv()

COLLECTION = "grocery_hybrid"
GEN_MODEL = "command-a-03-2025"
RERANK_MODEL = "rerank-v4.0-fast"

# Async clients — created once, reused across all requests.
co = cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])
qdrant = AsyncQdrantClient(url="http://localhost:6333")

app = FastAPI(title="Grocery RAG Assistant")

# ---- TOOL: volatile data (synthetic stand-in for a real price API/scraper) ----
PRICE_DB = {
    "spinach": (1.25, "Albert Heijn"), "paneer": (3.49, "Amazing Oriental"),
    "onion": (0.99, "Lidl"), "garlic": (0.79, "Albert Heijn"),
    "ginger": (0.89, "Albert Heijn"), "green chili": (0.99, "Amazing Oriental"),
    "garam masala": (2.19, "Jumbo"), "cream": (1.49, "Albert Heijn"),
    "toor dal": (3.20, "Amazing Oriental"), "ghee": (9.99, "Amazing Oriental"),
    "chicken": (6.49, "Lidl"), "yogurt": (1.89, "Albert Heijn"),
    "cashews": (3.99, "Jumbo"), "tomato": (1.49, "Albert Heijn"),
    "cumin": (1.79, "Jumbo"), "turmeric": (1.59, "Jumbo"),
}


async def check_price(product: str) -> dict:
    """Return the current price of a product (volatile data — lives behind a tool, not in Qdrant)."""
    key = product.strip().lower()
    for name, (price, store) in PRICE_DB.items():
        if name in key or key in name:
            return {"product": product, "price_eur": price, "store": store}
    return {"product": product, "price_eur": None, "note": "price not available"}


# ---- TOOL: your RAG retrieval, exposed so the agent can choose to search ----
async def search_products(query: str) -> dict:
    """Semantically search the grocery catalogue (hybrid + rerank)."""
    sources = await retrieve(query, k=5)   # reuses your existing pipeline
    return {"query": query, "results": [s.text for s in sources]}


# ---- TOOL: the "kitchen" (real Python that runs when the model asks) ----
RECIPE_DB = {
    "palak paneer": ["spinach", "paneer", "onion", "garlic", "ginger",
                     "green chili", "garam masala", "cream"],
    "dal tadka":    ["toor dal", "onion", "tomato", "cumin", "garlic", "ghee", "turmeric"],
    "chicken korma": ["chicken", "yogurt", "onion", "cashews", "garam masala", "cream"],
}


async def get_recipe_ingredients(dish: str) -> dict:   # add 'async'
    key = dish.strip().lower()
    return {"dish": dish, "ingredients": RECIPE_DB.get(key, [])}


TOOL_FUNCTIONS = {
    "get_recipe_ingredients": get_recipe_ingredients,
    "search_products": search_products,
    "check_price": check_price,
}

# ---- TOOL: the "menu" (what we TELL the model exists) ----
TOOLS = [
    {"type": "function", "function": {
        "name": "get_recipe_ingredients",
        "description": "Get the exact ingredient list to cook a named Indian dish. Use when asked what's needed to make a dish.",
        "parameters": {"type": "object",
            "properties": {"dish": {"type": "string", "description": "Dish name, e.g. 'palak paneer'"}},
            "required": ["dish"]}}},
    {"type": "function", "function": {
        "name": "search_products",
        "description": "Search the grocery catalogue for products matching a query. Use to find what's available to buy.",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for, e.g. 'basmati rice'"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "check_price",
        "description": "Get the current price and store for a product. Use when the user asks about cost or price.",
        "parameters": {"type": "object",
            "properties": {"product": {"type": "string", "description": "Product/ingredient name, e.g. 'paneer'"}},
            "required": ["product"]}}},
]


# ---- request/response shapes (Pydantic validates these automatically) ----
class AskRequest(BaseModel):
    question: str
    k: int = 4


class Source(BaseModel):
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

class AgentResponse(BaseModel):
    answer: str
    tool_plan: str | None = None   # expose the model's reasoning so we can SEE it
    tools_used: list[str] = []


# ---- the pipeline, now async ----
async def embed_query(q: str) -> list[float]:
    res = await co.embed(
        texts=[q], model="embed-v4.0", input_type="search_query",
        output_dimension=1024, embedding_types=["float"],
    )
    return (getattr(res.embeddings, "float", None) or res.embeddings.float_)[0]


async def retrieve(query: str, k: int) -> list[Source]:
    qvec = await embed_query(query)
    cands = (await qdrant.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=qvec, using="dense", limit=15),
            models.Prefetch(query=models.Document(text=query, model="Qdrant/bm25"),
                            using="bm25", limit=15),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=15, with_payload=True,
    )).points
    if not cands:
        return []
    docs = [c.payload["text"] for c in cands]
    rr = await co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=k)
    return [Source(text=docs[r.index], score=r.relevance_score) for r in rr.results]


async def generate(query: str, sources: list[Source]) -> str:
    system = (
        "You are a cooking and grocery assistant for an Indian foodie in the Netherlands. "
        "You have tools to look up recipe ingredients, search the product catalogue, and "
        "check prices. Use whatever tools help — you may use several to fully answer. "
        "Answer concisely."
    )
    ctx = "\n".join(f"- {s.text}" for s in sources)
    resp = await co.chat(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}],
        max_tokens=300,
    )
    return resp.message.content[0].text

async def run_agent(question: str) -> AgentResponse:
    system = (
        "You are a cooking and grocery assistant for an Indian foodie in the Netherlands. "
        "When the user asks what's needed to cook a specific dish, use the "
        "get_recipe_ingredients tool. Answer concisely."
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    tools_used: list[str] = []        
    captured_plan = None

    # 1) first call — the model may answer directly, OR ask for a tool
    res = await co.chat(model=GEN_MODEL, messages=messages, tools=TOOLS)
    
    # 2) loop while the model wants tools (usually one round for us)
    while res.message.tool_calls:
        captured_plan = res.message.tool_plan  # its natural-language reasoning
        # record the model's request in the conversation
        messages.append({
            "role": "assistant",
            "tool_calls": res.message.tool_calls,
            "tool_plan": res.message.tool_plan,
        })
        # 3) WE execute each requested call and hand back the result
        for tc in res.message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = await TOOL_FUNCTIONS[tc.function.name](**args)   # <-- add 'await'
            tools_used.append(tc.function.name)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })
        # 4) call again — now the model can use the tool results
        res = await co.chat(model=GEN_MODEL, messages=messages, tools=TOOLS)

    return AgentResponse(
        answer=res.message.content[0].text,
        tool_plan=captured_plan,
        tools_used=tools_used,
    )


# ---- the endpoint ----
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    sources = await retrieve(req.question, req.k)
    answer = await generate(req.question, sources)
    return AskResponse(answer=answer, sources=sources)

@app.post("/ask_agent", response_model=AgentResponse)
async def ask_agent_endpoint(req: AskRequest) -> AgentResponse:
    return await run_agent(req.question)

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

@app.get("/")
async def root() -> dict:
    return {"service": "Grocery RAG Assistant", "docs": "/docs", "ask": "POST /ask"}


#Question 1 : What do I need to make palak paneer, and roughly what will the ingredients cost?
#Question 2 : What paneer can I actually buy?
#Question 3 : What is paneer?