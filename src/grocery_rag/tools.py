from grocery_rag.retrieval import retrieve

# ---- tool backends (synthetic stand-ins; swap for real sources later) ----
RECIPE_DB = {
    "palak paneer": ["spinach", "paneer", "onion", "garlic", "ginger",
                     "green chili", "garam masala", "cream"],
    "dal tadka": ["toor dal", "onion", "tomato", "cumin", "garlic", "ghee", "turmeric"],
    "chicken korma": ["chicken", "yogurt", "onion", "cashews", "garam masala", "cream"],
}
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


async def get_recipe_ingredients(dish: str) -> dict:
    return {"dish": dish, "ingredients": RECIPE_DB.get(dish.strip().lower(), [])}


async def check_price(product: str) -> dict:
    key = product.strip().lower()
    for name, (price, store) in PRICE_DB.items():
        if name in key or key in name:
            return {"product": product, "price_eur": price, "store": store}
    return {"product": product, "price_eur": None, "note": "price not available"}


async def search_products(query: str) -> dict:
    results = await retrieve(query, k=5)          # reuses the ONE real retrieval
    return {"query": query, "results": [r["text"] for r in results]}


TOOL_FUNCTIONS = {
    "get_recipe_ingredients": get_recipe_ingredients,
    "search_products": search_products,
    "check_price": check_price,
}

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