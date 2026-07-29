import os

from dotenv import load_dotenv

load_dotenv()  # the ONLY place we load .env

# --- Cohere ---
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
EMBED_MODEL = "embed-v4.0"
EMBED_DIM = 1024
GEN_MODEL = "command-a-03-2025"
RERANK_MODEL = "rerank-v4.0-fast"

# --- Qdrant ---
QDRANT_URL = "http://localhost:6333"
COLLECTION = "grocery_hybrid"

# --- retrieval knobs ---
CANDIDATES = 20          # how many to retrieve before reranking
DEFAULT_K = 4            # how many to keep after rerank
RELEVANCE_THRESHOLD = 0.30
