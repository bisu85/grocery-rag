import os

from dotenv import load_dotenv

load_dotenv()  # the ONLY place we load .env

# --- Cohere ---
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
EMBED_MODEL = "embed-v4.0"
EMBED_DIM = 1024
GEN_MODEL = "command-a-03-2025"
RERANK_MODEL = "rerank-v4.0-fast"

# --- provider toggles (escape trial caps by going local) ---
RERANK_PROVIDER = "local"                       # "cohere" | "local"
LOCAL_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # multilingual (matches Indian↔Dutch)

# --- embedding provider (escape the embed cap by going local) ---
EMBED_PROVIDER = "local"                 # "cohere" | "local"
LOCAL_EMBED_MODEL = "bge-m3"             # Ollama, 1024-dim, multilingual

# --- Qdrant ---
QDRANT_URL = "http://localhost:6333"
COLLECTION = "grocery_hybrid_local" if EMBED_PROVIDER == "local" else "grocery_hybrid" ## for local ollama embeddings, use a separate collection to avoid collisions with cohere embeddings

# --- retrieval knobs ---
CANDIDATES = 20          # how many to retrieve before reranking
DEFAULT_K = 4            # how many to keep after rerank

# Cohere rerank returns calibrated 0–1 scores (threshold-able); the local
# bge-reranker returns raw, uncalibrated scores (not suitable for an absolute
# cutoff) — so locally we disable the threshold and rely on ranking + the LLM's
# grounding prompt to reject junk.
RELEVANCE_THRESHOLD = 0.0 if RERANK_PROVIDER == "local" else 0.30


# --- Anthropic (Claude) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"   # cheap, fast, strong at tool use

# --- chat / agent brain provider ---
CHAT_PROVIDER = "anthropic"   # "cohere" | "anthropic"

MAX_AGENT_STEPS = 6         # cap on model round-trips (serial reasoning depth)
MAX_AGENT_TOOL_CALLS = 10   # cap on TOTAL tool executions, summed across all rounds (fan-out work)

MAX_REFLECTIONS = 1   # bounded self-critique passes (reflection has its own leash)

# --- memory ---
SESSION_DB = "sessions.db"            # durable short-term session history (SQLite)
MEMORY_COLLECTION = "grocery_memory"  # long-term semantic fact memory (Qdrant)
MEMORY_TOP_K = 3                      # how many recalled facts to inject