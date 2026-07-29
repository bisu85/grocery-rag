import cohere
from qdrant_client import AsyncQdrantClient
from grocery_rag.config import COHERE_API_KEY, QDRANT_URL
import ollama


# Created once at import, shared everywhere. Async versions (for the API).
co = cohere.AsyncClientV2(api_key=COHERE_API_KEY)
qdrant = AsyncQdrantClient(url=QDRANT_URL)
ollama_client = ollama.AsyncClient()   # talks to localhost:11434
