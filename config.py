import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration Constants ---

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat History Configuration
CHATS_DIR = os.getenv("CHATS_DIR", "chats")

# RAG Configuration (Basic Example)
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 100
RAG_NUM_RETRIEVED_CHUNKS = 3 # Number of chunks to retrieve for context

# Web Search Configuration (Placeholder)
# Example: TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# UI Defaults
DEFAULT_MODEL = "llama3:latest" # Or choose another default
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 40

# File Handling
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
ALLOWED_FILE_TYPES = ALLOWED_IMAGE_TYPES + ["pdf"]

# Other
MAX_TITLE_GENERATION_MESSAGES = 4 # Use first N messages for title generation
TITLE_GENERATION_MODEL = "llama3:latest" # Use a fast model for titles