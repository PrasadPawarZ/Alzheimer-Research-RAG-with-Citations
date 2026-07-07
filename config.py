"""Central configuration, loaded from environment variables and .env."""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2").strip()

TOP_K = int(os.getenv("TOP_K", "5"))
VECTOR_POOL_MULTIPLIER = int(os.getenv("VECTOR_POOL_MULTIPLIER", "4"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.72"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.28"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

PAPERS_DIR = os.getenv("PAPERS_DIR", "./papers")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "alzheimer_papers"

CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "220"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))

# Backward-compatible aliases for older scripts that referenced character names.
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS
