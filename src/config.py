import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4


def require_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set GROQ_API_KEY in .env before asking a question.")
    return key
