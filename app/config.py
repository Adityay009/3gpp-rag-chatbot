import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    # LLM
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Embeddings / reranker
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Paths
    data_dir: Path = BASE_DIR / "data"
    index_dir: Path = BASE_DIR / "index"
    faiss_index_path: Path = index_dir / "faiss.index"
    chunks_path: Path = index_dir / "chunks.json"

    # Retrieval tuning
    top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", "20"))
    top_n_rerank: int = int(os.getenv("TOP_N_RERANK", "5"))

    # Hallucination-control thresholds
    min_rerank_score: float = float(os.getenv("MIN_RERANK_SCORE", "0.15"))
    min_supporting_chunks: int = int(os.getenv("MIN_SUPPORTING_CHUNKS", "1"))


settings = Settings()