from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.retrieval import Retriever
from app.generation import generate_answer

app = FastAPI(title="3GPP RAG Chatbot", description="Low-hallucination RAG chatbot over 3GPP TS 23.501")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        if not settings.faiss_index_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Index not built yet. Run: python -m app.ingest --pdf data/<spec>.pdf",
            )
        _retriever = Retriever()
    return _retriever


class ChatRequest(BaseModel):
    message: str


class Source(BaseModel):
    clause_id: str
    title: str
    pages: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    verified: bool
    refused: bool
    reason: str | None = None


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message.")

    retriever = get_retriever()
    chunks, is_sufficient = retriever.retrieve(req.message)
    result = generate_answer(req.message, chunks, is_sufficient)
    return result


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "index_built": settings.faiss_index_path.exists(),
    }


static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(static_dir / "index.html")
