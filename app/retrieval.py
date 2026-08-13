"""
Retrieval layer: dense FAISS search -> cross-encoder reranking -> confidence gate.

The reranking + threshold step is the primary hallucination-suppression mechanism
on the retrieval side: a bi-encoder (FAISS/sentence-transformers) is fast but
imprecise, so we over-fetch (top_k_retrieve) and let a cross-encoder, which scores
the query and each candidate chunk jointly, re-sort for actual relevance. If even
the best reranked chunk falls below `min_rerank_score`, we treat the corpus as not
containing the answer and signal the generation layer to refuse rather than guess.
"""
import json
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import settings


@dataclass
class RetrievedChunk:
    clause_id: str
    title: str
    text: str
    doc_id: str
    version: str
    page_start: int
    page_end: int
    score: float


class Retriever:
    def __init__(self):
        self.embed_model = SentenceTransformer(settings.embedding_model)
        self.reranker = CrossEncoder(settings.reranker_model)
        self.index = faiss.read_index(str(settings.faiss_index_path))
        with open(settings.chunks_path) as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str) -> tuple[list[RetrievedChunk], bool]:
        """Returns (ranked_chunks, is_sufficient). is_sufficient is False when the
        top reranked score doesn't clear the confidence floor, in which case the
        caller should refuse to answer rather than generate from weak context."""
        q_emb = self.embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, idxs = self.index.search(q_emb.astype(np.float32), settings.top_k_retrieve)

        candidates = [self.chunks[i] for i in idxs[0] if i != -1]
        if not candidates:
            return [], False

        pairs = [[query, f"{c['title']}. {c['text']}"] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        top = ranked[: settings.top_n_rerank]

        results = [
            RetrievedChunk(
                clause_id=c["clause_id"], title=c["title"], text=c["text"],
                doc_id=c["doc_id"], version=c["version"],
                page_start=c["page_start"], page_end=c["page_end"],
                score=float(s),
            )
            for c, s in top
        ]

        is_sufficient = len(results) > 0 and results[0].score >= settings.min_rerank_score
        return results, is_sufficient
