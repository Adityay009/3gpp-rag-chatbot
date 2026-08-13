"""
Ingests a 3GPP technical specification PDF (e.g. TS 23.501) into a clause-aware,
metadata-tagged chunk store, then builds a FAISS index over the chunks.

Why clause-aware chunking (not fixed token windows):
3GPP specs are organised as a strict clause hierarchy (e.g. "5.15.11.2 Network Slice
Admission Control for maximum number of PDU sessions"). A naive fixed-size sliding
window chunker will frequently split a clause mid-definition or merge unrelated
clauses together, which both hurts retrieval precision and makes it impossible to
give the reader an accurate citation. Chunking on clause boundaries means every
retrieved chunk corresponds to something a reviewer can look up directly in the spec.

Usage:
    python -m app.ingest --pdf data/ts_123501.pdf --doc-id "3GPP TS 23.501" --version "18.5.0"
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.config import settings

# Matches 3GPP clause headers like:
#   "4.2.3 Non-roaming reference architecture"
#   "5.15.11.2a Network Slice Admission Control for maximum number of PDU sessions"
# Deliberately does NOT match table-of-contents lines, which are filtered separately
# because they contain dot-leaders ("....") before the page number.
CLAUSE_HEADER_RE = re.compile(
    r"^(?P<num>\d+(?:\.\d+){0,5}[a-zA-Z]?)\s+(?P<title>[A-Z][^\n]{2,120})$"
)
DOT_LEADER_RE = re.compile(r"\.{4,}")
MAX_CHUNK_CHARS = 1400
MIN_CHUNK_CHARS = 40


@dataclass
class Chunk:
    id: str
    doc_id: str
    version: str
    clause_id: str
    title: str
    page_start: int
    page_end: int
    text: str


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
            text = page.extract_text() or ""
            pages.append((i + 1, text))
    return pages


def is_toc_line(line: str) -> bool:
    return bool(DOT_LEADER_RE.search(line))


def find_body_start_page(pages: list[tuple[int, str]]) -> int:
    """
    The Table of Contents precedes the actual body and would otherwise pollute
    chunking with fake "clause headers" that are really just TOC entries.
    Heuristic: find the first page where "1 Scope" or "1\tScope" appears as a
    line WITHOUT a dot-leader (i.e. it's the real clause, not a TOC row).
    """
    for page_num, text in pages:
        for line in text.splitlines():
            stripped = line.strip()
            if is_toc_line(stripped):
                continue
            if re.match(r"^1\s+Scope\s*$", stripped):
                return page_num
    return 1  # fallback: no TOC detected, index everything


def segment_into_clauses(pages: list[tuple[int, str]], body_start_page: int, doc_id: str, version: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_clause_id = "0"
    current_title = "Front matter"
    current_text_lines: list[str] = []
    current_page_start = body_start_page

    def flush(page_end: int):
        text = "\n".join(current_text_lines).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(
                id=f"{current_clause_id}::0",
                doc_id=doc_id,
                version=version,
                clause_id=current_clause_id,
                title=current_title,
                page_start=current_page_start,
                page_end=page_end,
                text=text,
            ))

    for page_num, text in pages:
        if page_num < body_start_page:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or is_toc_line(stripped):
                continue
            # Skip repeated running headers/footers e.g. "ETSI" / "3GPP TS 23.501 version..."
            if stripped in ("ETSI",) or stripped.startswith("3GPP TS") or stripped.startswith("ETSI TS"):
                continue
            match = CLAUSE_HEADER_RE.match(stripped)
            if match:
                flush(page_num)
                current_clause_id = match.group("num")
                current_title = match.group("title").strip()
                current_text_lines = []
                current_page_start = page_num
            else:
                current_text_lines.append(stripped)
    flush(pages[-1][0] if pages else body_start_page)

    return chunks


def split_oversized_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Sub-splits any clause whose body text exceeds MAX_CHUNK_CHARS on sentence
    boundaries, keeping the same clause_id/title so citations remain accurate."""
    result: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) <= MAX_CHUNK_CHARS:
            result.append(chunk)
            continue
        sentences = re.split(r"(?<=[.;])\s+", chunk.text)
        buf = []
        buf_len = 0
        part = 0
        for sent in sentences:
            if buf_len + len(sent) > MAX_CHUNK_CHARS and buf:
                result.append(Chunk(
                    id=f"{chunk.clause_id}::{part}",
                    doc_id=chunk.doc_id, version=chunk.version,
                    clause_id=chunk.clause_id, title=chunk.title,
                    page_start=chunk.page_start, page_end=chunk.page_end,
                    text=" ".join(buf),
                ))
                part += 1
                buf, buf_len = [], 0
            buf.append(sent)
            buf_len += len(sent)
        if buf:
            result.append(Chunk(
                id=f"{chunk.clause_id}::{part}",
                doc_id=chunk.doc_id, version=chunk.version,
                clause_id=chunk.clause_id, title=chunk.title,
                page_start=chunk.page_start, page_end=chunk.page_end,
                text=" ".join(buf),
            ))
    return result


def build_index(chunks: list[Chunk]):
    import faiss

    model = SentenceTransformer(settings.embedding_model)
    texts = [f"{c.title}. {c.text}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product on normalized vectors
    index.add(embeddings.astype(np.float32))

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.faiss_index_path))

    with open(settings.chunks_path, "w") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)

    print(f"Indexed {len(chunks)} chunks -> {settings.faiss_index_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the 3GPP spec PDF")
    parser.add_argument("--doc-id", default="3GPP TS 23.501")
    parser.add_argument("--version", default="18.5.0")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}. Download it and place it there first.")

    pages = extract_pages(pdf_path)
    body_start = find_body_start_page(pages)
    print(f"Body (post-TOC) detected starting at page {body_start}")

    raw_chunks = segment_into_clauses(pages, body_start, args.doc_id, args.version)
    chunks = split_oversized_chunks(raw_chunks)
    print(f"Segmented into {len(chunks)} clause-level chunks")

    build_index(chunks)


if __name__ == "__main__":
    main()
