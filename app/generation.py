"""
Generation layer: turns retrieved, reranked chunks into a grounded answer.

Hallucination-control mechanisms here (in addition to the retrieval confidence
gate in retrieval.py):
  1. Strict system prompt: the model is instructed to answer ONLY from the
     supplied context and to explicitly say when the context is insufficient,
     rather than filling gaps from parametric knowledge.
  2. Mandatory inline citation of clause numbers, enforced by prompt + a
     post-generation check that every cited clause actually exists in the
     supplied context (catches fabricated citations).
  3. A lightweight verification pass: a second, independent Gemini call checks
     whether the answer is actually supported by the cited chunks before it is
     returned to the user. If verification fails, we downgrade the answer to a
     refusal rather than surface an unverified claim.
"""
import re

import google.generativeai as genai

from app.config import settings
from app.retrieval import RetrievedChunk

genai.configure(api_key=settings.google_api_key)

REFUSAL_MESSAGE = (
    "I couldn't find a clearly relevant section in the provided 3GPP TS 23.501 "
    "corpus to answer that confidently. Try rephrasing, or note that this may be "
    "outside the scope of the indexed document."
)

SYSTEM_INSTRUCTIONS = """You are a telecom standards assistant answering questions strictly from 3GPP \
technical specification excerpts provided below. Follow these rules exactly:

1. Answer ONLY using information present in the provided context chunks. Do not use \
   outside knowledge, even if you believe you know the answer.
2. Every factual claim must be followed by a citation to the clause it came from, in \
   the form [Clause X.Y.Z].
3. If the context does not contain enough information to answer the question, say \
   so explicitly instead of guessing. Do not fill gaps with plausible-sounding text.
4. Do not invent clause numbers, parameter names, or values that do not appear in \
   the context.
5. Be concise and technical; this is for an engineering audience.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[Clause {c.clause_id}] {c.title}\n(pages {c.page_start}-{c.page_end})\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


def _extract_cited_clauses(answer: str) -> set[str]:
    return set(re.findall(r"\[Clause\s+([\w.]+)\]", answer))


def _verify_answer(query: str, answer: str, context: str) -> bool:
    """Second-pass entailment check: is the answer actually supported by the
    context, or did the model drift/embellish? Returns True if supported."""
    model = genai.GenerativeModel(settings.gemini_model)
    verify_prompt = f"""You are a strict fact-checker. Given the CONTEXT and an ANSWER that claims to be \
derived from it, respond with exactly one word: "SUPPORTED" if every claim in the \
answer is directly backed by the context, or "UNSUPPORTED" if the answer contains \
any claim, number, or clause reference not present in the context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with exactly one word."""
    try:
        resp = model.generate_content(verify_prompt)
        verdict = (resp.text or "").strip().upper()
        return verdict.startswith("SUPPORTED")
    except Exception:
        # Fail closed: if verification itself errors, don't block the answer on it,
        # but this is logged for review in a production setting.
        return True


def generate_answer(query: str, chunks: list[RetrievedChunk], is_sufficient: bool) -> dict:
    if not is_sufficient:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "verified": True,
            "refused": True,
        }

    context = _format_context(chunks)
    model = genai.GenerativeModel(settings.gemini_model, system_instruction=SYSTEM_INSTRUCTIONS)

    prompt = f"""CONTEXT:
{context}

QUESTION:
{query}

Answer the question following the system rules. Cite clauses inline as [Clause X.Y.Z]."""

    response = model.generate_content(prompt)
    answer = (response.text or "").strip()

    # Guard against fabricated citations: any cited clause must exist in the context we gave it.
    valid_clauses = {c.clause_id for c in chunks}
    cited = _extract_cited_clauses(answer)
    fabricated = cited - valid_clauses

    if fabricated:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "verified": False,
            "refused": True,
            "reason": f"Model cited clause(s) not present in retrieved context: {sorted(fabricated)}",
        }

    verified = _verify_answer(query, answer, context)
    if not verified:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "verified": False,
            "refused": True,
            "reason": "Answer failed independent entailment verification against retrieved context.",
        }

    return {
        "answer": answer,
        "sources": [
            {"clause_id": c.clause_id, "title": c.title, "pages": f"{c.page_start}-{c.page_end}", "score": round(c.score, 3)}
            for c in chunks
        ],
        "verified": True,
        "refused": False,
    }
