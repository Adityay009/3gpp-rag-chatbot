# 3GPP TS 23.501 RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions grounded in **3GPP
TS 23.501 (5G System Architecture)**, built for the Mavenir GET take-home
assignment. The design goal is **minimal to near-zero hallucination**: the bot
would rather say "not in the spec" than guess.

## Live demo
`<add your Render URL here after deploy>`

## Architecture

```
User question
     │
     ▼
[Sentence-Transformer embed] ──▶ [FAISS dense search, top 20]
     │
     ▼
[Cross-encoder rerank] ──▶ confidence gate (min_rerank_score)
     │                              │
     │ score too low                │ score sufficient
     ▼                              ▼
  Refuse to answer          [Gemini generation, grounded prompt + forced citations]
                                     │
                                     ▼
                          [Citation validity check: every cited
                           clause must exist in retrieved context]
                                     │
                                     ▼
                          [Independent verification pass: second
                           Gemini call checks entailment]
                                     │
                                     ▼
                              Answer + sources, or refusal
```

## Why this design suppresses hallucination

3GPP specs are dense, cross-referenced, and easy for an LLM to "fill in" plausible
but wrong detail for (e.g. inventing a QoS parameter value). Every layer below
exists to catch a different failure mode:

| Layer | Failure mode it catches |
|---|---|
| **Clause-aware chunking** (`app/ingest.py`) | Naive fixed-size chunking splits a clause mid-definition, so retrieval returns incomplete/misleading context. Chunking on the spec's own clause hierarchy keeps each chunk semantically whole and citable. |
| **Bi-encoder retrieval + cross-encoder rerank** (`app/retrieval.py`) | Bi-encoder (FAISS) search alone is fast but imprecise on dense technical text; reranking with a cross-encoder that scores query+chunk jointly recovers precision. |
| **Confidence gate** (`min_rerank_score` threshold) | If nothing in the corpus is actually relevant, the bot refuses instead of answering from the LLM's parametric (and possibly wrong or outdated) knowledge. |
| **Grounded prompt with forced citations** (`app/generation.py`) | Instructs Gemini to answer only from supplied context and cite `[Clause X.Y.Z]` for every claim, which also makes fabrication easier to detect downstream. |
| **Citation validity check** | If the model cites a clause number that wasn't actually in the retrieved context, that's direct evidence of fabrication — the answer is discarded and replaced with a refusal. |
| **Independent verification pass** | A second, separate Gemini call fact-checks the answer against the context before it's shown to the user, catching subtler drift that citation-checking alone would miss. |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Get the spec (either run the script or download manually and place at data/ts_123501.pdf)
python scripts/download_spec.py

# 2. Build the index (clause chunking + embedding + FAISS)
python -m app.ingest --pdf data/ts_123501.pdf --doc-id "3GPP TS 23.501" --version "18.5.0"

# 3. Set your Gemini API key
export GOOGLE_API_KEY=your_key_here

# 4. Run
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Deployment (Render)

This repo includes `render.yaml`. Connect the repo in Render, set `GOOGLE_API_KEY`
as a secret env var, and deploy — the build step downloads the spec and builds the
index automatically (the PDF itself isn't committed to the repo, since ETSI's
deliverable carries a redistribution restriction; we fetch it directly from ETSI's
own server at build time instead).

## API

`POST /api/chat`
```json
{ "message": "What are the RM-REGISTERED and RM-DEREGISTERED states?" }
```
Response:
```json
{
  "answer": "...grounded answer with [Clause X.Y.Z] citations...",
  "sources": [{"clause_id": "5.3.2.2", "title": "5GS Registration Management states", "pages": "94-96", "score": 0.71}],
  "verified": true,
  "refused": false
}
```

## Example: a refusal in action

Asking something outside the corpus (e.g. "What's the price of a Cisco 5G core
license?") returns a refusal rather than an invented answer, since nothing in the
retrieved chunks clears the confidence threshold — demonstrating the near-zero
hallucination behavior explicitly rather than just claiming it.

## Tuning knobs (`app/config.py`)

- `MIN_RERANK_SCORE` — raise for stricter refusal behavior, lower for more coverage.
- `TOP_K_RETRIEVE` / `TOP_N_RERANK` — retrieval breadth vs. context size trade-off.
- `EMBEDDING_MODEL` / `RERANKER_MODEL` — swappable for larger models if latency budget allows.

## Known limitations

- Currently indexes TS 23.501 only (single-document scope, per assignment framing).
- Clause-header regex is tuned to 3GPP's numbering convention; a spec with a very
  different formatting style would need the regex adjusted.
- The verification pass adds one extra LLM call per query (latency/cost trade-off
  for the accuracy gain).
