import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import google.generativeai as genai
from app.config import settings

genai.configure(api_key=settings.google_api_key)

CONTEXT = """[Clause 5.4.1.3] Mobile Initiated Connection Only (MICO) mode
(pages 116-117)
MICO mode is a feature designed to enable power saving for Mobile Terminated
(MT) reachability, particularly for devices like Cellular IoT. A UE may
indicate a preference for MICO mode during Initial Registration or Mobility
Registration Update procedures."""

ANSWER = (
    "MICO mode is a power-saving feature for Mobile Terminated reachability, "
    "primarily used by Cellular IoT devices [Clause 5.4.1.3]."
)

model = genai.GenerativeModel(settings.gemini_model)
prompt = f"""You are a strict fact-checker. Given the CONTEXT and an ANSWER that claims to be \
derived from it, respond with exactly one word: "SUPPORTED" if every claim in the \
answer is directly backed by the context, or "UNSUPPORTED" if the answer contains \
any claim, number, or clause reference not present in the context.

CONTEXT:
{CONTEXT}

ANSWER:
{ANSWER}

Respond with exactly one word."""

print(f"Using model: {settings.gemini_model}")
try:
    resp = model.generate_content(prompt)
    print("RAW response object:", resp)
    print("\n---")
    print("resp.text:", repr(resp.text))
    if hasattr(resp, "candidates"):
        for i, c in enumerate(resp.candidates):
            print(f"candidate[{i}].finish_reason:", c.finish_reason)
except Exception as e:
    print("EXCEPTION:", type(e).__name__, str(e))