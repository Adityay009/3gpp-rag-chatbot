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
Registration Update procedures. The AMF determines if MICO mode is allowed
for the UE based on local configuration, Expected UE Behaviour, network
configuration parameters, UE preferences, UE subscription information, and
network policies, and indicates this to the UE during the Registration
procedure. If the UE does not indicate a preference, the AMF will not
activate MICO mode."""

FAITHFUL_ANSWER = (
    "MICO mode is a power-saving feature for Mobile Terminated reachability, "
    "primarily used by Cellular IoT devices [Clause 5.4.1.3]. A UE can request "
    "it during Initial Registration, and the AMF decides whether to allow it "
    "based on configuration and subscription data [Clause 5.4.1.3]. If the UE "
    "doesn't request it, the AMF will not activate MICO mode [Clause 5.4.1.3]."
)

FABRICATED_ANSWER = (
    "MICO mode is a power-saving feature for Mobile Terminated reachability "
    "[Clause 5.4.1.3]. The MICO timer is fixed at exactly 45 minutes and cannot "
    "be configured by the operator [Clause 5.4.1.3]. Additionally, MICO mode "
    "requires a minimum 5G NR signal strength of -95 dBm to activate "
    "[Clause 5.4.1.3]."
)

NEW_PROMPT_TEMPLATE = """You are a fact-checker reviewing whether an ANSWER is faithfully grounded in \
the given CONTEXT.

Paraphrasing, summarizing, and rewording are all considered SUPPORTED as long \
as the meaning is preserved and no new specific facts, numbers, names, \
thresholds, or claims are introduced beyond what the context actually states.

Respond "UNSUPPORTED" only if the answer contains a specific fact, number, \
name, threshold, or claim that does NOT appear anywhere in the context, even \
in different words.

Respond "SUPPORTED" if the answer is a faithful paraphrase or summary of the \
context, even if wording differs.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with exactly one word: SUPPORTED or UNSUPPORTED."""


def test_case(name, answer):
    model = genai.GenerativeModel(settings.gemini_model)
    prompt = NEW_PROMPT_TEMPLATE.format(context=CONTEXT, answer=answer)
    resp = model.generate_content(prompt)
    print(f"{name}: {resp.text.strip()}")
    return resp.text.strip()


print(f"Using model: {settings.gemini_model}\n")
r1 = test_case("Faithful answer (expect SUPPORTED)", FAITHFUL_ANSWER)
r2 = test_case("Fabricated answer (expect UNSUPPORTED)", FABRICATED_ANSWER)

print("\n=== Result ===")
if r1.upper().startswith("SUPPORTED") and r2.upper().startswith("UNSUPPORTED"):
    print("PASS: revised prompt correctly discriminates faithful vs fabricated.")
else:
    print("Still not discriminating correctly, needs further tuning.")