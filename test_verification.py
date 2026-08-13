"""
Directly exercises the independent verification pass (`_verify_answer` in
app/generation.py) without relying on the live retrieval pipeline to happen to
produce a drifting answer. This is more reliable than testing through the UI,
since a clean retrieval pipeline may simply never trigger a bad first-pass
answer to catch.

Run from the project root:
    python -m tests.test_verification

Requires GOOGLE_API_KEY to be set (loaded from .env automatically via app.config).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.generation import _verify_answer

# Real context, lifted from an actual retrieved chunk during earlier testing.
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

CASE_1_SUPPORTED = {
    "name": "Faithful answer (should PASS verification)",
    "answer": (
        "MICO mode is a power-saving feature for Mobile Terminated reachability, "
        "primarily used by Cellular IoT devices [Clause 5.4.1.3]. A UE can request "
        "it during Initial Registration, and the AMF decides whether to allow it "
        "based on configuration and subscription data [Clause 5.4.1.3]. If the UE "
        "doesn't request it, the AMF will not activate MICO mode [Clause 5.4.1.3]."
    ),
}

CASE_2_FABRICATED = {
    "name": "Fabricated claim not in context (should FAIL verification)",
    "answer": (
        "MICO mode is a power-saving feature for Mobile Terminated reachability "
        "[Clause 5.4.1.3]. The MICO timer is fixed at exactly 45 minutes and cannot "
        "be configured by the operator [Clause 5.4.1.3]. Additionally, MICO mode "
        "requires a minimum 5G NR signal strength of -95 dBm to activate "
        "[Clause 5.4.1.3]."
    ),
}


def run_case(case: dict):
    print(f"\n--- {case['name']} ---")
    print(f"Answer under test: {case['answer'][:100]}...")
    is_supported = _verify_answer(
        query="What is MICO mode?",
        answer=case["answer"],
        context=CONTEXT,
    )
    verdict = "SUPPORTED" if is_supported else "UNSUPPORTED"
    print(f"Verification result: {verdict}")
    return is_supported


def main():
    result_1 = run_case(CASE_1_SUPPORTED)
    result_2 = run_case(CASE_2_FABRICATED)

    print("\n=== Summary ===")
    print(f"Faithful answer flagged as supported:   {result_1} (expected: True)")
    print(f"Fabricated answer flagged as supported: {result_2} (expected: False)")

    if result_1 and not result_2:
        print("\nPASS: Verification pass correctly distinguishes faithful answers from fabricated ones.")
    else:
        print("\nWARNING: Verification pass did not behave as expected. Review the cases above.")


if __name__ == "__main__":
    main()
