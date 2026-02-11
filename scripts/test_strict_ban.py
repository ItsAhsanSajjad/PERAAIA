"""
Test suite: STRICT_AUTHORITY intent + ban phrase enforcement.
Run: venv\\Scripts\\python.exe scripts/test_strict_ban.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answerer import (
    classify_intent, QueryIntent, _post_process_strict,
    _BANNED_HEDGE_PHRASES, evidence_relevance_gate,
)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")


# ═══════════════════════════════════════════════════════════════
# 1) INTENT CLASSIFICATION: STRICT_AUTHORITY
# ═══════════════════════════════════════════════════════════════
print("== Intent Classification: STRICT_AUTHORITY ==")

strict_queries = [
    ("CTO kis ko terminate kr skta hai", QueryIntent.STRICT_AUTHORITY),
    ("can CTO fire someone", QueryIntent.STRICT_AUTHORITY),
    ("who can dismiss an officer", QueryIntent.STRICT_AUTHORITY),
    ("appointment authority of PERA", QueryIntent.STRICT_AUTHORITY),
    ("powers to suspend", QueryIntent.STRICT_AUTHORITY),
    ("legal authority to terminate", QueryIntent.STRICT_AUTHORITY),
    ("competent authority for dismissal", QueryIntent.STRICT_AUTHORITY),
]

for query, expected in strict_queries:
    result = classify_intent(query)
    check(f"'{query}' -> STRICT_AUTHORITY", result == expected)


# ═══════════════════════════════════════════════════════════════
# 2) BAN PHRASES PRESENT IN LIST
# ═══════════════════════════════════════════════════════════════
print("\n== Ban Phrases in List ==")

required_bans = ["general rules", "typically", "depends", "work culture", "may apply"]
for phrase in required_bans:
    check(f"'{phrase}' in ban list", phrase in _BANNED_HEDGE_PHRASES)


# ═══════════════════════════════════════════════════════════════
# 3) _post_process_strict DETECTS BAN PHRASES
# ═══════════════════════════════════════════════════════════════
print("\n== Ban Phrase Detection ==")

ban_test_cases = [
    ("Based on general rules, the CTO can fire employees.", True),
    ("Typically in organizations, the CTO has such powers.", True),
    ("It depends on the organization's policies.", True),  
    ("The work culture of PERA determines this.", True),
    ("Standard practice may apply in this case.", True),
    ("The CTO's powers are defined in Section 12 of the PERA Act.", False),
    ("This information is clearly stated in the official documents.", False),
]

for answer, should_flag in ban_test_cases:
    _, needs_regen = _post_process_strict(answer, QueryIntent.STRICT_AUTHORITY, "can CTO fire?")
    check(f"ban detect: '{answer[:50]}...' -> regen={should_flag}", needs_regen == should_flag)


# ═══════════════════════════════════════════════════════════════
# 4) STRICT_AUTHORITY EVIDENCE GATE
# ═══════════════════════════════════════════════════════════════
print("\n== STRICT_AUTHORITY Evidence Gate ==")

# Dress code docs should be blocked for STRICT_AUTHORITY
fake_retrieval = {
    "question": "terminate",
    "has_evidence": True,
    "evidence": [
        {
            "doc_name": "PERA_Dress_Code_2024.pdf",
            "max_score": 0.5,
            "hits": [{"text": "Employees shall wear proper attire", "score": 0.5, "page_start": 1}]
        },
        {
            "doc_name": "PERA_Service_Rules_2024.pdf",
            "max_score": 0.6,
            "hits": [{"text": "terminate any employee under competent authority CTO powers", "score": 0.6, "page_start": 5}]
        },
    ]
}

gated = evidence_relevance_gate(fake_retrieval, "can CTO terminate employees", QueryIntent.STRICT_AUTHORITY)
doc_names = [d["doc_name"] for d in gated.get("evidence", [])]
check("Dress code blocked for STRICT_AUTHORITY", "PERA_Dress_Code_2024.pdf" not in doc_names)
check("Service rules allowed for STRICT_AUTHORITY", "PERA_Service_Rules_2024.pdf" in doc_names)


# ═══════════════════════════════════════════════════════════════
# 5) GREETING still works (not broken by new intents)
# ═══════════════════════════════════════════════════════════════
print("\n== Greeting Still Works ==")

greeting_checks = [
    ("ty", QueryIntent.GREETING_SMALLTALK),
    ("hello", QueryIntent.GREETING_SMALLTALK),
    ("tum kon ho", QueryIntent.GREETING_SMALLTALK),
]

for query, expected in greeting_checks:
    result = classify_intent(query)
    check(f"'{query}' -> GREETING_SMALLTALK", result == expected)


# ═══════════════════════════════════════════════════════════════
# 6) OUT_OF_SCOPE still works
# ═══════════════════════════════════════════════════════════════
print("\n== OUT_OF_SCOPE Still Works ==")

oos_checks = [
    ("kotlin kya hy", QueryIntent.OUT_OF_SCOPE),
    ("html css difference", QueryIntent.OUT_OF_SCOPE),
]

for query, expected in oos_checks:
    result = classify_intent(query)
    check(f"'{query}' -> OUT_OF_SCOPE", result == expected)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print("ALL PASS" if failed == 0 else "SOME FAILURES")
sys.exit(0 if failed == 0 else 1)
