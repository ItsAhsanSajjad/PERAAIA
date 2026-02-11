"""Trace each test_strict_ban test."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from answerer import (
    classify_intent, QueryIntent, _post_process_strict,
    _BANNED_HEDGE_PHRASES, evidence_relevance_gate, _extract_query_keywords,
)

# Test each STRICT_AUTHORITY intent
strict_queries = [
    ("CTO kis ko terminate kr skta hai", QueryIntent.STRICT_AUTHORITY),
    ("can CTO fire someone", QueryIntent.STRICT_AUTHORITY),
    ("who can dismiss an officer", QueryIntent.STRICT_AUTHORITY),
    ("appointment authority of PERA", QueryIntent.STRICT_AUTHORITY),
    ("powers to suspend", QueryIntent.STRICT_AUTHORITY),
    ("legal authority to terminate", QueryIntent.STRICT_AUTHORITY),
    ("competent authority for dismissal", QueryIntent.STRICT_AUTHORITY),
]
for q, exp in strict_queries:
    r = classify_intent(q)
    print(f"{'PASS' if r == exp else 'FAIL'} intent '{q}' -> {r}")

# Test ban phrases in list
required_bans = ["general rules", "typically", "depends", "work culture", "may apply"]
for p in required_bans:
    print(f"{'PASS' if p in _BANNED_HEDGE_PHRASES else 'FAIL'} ban '{p}' in list")

# Test ban detection
ban_tests = [
    ("Based on general rules, the CTO can fire employees.", True),
    ("Typically in organizations, the CTO has such powers.", True),
    ("It depends on the organization's policies.", True),
    ("The work culture of PERA determines this.", True),
    ("Standard practice may apply in this case.", True),
    ("The CTO's powers are defined in Section 12 of the PERA Act.", False),
    ("This information is clearly stated in the official documents.", False),
]
for a, should in ban_tests:
    _, regen = _post_process_strict(a, QueryIntent.STRICT_AUTHORITY, "x")
    print(f"{'PASS' if regen == should else 'FAIL'} ban detect '{a[:50]}' regen={regen} expected={should}")

# Test evidence gate
fake = {
    "question": "terminate",
    "has_evidence": True,
    "evidence": [
        {"doc_name": "PERA_Dress_Code_2024.pdf", "max_score": 0.5, "hits": [{"text": "Employees shall wear proper attire", "score": 0.5, "page_start": 1}]},
        {"doc_name": "PERA_Service_Rules_2024.pdf", "max_score": 0.6, "hits": [{"text": "terminate any employee under competent authority CTO powers", "score": 0.6, "page_start": 5}]},
    ]
}
# Debug: what keywords are extracted?
kw = _extract_query_keywords("can CTO terminate employees")
print(f"Keywords for 'can CTO terminate employees': {kw}")

gated = evidence_relevance_gate(fake, "can CTO terminate employees", QueryIntent.STRICT_AUTHORITY)
doc_names = [d["doc_name"] for d in gated.get("evidence", [])]
print(f"Docs after gate: {doc_names}")
print(f"{'PASS' if 'PERA_Dress_Code_2024.pdf' not in doc_names else 'FAIL'} dress code blocked")
print(f"{'PASS' if 'PERA_Service_Rules_2024.pdf' in doc_names else 'FAIL'} service rules kept")

# Greeting/OOS regression
for q, exp in [("ty", QueryIntent.GREETING_SMALLTALK), ("hello", QueryIntent.GREETING_SMALLTALK), ("tum kon ho", QueryIntent.GREETING_SMALLTALK), ("kotlin kya hy", QueryIntent.OUT_OF_SCOPE), ("html css difference", QueryIntent.OUT_OF_SCOPE)]:
    r = classify_intent(q)
    print(f"{'PASS' if r == exp else 'FAIL'} '{q}' -> {r}")
