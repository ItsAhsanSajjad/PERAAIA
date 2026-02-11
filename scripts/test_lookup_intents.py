"""
Test suite: DOC_LOOKUP intent routing + hybrid retrieval + ORG_OVERVIEW restoration.
Run: venv\\Scripts\\python.exe scripts/test_lookup_intents.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answerer import classify_intent, QueryIntent, evidence_relevance_gate
from retriever import _normalize_query, _get_keyword_boosts

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
# 1) INTENT CLASSIFICATION: DOC_LOOKUP
# ═══════════════════════════════════════════════════════════════
print("== Intent Classification: DOC_LOOKUP ==")

doc_lookup_queries = [
    ("schedule 1", QueryIntent.DOC_LOOKUP),
    ("schedule 3", QueryIntent.DOC_LOOKUP),
    ("SPPP-1 salary", QueryIntent.DOC_LOOKUP),
    ("sppp salary table", QueryIntent.DOC_LOOKUP),
    ("appendix A", QueryIntent.DOC_LOOKUP),
    ("annex H", QueryIntent.DOC_LOOKUP),
    ("pay scale of PERA", QueryIntent.DOC_LOOKUP),
    ("salary table PERA", QueryIntent.DOC_LOOKUP),
    ("what is PERA", QueryIntent.DOC_LOOKUP),
    ("PERA kya hai", QueryIntent.DOC_LOOKUP),
    ("PERA kia hai", QueryIntent.DOC_LOOKUP),
    ("purpose of PERA", QueryIntent.DOC_LOOKUP),
    ("about pera", QueryIntent.DOC_LOOKUP),
    ("mandate of PERA", QueryIntent.DOC_LOOKUP),
    ("goal of PERA", QueryIntent.DOC_LOOKUP),
]

for query, expected in doc_lookup_queries:
    result = classify_intent(query)
    check(f"'{query}' -> {expected}", result == expected)


# ═══════════════════════════════════════════════════════════════
# 2) QUERY NORMALIZATION
# ═══════════════════════════════════════════════════════════════
print("\n== Query Normalization ==")

check("sehdule -> schedule", "schedule" in _normalize_query("sehdule 1").lower())
check("sa;lary -> salary", "salary" in _normalize_query("sa;lary").lower())
check("sppp1 -> SPPP-1", "SPPP-1" in _normalize_query("sppp1 salary"))
check("bps14 -> BPS-14", "BPS-14" in _normalize_query("bps14"))


# ═══════════════════════════════════════════════════════════════
# 3) KEYWORD BOOSTS
# ═══════════════════════════════════════════════════════════════
print("\n== Keyword Boosts ==")

boosts_sched1 = _get_keyword_boosts("schedule 1")
check("schedule 1 boosts have 'offences'", any("offence" in b.lower() for b in boosts_sched1))
check("schedule 1 boosts have 'Schedule-I'", "Schedule-I" in boosts_sched1)

boosts_sched3 = _get_keyword_boosts("schedule 3")
check("schedule 3 boosts have 'SPPP'", "SPPP" in boosts_sched3)
check("schedule 3 boosts have 'Schedule-III'", "Schedule-III" in boosts_sched3)

boosts_pera = _get_keyword_boosts("what is PERA")
check("what is PERA boosts have 'established'", "established" in boosts_pera)


# ═══════════════════════════════════════════════════════════════
# 4) EVIDENCE GATE: DOC_LOOKUP is permissive
# ═══════════════════════════════════════════════════════════════
print("\n== Evidence Gate: DOC_LOOKUP Permissive ==")

# Simulate retrieval with a low-overlap doc (should PASS for DOC_LOOKUP)
fake_retrieval = {
    "question": "schedule 3",
    "has_evidence": True,
    "evidence": [{
        "doc_name": "PERA_Bill_Recommended_by_Standing_Committee.pdf",
        "max_score": 0.35,
        "hits": [
            {"text": "Schedule-III: Special Pay Package PERA (SPPP) minimum 350,082 maximum 1,000,082", "score": 0.35, "page_start": 42}
        ]
    }, {
        "doc_name": "Some_Notification_2024.pdf",
        "max_score": 0.15,
        "hits": [
            {"text": "PERA notification about establishment", "score": 0.15, "page_start": 1}
        ]
    }]
}

gated = evidence_relevance_gate(fake_retrieval, "schedule 3", QueryIntent.DOC_LOOKUP)
check("DOC_LOOKUP keeps evidence (not over-filtered)", gated.get("has_evidence", False))
check("DOC_LOOKUP keeps Bill doc", any("Bill" in d.get("doc_name", "") for d in gated.get("evidence", [])))


# ═══════════════════════════════════════════════════════════════
# 5) ORG_OVERVIEW now routes through DOC_LOOKUP
# ═══════════════════════════════════════════════════════════════
print("\n== ORG_OVERVIEW Restoration ==")

org_queries = [
    "what does PERA do",
    "functions of PERA",
    "PERA ka maqsad",
    "PERA objective",
]

for q in org_queries:
    result = classify_intent(q)
    check(f"'{q}' -> DOC_LOOKUP", result == QueryIntent.DOC_LOOKUP)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print("ALL PASS" if failed == 0 else "SOME FAILURES")
sys.exit(0 if failed == 0 else 1)
