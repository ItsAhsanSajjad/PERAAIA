"""Quick integration verification for citation contract."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from answerer import answer_question

empty_retrieval = {"has_evidence": False, "evidence": []}

tests = [
    ("ty", "greeting", True),
    ("tum kon ho", "greeting", True),
    ("hello", "greeting", True),
    ("aoa", "greeting", True),
    ("ok", "greeting", True),
    ("shukriya", "greeting", True),
    ("kotlin kya hy", "refuse", True),
    ("html css difference", "refuse", True),
]

all_pass = True
for q, expected_decision, expect_empty_refs in tests:
    res = answer_question(q, empty_retrieval)
    refs_ok = (res["references"] == []) == expect_empty_refs
    dec_ok = res["decision"] == expected_decision
    status = "PASS" if (refs_ok and dec_ok) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  {status} {q:30s} decision={res['decision']:10s} refs={len(res['references'])}")

print()
print("ALL PASS" if all_pass else "SOME FAILURES")
sys.exit(0 if all_pass else 1)
