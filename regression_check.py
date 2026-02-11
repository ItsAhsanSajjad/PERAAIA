"""Quick regression check after schedule + SSO fixes."""
from retriever import retrieve
from answerer import answer_question

queries = [
    ("SSO", "sso kia hota", ["system support officer"]),
    ("Schedule-3", "schedule 3", ["schedule-iii", "kpa"]),
    ("Schedule-4", "schedule 4", ["schedule-iv", "rules"]),
    ("Schedule-5", "schedule 5", ["schedule-v", "rating"]),
    ("Act-title", "What is the short title of the Act?", ["punjab enforcement"]),
    ("DG-name", "director general ka naam", ["rizwan"]),
    ("Section-32", "section 32 mein kya hai", ["magistrate"]),
]

out = []
for tc_id, q, must_kw in queries:
    ret = retrieve(q)
    res = answer_question(q, ret)
    answer = res.get("answer","").lower()
    decision = res.get("decision","")
    
    missing = [k for k in must_kw if k not in answer]
    status = "PASS" if not missing else f"FAIL missing: {missing}"
    out.append(f"{tc_id}: {status} | {decision}")
    out.append(f"  answer: {answer[:100]}")

with open("regression_check.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
