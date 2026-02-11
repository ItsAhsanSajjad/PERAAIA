"""Quick spot-check: verify 5 key queries still answer correctly after stricter prompt."""
from retriever import retrieve
from answerer import answer_question

queries = [
    ("TC-001", "What is the short title of the Act?", ["punjab enforcement", "2024"]),
    ("TC-012", "Who is the Chairperson of the Authority?", ["chief minister"]),
    ("TC-026", "Can the SDEO register FIRs and investigate?", ["fir"]),
    ("TC-038", "After arrest, what must happen next?", ["magistrate"]),
    ("SSO",    "sso kia hota", []),  # Should REFUSE
]

for tc_id, query, must_keywords in queries:
    ret = retrieve(query)
    res = answer_question(query, ret)
    answer = res.get("answer", "").lower()
    decision = res.get("decision", "")
    
    if not must_keywords:
        # Should refuse
        status = "PASS" if decision == "refuse" else "FAIL"
    else:
        missing = [k for k in must_keywords if k not in answer]
        status = "PASS" if not missing else f"FAIL missing: {missing}"
    
    print(f"  {tc_id}: {status} | {decision} | {answer[:80]}")
