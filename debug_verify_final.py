"""Verify all key queries work after system prompt fix."""
from retriever import retrieve, rewrite_contextual_query
from answerer import answer_question

out = []
tests = [
    ("CTO ki powers kia hain?", None, None),
    ("dg kon hai", None, None),
    ("director general name", None, None),
    ("who is director general pera", None, None),
    ("monitoring manager salary", None, None),
    ("What are the powers of PERA?", None, None),
    # Follow-up test
    ("salarey ktna hai", "monitoring manager kia krta hai", "Monitoring Manager MIS & GIS ka kaam hai..."),
]

for q, last_q, last_a in tests:
    rw = rewrite_contextual_query(q, last_q, last_a)
    ret = retrieve(rw)
    hist = [{"role":"user","content":last_q},{"role":"assistant","content":last_a}] if last_q else None
    res = answer_question(q, ret, conversation_history=hist)
    status = "✅" if res.get("decision") == "answer" else "❌"
    out.append(f"{status} '{q}' {'(follow-up)' if last_q else ''}")
    out.append(f"   Answer: {res.get('answer','')[:150]}")
    out.append("")

with open("debug_final_verify.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done. See debug_final_verify.txt")
