"""Verify: all three failing queries should now work."""
from retriever import retrieve, rewrite_contextual_query
from answerer import answer_question

out = []
queries = [
    ("who is director genral pera", None, None),
    ("who is head of pera", None, None),
    ("dg name?", None, None),
    ("director general name", None, None),
    # Follow-up: after asking "who is head of pera"
    ("nam kia hai", "who is head of pera", "PERA ka head Director General hota hai."),
]

for q, last_q, last_a in queries:
    rw = rewrite_contextual_query(q, last_q, last_a)
    is_followup = last_q is not None
    ret = retrieve(rw)
    res = answer_question(q, ret, conversation_history=[{"role":"user","content":last_q},{"role":"assistant","content":last_a}] if last_q else None)
    status = "✅" if res.get("decision") == "answer" else "❌"
    out.append(f"{status} '{q}' {'(follow-up)' if is_followup else ''}")
    out.append(f"   Rewritten: '{rw[:80]}'")
    out.append(f"   Decision: {res.get('decision')}")
    out.append(f"   Answer: {res.get('answer','')[:150]}")
    out.append("")

with open("debug_verify_all.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done. See debug_verify_all.txt")
