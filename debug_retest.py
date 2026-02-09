"""Quick re-test CTO and monitoring manager."""
from retriever import retrieve; from answerer import answer_question
out = []
for q in ["CTO ki powers kia hain?", "monitoring manager salary"]:
    r = retrieve(q)
    a = answer_question(q, r)
    s = "PASS" if a.get("decision") == "answer" else "FAIL"
    out.append(f"{s}: '{q}'")
    out.append(f"  {a.get('answer','')[:200]}")
    out.append("")
with open("debug_retest.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
