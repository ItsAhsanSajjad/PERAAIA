"""Verify CTO name queries work correctly now."""
from retriever import retrieve
from answerer import answer_question, _get_identity_answer

out = []

# 1. Test identity detection doesn't hijack CTO queries
tests_identity = ["CTO NAM KIA HAI", "CTO KA NAME", "DG ka naam", "nam kia hai", "tera naam"]
for q in tests_identity:
    result = _get_identity_answer(q)
    out.append(f"Identity check '{q}': {'BOT NAME' if result else 'PASS (not identity)'}")

out.append("")

# 2. Test full pipeline for CTO and DG name queries 
for q in ["CTO KA NAME", "CTO NAM KIA HAI", "DG KON HAI"]:
    ret = retrieve(q)
    res = answer_question(q, ret)
    decision = res.get("decision")
    answer = res.get("answer", "")[:200]
    has_ahsan = "ahsan" in answer.lower() or "sajjad" in answer.lower()
    out.append(f"Query: '{q}'")
    out.append(f"  Decision: {decision}")
    out.append(f"  Has Ahsan (should be NO for CTO): {'YES ❌' if has_ahsan else 'NO ✅'}")
    out.append(f"  Answer: {answer}")
    out.append("")

with open("debug_cto_name_verify.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
