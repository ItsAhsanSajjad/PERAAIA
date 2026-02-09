"""Re-test TC-038 after fix."""
from retriever import retrieve
from answerer import answer_question, _get_ranked_chunks

out = []
q = "After arrest, what must happen next?"
ret = retrieve(q)
ranked = _get_ranked_chunks(ret, q)
res = answer_question(q, ret)

out.append(f"Decision: {res.get('decision')}")
answer = res.get('answer', '')
out.append(f"Full answer:\n{answer}")

has_24h = "24" in answer.lower()
has_magistrate = "magistrate" in answer.lower()
out.append(f"\nHas '24': {has_24h}")
out.append(f"Has 'magistrate': {has_magistrate}")

out.append(f"\nTop 5 ranked chunks:")
for i, rc in enumerate(ranked[:5]):
    out.append(f"  [{i+1}] subj={rc['subject_match']} {rc['doc_name'][:30]} p.{rc['page']}")
    out.append(f"      {rc['text'][:100]}")

with open("debug_tc038_retest.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
