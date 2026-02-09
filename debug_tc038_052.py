"""Diagnose TC-038 and TC-052 failures."""
from retriever import retrieve
from answerer import answer_question, _get_ranked_chunks, format_evidence_for_llm

out = []

# TC-038: "After arrest, what must happen next?"
out.append("=" * 60)
out.append("TC-038: 'After arrest, what must happen next?'")
out.append("=" * 60)
q = "After arrest, what must happen next?"
ret = retrieve(q)
ranked = _get_ranked_chunks(ret, q)
res = answer_question(q, ret)
out.append(f"Decision: {res.get('decision')}")
out.append(f"Full answer: {res.get('answer', '')}")
out.append(f"\nRanked chunks (first 5):")
for i, rc in enumerate(ranked[:5]):
    out.append(f"  [{i+1}] {rc['doc_name'][:30]} p.{rc['page']} score={rc['score']:.3f}")
    out.append(f"      {rc['text'][:150]}")

# Check if any chunk mentions 24 hours / magistrate
out.append(f"\nChunks with '24 hour' or 'magistrate' (in ALL ranked):")
for i, rc in enumerate(ranked):
    text_l = rc.get("text","").lower()
    if "24 hour" in text_l or "magistrate" in text_l:
        out.append(f"  [{i+1}] {rc['doc_name'][:30]} p.{rc['page']}")
        out.append(f"      {rc['text'][:200]}")

out.append("\n" + "=" * 60)
out.append("TC-052: 'Give me the list of all Scheduled laws and their sections.'")
out.append("=" * 60)
q2 = "Give me the list of all Scheduled laws and their sections."
ret2 = retrieve(q2)
ranked2 = _get_ranked_chunks(ret2, q2)
res2 = answer_question(q2, ret2)
out.append(f"Decision: {res2.get('decision')}")
out.append(f"Full answer: {res2.get('answer', '')[:300]}")
out.append(f"\nRanked chunks (first 5):")
for i, rc in enumerate(ranked2[:5]):
    out.append(f"  [{i+1}] {rc['doc_name'][:30]} p.{rc['page']} score={rc['score']:.3f}")
    out.append(f"      {rc['text'][:150]}")

# Check for "schedule" in all ranked
out.append(f"\nChunks with 'schedule' (first 5 matches):")
cnt = 0
for i, rc in enumerate(ranked2):
    text_l = rc.get("text","").lower()
    if "schedule" in text_l and cnt < 5:
        out.append(f"  [{i+1}] {rc['doc_name'][:30]} p.{rc['page']}")
        out.append(f"      {rc['text'][:200]}")
        cnt += 1

with open("debug_tc038_052.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
