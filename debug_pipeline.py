"""End-to-end pipeline trace: monitoring manager salary"""
from retriever import retrieve, rewrite_contextual_query
from answerer import answer_question, _get_ranked_chunks
import re, json

last_q = "monitoring manager kia krta"
last_a = "Monitoring Manager (MIS & GIS) ke responsibilities ye hain..."
current_q = "salarey ktna hai"

out = []

out.append("=== STEP 1: REWRITE ===")
rewritten = rewrite_contextual_query(current_q, last_q, last_a)
out.append(f"  Original: '{current_q}'")
out.append(f"  Rewritten: '{rewritten}'")

out.append("\n=== STEP 2: RETRIEVE ===")
retrieval = retrieve(rewritten)
evidence = retrieval.get("evidence", [])
out.append(f"  Evidence groups: {len(evidence)}")

has_numbers = False
for eg in evidence:
    for hit in eg.get("hits", []):
        text = hit.get("text", "")
        if re.search(r'\d{2,3},\d{3}', text) and "SPPP" in text:
            has_numbers = True
            out.append(f"  ✅ SPPP numbers in: {eg['doc_name']} score={eg['max_score']:.3f}")
            out.append(f"     {text[:150]}")

if not has_numbers:
    out.append("  ❌ NO SPPP salary numbers in retrieval!")

out.append("\n=== STEP 3: RANKED CHUNKS (top 12) ===")
ranked = _get_ranked_chunks(retrieval, current_q)
out.append(f"  Total ranked: {len(ranked)}")
for i, rc in enumerate(ranked[:12]):
    has_num = bool(re.search(r'\d{2,3},\d{3}', rc.get("text", "")))
    flag = "💰" if has_num and "SPPP" in rc.get("text","") else "  "
    out.append(f"  [{i+1}]{flag} subj={rc['subject_match']} score={rc['score']:.3f} {rc['doc_name'][:35]} p.{rc['page']}")

out.append("\n=== STEP 4: ANSWER ===")
result = answer_question(current_q, retrieval)
ans = result.get("answer", "")[:500]
refs = result.get("references", [])
out.append(f"  Answer: {ans}")
out.append(f"  Refs: {len(refs)}")

with open("debug_pipeline_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done. See debug_pipeline_out.txt")
