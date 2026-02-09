"""Trace DG salary query: what chunks reach the LLM and is SPPP data there?"""
from retriever import retrieve, rewrite_contextual_query
from answerer import _get_ranked_chunks

# Simulate the follow-up
q = "salay of dg"
last_q = "explain"
last_a = "Punjab Enforcement and Regulatory Authority (PERA) ka structure aur reporting mechanism..."
rw = rewrite_contextual_query(q, last_q, last_a)

out = []
out.append(f"Original: '{q}'")
out.append(f"Rewritten: '{rw}'")

ret = retrieve(rw)
ranked = _get_ranked_chunks(ret, q)
out.append(f"Total ranked: {len(ranked)}")
out.append("")

# Check for SPPP data and DG salary info
for i, rc in enumerate(ranked[:20]):
    text = rc.get("text", "")
    has_sppp = "SPPP" in text or "sppp" in text.lower()
    has_salary_num = any(w in text for w in ["350,082", "115,561", "90,219", "296,324", "250,051"])
    has_dg = "director general" in text.lower() or "DG" in text
    has_bs20 = "BS-20" in text or "BPS-20" in text or "bs-20" in text.lower()
    
    markers = []
    if has_sppp: markers.append("SPPP")
    if has_salary_num: markers.append("$$")
    if has_dg: markers.append("DG")
    if has_bs20: markers.append("BS20")
    
    out.append(f"[{i+1}] subj={rc['subject_match']} score={rc['score']:.3f} {' '.join(markers) if markers else '-'}")
    out.append(f"    {rc['doc_name'][:30]} p.{rc['page']}")
    out.append(f"    {text[:150]}")
    out.append("")

with open("debug_dg_salary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done. See debug_dg_salary.txt")
