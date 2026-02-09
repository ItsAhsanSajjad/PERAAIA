"""Find the DG name chunk's exact position in ranked results."""
from retriever import retrieve
from answerer import _get_ranked_chunks

q = "director general pera name"
ret = retrieve(q)
ranked = _get_ranked_chunks(ret, q)

out = []
out.append(f"Query: '{q}'")
out.append(f"Total ranked: {len(ranked)}")
out.append("")

# Find DG name chunk (contains Farrukh)
dg_pos = None
for i, rc in enumerate(ranked):
    text = rc.get("text", "")
    has_farrukh = "farrukh" in text.lower() or "atiq" in text.lower()
    if has_farrukh:
        if dg_pos is None:
            dg_pos = i + 1
        out.append(f"*** DG NAME at [{i+1}] subj={rc['subject_match']} score={rc['score']:.3f}")
        out.append(f"    {rc['doc_name'][:30]} p.{rc['page']}")
        out.append(f"    {text[:200]}")
        out.append("")

if dg_pos:
    out.insert(2, f"DG name chunk at position: {dg_pos}")
else:
    out.insert(2, "DG name chunk NOT FOUND in any ranked chunks!")

# Show first 5 for comparison
out.append("\n--- Top 5 chunks ---")
for i in range(min(5, len(ranked))):
    rc = ranked[i]
    out.append(f"[{i+1}] subj={rc['subject_match']} score={rc['score']:.3f} {rc['doc_name'][:30]} p.{rc['page']}")
    out.append(f"    {rc['text'][:100]}")

with open("debug_dg_position.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
