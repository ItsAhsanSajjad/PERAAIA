"""Compare full LLM context for schedule 3 vs schedule 4."""
from retriever import retrieve, _expand_abbreviations
from answerer import _get_ranked_chunks, format_evidence_for_llm

out = []
for q in ["schedule 3", "schedule 4"]:
    expanded = _expand_abbreviations(q)
    out.append(f"=== Query: '{q}' -> '{expanded}' ===")
    ret = retrieve(q)
    ranked = _get_ranked_chunks(ret, q)
    context = format_evidence_for_llm(ranked)
    
    # Show first 500 chars of context
    out.append(f"Context length: {len(context)}")
    out.append(f"First 500 chars:")
    out.append(context[:500])
    out.append("")

with open("compare_schedule34.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
