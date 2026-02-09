"""Verify reference accuracy: check that citation [N] points to the correct chunk."""
from retriever import retrieve
from answerer import answer_question, _get_ranked_chunks, format_evidence_for_llm, extract_references_simple
import re

out = []
q = "What are the powers of PERA?"
ret = retrieve(q)
res = answer_question(q, ret)

answer = res.get("answer", "")
refs = res.get("references", [])

out.append(f"Query: '{q}'")
out.append(f"Answer (first 200): {answer[:200]}")
out.append(f"\nReferences returned: {len(refs)}")
for r in refs:
    out.append(f"  [{r['id']}] {r['document'][:30]} p.{r['page_start']} -> {r['open_url'][:80]}")
    out.append(f"      Snippet: {r['snippet'][:100]}")

# Now check: what did the LLM cite?
cited = re.findall(r'\[(\d+)\]', answer)
out.append(f"\nLLM cited: {cited}")

# Show the context chunks that the LLM saw
ranked = _get_ranked_chunks(ret, q)
out.append(f"\nContext chunks sent to LLM (first 5):")
for i in range(min(5, len(ranked))):
    rc = ranked[i]
    out.append(f"  [{i+1}] {rc['doc_name'][:30]} p.{rc['page']} score={rc['score']:.3f}")
    out.append(f"      {rc['text'][:100]}")

with open("debug_ref_verify.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
