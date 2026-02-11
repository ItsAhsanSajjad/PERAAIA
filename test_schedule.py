"""Test schedule queries after normalization fix - output to file."""
from retriever import retrieve, _expand_abbreviations
from answerer import answer_question, _get_ranked_chunks

out = []
queries = ["schedule 3", "schedule 4", "schedule 5"]
for q in queries:
    expanded = _expand_abbreviations(q)
    out.append(f"Query: '{q}' -> Expanded: '{expanded}'")
    ret = retrieve(q)
    ranked = _get_ranked_chunks(ret, q)
    res = answer_question(q, ret)
    decision = res.get('decision')
    answer = res.get('answer', '')[:300]
    refs = res.get('references', [])
    
    out.append(f"Decision: {decision}")
    out.append(f"Answer: {answer}")
    out.append(f"References: {len(refs)}")
    for r in refs[:3]:
        out.append(f"  {r}")
    out.append(f"Top 3 chunks:")
    for j, rc in enumerate(ranked[:3]):
        out.append(f"  [{j+1}] subj={rc['subject_match']} score={rc['score']:.3f} {rc['doc_name'][:30]} p.{rc['page']}")
        out.append(f"    {rc['text'][:120]}")
    out.append("")

with open("test_schedule_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
