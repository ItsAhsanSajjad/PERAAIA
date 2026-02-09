"""Find DG name chunk and test ranking for different queries."""
from index_store import load_index_and_chunks
from retriever import retrieve
from answerer import _get_ranked_chunks
import re

idx, chunks = load_index_and_chunks()

# 1. Find chunk(s) with DG's name
print("=== CHUNKS WITH DG NAME ===")
dg_chunks = []
for i, c in enumerate(chunks):
    text = c.get("text", "")
    if "Farrukh" in text or "farrukh" in text.lower():
        dg_chunks.append(i)
        print(f"  Chunk {i}: doc={c.get('doc_name','?')[:40]} page={c.get('loc_start','?')}")
        print(f"  Text: {text[:250]}")
        print()

# 2. Test different queries and see where the DG chunk ranks
test_queries = ["dg name?", "dg kon hai", "director general name", "who is director general pera"]
out = []
for q in test_queries:
    ret = retrieve(q)
    ranked = _get_ranked_chunks(ret, q)
    
    # Find DG name chunk position in ranked
    dg_pos = None
    for i, rc in enumerate(ranked):
        if "Farrukh" in rc.get("text", "") or "farrukh" in rc.get("text", "").lower():
            dg_pos = i + 1
            break
    
    has_evidence = ret.get("has_evidence", False)
    out.append(f"Query: '{q}'")
    out.append(f"  Evidence groups: {len(ret.get('evidence', []))}")
    out.append(f"  Total ranked: {len(ranked)}")
    out.append(f"  DG name chunk position: {dg_pos if dg_pos else 'NOT FOUND'}")
    if ranked:
        out.append(f"  Top 3 chunks:")
        for i, rc in enumerate(ranked[:3]):
            out.append(f"    [{i+1}] subj={rc['subject_match']} score={rc['score']:.3f} {rc['doc_name'][:30]} p.{rc['page']}")
            out.append(f"        {rc['text'][:100]}")
    out.append("")

with open("debug_dg_ranking.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print(f"Done. {len(dg_chunks)} DG chunks found. See debug_dg_ranking.txt")
