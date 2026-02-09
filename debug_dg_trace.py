"""Trace: why does 'who is director genral pera' fail?"""
from retriever import retrieve, rewrite_contextual_query
from answerer import answer_question, _get_ranked_chunks

out = []

# Test 1: The failing query
q1 = "who is director genral pera"
out.append(f"=== TEST 1: '{q1}' ===")
rw1 = rewrite_contextual_query(q1, None, None)
out.append(f"  Rewritten: '{rw1}'")
ret1 = retrieve(rw1)
out.append(f"  Evidence groups: {len(ret1.get('evidence', []))}")
out.append(f"  has_evidence: {ret1.get('has_evidence')}")
for eg in ret1.get("evidence", [])[:3]:
    out.append(f"  Doc: {eg['doc_name'][:40]} score={eg['max_score']:.3f}")
    for h in eg.get("hits", [])[:2]:
        out.append(f"    Hit: {h.get('text','')[:120]}")

res1 = answer_question(q1, ret1)
out.append(f"  Decision: {res1.get('decision')}")
out.append(f"  Answer: {res1.get('answer','')[:250]}")

# Test 2: The working query
q2 = "who is head of pera"
out.append(f"\n=== TEST 2: '{q2}' ===")
rw2 = rewrite_contextual_query(q2, None, None)
out.append(f"  Rewritten: '{rw2}'")
ret2 = retrieve(rw2)
out.append(f"  Evidence groups: {len(ret2.get('evidence', []))}")
out.append(f"  has_evidence: {ret2.get('has_evidence')}")
for eg in ret2.get("evidence", [])[:3]:
    out.append(f"  Doc: {eg['doc_name'][:40]} score={eg['max_score']:.3f}")

res2 = answer_question(q2, ret2)
out.append(f"  Decision: {res2.get('decision')}")
out.append(f"  Answer: {res2.get('answer','')[:250]}")

# Test 3: Direct without rewrite
q3 = "who is director general pera"
out.append(f"\n=== TEST 3: '{q3}' (no rewrite) ===")
ret3 = retrieve(q3)
out.append(f"  Evidence groups: {len(ret3.get('evidence', []))}")
out.append(f"  has_evidence: {ret3.get('has_evidence')}")

res3 = answer_question(q3, ret3)
out.append(f"  Decision: {res3.get('decision')}")
out.append(f"  Answer: {res3.get('answer','')[:250]}")

with open("debug_dg_trace.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done. See debug_dg_trace.txt")
