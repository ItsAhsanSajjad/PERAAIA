"""Debug: dump top retrieved docs for 'Powers of PERA' to see doc names."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retriever import retrieve
from answerer import classify_intent, evidence_relevance_gate, QueryIntent

q = "What are the powers of PERA?"
r = retrieve(q)

print("=== RAW RETRIEVAL ===")
for d in r["evidence"][:8]:
    print(f"  DOC: {d['doc_name']} (max_score={d['max_score']:.3f}, hits={len(d['hits'])})")
    for h in d["hits"][:3]:
        snippet = (h.get("text") or "")[:120].replace("\n", " ")
        print(f"    score={h.get('score',0):.3f} page={h.get('page_start','?')} | {snippet}")

intent = classify_intent(q)
print(f"\n=== INTENT: {intent} ===")

gated = evidence_relevance_gate(r, q, intent)
print(f"\n=== GATED RETRIEVAL (has_evidence={gated['has_evidence']}) ===")
print(f"  gate_reason: {gated.get('_gate_reason')}")
for d in gated.get("evidence", [])[:5]:
    print(f"  DOC: {d['doc_name']} (max_score={d['max_score']:.3f}, hits={len(d['hits'])})")
    for h in d["hits"][:2]:
        snippet = (h.get("text") or "")[:120].replace("\n", " ")
        print(f"    score={h.get('score',0):.3f} page={h.get('page_start','?')} | {snippet}")
