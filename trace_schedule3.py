"""Trace schedule 3 LLM call."""
from retriever import retrieve, _expand_abbreviations
from answerer import answer_question

q = "schedule 3"
expanded = _expand_abbreviations(q)
print(f"LLM will see: '{expanded}'")

ret = retrieve(q)
res = answer_question(q, ret)

print(f"Decision: {res.get('decision')}")
print(f"Answer: {res.get('answer','')[:300]}")
print(f"Refs: {len(res.get('references', []))}")
