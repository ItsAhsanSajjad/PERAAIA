"""Detailed diagnostic: what messages does the LLM actually receive?"""
from retriever import retrieve
from answerer import _get_ranked_chunks, format_evidence_for_llm, get_client, ANSWER_MODEL, MAX_DOCS, answer_question

out = []
q = "director general pera name"
ret = retrieve(q)
ranked = _get_ranked_chunks(ret, q)
ctx = format_evidence_for_llm(ranked)

# Test 1: Without history
res1 = answer_question(q, ret, conversation_history=None)
out.append(f"No history: {res1.get('decision')} -> {res1.get('answer','')[:200]}")

# Test 2: With poisoned history
history = [
    {"role": "user", "content": "name of cto"},
    {"role": "assistant", "content": "Yeh information PERA documents mein available nahi hai."},
    {"role": "user", "content": "name of cto"},
    {"role": "assistant", "content": "Yeh information PERA documents mein available nahi hai."},
]
res2 = answer_question(q, ret, conversation_history=history)
out.append(f"Poisoned: {res2.get('decision')} -> {res2.get('answer','')[:200]}")

out.append(f"\nRanked: {len(ranked)}, Context: {len(ctx)} chars, MAX_DOCS: {MAX_DOCS}")
if ranked:
    out.append(f"Top chunk: {ranked[0]['doc_name'][:30]} p.{ranked[0]['page']}")
    out.append(f"  text: {ranked[0]['text'][:150]}")

with open("debug_detailed_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
