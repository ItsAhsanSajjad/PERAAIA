"""Diagnose: raw LLM call with CTO evidence."""
from answerer import _get_ranked_chunks, format_evidence_for_llm, get_client, ANSWER_MODEL
from retriever import retrieve

q = "CTO ki powers kia hain?"
ret = retrieve(q)
ranked = _get_ranked_chunks(ret, q)
context = format_evidence_for_llm(ranked)

client = get_client()

# Minimal system prompt - just give context and ask
system = f"Answer the user's question based on this context:\n\n{context}"
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": q}
]

response = client.chat.completions.create(
    model=ANSWER_MODEL,
    messages=messages,
    temperature=0.3,
)
answer = response.choices[0].message.content

out = []
out.append(f"Model: {ANSWER_MODEL}")
out.append(f"Context chars: {len(context)}")
out.append(f"Ranked chunks: {len(ranked)}")
out.append(f"\nRAW LLM Answer:\n{answer}")

# Now test with full system prompt
from answerer import answer_question
res2 = answer_question(q, ret)
out.append(f"\n\nFULL PIPELINE Answer:")
out.append(f"Decision: {res2.get('decision')}")
out.append(f"Answer: {res2.get('answer','')[:300]}")

with open("debug_cto_llm.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done. See debug_cto_llm.txt")
