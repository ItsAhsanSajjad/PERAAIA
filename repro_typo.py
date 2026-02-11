from retriever import retrieve, _expand_abbreviations, _ABBREV_MAP
from answerer import answer_question, _get_ranked_chunks

# Test case 1: Typo "schedudle 4"
q_typo = "schedudle 4"
print(f"--- Testing typo query: '{q_typo}' ---")

# Check expansion
expanded = _expand_abbreviations(q_typo)
print(f"Expansion result: '{expanded}'")

# Check retrieval
ret = retrieve(q_typo)
chunks = ret.get("evidence", [])
print(f"Top 3 retrieved chunks:")
for i, c in enumerate(chunks[:3]):
    score = c.get('score') or 0.0
    print(f"  [{i+1}] {c.get('doc_name')} p.{c.get('page')} (Score: {score:.4f})")
    text = c.get('text') or ""
    print(f"      Snippet: {text[:100]}...")

# Check Answerer decision
res = answer_question(q_typo, ret)
print(f"Decision: {res.get('decision')}")
print(f"Answer: {res.get('answer')[:100]}...")


# Test case 2: Correct "schedule 4"
q_correct = "schedule 4"
print(f"\n--- Testing correct query: '{q_correct}' ---")
expanded_correct = _expand_abbreviations(q_correct)
print(f"Expansion result: '{expanded_correct}'") 

ret_correct = retrieve(q_correct)
chunks_correct = ret_correct.get("evidence", [])
print(f"Top 3 retrieved chunks:")
for i, c in enumerate(chunks_correct[:3]):
    score = c.get('score') or 0.0
    print(f"  [{i+1}] {c.get('doc_name')} p.{c.get('page')} (Score: {score:.4f})")
    text = c.get('text') or ""
    print(f"      Snippet: {text[:100]}...")
