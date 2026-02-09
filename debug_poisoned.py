"""Simulate poisoned history + verify fix."""
from retriever import retrieve
from answerer import answer_question

# Simulate conversation with multiple refusals in history
poisoned_history = [
    {"role": "user", "content": "name of cto"},
    {"role": "assistant", "content": "Yeh information PERA documents mein available nahi hai. Mazeed madad ke liye PERA se contact karein."},
    {"role": "user", "content": "name of cto"},  
    {"role": "assistant", "content": "Yeh information PERA documents mein available nahi hai. Mazeed madad ke liye PERA se contact karein."},
]

q = "director general pera name"
ret = retrieve(q)

# With poisoned history
res = answer_question(q, ret, conversation_history=poisoned_history)
print(f"WITH poisoned history:")
print(f"  Decision: {res.get('decision')}")
print(f"  Answer: {res.get('answer','')[:200]}")

# Without history  
res2 = answer_question(q, ret)
print(f"\nWITHOUT history:")
print(f"  Decision: {res2.get('decision')}")
print(f"  Answer: {res2.get('answer','')[:200]}")
