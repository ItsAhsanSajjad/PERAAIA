import os
from dotenv import load_dotenv
from retriever import rewrite_contextual_query

load_dotenv()

def test_context_rewrite():
    # Simulate history
    last_q = "What are the other benefits for CTO?"
    last_a = "The salary and benefits for the **Android Developer** position are as follows:\n- Salary: SPPP-4\n- Additional Benefits: Any other benefit admissible to SPPP-4 as mentioned in Schedule-III [1] [2]."
    
    current_q = "what are tbat benefits"
    
    print(f"--- Context ---")
    print(f"User: {last_q}")
    print(f"Bot: {last_a}")
    print(f"User: {current_q}")
    
    rewritten = rewrite_contextual_query(current_q, last_q, last_a)
    print(f"\n--- Result ---")
    print(f"Rewritten: '{rewritten}'")
    
    if "benefit" in rewritten.lower() or "schedule" in rewritten.lower() or "sppp" in rewritten.lower():
        print("✅ SUCCESS: Query rewritten to specific terms.")
    else:
        print("❌ FAILURE: Query remains vague or incorrect.")

if __name__ == "__main__":
    test_context_rewrite()
