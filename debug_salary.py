
from retriever import retrieve
from answerer import answer_question
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_salary():
    queries = [
        "What is the salary of CTO?",
        "What is the pay scale for SPPP-1?",
        "CTO salary SPPP-1"
    ]
    
    for q in queries:
        print(f"\n--- Query: {q} ---")
        ret = retrieve(q)
        evidence = ret.get("evidence", [])
        if evidence:
            first_chunk = evidence[0]
            # Write to file to avoid truncation
            with open("debug_chunk.json", "w", encoding="utf-8") as f:
                json.dump(first_chunk, f, indent=2, default=str)
            print("Wrote first chunk to debug_chunk.json")
            
            # Try to get content
            content_key = "content"
            if "content" not in first_chunk:
                content_key = "text" # common fallback
            
            for i, chunk in enumerate(evidence):
                content = chunk.get(content_key, "")
                if not content: continue
                
                if "SPPP-1" in content or "CTO" in content:
                    doc_name = chunk.get('document', chunk.get('source', 'unknown_doc'))
                    page_num = chunk.get('page', chunk.get('page_number', '?'))
                    print(f"[Evidence {i}] {doc_name} p.{page_num}: {content[:300]}...")
        else:
            print("No evidence found")

if __name__ == "__main__":
    test_salary()
