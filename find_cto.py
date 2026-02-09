import os, sys
os.environ["RETRIEVER_LLM_QUERY_REWRITE_ALWAYS"] = "0"
sys.path.insert(0, '.')
from retriever import retrieve
from answerer import _get_ranked_chunks, format_evidence_for_llm

query = "Chief Technology Officer pay and allowances SPPP Schedule pay and allowances"
retrieval = retrieve(query)

chunks = _get_ranked_chunks(retrieval, "cto pay in numbers")
context = format_evidence_for_llm(chunks)

with open("cto_context.txt", "w", encoding="utf-8") as f:
    f.write(f"has_evidence: {retrieval['has_evidence']}\n")
    f.write(f"Chunks used: {len(chunks)}\n")
    f.write(f"Context length: {len(context)} chars\n\n")
    f.write("--- CONTEXT ---\n")
    f.write(context[:3000])

print("Done - see cto_context.txt")
