"""Quick diagnostic: check if pipeline works at all."""
import traceback
try:
    from retriever import retrieve
    from answerer import answer_question, _get_ranked_chunks, format_evidence_for_llm
    
    q = "director general name"
    ret = retrieve(q)
    print(f"has_evidence: {ret.get('has_evidence')}")
    print(f"evidence count: {len(ret.get('evidence', []))}")
    
    ranked = _get_ranked_chunks(ret, q)
    print(f"ranked chunks: {len(ranked)}")
    
    if ranked:
        ctx = format_evidence_for_llm(ranked)
        print(f"context chars: {len(ctx)}")
        print(f"context empty: {len(ctx.strip()) == 0}")
    
    res = answer_question(q, ret)
    print(f"decision: {res.get('decision')}")
    print(f"answer: {res.get('answer','')[:200]}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
