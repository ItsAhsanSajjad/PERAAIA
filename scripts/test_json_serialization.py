"""
Regression test: ensure retrieve() output is JSON-serializable (no numpy types).
Covers SEV-0 fix for PydanticSerializationError: numpy.int64.
"""
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_retrieve_json_serializable():
    """retrieve() output must survive json.dumps without TypeError."""
    from retriever import retrieve

    # These queries exercise the FAISS search path (not table registry bypass)
    queries = [
        "What is PERA?",
        "Director General responsibilities",
        "salary of CTO",
    ]

    for q in queries:
        print(f"\n[Test] Query: '{q}'")
        result = retrieve(q)

        # Must not raise TypeError: Object of type int64 is not JSON serializable
        try:
            serialized = json.dumps(result, ensure_ascii=False, default=str)
            print(f"  ✅ json.dumps OK ({len(serialized)} chars)")
        except TypeError as e:
            print(f"  ❌ FAIL: {e}")
            raise AssertionError(f"retrieve() output not JSON-serializable for '{q}': {e}")

        # Verify all chunk_ids are native int (not numpy.int64)
        for dg in result.get("evidence", []):
            for hit in dg.get("hits", []):
                cid = hit.get("chunk_id")
                if cid is not None:
                    assert isinstance(cid, int), \
                        f"chunk_id is {type(cid).__name__}, expected int"
                score = hit.get("score")
                assert isinstance(score, (int, float)), \
                    f"score is {type(score).__name__}, expected float"
                ps = hit.get("page_start")
                assert not hasattr(ps, 'dtype'), \
                    f"page_start is numpy type: {type(ps).__name__}"

        ev_count = sum(len(d.get("hits", [])) for d in result.get("evidence", []))
        print(f"  Evidence docs: {len(result.get('evidence', []))}, hits: {ev_count}")

    print("\n" + "=" * 50)
    print("ALL JSON SERIALIZATION TESTS PASSED ✅")


if __name__ == "__main__":
    test_retrieve_json_serializable()
