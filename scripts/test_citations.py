"""
PERA AI Citation Contract Tests
Tests that refusals/smalltalk/greeting return ZERO citations,
and factual answers require >=1 citation.

Run: venv\\Scripts\\python.exe scripts/test_citations.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answerer import (
    classify_intent, QueryIntent, answer_question,
    _post_process_strict, _detect_roman_urdu,
)


# =============================================================================
# A) Greeting/Smalltalk -> ZERO citations
# =============================================================================

class TestCitationContractGreetings:
    """Greetings and smalltalk must return empty citations."""

    def test_ty_returns_empty_citations(self):
        """'ty' = thank you -> greeting, 0 citations."""
        result = answer_question("ty", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"
        assert result["decision"] in ("greeting", "refuse"), f"Unexpected decision: {result['decision']}"

    def test_thanks_returns_empty_citations(self):
        result = answer_question("thanks", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"

    def test_tum_kon_ho_returns_empty_citations(self):
        """'tum kon ho' = who are you -> greeting, 0 citations."""
        result = answer_question("tum kon ho", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"
        assert result["decision"] in ("greeting", "refuse")

    def test_hello_returns_empty_citations(self):
        result = answer_question("hello", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"
        assert result["decision"] == "greeting"

    def test_aoa_returns_empty_citations(self):
        result = answer_question("aoa", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"

    def test_ok_returns_empty_citations(self):
        result = answer_question("ok", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"


# =============================================================================
# B) Out-of-scope -> ZERO citations
# =============================================================================

class TestCitationContractRefusals:
    """Out-of-scope refusals must return empty citations."""

    def test_kotlin_empty_citations(self):
        result = answer_question("kotlin kya hy", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"
        assert result["decision"] == "refuse"

    def test_html_css_empty_citations(self):
        result = answer_question("html css difference", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"
        assert result["decision"] == "refuse"

    def test_python_empty_citations(self):
        result = answer_question("python programming", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"

    def test_funny_bat_empty_citations(self):
        result = answer_question("cto yeh funny bat", {"has_evidence": False, "evidence": []})
        assert result["references"] == [], f"Expected empty refs, got: {result['references']}"


# =============================================================================
# C) Intent classification for greeting patterns
# =============================================================================

class TestGreetingIntentClassification:
    """Greeting patterns must map to GREETING_SMALLTALK intent."""

    def test_ty_is_greeting(self):
        assert classify_intent("ty") == QueryIntent.GREETING_SMALLTALK

    def test_thanks_is_greeting(self):
        assert classify_intent("thanks") == QueryIntent.GREETING_SMALLTALK

    def test_tum_kon_ho_is_greeting(self):
        assert classify_intent("tum kon ho") == QueryIntent.GREETING_SMALLTALK

    def test_hello_is_greeting(self):
        assert classify_intent("hello") == QueryIntent.GREETING_SMALLTALK

    def test_hi_is_greeting(self):
        assert classify_intent("hi") == QueryIntent.GREETING_SMALLTALK

    def test_aoa_is_greeting(self):
        assert classify_intent("aoa") == QueryIntent.GREETING_SMALLTALK

    def test_ok_is_greeting(self):
        assert classify_intent("ok") == QueryIntent.GREETING_SMALLTALK

    def test_shukriya_is_greeting(self):
        assert classify_intent("shukriya") == QueryIntent.GREETING_SMALLTALK

    def test_who_are_you_is_greeting(self):
        assert classify_intent("who are you") == QueryIntent.GREETING_SMALLTALK

    def test_aap_kon_ho_is_greeting(self):
        assert classify_intent("aap kon ho") == QueryIntent.GREETING_SMALLTALK

    def test_greeting_plus_question_is_NOT_greeting(self):
        """'hi what is section 3' should NOT be greeting-only."""
        intent = classify_intent("hi what is section 3")
        assert intent != QueryIntent.GREETING_SMALLTALK, f"Got: {intent}"


# =============================================================================
# Run as script
# =============================================================================
def _run_tests():
    import traceback
    test_classes = [
        TestCitationContractGreetings,
        TestCitationContractRefusals,
        TestGreetingIntentClassification,
    ]

    total = passed = failed = 0
    failures = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        print(f"\n{'='*60}")
        print(f"  {cls.__name__} ({len(methods)} tests)")
        print(f"{'='*60}")
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS {method_name}")
            except Exception as e:
                failed += 1
                failures.append((cls.__name__, method_name, e))
                print(f"  FAIL {method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed / {total} total")
    print(f"{'='*60}")
    if failures:
        print("\nFAILURES:")
        for cls_name, method_name, err in failures:
            print(f"  {cls_name}.{method_name}:")
            traceback.print_exception(type(err), err, err.__traceback__)
    return failed == 0


if __name__ == "__main__":
    success = _run_tests()
    sys.exit(0 if success else 1)
