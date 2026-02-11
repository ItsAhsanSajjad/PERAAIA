"""
PERA AI Strict Mode Tests
Tests the Intent Router, Evidence Relevance Gate, and Strict Mode enforcement.

Run: python -m pytest scripts/test_strict_mode.py -v
  OR: python scripts/test_strict_mode.py
"""
from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answerer import (
    classify_intent,
    QueryIntent,
    detect_sso_ambiguity,
    evidence_relevance_gate,
    _extract_query_keywords,
    _post_process_strict,
    _detect_roman_urdu,
    answer_question,
)


# =============================================================================
# A) Intent Classification Tests
# =============================================================================

class TestIntentClassification:
    """Tests for classify_intent()."""

    def test_org_overview_what_is_pera(self):
        assert classify_intent("What is PERA?") == QueryIntent.ORG_OVERVIEW

    def test_org_overview_urdu(self):
        assert classify_intent("PERA kya hai?") == QueryIntent.ORG_OVERVIEW

    def test_org_overview_powers(self):
        assert classify_intent("What are the powers of PERA?") == QueryIntent.ORG_OVERVIEW

    def test_org_overview_functions(self):
        assert classify_intent("PERA kia kam karti hai?") == QueryIntent.ORG_OVERVIEW

    def test_org_overview_purpose(self):
        assert classify_intent("Purpose of PERA") == QueryIntent.ORG_OVERVIEW

    def test_job_role_cto(self):
        assert classify_intent("cto?") == QueryIntent.JOB_ROLE

    def test_job_role_cto_salary(self):
        assert classify_intent("CTO ki salary kya hai?") == QueryIntent.JOB_ROLE

    def test_job_role_sso_system(self):
        assert classify_intent("System Support Officer duties") == QueryIntent.JOB_ROLE

    def test_job_role_qualification(self):
        assert classify_intent("Manager Development ki qualification?") == QueryIntent.JOB_ROLE

    def test_enforcement_eo_powers(self):
        assert classify_intent("EO powers kya hain?") == QueryIntent.ENFORCEMENT_LEGAL

    def test_enforcement_section(self):
        assert classify_intent("Section 12 kya kehta hai?") == QueryIntent.ENFORCEMENT_LEGAL

    def test_enforcement_arrest(self):
        assert classify_intent("Can PERA arrest someone?") == QueryIntent.ENFORCEMENT_LEGAL

    def test_enforcement_fir(self):
        assert classify_intent("PERA FIR kar sakti hai?") == QueryIntent.ENFORCEMENT_LEGAL

    def test_out_of_scope_html(self):
        assert classify_intent("html css difference") == QueryIntent.OUT_OF_SCOPE

    def test_out_of_scope_kotlin(self):
        assert classify_intent("kotlin kya hai") == QueryIntent.OUT_OF_SCOPE

    def test_out_of_scope_python(self):
        assert classify_intent("python programming") == QueryIntent.OUT_OF_SCOPE

    def test_out_of_scope_cooking(self):
        assert classify_intent("biryani recipe batao") == QueryIntent.OUT_OF_SCOPE

    def test_out_of_scope_funny_bat(self):
        assert classify_intent("cto yeh funny baat") == QueryIntent.OUT_OF_SCOPE

    def test_out_of_scope_funny_bat_variant(self):
        assert classify_intent("cto se funny bat karo") == QueryIntent.OUT_OF_SCOPE

    def test_out_of_scope_joke(self):
        assert classify_intent("ek joke sunao") == QueryIntent.OUT_OF_SCOPE

    def test_out_of_scope_ogra(self):
        assert classify_intent("OGRA ke rules kya hain?") == QueryIntent.OUT_OF_SCOPE

    def test_general_pera_fallback(self):
        assert classify_intent("PERA ke documents mein kya hai?") == QueryIntent.GENERAL_PERA


# =============================================================================
# B) SSO Ambiguity Detection Tests
# =============================================================================

class TestSSOAmbiguity:
    """Tests for detect_sso_ambiguity()."""

    def test_ambiguous_sso(self):
        assert detect_sso_ambiguity("SSO ki duties kya hain?") is True

    def test_disambiguated_system_support(self):
        assert detect_sso_ambiguity("System Support Officer ki duties kya hain?") is False

    def test_disambiguated_senior_staff(self):
        assert detect_sso_ambiguity("Senior Staff Officer ki duties?") is False

    def test_no_sso(self):
        assert detect_sso_ambiguity("CTO ki duties?") is False


# =============================================================================
# C) Evidence Relevance Gate Tests
# =============================================================================

class TestEvidenceRelevanceGate:
    """Tests for evidence_relevance_gate()."""

    def test_relevant_chunks_pass(self):
        """Chunks with keyword overlap should pass the gate."""
        retrieval = {
            "question": "Powers of PERA",
            "has_evidence": True,
            "evidence": [
                {
                    "doc_name": "PERA Act 2022",
                    "max_score": 0.65,
                    "hits": [
                        {"text": "The Punjab Enforcement and Regulatory Authority shall have the following powers and functions...", "score": 0.65, "page_start": 5},
                        {"text": "The Authority is established under the PERA Act to enforce regulatory compliance.", "score": 0.60, "page_start": 1},
                    ]
                }
            ]
        }
        result = evidence_relevance_gate(retrieval, "Powers of PERA", QueryIntent.ORG_OVERVIEW)
        assert result["has_evidence"] is True
        assert len(result["evidence"]) > 0

    def test_irrelevant_chunks_filtered(self):
        """Chunks without keyword overlap should be filtered out."""
        retrieval = {
            "question": "Powers of PERA",
            "has_evidence": True,
            "evidence": [
                {
                    "doc_name": "HR Manual",
                    "max_score": 0.55,
                    "hits": [
                        {"text": "Dress code: All employees must wear formal attire during office hours.", "score": 0.55, "page_start": 42},
                        {"text": "Leave policy: Casual leave is 10 days per year.", "score": 0.50, "page_start": 43},
                    ]
                }
            ]
        }
        result = evidence_relevance_gate(retrieval, "Powers of PERA", QueryIntent.ORG_OVERVIEW)
        # Should have no evidence because dress code/leave are irrelevant
        assert result["has_evidence"] is False

    def test_empty_retrieval(self):
        """Empty retrieval should pass through."""
        retrieval = {"question": "test", "has_evidence": False, "evidence": []}
        result = evidence_relevance_gate(retrieval, "test", QueryIntent.GENERAL_PERA)
        assert result["has_evidence"] is False

    def test_out_of_scope_html_no_relevant_chunks(self):
        """HTML question with random PERA chunks should be filtered."""
        retrieval = {
            "question": "html css difference",
            "has_evidence": True,
            "evidence": [
                {
                    "doc_name": "PERA Act 2022",
                    "max_score": 0.30,
                    "hits": [
                        {"text": "The Authority shall establish offices as required.", "score": 0.30, "page_start": 10},
                    ]
                }
            ]
        }
        result = evidence_relevance_gate(retrieval, "html css difference", QueryIntent.OUT_OF_SCOPE)
        # Keywords "html" and "css" won't match PERA text
        assert result["has_evidence"] is False


# =============================================================================
# D) Keyword Extraction Tests
# =============================================================================

class TestKeywordExtraction:
    """Tests for _extract_query_keywords()."""

    def test_basic_extraction(self):
        kws = _extract_query_keywords("What are the powers of PERA?")
        assert "powers" in kws
        # PERA should expand
        assert any(w in kws for w in ["punjab", "enforcement", "regulatory", "authority"])

    def test_cto_expansion(self):
        kws = _extract_query_keywords("CTO salary?")
        assert "chief" in kws
        assert "technology" in kws
        assert "officer" in kws
        assert "salary" in kws

    def test_stops_removed(self):
        kws = _extract_query_keywords("What is the purpose of PERA?")
        assert "what" not in kws
        assert "the" not in kws
        assert "is" not in kws


# =============================================================================
# E) Post-Processing Strict Mode Tests
# =============================================================================

class TestPostProcessStrict:
    """Tests for _post_process_strict()."""

    def test_hedge_phrase_caught(self):
        ans = "The general rules may apply for this position."
        result = _post_process_strict(ans, QueryIntent.JOB_ROLE, "cto powers?")
        assert result is not None  # should override
        assert "outside my scope" in result.lower() or "sorry" in result.lower()

    def test_clean_answer_passes(self):
        ans = "The CTO has the following powers as defined in the PERA Act..."
        result = _post_process_strict(ans, QueryIntent.JOB_ROLE, "cto powers?")
        assert result is None  # should NOT override

    def test_general_workplace_caught(self):
        ans = "Based on general workplace etiquette, the CTO should..."
        result = _post_process_strict(ans, QueryIntent.JOB_ROLE, "cto?")
        assert result is not None


# =============================================================================
# F) Integration Tests (require OpenAI key + index)
# These test the full answer_question pipeline.
# Skip if no API key configured.
# =============================================================================

class TestIntegration:
    """
    Full pipeline integration tests.
    These require OPENAI_API_KEY and a working FAISS index.
    They are skipped gracefully if not available.
    """

    @staticmethod
    def _can_run():
        return bool(os.getenv("OPENAI_API_KEY"))

    def test_powers_of_pera_citations(self):
        """'Powers of PERA' must cite Act/Bill, NOT HR dress code or Squads."""
        if not self._can_run():
            print("SKIP: No OPENAI_API_KEY")
            return

        from retriever import retrieve
        retrieval = retrieve("What are the powers of PERA?")
        result = answer_question("What are the powers of PERA?", retrieval)

        assert result["decision"] in ("answer", "refuse"), f"Unexpected decision: {result['decision']}"
        if result["decision"] == "answer":
            # Check that citations don't reference dress code or weapons
            for ref in result.get("references", []):
                doc = (ref.get("document") or "").lower()
                snippet = (ref.get("snippet") or "").lower()
                assert "dress code" not in snippet, f"Dress code citation leaked: {ref}"
                assert "squads" not in doc or "weapon" not in snippet, f"Squads/Weapons citation leaked: {ref}"

    def test_cto_no_general_rules(self):
        """'cto?' must not say 'general rules may apply'."""
        if not self._can_run():
            print("SKIP: No OPENAI_API_KEY")
            return

        from retriever import retrieve
        retrieval = retrieve("cto?")
        result = answer_question("cto?", retrieval)

        lower_ans = result["answer"].lower()
        assert "general rules may apply" not in lower_ans, f"Hedge leak: {result['answer'][:200]}"

    def test_cto_funny_bat_refuses(self):
        """'cto yeh… funny bat' must strictly refuse."""
        if not self._can_run():
            print("SKIP: No OPENAI_API_KEY")
            return

        result = answer_question("cto yeh funny bat", {"has_evidence": False, "evidence": []})
        assert result["decision"] == "refuse", f"Should refuse, got: {result['decision']}"
        assert result["references"] == [], f"Should have no refs, got: {result['references']}"

    def test_html_css_strict_refuse(self):
        """'html css difference' must strict refuse with no PERA citations."""
        if not self._can_run():
            print("SKIP: No OPENAI_API_KEY")
            return

        result = answer_question("html css difference", {"has_evidence": False, "evidence": []})
        assert result["decision"] == "refuse", f"Should refuse, got: {result['decision']}"
        assert result["references"] == [], f"Should have no refs, got: {result['references']}"

    def test_kotlin_strict_refuse(self):
        """'kotlin kya hai' must strict refuse with no unrelated citations."""
        if not self._can_run():
            print("SKIP: No OPENAI_API_KEY")
            return

        result = answer_question("kotlin kya hai", {"has_evidence": False, "evidence": []})
        assert result["decision"] == "refuse", f"Should refuse, got: {result['decision']}"
        assert result["references"] == [], f"Should have no refs, got: {result['references']}"


# =============================================================================
# Run as script
# =============================================================================
def _run_tests():
    """Simple test runner (no pytest needed)."""
    import traceback

    test_classes = [
        TestIntentClassification,
        TestSSOAmbiguity,
        TestEvidenceRelevanceGate,
        TestKeywordExtraction,
        TestPostProcessStrict,
        TestIntegration,
    ]

    total = 0
    passed = 0
    failed = 0
    skipped = 0
    failures = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        print(f"\n{'='*60}")
        print(f"  {cls.__name__} ({len(methods)} tests)")
        print(f"{'='*60}")

        for method_name in sorted(methods):
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
                print(f"  PASS {method_name}")
            except Exception as e:
                if "SKIP" in str(e):
                    skipped += 1
                    print(f"  SKIP {method_name} (skipped)")
                else:
                    failed += 1
                    failures.append((cls.__name__, method_name, e))
                    print(f"  FAIL {method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped / {total} total")
    print(f"{'='*60}")

    if failures:
        print("\nFAILURES:")
        for cls_name, method_name, err in failures:
            print(f"\n  {cls_name}.{method_name}:")
            traceback.print_exception(type(err), err, err.__traceback__)

    return failed == 0


if __name__ == "__main__":
    success = _run_tests()
    sys.exit(0 if success else 1)
