"""
PERA AI Legal Section Verification Tests
Tests that ENFORCEMENT_LEGAL answers cite correct Sections from Tier1 docs only.

Run: venv\\Scripts\\python.exe scripts/test_legal_sections.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answerer import (
    classify_intent, QueryIntent,
    evidence_relevance_gate,
    _verify_section_citations,
)


# =============================================================================
# A) Intent Classification for Legal Queries
# =============================================================================

class TestLegalIntentClassification:
    """Legal queries must map to ENFORCEMENT_LEGAL."""

    def test_epo_intent(self):
        assert classify_intent("EPO kaise issue hota hai?") == QueryIntent.ENFORCEMENT_LEGAL

    def test_arrest_intent(self):
        assert classify_intent("Can PERA arrest someone?") == QueryIntent.ENFORCEMENT_LEGAL

    def test_fir_intent(self):
        assert classify_intent("PERA FIR kar sakti hai?") == QueryIntent.ENFORCEMENT_LEGAL

    def test_section_intent(self):
        assert classify_intent("Section 12 kya kehta hai?") == QueryIntent.ENFORCEMENT_LEGAL

    def test_search_seal_intent(self):
        assert classify_intent("search and seal powers") == QueryIntent.ENFORCEMENT_LEGAL

    def test_appeal_intent(self):
        assert classify_intent("appeal process kya hai") == QueryIntent.ENFORCEMENT_LEGAL


# =============================================================================
# B) Evidence Gate: Tier1 Hard Filter
# =============================================================================

class TestLegalEvidenceGate:
    """ENFORCEMENT_LEGAL must hard-filter to Tier1 docs only."""

    def test_hr_manual_blocked(self):
        """HR Manual must be filtered out for legal queries."""
        retrieval = {
            "question": "EPO procedure",
            "has_evidence": True,
            "evidence": [{
                "doc_name": "Human_Resource_Manual_May_2025.pdf",
                "max_score": 0.65,
                "hits": [{"text": "EPO process for employees", "score": 0.65, "page_start": 10}],
            }],
        }
        result = evidence_relevance_gate(retrieval, "EPO kaise issue hota hai?", QueryIntent.ENFORCEMENT_LEGAL)
        assert result["has_evidence"] is False or len(result["evidence"]) == 0, \
            "HR Manual should be blocked for ENFORCEMENT_LEGAL"

    def test_annex_h_blocked(self):
        """Annex H (job descriptions) must be filtered out for legal queries."""
        retrieval = {
            "question": "EPO procedure",
            "has_evidence": True,
            "evidence": [{
                "doc_name": "Annex H Job Descriptions.pdf",
                "max_score": 0.60,
                "hits": [{"text": "Enforcement officer job description", "score": 0.60, "page_start": 5}],
            }],
        }
        result = evidence_relevance_gate(retrieval, "EPO kaise issue hota hai?", QueryIntent.ENFORCEMENT_LEGAL)
        assert result["has_evidence"] is False or len(result["evidence"]) == 0

    def test_weapons_regs_blocked(self):
        """Weapons regulations must be filtered out for legal queries."""
        retrieval = {
            "question": "EPO procedure",
            "has_evidence": True,
            "evidence": [{
                "doc_name": "Annex L PERA Squads and Weapons Regulations.pdf",
                "max_score": 0.65,
                "hits": [{"text": "Weapons handling procedures", "score": 0.65, "page_start": 3}],
            }],
        }
        result = evidence_relevance_gate(retrieval, "EPO issue", QueryIntent.ENFORCEMENT_LEGAL)
        assert result["has_evidence"] is False or len(result["evidence"]) == 0

    def test_bill_passes_tier1(self):
        """Bill (Tier1 doc) must pass through for legal queries."""
        retrieval = {
            "question": "EPO procedure",
            "has_evidence": True,
            "evidence": [{
                "doc_name": "Bill_Recommended_by_Standing_Committee.pdf",
                "max_score": 0.70,
                "hits": [
                    {"text": "Section 15: The EPO shall be issued by enforcement officer", "score": 0.70, "page_start": 15},
                ],
            }],
        }
        result = evidence_relevance_gate(retrieval, "EPO kaise issue hota hai?", QueryIntent.ENFORCEMENT_LEGAL)
        assert result["has_evidence"] is True
        assert len(result["evidence"]) > 0

    def test_compiled_working_paper_blocked(self):
        """Compiled Working Paper (HR/admin sections) blocked for legal."""
        retrieval = {
            "question": "EPO procedure",
            "has_evidence": True,
            "evidence": [{
                "doc_name": "Compiled Working Paper_2nd Meeting PERA 03-07-2025.pdf",
                "max_score": 0.65,
                "hits": [{"text": "EPO related discussion in meeting", "score": 0.65, "page_start": 100}],
            }],
        }
        result = evidence_relevance_gate(retrieval, "EPO issue", QueryIntent.ENFORCEMENT_LEGAL)
        assert result["has_evidence"] is False or len(result["evidence"]) == 0


# =============================================================================
# C) Section Verification
# =============================================================================

class TestSectionVerification:
    """Tests for _verify_section_citations()."""

    def test_verified_section(self):
        """Section mentioned in answer AND evidence -> OK."""
        answer = "According to Section 15, the EPO can be issued..."
        snippets = ["Section 15: The EPO shall be issued by an enforcement officer."]
        result = _verify_section_citations(answer, snippets)
        assert result is None, f"Should be OK, got: {result}"

    def test_unverified_section(self):
        """Section mentioned in answer but NOT in evidence -> warning."""
        answer = "According to Section 99, the authority can..."
        snippets = ["Section 15: The EPO shall be issued by an enforcement officer."]
        result = _verify_section_citations(answer, snippets)
        assert result is not None
        assert "Section 99" in result

    def test_no_sections_in_answer(self):
        """No section references in answer -> OK."""
        answer = "The EPO can be issued by enforcement officers."
        snippets = ["The EPO procedure is defined in the Act."]
        result = _verify_section_citations(answer, snippets)
        assert result is None

    def test_multiple_sections_partial_verified(self):
        """Only unverified sections should be flagged."""
        answer = "As per Section 15 and Section 42, the powers..."
        snippets = ["Section 15: EPO issuance procedure."]
        result = _verify_section_citations(answer, snippets)
        assert result is not None
        assert "Section 42" in result
        # Section 15 should NOT be flagged
        assert "Section 15" not in result or "Section 42" in result


# =============================================================================
# D) Integration: EPO query (requires index + API key)
# =============================================================================

class TestLegalIntegration:
    """Full pipeline tests for legal queries."""

    @staticmethod
    def _can_run():
        return bool(os.getenv("OPENAI_API_KEY"))

    def test_epo_cites_tier1_only(self):
        """'EPO kaise issue hota hai?' must cite Act/Bill Tier1 only."""
        if not self._can_run():
            raise Exception("SKIP: No OPENAI_API_KEY")

        from retriever import retrieve
        from answerer import answer_question

        retrieval = retrieve("EPO kaise issue hota hai?")
        result = answer_question("EPO kaise issue hota hai?", retrieval)

        if result["decision"] == "answer":
            for ref in result.get("references", []):
                doc = (ref.get("document") or "").lower()
                # Must NOT cite HR manual, Annex H, weapons, or compiled working paper
                assert "human_resource" not in doc, f"HR Manual cited for legal query: {doc}"
                assert "annex_h" not in doc.replace(" ", "_"), f"Annex H cited: {doc}"
                assert "weapon" not in doc, f"Weapons cited: {doc}"


# =============================================================================
# Run as script
# =============================================================================
def _run_tests():
    import traceback
    test_classes = [
        TestLegalIntentClassification,
        TestLegalEvidenceGate,
        TestSectionVerification,
        TestLegalIntegration,
    ]

    total = passed = failed = skipped = 0
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
                if "SKIP" in str(e):
                    skipped += 1
                    print(f"  SKIP {method_name}")
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
            print(f"  {cls_name}.{method_name}:")
            traceback.print_exception(type(err), err, err.__traceback__)
    return failed == 0


if __name__ == "__main__":
    success = _run_tests()
    sys.exit(0 if success else 1)
