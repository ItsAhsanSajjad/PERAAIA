"""
PERA AI Compensation/Salary Routing Tests
Tests compensation type detection (SPPP vs BPS) and salary routing.

Run: venv\\Scripts\\python.exe scripts/test_compensation.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answerer import (
    detect_compensation_type, extract_compensation_grade,
    CompensationType,
)


# =============================================================================
# A) Compensation Type Detection
# =============================================================================

class TestCompensationTypeDetection:
    """Tests for detect_compensation_type()."""

    def test_sppp_only(self):
        snippets = [
            "System Support Officer - SPPP-6, Qualification: Bachelor's degree",
            "The officer shall report to the Director IT."
        ]
        assert detect_compensation_type(snippets) == CompensationType.SPPP

    def test_bps_only(self):
        snippets = [
            "Naib Qasid - BPS-14, Qualification: Matric",
            "Duties: Office support and maintenance."
        ]
        assert detect_compensation_type(snippets) == CompensationType.BPS

    def test_sppp_takes_precedence_over_bps(self):
        """When both SPPP and BPS appear, SPPP wins per requirements."""
        snippets = [
            "CTO - SPPP-1, equivalent senior position",
            "For comparison, BPS-20 officers receive...",
        ]
        assert detect_compensation_type(snippets) == CompensationType.SPPP

    def test_neither(self):
        snippets = [
            "The Director General shall oversee all operations.",
            "Powers include regulatory enforcement.",
        ]
        assert detect_compensation_type(snippets) == CompensationType.NOT_SPECIFIED

    def test_empty_snippets(self):
        assert detect_compensation_type([]) == CompensationType.NOT_SPECIFIED


# =============================================================================
# B) Compensation Grade Extraction
# =============================================================================

class TestCompensationGradeExtraction:
    """Tests for extract_compensation_grade()."""

    def test_sppp_grade(self):
        snippets = ["CTO - SPPP-1, Qualification: PhD"]
        grade = extract_compensation_grade(snippets, CompensationType.SPPP)
        assert grade == "SPPP-1", f"Expected SPPP-1, got {grade}"

    def test_bps_grade(self):
        snippets = ["Naib Qasid - BPS-4, Qualification: Matric"]
        grade = extract_compensation_grade(snippets, CompensationType.BPS)
        assert grade == "BPS-4", f"Expected BPS-4, got {grade}"

    def test_sppp_6(self):
        snippets = ["System Support Officer, Grade: SPPP 6"]
        grade = extract_compensation_grade(snippets, CompensationType.SPPP)
        assert grade == "SPPP-6", f"Expected SPPP-6, got {grade}"

    def test_no_grade_found(self):
        snippets = ["The officer shall have powers..."]
        grade = extract_compensation_grade(snippets, CompensationType.SPPP)
        assert grade is None

    def test_not_specified_returns_none(self):
        snippets = ["Some text about PERA functions."]
        grade = extract_compensation_grade(snippets, CompensationType.NOT_SPECIFIED)
        assert grade is None


# =============================================================================
# C) Integration: salary query routing (requires index + API key)
# =============================================================================

class TestSalaryRouting:
    """Integration tests for salary query routing."""

    @staticmethod
    def _can_run():
        return bool(os.getenv("OPENAI_API_KEY"))

    def test_sso_salary_not_bps(self):
        """SSO salary must follow Annex H; must NOT output BPS min/max if JD says SPPP."""
        if not self._can_run():
            raise Exception("SKIP: No OPENAI_API_KEY")

        from retriever import retrieve
        from answerer import answer_question

        retrieval = retrieve("System Support Officer salary")
        result = answer_question("System Support Officer salary", retrieval)

        answer_lower = result["answer"].lower()
        # If evidence says SPPP -> answer should NOT say "BPS-14" equivalence
        if "sppp" in answer_lower:
            assert "bps-14" not in answer_lower or "equivalent" not in answer_lower, \
                f"SSO salary should not map SPPP to BPS-14. Answer: {result['answer'][:300]}"

    def test_cto_salary_sppp(self):
        """CTO salary must say SPPP-1 if Annex H says so; not BPS."""
        if not self._can_run():
            raise Exception("SKIP: No OPENAI_API_KEY")

        from retriever import retrieve
        from answerer import answer_question

        retrieval = retrieve("CTO salary in number")
        result = answer_question("CTO salary in number", retrieval)

        # Should mention SPPP or say info not available; should NOT invent BPS
        answer_lower = result["answer"].lower()
        if result["decision"] == "answer":
            # If answering, should reference SPPP (or specific amount) not BPS
            if "bps" in answer_lower:
                assert "sppp" in answer_lower, \
                    f"CTO salary mentions BPS without SPPP context: {result['answer'][:300]}"


# =============================================================================
# Run as script
# =============================================================================
def _run_tests():
    import traceback
    test_classes = [
        TestCompensationTypeDetection,
        TestCompensationGradeExtraction,
        TestSalaryRouting,
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
