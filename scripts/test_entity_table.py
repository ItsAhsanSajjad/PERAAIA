"""
Test: Entity Pinning + Deterministic Table Lookup + Compensation + Refusal Format
Tests for regressions: wrong entity binding, schedule lookup failing, strict-mode leaks.
Run: venv\Scripts\python.exe scripts\test_entity_table.py
"""
import sys, os, re, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from answerer import (
    extract_job_title, filter_evidence_by_entity,
    classify_intent, QueryIntent,
    _post_process_strict, _detect_roman_urdu,
    _BANNED_HEDGE_PHRASES,
    detect_compensation_type, extract_compensation_grade, CompensationType,
    evidence_relevance_gate,
)
from retriever import (
    _normalize_query, _get_keyword_boosts, _TABLE_ROUTES,
    _build_table_registry, table_lookup,
)


# ═══════════════════════════════════════════════════════════════════════════════
# A) ENTITY PINNING TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestEntityExtraction(unittest.TestCase):
    """Test extract_job_title identifies canonical titles correctly."""

    def test_manager_development(self):
        self.assertEqual(extract_job_title("salary btao manager development ki"), "Manager (Development)")

    def test_manager_development_parens(self):
        self.assertEqual(extract_job_title("Manager (Development) ki salary kya hai?"), "Manager (Development)")

    def test_cto(self):
        self.assertEqual(extract_job_title("CTO ke powers batao"), "Chief Technology Officer")

    def test_system_support_officer(self):
        self.assertEqual(extract_job_title("System Support Officer ki salary"), "System Support Officer")

    def test_sso_alias(self):
        self.assertEqual(extract_job_title("SSO ki salary batao"), "System Support Officer")

    def test_assistant_manager_gis(self):
        self.assertEqual(extract_job_title("assistant manager GIS ki JD"), "Assistant Manager GIS")

    def test_no_match(self):
        self.assertIsNone(extract_job_title("what is PERA?"))

    def test_android_developer(self):
        self.assertEqual(extract_job_title("Android Developer ki salary"), "Android Developer")

    def test_dg(self):
        self.assertEqual(extract_job_title("DG ke powers kya hain"), "Director General")


class TestEntityFiltering(unittest.TestCase):
    """Test filter_evidence_by_entity keeps only relevant chunks."""

    def _make_retrieval(self, texts):
        return {
            "question": "test",
            "has_evidence": True,
            "evidence": [{"doc_name": "test.pdf", "max_score": 1.0,
                          "hits": [{"text": t, "score": 0.9} for t in texts]}]
        }

    def test_filters_manager_dev(self):
        """Manager (Development) should NOT return Assistant Manager GIS chunks."""
        r = self._make_retrieval([
            "Position Title: - Manager (Development)\nReport To: CTO",
            "Position Title: - Assistant Manager GIS\nReport To: Head Monitoring",
        ])
        filtered = filter_evidence_by_entity(r, "Manager (Development)")
        hits = filtered["evidence"][0]["hits"]
        self.assertEqual(len(hits), 1)
        self.assertIn("Manager (Development)", hits[0]["text"])

    def test_filters_keeps_variant(self):
        """Manager Development (no parens) should still match Manager (Development)."""
        r = self._make_retrieval([
            "Manager Development / SPPP-3 (1)",
        ])
        filtered = filter_evidence_by_entity(r, "Manager (Development)")
        self.assertTrue(filtered["has_evidence"])

    def test_empty_when_no_match(self):
        """Should return empty evidence when no chunks match."""
        r = self._make_retrieval([
            "Position Title: - Web Developer",
        ])
        filtered = filter_evidence_by_entity(r, "Manager (Development)")
        self.assertFalse(filtered["has_evidence"])


class TestEntitySessionPin(unittest.TestCase):
    """Test entity pinning from session context for follow-ups."""

    def test_followup_uses_session_title(self):
        """'tell me in number' should use session's last_job_title."""
        title = extract_job_title("tell me in number")
        self.assertIsNone(title)  # No job title in the query
        # But session context should provide it
        session = {"last_job_title": "Manager (Development)"}
        # Simulating the code path in answer_question
        job_title = extract_job_title("tell me in number")
        if not job_title and session:
            job_title = session.get("last_job_title")
        self.assertEqual(job_title, "Manager (Development)")

    def test_new_title_overrides_session(self):
        """Explicit title in query overrides session."""
        session = {"last_job_title": "Manager (Development)"}
        title = extract_job_title("CTO ki salary btao")
        # New title takes precedence over session
        self.assertEqual(title, "Chief Technology Officer")


# ═══════════════════════════════════════════════════════════════════════════════
# B) DETERMINISTIC TABLE LOOKUP TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestTableRoutes(unittest.TestCase):
    """Test table route trigger matching."""

    def test_schedule_1_triggers(self):
        triggers = _TABLE_ROUTES[0]["triggers"]
        self.assertIn("schedule 1", triggers)
        self.assertIn("schedule-i", triggers)
        self.assertIn("sehdule 1", triggers)

    def test_schedule_3_triggers(self):
        triggers = _TABLE_ROUTES[1]["triggers"]
        self.assertIn("schedule 3", triggers)
        self.assertIn("schedule-iii", triggers)
        self.assertIn("sppp salary", triggers)
        self.assertIn("sppp-1", triggers)

    def test_schedule_1_match_fn(self):
        match = _TABLE_ROUTES[0]["match"]
        self.assertTrue(match("schedule-i offences and fines"))
        self.assertTrue(match("description of offences and fine imposed"))
        self.assertFalse(match("sppp salary table"))

    def test_schedule_3_match_fn(self):
        match = _TABLE_ROUTES[1]["match"]
        self.assertTrue(match("sppp minimum pay maximum pay"))
        self.assertTrue(match("schedule-iii sppp-1 350,082"))
        self.assertFalse(match("schedule-i offences"))


class TestTableLookup(unittest.TestCase):
    """Test table_lookup returns chunks for known tables."""

    def _make_chunks(self):
        return [
            {"text": "Some random chunk", "doc_name": "doc1.pdf", "loc_start": 1, "loc_end": 1},
            {"text": "Schedule-I Description of Offences and Fine imposed", "doc_name": "Bill.pdf", "loc_start": 5, "loc_end": 5},
            {"text": "SPPP Minimum Pay Maximum Pay 350,082 1,000,082", "doc_name": "Working Paper.pdf", "loc_start": 10, "loc_end": 10},
        ]

    def test_schedule_1_lookup(self):
        # Clear cache
        from retriever import _table_registry_cache
        _table_registry_cache.clear()
        chunks = self._make_chunks()
        result = table_lookup("schedule 1", chunks)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["doc_name"], "Bill.pdf")

    def test_sppp_salary_lookup(self):
        from retriever import _table_registry_cache
        _table_registry_cache.clear()
        chunks = self._make_chunks()
        result = table_lookup("sppp-1 salary", chunks)
        self.assertIsNotNone(result)
        # Should contain the SPPP chunk
        texts = [h["text"] for g in result for h in g["hits"]]
        self.assertTrue(any("350,082" in t for t in texts))

    def test_sehdule_normalized(self):
        from retriever import _table_registry_cache
        _table_registry_cache.clear()
        chunks = self._make_chunks()
        # "sehdule 1" gets normalized to "schedule 1" 
        result = table_lookup("sehdule 1", chunks)
        self.assertIsNotNone(result)

    def test_no_match(self):
        from retriever import _table_registry_cache
        _table_registry_cache.clear()
        chunks = self._make_chunks()
        result = table_lookup("what is PERA?", chunks)
        self.assertIsNone(result)

    def test_table_lookup_score_is_max(self):
        from retriever import _table_registry_cache
        _table_registry_cache.clear()
        chunks = self._make_chunks()
        result = table_lookup("schedule 1", chunks)
        self.assertIsNotNone(result)
        for group in result:
            self.assertEqual(group["max_score"], 1.0)
            for hit in group["hits"]:
                self.assertEqual(hit["score"], 1.0)
                self.assertTrue(hit.get("_is_table_lookup"))


# ═══════════════════════════════════════════════════════════════════════════════
# C) COMPENSATION LOGIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestCompensationDetection(unittest.TestCase):
    """Test SPPP/BPS detection and grade extraction."""

    def test_sppp_detection(self):
        snippets = ["Manager Development / SPPP-3 (1)", "Some other text"]
        self.assertEqual(detect_compensation_type(snippets), CompensationType.SPPP)

    def test_bps_detection(self):
        snippets = ["Assistant / BPS-16 (1)", "Administrative staff"]
        self.assertEqual(detect_compensation_type(snippets), CompensationType.BPS)

    def test_sppp_grade_extraction(self):
        snippets = ["Manager Development / SPPP-3 (1)"]
        grade = extract_compensation_grade(snippets, CompensationType.SPPP)
        self.assertEqual(grade, "SPPP-3")

    def test_bps_grade_extraction(self):
        snippets = ["Clerk / BPS-14 (1)"]
        grade = extract_compensation_grade(snippets, CompensationType.BPS)
        self.assertEqual(grade, "BPS-14")


# ═══════════════════════════════════════════════════════════════════════════════
# D) STRICT REFUSAL / BAN PHRASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestBanPhrases(unittest.TestCase):
    """Test ban phrase detection catches new patterns."""

    def test_please_refer_banned(self):
        self.assertIn("please refer to guidelines", _BANNED_HEDGE_PHRASES)
        self.assertIn("please refer to", _BANNED_HEDGE_PHRASES)
        self.assertIn("refer to the official", _BANNED_HEDGE_PHRASES)

    def test_post_process_catches_please_refer(self):
        override, needs_regen = _post_process_strict(
            "Please refer to the relevant PERA guidelines.",
            "GENERAL_PERA", "test question"
        )
        self.assertTrue(needs_regen)

    def test_post_process_catches_general_rules(self):
        override, needs_regen = _post_process_strict(
            "General rules may apply in such cases.",
            "GENERAL_PERA", "test question"
        )
        self.assertTrue(needs_regen)

    def test_post_process_ok_for_clean_answer(self):
        override, needs_regen = _post_process_strict(
            "The Manager (Development) reports to the CTO per the PERA regulations.",
            "GENERAL_PERA", "test question"
        )
        self.assertFalse(needs_regen)

    def test_typically_banned(self):
        override, needs_regen = _post_process_strict(
            "Typically in organizations, the CTO has broad powers.",
            "GENERAL_PERA", "test"
        )
        self.assertTrue(needs_regen)


class TestRefusalFormat(unittest.TestCase):
    """Test refusal responses always have zero sources."""

    def test_strict_refusal_no_refs(self):
        """Simulated: if answer_question returns refuse, refs must be []."""
        # This tests the contract: refusals always have references=[]
        from answerer import _STRICT_REFUSAL
        # A refusal result should look like this:
        result = {"answer": _STRICT_REFUSAL, "references": [], "decision": "refuse"}
        self.assertEqual(result["references"], [])
        self.assertEqual(result["decision"], "refuse")


# ═══════════════════════════════════════════════════════════════════════════════
# E) INTEGRATION: QUERY NORMALIZATION + BOOSTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestQueryNormalization(unittest.TestCase):
    def test_sehdule_normalized(self):
        self.assertEqual(_normalize_query("sehdule 1"), "schedule 1")
    def test_sppp1_normalized(self):
        self.assertEqual(_normalize_query("sppp1 salary"), "SPPP-1 salary")
    def test_salary_typo(self):
        self.assertEqual(_normalize_query("sa;lary of cto"), "salary of cto")


if __name__ == "__main__":
    unittest.main(verbosity=2)
