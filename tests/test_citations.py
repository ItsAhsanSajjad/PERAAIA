"""Step 4 — citation alignment + prompt-injection guard tests.

Exercises pure helpers in answerer.py. No OpenAI calls, no indexing.
"""
import os
import unittest
from unittest import mock

from tests import _bootstrap  # noqa: F401

import answerer
from answerer import (
    _number_evidence_and_sources,
    _reconcile_citations,
    _build_cited_references,
)


def _read(name):
    with open(os.path.join(_bootstrap.PROJECT_ROOT, name), "r", encoding="utf-8") as f:
        return f.read()


class EvidenceNumberingTests(unittest.TestCase):
    def test_numbers_each_evidence_block(self):
        ctx = (
            '<evidence doc="a.pdf" page="3">alpha</evidence>\n'
            '<evidence doc="b.pdf" page="5">beta</evidence>'
        )
        numbered, sources = _number_evidence_and_sources(ctx)
        self.assertIn('<evidence n="1"', numbered)
        self.assertIn('<evidence n="2"', numbered)
        self.assertEqual([s["id"] for s in sources], [1, 2])
        self.assertEqual(sources[0]["document"], "a.pdf")
        self.assertEqual(sources[0]["page_start"], 3)
        self.assertIn("#page=3", sources[0]["open_url"])

    def test_empty_context(self):
        numbered, sources = _number_evidence_and_sources("")
        self.assertEqual(sources, [])


class ReconcileCitationsTests(unittest.TestCase):
    def setUp(self):
        _, self.sources = _number_evidence_and_sources(
            '<evidence doc="a.pdf" page="1">a</evidence>'
            '<evidence doc="b.pdf" page="2">b</evidence>'
        )

    def test_keeps_valid_citation(self):
        clean, used = _reconcile_citations("Answer with cite [1].", self.sources)
        self.assertIn("[1]", clean)
        self.assertEqual([r["id"] for r in used], [1])

    def test_removes_invalid_citation(self):
        clean, used = _reconcile_citations("Bad cite [99] here.", self.sources)
        self.assertNotIn("[99]", clean)
        self.assertEqual(used, [])

    def test_mixed_valid_and_invalid(self):
        clean, used = _reconcile_citations("Foo [1] bar [99] baz [2].", self.sources)
        self.assertNotIn("[99]", clean)
        self.assertEqual([r["id"] for r in used], [1, 2])

    def test_used_refs_have_matching_ids(self):
        clean, used = _reconcile_citations("See [2] and [1].", self.sources)
        # Returned ascending and id matches the cited token.
        self.assertEqual([r["id"] for r in used], [1, 2])
        for r in used:
            self.assertIn(f"[{r['id']}]", clean)

    def test_no_citations_returns_empty(self):
        clean, used = _reconcile_citations("No citations at all.", self.sources)
        self.assertEqual(used, [])


class SourceLevelGuaranteeTests(unittest.TestCase):
    """Behaviors that can't be unit-run without OpenAI are asserted at the
    source level so a regression is still caught by the suite."""

    def setUp(self):
        self.src = _read("answerer.py")

    def test_refusal_returns_empty_references(self):
        # The refusal gate must return an empty references list.
        self.assertIn('grounding.score < 0.25', self.src)
        self.assertIn('"references": []', self.src)

    def test_anti_injection_instruction_in_prompt(self):
        # Evidence must be framed as untrusted data to the model.
        self.assertIn("UNTRUSTED EVIDENCE", self.src)


class NumberEvidenceMetadataTests(unittest.TestCase):
    """Step 2 — sources must expose eid + synthetic so the cited path can
    remap [n] back to the originating retrieval hit."""

    def test_eid_and_synthetic_captured(self):
        ctx = (
            '<evidence doc="a.pdf" page="3" eid="e1">alpha</evidence>\n'
            '<evidence doc="b.pdf" page="5" eid="e2" '
            'role="authoritative-role-salary">beta</evidence>'
        )
        _, sources = _number_evidence_and_sources(ctx)
        self.assertEqual(sources[0]["eid"], "e1")
        self.assertFalse(sources[0]["synthetic"])
        self.assertEqual(sources[1]["eid"], "e2")
        self.assertTrue(sources[1]["synthetic"])


class BuildCitedReferencesTests(unittest.TestCase):
    """Step 2 — inline [n] citations resolve to refined reference cards
    (accurate page + snippet + score), not the raw chunk page_start."""

    def _retrieval(self):
        hit = {
            "evidence_id": "e1",
            "text": (
                "Position Title: Software Engineer\n"
                "The minimum pay per month for this post is set by the "
                "special pay package of PERA as described herein."
            ),
            "page_start": 3,
            "page_end": 7,
            "_blend": 0.87,
            "public_path": "/assets/data/a.pdf",
        }
        return {"evidence": [{"doc_name": "a.pdf", "hits": [hit]}]}

    def test_cited_ref_uses_refined_page_not_raw_page_start(self):
        cited = [{"id": 1, "eid": "e1", "document": "a.pdf",
                  "page_start": 3, "synthetic": False}]
        with mock.patch.object(answerer, "_find_pdf_path_for_hit",
                               return_value="/fake/a.pdf"), \
             mock.patch.object(answerer, "_resolve_best_page",
                               return_value=(5, "minimum pay per month")), \
             mock.patch.object(answerer, "_resolve_exact_page",
                               return_value=5):
            out = _build_cited_references(
                cited, self._retrieval(),
                question="what is the minimum pay per month",
                answer_text="The minimum pay per month is defined.",
            )
        self.assertEqual(len(out), 1)
        # Raw chunk page_start was 3; refined page must be 5.
        self.assertEqual(out[0]["page_start"], 5)
        self.assertNotEqual(out[0]["page_start"], 3)

    def test_cited_ref_includes_snippet_and_score(self):
        cited = [{"id": 1, "eid": "e1", "document": "a.pdf",
                  "page_start": 3, "synthetic": False}]
        with mock.patch.object(answerer, "_find_pdf_path_for_hit",
                               return_value="/fake/a.pdf"), \
             mock.patch.object(answerer, "_resolve_best_page",
                               return_value=(5, "minimum pay per month")), \
             mock.patch.object(answerer, "_resolve_exact_page",
                               return_value=5):
            out = _build_cited_references(
                cited, self._retrieval(),
                question="minimum pay per month",
                answer_text="The minimum pay per month is defined.",
            )
        self.assertTrue(out[0]["snippet"])
        self.assertEqual(out[0]["score"], 0.87)

    def test_id_alignment_preserved(self):
        # Citation [2] must map to a reference card whose id == 2.
        cited = [{"id": 2, "eid": "e1", "document": "a.pdf",
                  "page_start": 3, "synthetic": False}]
        with mock.patch.object(answerer, "_find_pdf_path_for_hit",
                               return_value=""):
            out = _build_cited_references(cited, self._retrieval())
        self.assertEqual(out[0]["id"], 2)

    def test_synthetic_remaps_to_real_hit(self):
        # A synthetic block carries the real hit's eid → exposes the real doc.
        cited = [{"id": 1, "eid": "e1", "document": "a.pdf",
                  "page_start": 3, "synthetic": True}]
        with mock.patch.object(answerer, "_find_pdf_path_for_hit",
                               return_value=""):
            out = _build_cited_references(cited, self._retrieval())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["document"], "a.pdf")
        self.assertEqual(out[0]["id"], 1)

    def test_unresolvable_synthetic_without_document_is_dropped(self):
        # No matching eid and no real document → no broken card emitted.
        cited = [{"id": 1, "eid": "missing", "document": "",
                  "synthetic": True}]
        out = _build_cited_references(cited, self._retrieval())
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
