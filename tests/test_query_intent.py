"""Command 3 — abbreviation expansion, salary intent, and grounding-bypass
safety. Pure helpers in answerer.py. No OpenAI, no indexing, no network."""
import os
import unittest

from tests import _bootstrap  # noqa: F401

import answerer
from answerer import (
    _expand_query_abbreviations,
    _is_salary_query,
    _is_sensitive_query,
)


def _read(name):
    with open(os.path.join(_bootstrap.PROJECT_ROOT, name), "r", encoding="utf-8") as f:
        return f.read()


class AbbreviationExpansionTests(unittest.TestCase):
    # Natural-language lowercase "it" must NOT expand.
    def test_natural_it_not_expanded(self):
        for q in ("what is it about", "tell me about it",
                  "is it valid", "what does it mean",
                  "how does it work", "explain it"):
            out = _expand_query_abbreviations(q)
            self.assertNotIn("information technology", out, q)

    # Clear role/department context SHOULD expand ambiguous abbreviations.
    def test_context_expands_ambiguous(self):
        self.assertIn("information technology",
                      _expand_query_abbreviations("IT officer salary"))
        self.assertIn("information technology",
                      _expand_query_abbreviations("IT wing kya hai"))
        self.assertIn("software engineer",
                      _expand_query_abbreviations("SE post eligibility"))
        self.assertIn("enforcement officer",
                      _expand_query_abbreviations("EO job description"))
        self.assertIn("director general",
                      _expand_query_abbreviations("DG powers under PERA"))

    # Uppercase ambiguous token expands even without other context words.
    def test_uppercase_expands(self):
        self.assertIn("human resources",
                      _expand_query_abbreviations("HR department role"))

    # Unambiguous abbreviations always expand.
    def test_unambiguous_always_expands(self):
        self.assertIn("system support officer",
                      _expand_query_abbreviations("sso ki tankhwah"))
        self.assertIn("chief technology officer",
                      _expand_query_abbreviations("cto duties"))


class SalaryIntentTests(unittest.TestCase):
    def test_salary_triggers(self):
        for q in ("EO salary kya hai",
                  "Enforcement Officer ki tankhwah kitni hai",
                  "Assistant Director pay scale",
                  "DG ki salary",
                  "SPPP scale of Enforcement Officer",
                  "basic pay btao",
                  "allowance kitna hai"):
            self.assertTrue(_is_salary_query(q), q)

    def test_salary_does_not_trigger(self):
        for q in ("PERA kya hai",
                  "EO kya karta hai",
                  "complaint kaise submit karni hai",
                  "DG powers btao",
                  "IT wing kya hai",
                  "enforcement officer duties kya hain",
                  "authority ka purpose kya hai"):
            self.assertFalse(_is_salary_query(q), q)

    def test_bare_generic_words_never_trigger(self):
        for q in ("kya", "kitni", "btao", "kaise", "kon", "kis", "kya hai"):
            self.assertFalse(_is_salary_query(q), q)


class SensitiveQueryTests(unittest.TestCase):
    def test_sensitive_categories_detected(self):
        for q in ("What are DG punishment powers?",
                  "Enforcement Officer duties",
                  "complaint procedure",
                  "eligibility for appointment",
                  "EO salary kya hai",
                  "powers under the act"):
            self.assertTrue(_is_sensitive_query(q), q)

    def test_non_sensitive_not_flagged(self):
        for q in ("any update on the project",
                  "tell me about it",
                  "what is PERA"):
            self.assertFalse(_is_sensitive_query(q), q)

    # Command 5 — every salary query is sensitive, incl. Roman/Urdu forms
    # that carry no English sensitive keyword.
    def test_roman_urdu_salary_is_sensitive(self):
        for q in ("Enforcement Officer ki tankhwah kitni hai",
                  "EO ki tankhwah kitni hai",
                  "basic pay btao",
                  "allowance kitna hai"):
            self.assertTrue(_is_sensitive_query(q), q)

    def test_english_salary_is_sensitive(self):
        for q in ("EO salary kya hai", "Assistant Director pay scale"):
            self.assertTrue(_is_sensitive_query(q), q)

    def test_non_salary_still_not_sensitive_without_keyword(self):
        for q in ("PERA kya hai", "what is it about"):
            self.assertFalse(_is_sensitive_query(q), q)


class GroundingBypassSafetyTests(unittest.TestCase):
    """The bypass is a nested closure; assert its safety guards at the
    source level so a regression is still caught by the suite."""

    def setUp(self):
        self.src = _read("answerer.py")

    def test_bypass_disabled_for_sensitive_queries(self):
        self.assertIn("if _is_sensitive_query(current_question):", self.src)

    def test_bypass_has_score_floor(self):
        self.assertIn("if grounding.score < 0.12:", self.src)

    def test_bypass_requires_top_chunk_score(self):
        # A weak chunk must not wave an answer through.
        self.assertIn("if score < HIT_MIN_SCORE:", self.src)

    def test_bypass_uses_phrase_level_bigrams(self):
        self.assertIn("bigrams = [", self.src)


if __name__ == "__main__":
    unittest.main()
