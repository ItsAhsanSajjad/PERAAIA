"""Step 2 — OpenAI env parsing tests (no client instantiation, no network)."""
import os
import unittest

from tests import _bootstrap  # noqa: F401

import openai_clients
from openai_clients import _env_float, _env_int


class EnvParsingTests(unittest.TestCase):
    """_env_float/_env_int read os.getenv at call time, so we can set
    throwaway env vars and assert fallback behavior without API keys."""

    def setUp(self):
        self._keys = ["PERA_TEST_TIMEOUT", "PERA_TEST_RETRIES"]
        for k in self._keys:
            os.environ.pop(k, None)

    def tearDown(self):
        for k in self._keys:
            os.environ.pop(k, None)

    def test_invalid_float_falls_back(self):
        os.environ["PERA_TEST_TIMEOUT"] = "not-a-number"
        self.assertEqual(_env_float("PERA_TEST_TIMEOUT", 30.0), 30.0)

    def test_nonpositive_float_falls_back(self):
        os.environ["PERA_TEST_TIMEOUT"] = "0"
        self.assertEqual(_env_float("PERA_TEST_TIMEOUT", 30.0), 30.0)

    def test_valid_float_parsed(self):
        os.environ["PERA_TEST_TIMEOUT"] = "12.5"
        self.assertEqual(_env_float("PERA_TEST_TIMEOUT", 30.0), 12.5)

    def test_invalid_int_falls_back(self):
        os.environ["PERA_TEST_RETRIES"] = "abc"
        self.assertEqual(_env_int("PERA_TEST_RETRIES", 1), 1)

    def test_negative_int_falls_back(self):
        os.environ["PERA_TEST_RETRIES"] = "-3"
        self.assertEqual(_env_int("PERA_TEST_RETRIES", 1), 1)

    def test_zero_int_allowed(self):
        os.environ["PERA_TEST_RETRIES"] = "0"
        self.assertEqual(_env_int("PERA_TEST_RETRIES", 1), 0)

    def test_max_token_defaults_available(self):
        self.assertGreater(openai_clients.OPENAI_ANSWER_MAX_TOKENS, 0)
        self.assertGreater(openai_clients.OPENAI_REFINE_MAX_TOKENS, 0)
        self.assertGreater(openai_clients.OPENAI_TIMEOUT_SECONDS, 0)
        self.assertGreaterEqual(openai_clients.OPENAI_MAX_RETRIES, 0)


if __name__ == "__main__":
    unittest.main()
