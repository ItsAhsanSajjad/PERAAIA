"""Step 2 — CORS config regression tests (pure function, no network)."""
import unittest

from tests import _bootstrap  # noqa: F401

import fastapi_app
from fastapi_app import parse_cors_origins


class CorsConfigTests(unittest.TestCase):
    def test_dev_default_includes_localhost(self):
        allow_all, origins = parse_cors_origins("", "development", False)
        self.assertFalse(allow_all)
        self.assertIn("http://localhost:3000", origins)
        self.assertIn("http://127.0.0.1:5173", origins)

    def test_dev_wildcard_without_credentials(self):
        allow_all, origins = parse_cors_origins("*", "development", False)
        self.assertTrue(allow_all)
        self.assertEqual(origins, ["*"])

    def test_wildcard_with_credentials_not_allowed(self):
        allow_all, origins = parse_cors_origins("*", "development", True)
        self.assertFalse(allow_all)
        self.assertNotIn("*", origins)

    def test_production_wildcard_ignored(self):
        allow_all, origins = parse_cors_origins("*", "production", False)
        self.assertFalse(allow_all)
        self.assertNotIn("*", origins)

    def test_production_explicit_origin(self):
        allow_all, origins = parse_cors_origins(
            "https://ask.pera.gop.pk", "production", False
        )
        self.assertFalse(allow_all)
        self.assertEqual(origins, ["https://ask.pera.gop.pk"])
        # No localhost defaults leak into production.
        self.assertNotIn("http://localhost:3000", origins)

    def test_deduplication_preserves_order(self):
        allow_all, origins = parse_cors_origins(
            "https://a.example, https://a.example, https://b.example",
            "production",
            False,
        )
        self.assertFalse(allow_all)
        self.assertEqual(origins, ["https://a.example", "https://b.example"])

    def test_empties_and_whitespace_dropped(self):
        allow_all, origins = parse_cors_origins(
            " , https://a.example ,  ", "production", False
        )
        self.assertEqual(origins, ["https://a.example"])

    def test_dev_default_origins_constant_present(self):
        self.assertIn("http://localhost:3000", fastapi_app._DEV_DEFAULT_ORIGINS)


if __name__ == "__main__":
    unittest.main()
