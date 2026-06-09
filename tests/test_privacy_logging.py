"""Step 5 — privacy-safe logging/redaction tests (no network, no secrets)."""
import importlib
import os
import unittest

from tests import _bootstrap  # noqa: F401


def _reload_audit(env):
    """Reload audit_trail with the given env so module-level privacy flags
    pick up the test values. Restores prior env in the caller's tearDown."""
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    import audit_trail
    return importlib.reload(audit_trail)


class RedactionTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("LOG_RAW_QUERIES", "LOG_CLIENT_IP", "AUDIT_HASH_SALT")}

    def tearDown(self):
        # Restore env and reload to default state for other suites.
        env = {k: (v if v is not None else None) for k, v in self._saved.items()}
        _reload_audit(env)

    def test_query_redacted_by_default(self):
        at = _reload_audit({"LOG_RAW_QUERIES": "0", "AUDIT_HASH_SALT": "s"})
        self.assertEqual(at.redact_query("what is my pension?"), "")
        # Hash is non-empty and not the raw text.
        h = at._short_hash("what is my pension?")
        self.assertTrue(h)
        self.assertNotIn("pension", h)

    def test_query_kept_when_enabled(self):
        at = _reload_audit({"LOG_RAW_QUERIES": "1"})
        self.assertEqual(at.redact_query("hello"), "hello")

    def test_ip_hashed_by_default(self):
        at = _reload_audit({"LOG_CLIENT_IP": "0", "AUDIT_HASH_SALT": "s"})
        out = at.redact_ip("203.0.113.7")
        self.assertTrue(out.startswith("h:"))
        self.assertNotIn("203.0.113.7", out)

    def test_ip_kept_when_enabled(self):
        at = _reload_audit({"LOG_CLIENT_IP": "1"})
        self.assertEqual(at.redact_ip("203.0.113.7"), "203.0.113.7")

    def test_empty_inputs(self):
        at = _reload_audit({"LOG_RAW_QUERIES": "0", "LOG_CLIENT_IP": "0"})
        self.assertEqual(at.redact_query(""), "")
        self.assertEqual(at.redact_ip(""), "")
        self.assertEqual(at._short_hash(""), "")

    def test_salt_changes_hash(self):
        at1 = _reload_audit({"AUDIT_HASH_SALT": "salt-a"})
        h1 = at1._short_hash("same-value")
        at2 = _reload_audit({"AUDIT_HASH_SALT": "salt-b"})
        h2 = at2._short_hash("same-value")
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
