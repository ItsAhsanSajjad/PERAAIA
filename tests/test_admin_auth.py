"""Step 1 — admin auth regression tests (no network, no secrets)."""
import os
import unittest

from tests import _bootstrap  # noqa: F401  (sys.path side effect)

import admin_auth

# Throwaway test-only values — NOT real credentials.
_TEST_EMAIL = "tester@example.test"
_TEST_PASSWORD = "throwaway-pw-1234"
_WRONG_PASSWORD = "wrong-pw-9999"


class HashPasswordTests(unittest.TestCase):
    def test_hash_password_format(self):
        h = admin_auth.hash_password(_TEST_PASSWORD)
        parts = h.split("$")
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "pbkdf2_sha256")
        self.assertTrue(int(parts[1]) > 0)
        # Salt is random → two hashes of same password differ.
        self.assertNotEqual(h, admin_auth.hash_password(_TEST_PASSWORD))

    def test_correct_password_verifies(self):
        h = admin_auth.hash_password(_TEST_PASSWORD)
        self.assertTrue(admin_auth._verify_password(_TEST_PASSWORD, h))

    def test_wrong_password_fails(self):
        h = admin_auth.hash_password(_TEST_PASSWORD)
        self.assertFalse(admin_auth._verify_password(_WRONG_PASSWORD, h))

    def test_malformed_hash_fails_safely(self):
        self.assertFalse(admin_auth._verify_password(_TEST_PASSWORD, "not-a-hash"))
        self.assertFalse(admin_auth._verify_password(_TEST_PASSWORD, ""))


class VerifyCredentialsTests(unittest.TestCase):
    """verify_credentials reads module globals — patch them per test."""

    def setUp(self):
        self._saved = (admin_auth.ADMIN_EMAIL, admin_auth.ADMIN_PASSWORD_HASH)

    def tearDown(self):
        admin_auth.ADMIN_EMAIL, admin_auth.ADMIN_PASSWORD_HASH = self._saved

    def test_missing_config_denies_login(self):
        admin_auth.ADMIN_EMAIL = ""
        admin_auth.ADMIN_PASSWORD_HASH = ""
        self.assertFalse(admin_auth.admin_auth_configured())
        self.assertFalse(admin_auth.verify_credentials(_TEST_EMAIL, _TEST_PASSWORD))

    def test_correct_credentials_pass(self):
        admin_auth.ADMIN_EMAIL = _TEST_EMAIL
        admin_auth.ADMIN_PASSWORD_HASH = admin_auth.hash_password(_TEST_PASSWORD)
        self.assertTrue(admin_auth.admin_auth_configured())
        self.assertTrue(admin_auth.verify_credentials(_TEST_EMAIL, _TEST_PASSWORD))

    def test_wrong_password_denied(self):
        admin_auth.ADMIN_EMAIL = _TEST_EMAIL
        admin_auth.ADMIN_PASSWORD_HASH = admin_auth.hash_password(_TEST_PASSWORD)
        self.assertFalse(admin_auth.verify_credentials(_TEST_EMAIL, _WRONG_PASSWORD))

    def test_wrong_email_denied(self):
        admin_auth.ADMIN_EMAIL = _TEST_EMAIL
        admin_auth.ADMIN_PASSWORD_HASH = admin_auth.hash_password(_TEST_PASSWORD)
        self.assertFalse(admin_auth.verify_credentials("other@example.test", _TEST_PASSWORD))


class NoHardcodedCredentialsTests(unittest.TestCase):
    def test_no_old_plaintext_credentials_in_source(self):
        path = os.path.join(_bootstrap.PROJECT_ROOT, "admin_auth.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        # Old hardcoded password must be gone.
        self.assertNotIn("Pera@112233", src)
        # Credentials must come from the environment.
        self.assertIn('os.getenv("ADMIN_EMAIL"', src)
        self.assertIn('os.getenv("ADMIN_PASSWORD_HASH"', src)


if __name__ == "__main__":
    unittest.main()
