#!/usr/bin/env python3
"""Generate an ADMIN_PASSWORD_HASH for Ask PERA admin auth.

Prompts for the admin password (hidden input via getpass), produces a
PBKDF2-HMAC-SHA256 hash compatible with ``admin_auth.verify_credentials``,
and prints ONLY the resulting hash string.

The plaintext password is never stored, never logged, and never echoed.

Usage:
    python scripts/generate_admin_password_hash.py

Then set the printed value as ADMIN_PASSWORD_HASH in your .env (do NOT
commit .env). Also set ADMIN_EMAIL to the admin login email.
"""
from __future__ import annotations

import getpass
import os
import sys

# Allow running from anywhere: ensure repo root (parent of scripts/) is
# importable so ``admin_auth`` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from admin_auth import hash_password  # noqa: E402


def main() -> int:
    pw1 = getpass.getpass("Admin password: ")
    if not pw1:
        print("Error: empty password.", file=sys.stderr)
        return 1
    pw2 = getpass.getpass("Confirm password: ")
    if pw1 != pw2:
        print("Error: passwords do not match.", file=sys.stderr)
        return 1

    digest = hash_password(pw1)
    # Print ONLY the hash so it can be piped/copied. No plaintext output.
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
