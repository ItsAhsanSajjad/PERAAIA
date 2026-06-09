"""Shared test bootstrap: put the project root on sys.path.

Importing this module (``from tests import _bootstrap``) guarantees the
backend modules (admin_auth, fastapi_app, ...) are importable regardless
of the directory unittest discovery is launched from.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PROJECT_ROOT = _PROJECT_ROOT
