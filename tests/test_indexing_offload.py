"""Step 3 — verify indexing does not block the event loop.

Static source inspection only — does NOT run indexing, upload, or delete.
"""
import os
import unittest

from tests import _bootstrap  # noqa: F401


def _read(name):
    with open(os.path.join(_bootstrap.PROJECT_ROOT, name), "r", encoding="utf-8") as f:
        return f.read()


class IndexingOffloadTests(unittest.TestCase):
    def setUp(self):
        self.src = _read("admin_routes.py")

    def test_run_in_threadpool_imported(self):
        self.assertIn("from fastapi.concurrency import run_in_threadpool", self.src)

    def test_upload_offloads_indexing(self):
        # Upload handler must offload the blocking index call.
        self.assertIn("await run_in_threadpool(indexer.ensure_index_once)", self.src)

    def test_upload_handler_is_async(self):
        self.assertIn("async def upload_document", self.src)

    def test_delete_handler_is_sync_def(self):
        # Sync def → FastAPI runs it in its own threadpool, so the call to
        # ensure_index_once() does not block the event loop.
        self.assertIn("def delete_document", self.src)
        self.assertNotIn("async def delete_document", self.src)


if __name__ == "__main__":
    unittest.main()
