import unittest
import os
import sys
# Add parent directory to sys.path to import memory_store
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import sqlite3
import json
from datetime import datetime
from memory_store import SQLiteMemoryStore

import traceback

import uuid
import tempfile

class TestSQLiteMemoryStore(unittest.TestCase):
    def setUp(self):
        try:
            # Use unique temp file to avoid WinError 32 locks
            self.test_db = os.path.join(tempfile.gettempdir(), f"test_memory_{uuid.uuid4().hex}.db")
            # db_path usually expects forward slashes or raw string, but os.path.join handles it. 
            # SQLiteMemoryStore replaces \ with / anyway.
            
            self.store = SQLiteMemoryStore(self.test_db)
            self.user_id = "test_user"
            self.conv_id = "conv_123"
        except Exception:
            print("SETUP FAILED:")
            traceback.print_exc()
            raise

    def tearDown(self):
        try:
            if os.path.exists(self.test_db):
                try:
                    os.remove(self.test_db)
                except PermissionError:
                    print("TEARDOWN: Could not remove db file (locked?)")
        except Exception:
            traceback.print_exc()

    def test_initialization(self):
        """Test that tables are created correctly."""
        try:
            conn = sqlite3.connect(self.test_db)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            # Correct table names based on memory_store.py
            self.assertIn("conversation_messages", tables)
            self.assertIn("conversation_evidence", tables)
            self.assertIn("conversation_state", tables)
        except Exception:
            print("TEST INITIALIZATION FAILED:")
            traceback.print_exc()
            raise

    def test_append_and_load(self):
        """Test appending messages and loading them back."""
        self.store.append_user(self.user_id, self.conv_id, "Hello")
        self.store.append_assistant(self.user_id, self.conv_id, "Hi there")
        
        memory = self.store.load(self.user_id, self.conv_id)
        messages = memory["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "Hi there")

    def test_trimming(self):
        """Test that memory trims to last 5 turns (10 messages)."""
        # Add 15 messages (7.5 turns)
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            self.store.append_user(self.user_id, self.conv_id, f"Msg {i}") if role == "user" else \
            self.store.append_assistant(self.user_id, self.conv_id, f"Msg {i}")

        memory = self.store.load(self.user_id, self.conv_id)
        messages = memory["messages"]
        self.assertEqual(len(messages), 10)
        # Should contain Msg 5 to Msg 14
        self.assertEqual(messages[0]["content"], "Msg 5")
        self.assertEqual(messages[-1]["content"], "Msg 14")

    def test_state_update(self):
        """Test updating and loading conversation state."""
        updates = {"topic": "salary", "entity": "DG"}
        self.store.update_state(self.user_id, self.conv_id, updates)
        
        memory = self.store.load(self.user_id, self.conv_id)
        state = memory["state"]
        self.assertEqual(state["topic"], "salary")
        self.assertEqual(state["entity"], "DG")
        
        # Partial update
        self.store.update_state(self.user_id, self.conv_id, {"grade": "BPS-17"})
        memory = self.store.load(self.user_id, self.conv_id)
        state = memory["state"]
        self.assertEqual(state["topic"], "salary")
        self.assertEqual(state["grade"], "BPS-17")

    def test_evidence_persistence(self):
        """Test saving and loading evidence traces."""
        trace = {"chunks": [1, 2, 3], "job_title": "Director"}
        self.store.append_assistant(self.user_id, self.conv_id, "Here is the info", evidence_trace=trace)
        
        memory = self.store.load(self.user_id, self.conv_id)
        pinned = memory["pinned_evidence"]
        # Should contain the evidence trace
        self.assertEqual(len(pinned), 1)
        self.assertEqual(pinned[0]["job_title"], "Director")

    def test_isolation(self):
        """Test that different conversation IDs are isolated."""
        conv2 = "conv_456"
        self.store.append_user(self.user_id, self.conv_id, "User 1 msg")
        self.store.append_user(self.user_id, conv2, "User 2 msg")
        
        mem1 = self.store.load(self.user_id, self.conv_id)
        mem2 = self.store.load(self.user_id, conv2)
        
        self.assertEqual(len(mem1["messages"]), 1)
        self.assertEqual(mem1["messages"][0]["content"], "User 1 msg")
        self.assertEqual(len(mem2["messages"]), 1)
        self.assertEqual(mem2["messages"][0]["content"], "User 2 msg")

if __name__ == "__main__":
    unittest.main()
