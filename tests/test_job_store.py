"""
Tests for the SQLite-backed JobStore.
"""
import os
import tempfile
import pytest
from backend.services.job_store import JobStore


@pytest.fixture
def store(tmp_path):
    """Create a temporary JobStore for testing."""
    db = str(tmp_path / "test_jobs.db")
    return JobStore(db_path=db, max_jobs=5)


class TestJobStore:
    def test_put_and_get(self, store):
        store.put("j1", {"status": "uploaded", "progress": 0})
        result = store.get("j1")
        assert result is not None
        assert result["status"] == "uploaded"

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_contains(self, store):
        assert not store.contains("j1")
        store.put("j1", {"status": "ok"})
        assert store.contains("j1")

    def test_update(self, store):
        store.put("j1", {"status": "uploaded", "progress": 0})
        store.update("j1", {"status": "processing", "progress": 50})
        result = store.get("j1")
        assert result["status"] == "processing"
        assert result["progress"] == 50

    def test_update_preserves_existing_keys(self, store):
        store.put("j1", {"status": "uploaded", "filename": "test.pdf"})
        store.update("j1", {"progress": 10})
        result = store.get("j1")
        assert result["filename"] == "test.pdf"
        assert result["progress"] == 10

    def test_lru_eviction(self, store):
        """When over max_jobs, least-recently-accessed jobs are evicted."""
        for i in range(6):
            store.put(f"j{i}", {"status": f"job{i}"})

        # j0 should have been evicted (max_jobs=5)
        assert store.count() <= 5

    def test_setdefault_nested(self, store):
        store.put("j1", {"status": "completed"})
        store.setdefault_nested("j1", "result", "files", {"rvt": "/path/to.rvt"})
        result = store.get("j1")
        assert result["result"]["files"] == {"rvt": "/path/to.rvt"}

    def test_get_updates_access_time(self, store):
        """Accessing a job should keep it alive during eviction."""
        # Fill store
        for i in range(5):
            store.put(f"j{i}", {"status": f"job{i}"})

        # Access j0 so it becomes recently-accessed
        store.get("j0")

        # Add one more to trigger eviction
        store.put("j5", {"status": "job5"})

        # j0 should survive because it was recently accessed
        assert store.contains("j0")
