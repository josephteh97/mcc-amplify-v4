"""
Persistent job status store backed by SQLite.

Replaces the in-memory dict in api/routes.py.  Jobs survive process restarts
and are evicted by LRU (least-recently-accessed) policy when the store
exceeds MAX_JOBS.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger


class JobStore:
    """Thread-safe, SQLite-backed job status store with LRU eviction."""

    def __init__(self, db_path: str = "data/jobs.db", max_jobs: int = 100):
        self._db_path = db_path
        self._max_jobs = max_jobs
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id      TEXT PRIMARY KEY,
                    data        TEXT NOT NULL,
                    created_at  REAL NOT NULL,
                    accessed_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_accessed
                ON jobs (accessed_at)
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def get(self, job_id: str) -> Optional[dict]:
        """Return job data and update last-access time, or None."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE jobs SET accessed_at = ? WHERE job_id = ?",
                (time.time(), job_id),
            )
            return json.loads(row[0])

    def put(self, job_id: str, data: dict) -> None:
        """Insert or replace job data."""
        now = time.time()
        blob = json.dumps(data, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs (job_id, data, created_at, accessed_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET
                       data = excluded.data,
                       accessed_at = excluded.accessed_at""",
                (job_id, blob, now, now),
            )
        self._evict()

    def update(self, job_id: str, patch: dict) -> None:
        """Merge *patch* into the existing job data dict."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return
            existing = json.loads(row[0])
            existing.update(patch)
            conn.execute(
                "UPDATE jobs SET data = ?, accessed_at = ? WHERE job_id = ?",
                (json.dumps(existing, default=str), time.time(), job_id),
            )

    def contains(self, job_id: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return row is not None

    def setdefault_nested(self, job_id: str, key: str, subkey: str, value) -> None:
        """Set data[key][subkey] = value, creating intermediate dicts if needed."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return
            existing = json.loads(row[0])
            existing.setdefault(key, {}).setdefault(subkey, {})
            if isinstance(existing[key], dict):
                existing[key][subkey] = value
            conn.execute(
                "UPDATE jobs SET data = ?, accessed_at = ? WHERE job_id = ?",
                (json.dumps(existing, default=str), time.time(), job_id),
            )

    # ── Eviction ─────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Drop least-recently-accessed jobs when over capacity."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            if count <= self._max_jobs:
                return
            n_remove = count - self._max_jobs + 1
            conn.execute(
                """DELETE FROM jobs WHERE job_id IN (
                       SELECT job_id FROM jobs ORDER BY accessed_at ASC LIMIT ?
                   )""",
                (n_remove,),
            )
            logger.debug(f"JobStore: evicted {n_remove} LRU job(s)")

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
