from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import EventDecision


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection = connection
        return connection

    def _initialize(self) -> None:
        connection = self._connection()
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS violation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL UNIQUE,
                camera_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                violation_type TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active', 'resolved')),
                started_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                ended_at TEXT,
                confidence REAL NOT NULL,
                evidence_path TEXT,
                end_reason TEXT,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_events_started
                ON violation_events(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_events_camera_status
                ON violation_events(camera_id, status);

            CREATE TABLE IF NOT EXISTS processing_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                source TEXT NOT NULL,
                profile TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                frames_processed INTEGER NOT NULL DEFAULT 0,
                average_fps REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL
            );
            """
        )
        connection.commit()

    def apply(self, decision: EventDecision, evidence_path: str | None = None) -> None:
        connection = self._connection()
        now = utc_now()
        if decision.action == "start":
            connection.execute(
                """
                INSERT OR IGNORE INTO violation_events (
                    event_key, camera_id, track_id, violation_type, status,
                    started_at, last_seen_at, confidence, evidence_path, details_json
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
                """,
                (
                    decision.event_key,
                    decision.camera_id,
                    decision.track_id,
                    decision.violation_type,
                    now,
                    now,
                    decision.confidence,
                    evidence_path,
                    json.dumps(decision.details, ensure_ascii=False),
                ),
            )
        elif decision.action == "update":
            connection.execute(
                """
                UPDATE violation_events
                SET last_seen_at = ?, confidence = MAX(confidence, ?), details_json = ?
                WHERE event_key = ? AND status = 'active'
                """,
                (
                    now,
                    decision.confidence,
                    json.dumps(decision.details, ensure_ascii=False),
                    decision.event_key,
                ),
            )
        elif decision.action == "end":
            connection.execute(
                """
                UPDATE violation_events
                SET status = 'resolved', last_seen_at = ?, ended_at = ?, end_reason = ?
                WHERE event_key = ? AND status = 'active'
                """,
                (now, now, decision.details.get("reason"), decision.event_key),
            )
        else:
            raise ValueError(f"Unknown event action: {decision.action}")
        connection.commit()

    def list_events(
        self,
        limit: int = 100,
        camera_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 1000)))
        rows = self._connection().execute(
            f"""
            SELECT * FROM violation_events
            {where}
            ORDER BY id DESC LIMIT ?
            """,  # nosec B608: WHERE only contains controlled fragments
            params,
        )
        return [self._decode(row) for row in rows]

    def latest_event_id(self) -> int:
        row = self._connection().execute(
            "SELECT COALESCE(MAX(id), 0) AS latest_id FROM violation_events"
        ).fetchone()
        return int(row["latest_id"])

    def list_events_after_id(
        self, event_id: int, camera_id: str | None = None
    ) -> list[dict[str, Any]]:
        params: list[Any] = [event_id]
        camera_filter = ""
        if camera_id:
            camera_filter = "AND camera_id = ?"
            params.append(camera_id)
        rows = self._connection().execute(
            f"""
            SELECT * FROM violation_events
            WHERE id > ? {camera_filter}
            ORDER BY id ASC
            """,  # nosec B608: camera_filter is a controlled fragment
            params,
        )
        return [self._decode(row) for row in rows]

    def stats(self) -> dict[str, int]:
        row = self._connection().execute(
            """
            SELECT
                COUNT(*) AS total_events,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_events,
                SUM(CASE WHEN violation_type = 'no_helmet' THEN 1 ELSE 0 END) AS no_helmet,
                SUM(CASE WHEN violation_type = 'no_vest' THEN 1 ELSE 0 END) AS no_vest
            FROM violation_events
            """
        ).fetchone()
        return {key: int(value or 0) for key, value in dict(row).items()}

    def start_run(self, camera_id: str, source: str, profile: str) -> int:
        cursor = self._connection().execute(
            """
            INSERT INTO processing_runs(camera_id, source, profile, started_at, status)
            VALUES (?, ?, ?, ?, 'running')
            """,
            (camera_id, source, profile, utc_now()),
        )
        self._connection().commit()
        return int(cursor.lastrowid)

    def finish_run(
        self, run_id: int, frames_processed: int, average_fps: float, status: str
    ) -> None:
        self._connection().execute(
            """
            UPDATE processing_runs
            SET ended_at = ?, frames_processed = ?, average_fps = ?, status = ?
            WHERE id = ?
            """,
            (utc_now(), frames_processed, average_fps, status, run_id),
        )
        self._connection().commit()

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["details"] = json.loads(value.pop("details_json") or "{}")
        return value
