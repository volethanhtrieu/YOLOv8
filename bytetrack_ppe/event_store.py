from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any


ROOT = Path(__file__).resolve().parent

LEGACY_GLASS_FIELDS = {
    "glass",
    "glass_conf",
    "glass_detections",
    "has_glass",
}


def _without_legacy_glass(raw: dict[str, Any]) -> dict[str, Any]:
    """Read old artifacts without exposing retired glass fields through V3."""

    return {
        key: value
        for key, value in raw.items()
        if key not in LEGACY_GLASS_FIELDS
    }


@dataclass(frozen=True)
class StorePaths:
    events_json: Path
    event_summary_json: Path
    temporal_states_csv: Path
    track_rows_csv: Path

    @classmethod
    def default(cls) -> "StorePaths":
        return cls(
            events_json=(
                ROOT
                / "outputs"
                / "event_engine_v2"
                / "events.json"
            ),
            event_summary_json=(
                ROOT
                / "outputs"
                / "event_engine_v2"
                / "summary.json"
            ),
            temporal_states_csv=(
                ROOT
                / "outputs"
                / "event_engine_v2"
                / "ppe_temporal_states.csv"
            ),
            track_rows_csv=(
                ROOT
                / "outputs"
                / "tiled_ppe_pipeline_v2"
                / "track_ppe_rows.csv"
            ),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


class EventStore:
    """
    Read-only store over Event Engine V2 outputs.

    The store reloads automatically when one of the source files changes.
    No database is required for the current offline backend stage.
    """

    def __init__(
        self,
        paths: StorePaths | None = None,
    ) -> None:
        self.paths = paths or StorePaths.default()

        self._lock = RLock()
        self._mtimes: dict[Path, float | None] = {}

        self._raw_events: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._events_by_id: dict[str, dict[str, Any]] = {}

        self._event_summary: dict[str, Any] = {}

        self._temporal_rows: list[dict[str, Any]] = []
        self._temporal_by_track: dict[int, list[dict[str, Any]]] = {}

        self._track_rows: list[dict[str, Any]] = []
        self._track_by_id: dict[int, list[dict[str, Any]]] = {}

        self.reload(force=True)

    def source_status(self) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for name, path in {
            "events_json": self.paths.events_json,
            "event_summary_json": self.paths.event_summary_json,
            "temporal_states_csv": self.paths.temporal_states_csv,
            "track_rows_csv": self.paths.track_rows_csv,
        }.items():
            exists = path.is_file()

            result[name] = {
                "path": str(path),
                "exists": exists,
                "size_bytes": (
                    path.stat().st_size
                    if exists
                    else None
                ),
            }

        return result

    def _mtime(
        self,
        path: Path,
    ) -> float | None:
        if not path.is_file():
            return None

        return path.stat().st_mtime

    def _needs_reload(self) -> bool:
        for path in (
            self.paths.events_json,
            self.paths.event_summary_json,
            self.paths.temporal_states_csv,
            self.paths.track_rows_csv,
        ):
            current = self._mtime(path)

            if self._mtimes.get(path) != current:
                return True

        return False

    def _update_mtimes(self) -> None:
        for path in (
            self.paths.events_json,
            self.paths.event_summary_json,
            self.paths.temporal_states_csv,
            self.paths.track_rows_csv,
        ):
            self._mtimes[path] = self._mtime(path)

    def reload(
        self,
        force: bool = False,
    ) -> bool:
        with self._lock:
            if (
                not force
                and not self._needs_reload()
            ):
                return False

            self._raw_events = self._read_json_list(
                self.paths.events_json
            )

            self._event_summary = self._read_json_object(
                self.paths.event_summary_json
            )

            self._temporal_rows = self._read_temporal_rows(
                self.paths.temporal_states_csv
            )

            self._track_rows = self._read_track_rows(
                self.paths.track_rows_csv
            )

            (
                self._events,
                self._events_by_id,
            ) = self._build_events(
                self._raw_events
            )

            temporal_by_track: dict[
                int,
                list[dict[str, Any]],
            ] = defaultdict(list)

            for row in self._temporal_rows:
                temporal_by_track[
                    row["track_id"]
                ].append(row)

            self._temporal_by_track = dict(
                temporal_by_track
            )

            track_by_id: dict[
                int,
                list[dict[str, Any]],
            ] = defaultdict(list)

            for row in self._track_rows:
                track_by_id[
                    row["track_id"]
                ].append(row)

            self._track_by_id = dict(
                track_by_id
            )

            self._update_mtimes()

            return True

    def _refresh(self) -> None:
        self.reload(force=False)

    @staticmethod
    def _read_json_list(
        path: Path,
    ) -> list[dict[str, Any]]:
        if not path.is_file():
            return []

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected JSON list: {path}"
            )

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    @staticmethod
    def _read_json_object(
        path: Path,
    ) -> dict[str, Any]:
        if not path.is_file():
            return {}

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected JSON object: {path}"
            )

        return data

    @staticmethod
    def _read_temporal_rows(
        path: Path,
    ) -> list[dict[str, Any]]:
        if not path.is_file():
            return []

        rows: list[dict[str, Any]] = []

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            for raw in reader:
                track_id = _optional_int(
                    raw.get("track_id")
                )

                frame_index = _optional_int(
                    raw.get("frame_index")
                )

                timestamp_s = _optional_float(
                    raw.get("timestamp_s")
                )

                if (
                    track_id is None
                    or frame_index is None
                    or timestamp_s is None
                ):
                    continue

                rows.append(
                    {
                        **_without_legacy_glass(raw),
                        "track_id": track_id,
                        "frame_index": frame_index,
                        "timestamp_s": timestamp_s,
                        "person_conf": _optional_float(
                            raw.get("person_conf")
                        ),
                        "person_width": _optional_float(
                            raw.get("person_width")
                        ),
                        "person_height": _optional_float(
                            raw.get("person_height")
                        ),
                        "head_conf": _optional_float(
                            raw.get("head_conf")
                        ),
                        "helmet_conf": _optional_float(
                            raw.get("helmet_conf")
                        ),
                        "vest_conf": _optional_float(
                            raw.get("vest_conf")
                        ),
                        "torso_person_occlusion_ratio": (
                            _optional_float(
                                raw.get(
                                    "torso_person_occlusion_ratio"
                                )
                            )
                        ),
                        "helmet_valid_samples": _optional_int(
                            raw.get(
                                "helmet_valid_samples"
                            )
                        ),
                        "helmet_violation_ratio": (
                            _optional_float(
                                raw.get(
                                    "helmet_violation_ratio"
                                )
                            )
                        ),
                        "vest_valid_samples": _optional_int(
                            raw.get(
                                "vest_valid_samples"
                            )
                        ),
                        "vest_absent_ratio": (
                            _optional_float(
                                raw.get(
                                    "vest_absent_ratio"
                                )
                            )
                        ),
                    }
                )

        rows.sort(
            key=lambda row: (
                row["frame_index"],
                row["track_id"],
            )
        )

        return rows

    @staticmethod
    def _read_track_rows(
        path: Path,
    ) -> list[dict[str, Any]]:
        if not path.is_file():
            return []

        rows: list[dict[str, Any]] = []

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            for raw in reader:
                track_id = _optional_int(
                    raw.get("track_id")
                )

                frame_index = _optional_int(
                    raw.get("frame_index")
                )

                timestamp_s = _optional_float(
                    raw.get("timestamp_s")
                )

                if (
                    track_id is None
                    or frame_index is None
                    or timestamp_s is None
                ):
                    continue

                rows.append(
                    {
                        **_without_legacy_glass(raw),
                        "track_id": track_id,
                        "frame_index": frame_index,
                        "timestamp_s": timestamp_s,
                        "person_conf": _optional_float(
                            raw.get("person_conf")
                        ),
                        "x1": _optional_float(
                            raw.get("x1")
                        ),
                        "y1": _optional_float(
                            raw.get("y1")
                        ),
                        "x2": _optional_float(
                            raw.get("x2")
                        ),
                        "y2": _optional_float(
                            raw.get("y2")
                        ),
                        "head_conf": _optional_float(
                            raw.get("head_conf")
                        ),
                        "helmet_conf": _optional_float(
                            raw.get("helmet_conf")
                        ),
                        "vest_conf": _optional_float(
                            raw.get("vest_conf")
                        ),
                    }
                )

        rows.sort(
            key=lambda row: (
                row["frame_index"],
                row["track_id"],
            )
        )

        return rows

    @staticmethod
    def _build_events(
        lifecycle_rows: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        grouped: dict[
            str,
            list[dict[str, Any]],
        ] = defaultdict(list)

        for row in lifecycle_rows:
            event_id = str(
                row.get("event_id", "")
            ).strip()

            if event_id == "":
                continue

            grouped[event_id].append(row)

        events: list[dict[str, Any]] = []

        for event_id, lifecycle in grouped.items():
            lifecycle.sort(
                key=lambda row: (
                    _optional_float(
                        row.get("timestamp_s")
                    )
                    or 0.0
                )
            )

            opened = next(
                (
                    row
                    for row in lifecycle
                    if row.get("event")
                    == "OPEN"
                ),
                lifecycle[0],
            )

            closed = next(
                (
                    row
                    for row in reversed(lifecycle)
                    if row.get("event")
                    == "CLOSE"
                ),
                None,
            )

            start_s = _optional_float(
                opened.get("timestamp_s")
            )

            end_s = (
                _optional_float(
                    closed.get("timestamp_s")
                )
                if closed is not None
                else None
            )

            duration_s = (
                end_s - start_s
                if (
                    start_s is not None
                    and end_s is not None
                )
                else None
            )

            event = {
                "event_id": event_id,
                "track_id": _optional_int(
                    opened.get("track_id")
                ),
                "ppe_type": opened.get(
                    "ppe_type"
                ),
                "event_type": opened.get(
                    "event_type"
                ),
                "status": opened.get(
                    "status"
                ),
                "state": (
                    "CLOSED"
                    if closed is not None
                    else "OPEN"
                ),
                "start_frame": _optional_int(
                    opened.get("frame_index")
                ),
                "start_s": start_s,
                "end_frame": (
                    _optional_int(
                        closed.get("frame_index")
                    )
                    if closed is not None
                    else None
                ),
                "end_s": end_s,
                "duration_s": duration_s,
                "open_reason": opened.get(
                    "reason"
                ),
                "close_reason": (
                    closed.get("reason")
                    if closed is not None
                    else None
                ),
                "open_evidence_ratio": (
                    _optional_float(
                        opened.get(
                            "evidence_ratio"
                        )
                    )
                ),
                "open_valid_samples": (
                    _optional_int(
                        opened.get(
                            "valid_samples"
                        )
                    )
                ),
                "lifecycle": lifecycle,
            }

            events.append(event)

        events.sort(
            key=lambda event: (
                event["start_s"]
                if event["start_s"]
                is not None
                else -1.0
            ),
            reverse=True,
        )

        by_id = {
            event["event_id"]: event
            for event in events
        }

        return events, by_id

    def health(self) -> dict[str, Any]:
        self._refresh()

        source_status = self.source_status()

        required_ok = all(
            item["exists"]
            for item in source_status.values()
        )

        return {
            "status": (
                "ok"
                if required_ok
                else "degraded"
            ),
            "sources": source_status,
            "counts": {
                "events": len(
                    self._events
                ),
                "tracks": len(
                    self._track_by_id
                ),
                "temporal_rows": len(
                    self._temporal_rows
                ),
                "track_rows": len(
                    self._track_rows
                ),
            },
        }

    def list_events(
        self,
        *,
        status: str | None = None,
        ppe_type: str | None = None,
        event_type: str | None = None,
        state: str | None = None,
        track_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._refresh()

        events = self._events

        if status:
            wanted = status.upper()

            events = [
                event
                for event in events
                if str(
                    event.get(
                        "status",
                        "",
                    )
                ).upper()
                == wanted
            ]

        if ppe_type:
            wanted = ppe_type.lower()

            events = [
                event
                for event in events
                if str(
                    event.get(
                        "ppe_type",
                        "",
                    )
                ).lower()
                == wanted
            ]

        if event_type:
            wanted = event_type.upper()

            events = [
                event
                for event in events
                if str(
                    event.get(
                        "event_type",
                        "",
                    )
                ).upper()
                == wanted
            ]

        if state:
            wanted = state.upper()

            events = [
                event
                for event in events
                if str(
                    event.get(
                        "state",
                        "",
                    )
                ).upper()
                == wanted
            ]

        if track_id is not None:
            events = [
                event
                for event in events
                if event.get(
                    "track_id"
                )
                == track_id
            ]

        total = len(events)

        limit = max(
            1,
            min(limit, 500),
        )

        offset = max(
            0,
            offset,
        )

        page = events[
            offset:
            offset + limit
        ]

        return {
            "items": [
                {
                    key: value
                    for key, value
                    in event.items()
                    if key != "lifecycle"
                }
                for event in page
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_event(
        self,
        event_id: str,
    ) -> dict[str, Any] | None:
        self._refresh()

        return self._events_by_id.get(
            event_id
        )

    def get_track(
        self,
        track_id: int,
    ) -> dict[str, Any] | None:
        self._refresh()

        rows = self._track_by_id.get(
            track_id
        )

        if not rows:
            return None

        temporal = (
            self._temporal_by_track.get(
                track_id,
                [],
            )
        )

        latest_state = (
            temporal[-1]
            if temporal
            else None
        )

        first = rows[0]
        last = rows[-1]

        duration_s = (
            last["timestamp_s"]
            - first["timestamp_s"]
        )

        person_conf_values = [
            row["person_conf"]
            for row in rows
            if row["person_conf"]
            is not None
        ]

        ppe_presence = {
            "head": sum(
                row["head_conf"]
                is not None
                for row in rows
            ),
            "helmet": sum(
                row["helmet_conf"]
                is not None
                for row in rows
            ),
            "vest": sum(
                row["vest_conf"]
                is not None
                for row in rows
            ),
        }

        track_events = [
            {
                key: value
                for key, value
                in event.items()
                if key != "lifecycle"
            }
            for event in self._events
            if event.get("track_id")
            == track_id
        ]

        return {
            "track_id": track_id,
            "first_frame": first[
                "frame_index"
            ],
            "last_frame": last[
                "frame_index"
            ],
            "first_s": first[
                "timestamp_s"
            ],
            "last_s": last[
                "timestamp_s"
            ],
            "duration_s": duration_s,
            "observation_count": len(
                rows
            ),
            "mean_person_conf": (
                sum(
                    person_conf_values
                )
                / len(
                    person_conf_values
                )
                if person_conf_values
                else None
            ),
            "ppe_presence_counts": (
                ppe_presence
            ),
            "latest_temporal_state": (
                latest_state
            ),
            "events": track_events,
        }

    def stats(self) -> dict[str, Any]:
        self._refresh()

        status_counts = Counter(
            str(
                event.get(
                    "status",
                    "UNKNOWN",
                )
            )
            for event in self._events
        )

        event_type_counts = Counter(
            str(
                event.get(
                    "event_type",
                    "UNKNOWN",
                )
            )
            for event in self._events
        )

        state_counts = Counter(
            str(
                event.get(
                    "state",
                    "UNKNOWN",
                )
            )
            for event in self._events
        )

        latest_helmet_states = Counter()
        latest_vest_states = Counter()

        for track_id, rows in (
            self._temporal_by_track.items()
        ):
            if not rows:
                continue

            latest = rows[-1]

            latest_helmet_states[
                str(
                    latest.get(
                        "helmet_state",
                        "UNKNOWN",
                    )
                )
            ] += 1

            latest_vest_states[
                str(
                    latest.get(
                        "vest_state",
                        "UNKNOWN",
                    )
                )
            ] += 1

        source_events = (
            self._event_summary.get(
                "events",
                {}
            )
        )

        return {
            "events": {
                "total": len(
                    self._events
                ),
                "by_status": dict(
                    status_counts
                ),
                "by_type": dict(
                    event_type_counts
                ),
                "by_state": dict(
                    state_counts
                ),
            },
            "tracks": {
                "total": len(
                    self._track_by_id
                ),
                "latest_helmet_states": (
                    dict(
                        latest_helmet_states
                    )
                ),
                "latest_vest_states": (
                    dict(
                        latest_vest_states
                    )
                ),
            },
            "engine_summary": (
                source_events
            ),
            "source_status": (
                self.source_status()
            ),
        }
