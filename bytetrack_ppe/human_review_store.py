from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


ROOT = Path(__file__).resolve().parent

DEFAULT_REVIEW_PATH = (
    ROOT
    / "outputs"
    / "reviews"
    / "reviews.json"
)

ALLOWED_DECISIONS = {
    "CONFIRMED_VIOLATION",
    "FALSE_ALARM",
    "NEEDS_REVIEW",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


class HumanReviewStore:
    """
    Persistent human-review overlay for PPE events.

    Reviews never modify Event Engine output files. They are stored
    separately so model output and human decisions remain auditable.
    """

    def __init__(
        self,
        path: Path | None = None,
    ) -> None:
        self.path = (
            path or DEFAULT_REVIEW_PATH
        )
        self._lock = RLock()

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.path.is_file():
            self._write(
                {
                    "version": 1,
                    "reviews": {},
                }
            )

    @staticmethod
    def _key(
        scope: str,
        event_id: str,
    ) -> str:
        return (
            f"{scope}::{event_id}"
        )

    def _read(
        self,
    ) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "version": 1,
                "reviews": {},
            }

        try:
            data = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            data = {
                "version": 1,
                "reviews": {},
            }

        if not isinstance(
            data,
            dict,
        ):
            data = {}

        reviews = data.get(
            "reviews"
        )

        if not isinstance(
            reviews,
            dict,
        ):
            reviews = {}

        return {
            "version": 1,
            "reviews": reviews,
        }

    def _write(
        self,
        data: dict[str, Any],
    ) -> None:
        temp = self.path.with_suffix(
            ".json.tmp"
        )

        temp.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temp.replace(
            self.path
        )

    def get(
        self,
        scope: str,
        event_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            data = self._read()

            return (
                data["reviews"]
                .get(
                    self._key(
                        scope,
                        event_id,
                    )
                )
            )

    def list(
        self,
        *,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()

            items = list(
                data["reviews"].values()
            )

            if scope is not None:
                items = [
                    item
                    for item in items
                    if item.get(
                        "scope"
                    )
                    == scope
                ]

            items.sort(
                key=lambda item: (
                    item.get(
                        "updated_at_utc",
                        "",
                    )
                ),
                reverse=True,
            )

            return items

    def upsert(
        self,
        *,
        scope: str,
        event_id: str,
        decision: str,
        reviewer: str = "",
        note: str = "",
        event_type: str | None = None,
        track_id: int | None = None,
    ) -> dict[str, Any]:
        scope = str(
            scope
        ).strip()

        event_id = str(
            event_id
        ).strip()

        decision = str(
            decision
        ).strip().upper()

        reviewer = str(
            reviewer or ""
        ).strip()

        note = str(
            note or ""
        ).strip()

        if scope == "":
            raise ValueError(
                "scope is required"
            )

        if event_id == "":
            raise ValueError(
                "event_id is required"
            )

        if (
            decision
            not in ALLOWED_DECISIONS
        ):
            raise ValueError(
                "decision must be one of: "
                + ", ".join(
                    sorted(
                        ALLOWED_DECISIONS
                    )
                )
            )

        with self._lock:
            data = self._read()

            key = self._key(
                scope,
                event_id,
            )

            previous = (
                data["reviews"]
                .get(key)
            )

            now = utc_now()

            history = []

            created_at = now

            if previous:
                created_at = (
                    previous.get(
                        "created_at_utc",
                        now,
                    )
                )

                old_history = (
                    previous.get(
                        "history",
                        [],
                    )
                )

                if isinstance(
                    old_history,
                    list,
                ):
                    history.extend(
                        old_history
                    )

                history.append(
                    {
                        "decision": (
                            previous.get(
                                "decision"
                            )
                        ),
                        "reviewer": (
                            previous.get(
                                "reviewer",
                                "",
                            )
                        ),
                        "note": (
                            previous.get(
                                "note",
                                "",
                            )
                        ),
                        "updated_at_utc": (
                            previous.get(
                                "updated_at_utc"
                            )
                        ),
                    }
                )

            record = {
                "scope": scope,
                "event_id": event_id,
                "event_type": event_type,
                "track_id": track_id,
                "decision": decision,
                "reviewer": reviewer,
                "note": note,
                "created_at_utc": (
                    created_at
                ),
                "updated_at_utc": now,
                "history": history,
            }

            data[
                "reviews"
            ][key] = record

            self._write(data)

            return record

    def stats(
        self,
    ) -> dict[str, Any]:
        items = self.list()

        counts = {
            decision: 0
            for decision
            in sorted(
                ALLOWED_DECISIONS
            )
        }

        for item in items:
            decision = item.get(
                "decision"
            )

            if decision in counts:
                counts[
                    decision
                ] += 1

        return {
            "total_reviews": (
                len(items)
            ),
            "by_decision": counts,
            "path": str(
                self.path
            ),
        }
