from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import EventConfig
from .types import EventDecision, PersonPPE


@dataclass(slots=True)
class _ViolationState:
    samples: deque[tuple[float, bool]] = field(default_factory=deque)
    violation_since: float | None = None
    compliant_since: float | None = None
    active: bool = False
    event_key: str | None = None
    last_seen: float = 0.0


class EventEngine:
    """Converts noisy per-frame PPE observations into deduplicated events."""

    def __init__(self, config: EventConfig, missing_timeout_seconds: float = 2.0):
        self.config = config
        self.missing_timeout_seconds = missing_timeout_seconds
        self._states: dict[tuple[str, int, str], _ViolationState] = {}
        self._sequence = 0

    def process(
        self, camera_id: str, people: list[PersonPPE], observed_at: float
    ) -> list[EventDecision]:
        decisions: list[EventDecision] = []
        seen: set[tuple[str, int, str]] = set()
        for person in people:
            for ppe in self.config.required_ppe:
                violation_type = f"no_{ppe}"
                key = (camera_id, person.track_id, violation_type)
                seen.add(key)
                is_violation = getattr(person, ppe, None) is None
                decisions.extend(
                    self._observe(
                        key,
                        is_violation,
                        person.person.confidence,
                        observed_at,
                    )
                )
        decisions.extend(self._expire_missing(seen, observed_at))
        return decisions

    def close_camera(
        self, camera_id: str, observed_at: float, reason: str = "source_ended"
    ) -> list[EventDecision]:
        """Close active events and clear state when a video/camera source ends."""
        decisions: list[EventDecision] = []
        keys = [key for key in self._states if key[0] == camera_id]
        for key in keys:
            state = self._states.pop(key)
            if not state.active or not state.event_key:
                continue
            _, track_id, violation_type = key
            decisions.append(
                EventDecision(
                    "end",
                    state.event_key,
                    camera_id,
                    track_id,
                    violation_type,
                    0.0,
                    observed_at,
                    {"reason": reason},
                )
            )
        return decisions

    def _observe(
        self,
        key: tuple[str, int, str],
        is_violation: bool,
        confidence: float,
        now: float,
    ) -> list[EventDecision]:
        camera_id, track_id, violation_type = key
        if not self.config.enabled:
            if not is_violation:
                return []
            self._sequence += 1
            return [
                EventDecision(
                    "start",
                    f"{camera_id}:{track_id}:{violation_type}:frame:{self._sequence}",
                    camera_id,
                    track_id,
                    violation_type,
                    confidence,
                    now,
                    {"mode": "immediate"},
                )
            ]

        state = self._states.setdefault(key, _ViolationState())
        state.last_seen = now
        state.samples.append((now, is_violation))
        cutoff = now - self.config.voting_window_seconds
        while state.samples and state.samples[0][0] < cutoff:
            state.samples.popleft()

        if is_violation:
            state.compliant_since = None
            if state.violation_since is None:
                state.violation_since = now
        else:
            state.violation_since = None
            if state.compliant_since is None:
                state.compliant_since = now

        confirmed = self._confirmed(state, now)
        if confirmed and not state.active:
            self._sequence += 1
            state.active = True
            state.event_key = (
                f"{camera_id}:{track_id}:{violation_type}:{int(now * 1000)}:{self._sequence}"
            )
            return [
                EventDecision(
                    "start",
                    state.event_key,
                    camera_id,
                    track_id,
                    violation_type,
                    confidence,
                    now,
                    self._details(state, now),
                )
            ]

        if state.active and is_violation:
            return [
                EventDecision(
                    "update",
                    state.event_key or "",
                    camera_id,
                    track_id,
                    violation_type,
                    confidence,
                    now,
                    self._details(state, now),
                )
            ]

        recovered = (
            state.active
            and state.compliant_since is not None
            and now - state.compliant_since >= self.config.recovery_seconds
        )
        if recovered:
            decision = EventDecision(
                "end",
                state.event_key or "",
                camera_id,
                track_id,
                violation_type,
                confidence,
                now,
                {"reason": "ppe_recovered"},
            )
            state.active = False
            state.event_key = None
            return [decision]
        return []

    def _confirmed(self, state: _ViolationState, now: float) -> bool:
        if self.config.mode == "majority":
            if len(state.samples) < self.config.min_voting_samples:
                return False
            ratio = sum(int(value) for _, value in state.samples) / len(state.samples)
            covered = state.samples[-1][0] - state.samples[0][0]
            return (
                ratio >= self.config.voting_ratio
                and covered >= min(
                    self.config.violation_seconds,
                    self.config.voting_window_seconds,
                )
            )
        return (
            state.violation_since is not None
            and now - state.violation_since >= self.config.violation_seconds
        )

    def _details(self, state: _ViolationState, now: float) -> dict[str, float | int | str]:
        ratio = (
            sum(int(value) for _, value in state.samples) / len(state.samples)
            if state.samples
            else 0.0
        )
        return {
            "mode": self.config.mode,
            "violation_duration": (
                now - state.violation_since if state.violation_since is not None else 0.0
            ),
            "violation_ratio": ratio,
            "sample_count": len(state.samples),
        }

    def _expire_missing(
        self, seen: set[tuple[str, int, str]], now: float
    ) -> list[EventDecision]:
        decisions: list[EventDecision] = []
        expired: list[tuple[str, int, str]] = []
        for key, state in self._states.items():
            if key in seen or now - state.last_seen < self.missing_timeout_seconds:
                continue
            if state.active and state.event_key:
                camera_id, track_id, violation_type = key
                decisions.append(
                    EventDecision(
                        "end",
                        state.event_key,
                        camera_id,
                        track_id,
                        violation_type,
                        0.0,
                        now,
                        {"reason": "track_lost"},
                    )
                )
            expired.append(key)
        for key in expired:
            del self._states[key]
        return decisions
