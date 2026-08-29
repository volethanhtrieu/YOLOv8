from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(slots=True)
class PersonPPE:
    person: Detection
    helmet: Detection | None = None
    vest: Detection | None = None
    head: Detection | None = None
    extras: dict[str, Detection] = field(default_factory=dict)

    @property
    def track_id(self) -> int:
        if self.person.track_id is None:
            raise ValueError("Person detection has no track_id")
        return self.person.track_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "person_confidence": self.person.confidence,
            "bbox": list(self.person.bbox),
            "head": self.head is not None,
            "head_confidence": self.head.confidence if self.head else None,
            "helmet": self.helmet is not None,
            "helmet_class": self.helmet.class_name if self.helmet else None,
            "helmet_confidence": self.helmet.confidence if self.helmet else None,
            "vest": self.vest is not None,
            "vest_confidence": self.vest.confidence if self.vest else None,
        }


@dataclass(slots=True)
class EventDecision:
    action: str  # start | update | end
    event_key: str
    camera_id: str
    track_id: int
    violation_type: str
    confidence: float
    observed_at: float
    details: dict[str, Any] = field(default_factory=dict)
