from __future__ import annotations

from collections.abc import Iterable

from .config import AssociationConfig, ClassConfig
from .types import BBox, Detection, PersonPPE


def intersection_area(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def crop_vertical(box: BBox, top_ratio: float, bottom_ratio: float) -> BBox:
    x1, y1, x2, y2 = box
    height = y2 - y1
    return (x1, y1 + height * top_ratio, x2, y1 + height * bottom_ratio)


class PPEAssociation:
    """Greedily assigns each PPE detection to at most one tracked person."""

    def __init__(self, config: AssociationConfig, classes: ClassConfig):
        self.config = config
        self.classes = classes
        self._person_names = {name.lower() for name in classes.person}
        self._head_names = {name.lower() for name in classes.head}
        self._helmet_names = {name.lower() for name in classes.helmet}
        self._vest_names = {name.lower() for name in classes.vest}

    def associate(self, detections: Iterable[Detection]) -> list[PersonPPE]:
        items = list(detections)
        persons = [
            PersonPPE(det)
            for det in items
            if det.class_name.lower() in self._person_names and det.track_id is not None
        ]
        if not persons:
            return []

        groups = (
            ("head", self._head_names),
            ("helmet", self._helmet_names),
            ("vest", self._vest_names),
        )
        for attribute, names in groups:
            candidates = [det for det in items if det.class_name.lower() in names]
            self._assign_group(persons, candidates, attribute)
        return persons

    def _assign_group(
        self, persons: list[PersonPPE], candidates: list[Detection], attribute: str
    ) -> None:
        scored: list[tuple[float, float, int, int]] = []
        for item_index, item in enumerate(candidates):
            for person_index, person in enumerate(persons):
                region = self._region(person, attribute)
                inside = intersection_area(item.bbox, region) / max(item.area, 1e-9)
                if inside < self.config.min_item_inside_ratio:
                    continue
                cx, cy = item.center
                rx1, ry1, rx2, ry2 = region
                center_inside = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                score = inside + (0.25 if center_inside else 0.0)
                scored.append((score, item.confidence, item_index, person_index))

        used_items: set[int] = set()
        used_persons: set[int] = set()
        for _, _, item_index, person_index in sorted(scored, reverse=True):
            if item_index in used_items or person_index in used_persons:
                continue
            setattr(persons[person_index], attribute, candidates[item_index])
            used_items.add(item_index)
            used_persons.add(person_index)

    def _region(self, person: PersonPPE, attribute: str) -> BBox:
        box = person.person.bbox
        if not self.config.enabled or self.config.mode == "full_body":
            return box
        if attribute in {"helmet", "head"}:
            if attribute == "helmet" and person.head is not None:
                return person.head.bbox
            return crop_vertical(box, 0.0, self.config.head_ratio)
        if attribute == "vest":
            return crop_vertical(
                box,
                self.config.torso_top_ratio,
                self.config.torso_bottom_ratio,
            )
        return box
