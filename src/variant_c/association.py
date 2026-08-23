from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    class_name: str
    bbox: BBox
    confidence: float


@dataclass
class TrackedPerson:
    track_id: int
    bbox: BBox
    confidence: float


def area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def containment(item: BBox, region: BBox) -> float:
    item_area = area(item)

    if item_area == 0:
        return 0.0

    return intersection(item, region) / item_area


def center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_inside(item: BBox, region: BBox) -> bool:
    cx, cy = center(item)
    x1, y1, x2, y2 = region

    return x1 <= cx <= x2 and y1 <= cy <= y2


def head_region(person_box: BBox) -> BBox:
    x1, y1, x2, y2 = person_box
    h = y2 - y1

    return (
        x1,
        y1,
        x2,
        y1 + 0.42 * h,
    )


def torso_region(person_box: BBox) -> BBox:
    x1, y1, x2, y2 = person_box
    h = y2 - y1

    return (
        x1,
        y1 + 0.18 * h,
        x2,
        y1 + 0.78 * h,
    )


def expand_box(box: BBox, scale: float = 0.25) -> BBox:
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    return (
        x1 - w * scale,
        y1 - h * scale,
        x2 + w * scale,
        y2 + h * scale,
    )


def association_region(
    person: TrackedPerson,
    class_name: str,
    matched_head: Optional[BBox] = None,
) -> BBox:

    if class_name == "helmet":
        if matched_head is not None:
            return expand_box(matched_head)

        return head_region(person.bbox)

    if class_name == "head":
        return head_region(person.bbox)

    if class_name == "vest":
        return torso_region(person.bbox)

    return person.bbox


def association_score(
    person: TrackedPerson,
    item: Detection,
    matched_head: Optional[BBox] = None,
) -> float:

    region = association_region(
        person,
        item.class_name,
        matched_head,
    )

    overlap = containment(item.bbox, region)

    if overlap < 0.15 and not center_inside(item.bbox, region):
        return 0.0

    icx, icy = center(item.bbox)
    rx1, ry1, rx2, ry2 = region

    rcx = (rx1 + rx2) / 2
    rcy = (ry1 + ry2) / 2

    rw = max(rx2 - rx1, 1.0)
    rh = max(ry2 - ry1, 1.0)

    distance = (
        abs(icx - rcx) / rw
        + abs(icy - rcy) / rh
    ) / 2

    proximity = max(0.0, 1.0 - distance)

    return (
        0.65 * overlap
        + 0.25 * proximity
        + 0.10 * item.confidence
    )


def match_class(
    persons: List[TrackedPerson],
    detections: List[Detection],
    class_name: str,
    heads: Dict[int, Detection],
    min_score: float = 0.25,
) -> Dict[int, Detection]:

    items = [
        d for d in detections
        if d.class_name == class_name
    ]

    candidates = []

    for item_index, item in enumerate(items):

        for person in persons:

            head = heads.get(person.track_id)

            score = association_score(
                person,
                item,
                head.bbox if head else None,
            )

            if score >= min_score:
                candidates.append(
                    (
                        score,
                        person.track_id,
                        item_index,
                        item,
                    )
                )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    used_people = set()
    used_items = set()

    matches = {}

    for score, track_id, item_index, item in candidates:

        if track_id in used_people:
            continue

        if item_index in used_items:
            continue

        matches[track_id] = item

        used_people.add(track_id)
        used_items.add(item_index)

    return matches


def associate_ppe(
    persons: List[TrackedPerson],
    detections: List[Detection],
) -> List[dict]:

    heads = match_class(
        persons,
        detections,
        "head",
        {},
    )

    helmets = match_class(
        persons,
        detections,
        "helmet",
        heads,
    )

    vests = match_class(
        persons,
        detections,
        "vest",
        heads,
    )

    output = []

    for person in persons:

        tid = person.track_id

        output.append(
            {
                "track_id": tid,
                "person_bbox": person.bbox,
                "person_conf": person.confidence,

                "head": heads.get(tid),
                "helmet": helmets.get(tid),
                "vest": vests.get(tid),

                "has_head": tid in heads,
                "has_helmet": tid in helmets,
                "has_vest": tid in vests,
            }
        )

    return output
