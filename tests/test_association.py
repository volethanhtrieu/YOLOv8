from src.variant_c.association import (
    Detection,
    TrackedPerson,
    associate_ppe,
)


persons = [
    TrackedPerson(
        track_id=10,
        bbox=(100, 100, 300, 500),
        confidence=0.95,
    ),
    TrackedPerson(
        track_id=20,
        bbox=(350, 100, 550, 500),
        confidence=0.94,
    ),
]


detections = [
    Detection(
        "head",
        (160, 110, 230, 180),
        0.95,
    ),
    Detection(
        "helmet",
        (155, 100, 235, 155),
        0.92,
    ),
    Detection(
        "vest",
        (135, 210, 265, 360),
        0.90,
    ),

    Detection(
        "head",
        (410, 110, 480, 180),
        0.94,
    ),
]


result = associate_ppe(
    persons,
    detections,
)


assert result[0]["track_id"] == 10
assert result[0]["has_helmet"] is True
assert result[0]["has_vest"] is True

assert result[1]["track_id"] == 20
assert result[1]["has_helmet"] is False
assert result[1]["has_vest"] is False


print("Association test PASSED")