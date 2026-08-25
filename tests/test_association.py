from backend.association import PPEAssociation
from backend.config import AssociationConfig, ClassConfig
from backend.types import Detection


def det(name, box, track_id=None, confidence=0.9):
    return Detection(box, confidence, 0, name, track_id)


def test_roi_association_assigns_ppe_to_correct_person():
    association = PPEAssociation(AssociationConfig(), ClassConfig())
    detections = [
        det("person", (0, 0, 100, 300), 1),
        det("person", (120, 0, 220, 300), 2),
        det("helmet", (20, 10, 80, 70)),
        det("helmet", (140, 12, 200, 72)),
        det("vest", (20, 100, 85, 240)),
    ]

    people = association.associate(detections)

    assert [person.track_id for person in people] == [1, 2]
    assert people[0].helmet.class_name == "helmet"
    assert people[1].helmet.class_name == "helmet"
    assert people[0].vest is not None
    assert people[1].vest is None


def test_one_ppe_detection_cannot_be_assigned_twice():
    config = AssociationConfig(mode="full_body")
    association = PPEAssociation(config, ClassConfig())
    detections = [
        det("person", (0, 0, 100, 300), 1),
        det("person", (50, 0, 150, 300), 2),
        det("helmet", (65, 10, 90, 50)),
    ]

    people = association.associate(detections)

    assert sum(person.helmet is not None for person in people) == 1
