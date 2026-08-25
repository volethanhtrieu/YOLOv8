from backend.config import load_config


def test_full_profile_uses_chvg4_and_two_second_delay():
    config = load_config("config.yaml", profile="D_full_system")

    assert config.model.path == "weights/S-N0-coco-best.pt"
    assert config.classes.person == ["person"]
    assert config.classes.head == ["head"]
    assert config.classes.helmet == ["helmet"]
    assert config.classes.vest == ["vest"]
    assert config.event.enabled is True
    assert config.event.mode == "consecutive"
    assert config.event.violation_seconds == 2.0
    assert config.event.required_ppe == ["helmet", "vest"]
