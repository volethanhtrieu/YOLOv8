from backend.config import EventConfig
from backend.event_engine import EventEngine
from backend.types import Detection, PersonPPE


def person(track_id=7, helmet=None, vest=None):
    return PersonPPE(
        person=Detection((0, 0, 100, 200), 0.95, 0, "person", track_id),
        helmet=helmet,
        vest=vest,
    )


def ppe(name):
    return Detection((10, 10, 50, 50), 0.9, 1, name)


def test_consecutive_engine_starts_once_and_recovers():
    config = EventConfig(
        mode="consecutive",
        violation_seconds=2.0,
        recovery_seconds=0.5,
        required_ppe=["helmet"],
    )
    engine = EventEngine(config)

    assert engine.process("cam", [person(helmet=None)], 0.0) == []
    starts = engine.process("cam", [person(helmet=None)], 2.0)
    assert [item.action for item in starts] == ["start"]

    updates = engine.process("cam", [person(helmet=None)], 2.2)
    assert [item.action for item in updates] == ["update"]

    assert engine.process("cam", [person(helmet=ppe("helmet"))], 2.3) == []
    ends = engine.process("cam", [person(helmet=ppe("helmet"))], 2.8)
    assert [item.action for item in ends] == ["end"]
    assert ends[0].event_key == starts[0].event_key


def test_majority_voting_tolerates_one_good_sample():
    config = EventConfig(
        mode="majority",
        violation_seconds=0.3,
        voting_window_seconds=1.0,
        voting_ratio=0.70,
        min_voting_samples=4,
        required_ppe=["helmet"],
    )
    engine = EventEngine(config)
    decisions = []
    for timestamp, has_helmet in [(0.0, False), (0.1, False), (0.2, True), (0.4, False)]:
        decisions.extend(
            engine.process(
                "cam",
                [person(helmet=ppe("helmet") if has_helmet else None)],
                timestamp,
            )
        )
    assert any(item.action == "start" for item in decisions)


def test_active_event_closes_when_track_is_lost():
    config = EventConfig(
        violation_seconds=0.0,
        required_ppe=["helmet"],
    )
    engine = EventEngine(config, missing_timeout_seconds=1.0)
    starts = engine.process("cam", [person(helmet=None)], 0.0)
    assert starts[0].action == "start"
    assert engine.process("cam", [], 0.5) == []
    ends = engine.process("cam", [], 1.1)
    assert ends[0].action == "end"
    assert ends[0].details["reason"] == "track_lost"


def test_active_event_closes_when_source_ends():
    config = EventConfig(violation_seconds=0.0, required_ppe=["helmet"])
    engine = EventEngine(config)
    starts = engine.process("cam", [person(helmet=None)], 0.0)
    assert starts[0].action == "start"

    ends = engine.close_camera("cam", 2.0)

    assert len(ends) == 1
    assert ends[0].action == "end"
    assert ends[0].event_key == starts[0].event_key
    assert ends[0].details["reason"] == "source_ended"
    assert engine.close_camera("cam", 3.0) == []
