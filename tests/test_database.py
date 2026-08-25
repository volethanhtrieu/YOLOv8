from backend.database import EventRepository
from backend.types import EventDecision


def decision(action, key="event-1", details=None):
    return EventDecision(
        action=action,
        event_key=key,
        camera_id="cam",
        track_id=5,
        violation_type="no_helmet",
        confidence=0.9,
        observed_at=1.0,
        details=details or {},
    )


def test_event_lifecycle_is_persisted(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    repository.apply(decision("start"), "evidence/test.jpg")
    repository.apply(decision("update"))
    repository.apply(decision("end", details={"reason": "ppe_recovered"}))

    events = repository.list_events()
    assert len(events) == 1
    assert events[0]["status"] == "resolved"
    assert events[0]["end_reason"] == "ppe_recovered"
    assert events[0]["evidence_path"] == "evidence/test.jpg"


def test_stats_count_violation_types(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    repository.apply(decision("start", "one"))
    repository.apply(decision("start", "two"))
    assert repository.stats()["no_helmet"] == 2


def test_list_events_after_id_only_returns_current_run(tmp_path):
    repository = EventRepository(tmp_path / "events.db")
    repository.apply(decision("start", "old"))
    previous_id = repository.latest_event_id()
    repository.apply(decision("start", "new"))

    events = repository.list_events_after_id(previous_id, camera_id="cam")

    assert [event["event_key"] for event in events] == ["new"]
