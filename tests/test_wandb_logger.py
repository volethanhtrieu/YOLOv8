from backend.wandb_logger import WandbVideoLogger


class FakeTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *values):
        self.rows.append(values)


class FakeWandb:
    Table = FakeTable


class FakeRun:
    def __init__(self):
        self.metrics = []
        self.logs = []

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def log(self, values, step=None):
        self.logs.append((values, step))


def test_wandb_logger_collects_frame_metrics_and_track_table():
    run = FakeRun()
    logger = WandbVideoLogger(FakeWandb(), run, log_every=1)
    payload = {
        "people": [
            {
                "track_id": 7,
                "person_confidence": 0.9,
                "head": False,
                "helmet": True,
                "helmet_confidence": 0.8,
                "vest": True,
                "vest_confidence": 0.7,
            }
        ],
        "events": [
            {
                "action": "start",
                "violation_type": "no_helmet",
            }
        ],
        "counts": {"tracked_people": 1, "no_helmet": 1, "no_vest": 0},
        "inference_ms": 50.0,
        "class_counts": {"person": 1, "head": 0, "helmet": 1, "vest": 1},
        "class_confidence_means": {
            "person": 0.9,
            "head": 0.0,
            "helmet": 0.8,
            "vest": 0.7,
        },
    }

    logger.observe_frame(payload, 1, video_time_seconds=0.04, elapsed_seconds=0.1)

    frame_metrics, step = run.logs[-1]
    assert step == 1
    assert frame_metrics["runtime/latency_ms"] == 50.0
    assert frame_metrics["runtime/instant_fps"] == 20.0
    assert frame_metrics["tracking/unique_person_tracks_so_far"] == 1
    assert frame_metrics["confidence/helmet_mean_in_frame"] == 0.8
    assert frame_metrics["violations/events_started_so_far"] == 1

    table = logger.log_tracking_table()
    assert table.columns == WandbVideoLogger.TRACK_COLUMNS
    assert len(table.rows) == 1
    assert table.rows[0][0] == 7
    assert table.rows[0][1] == 1

