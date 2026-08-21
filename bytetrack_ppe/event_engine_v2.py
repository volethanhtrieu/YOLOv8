import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULT_TRACK_CSV = (
    ROOT
    / "outputs"
    / "tiled_ppe_pipeline_v2"
    / "track_ppe_rows.csv"
)

DEFAULT_SUMMARY_JSON = (
    ROOT
    / "outputs"
    / "tiled_ppe_pipeline_v2"
    / "summary.json"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "event_engine_v2"
)

# ---------------------------------------------------------
# Shared temporal settings
# ---------------------------------------------------------

WINDOW_S = 3.0
TRACK_LOST_GRACE_S = 1.0

PERSON_CONF_MIN = 0.55
PERSON_HEIGHT_MIN = 180.0
PERSON_WIDTH_MIN = 50.0

PPE_CONF_MIN = 0.20

# ---------------------------------------------------------
# Helmet
#
# Helmet violation has positive evidence:
# CHVG "head" means an exposed / bare head.
# Therefore helmet events can be CONFIRMED.
# ---------------------------------------------------------

HELMET_OPEN_RATIO = 0.70
HELMET_CLOSE_RATIO = 0.30

HELMET_MIN_VALID_OPEN = 15
HELMET_MIN_VALID_CLOSE = 10

HELMET_MIN_SPAN_S = 1.0
HELMET_RECOVERY_HOLD_S = 1.0

# ---------------------------------------------------------
# Vest
#
# There is no explicit "no_vest" detector class.
# Missing vest detection is therefore negative evidence only.
#
# V2 never promotes absence-only evidence to CONFIRMED.
# It opens SUSPECTED_NO_VEST for review.
# ---------------------------------------------------------

VEST_SUSPECT_RATIO = 0.80
VEST_RECOVER_RATIO = 0.30

VEST_MIN_VALID_SUSPECT = 20
VEST_MIN_SPAN_S = 1.5
VEST_RECOVERY_HOLD_S = 0.5

# If the same track had a positive vest detection recently,
# a short missing-detection period is treated as UNKNOWN.
VEST_RECENT_POSITIVE_GRACE_S = 2.0

# ---------------------------------------------------------
# Torso visibility proxy
#
# This only handles person-on-person occlusion.
# It cannot prove visibility against static objects such as
# rebar, formwork, walls, machinery, etc.
# ---------------------------------------------------------

TORSO_X_MARGIN_RATIO = 0.15
TORSO_TOP_RATIO = 0.20
TORSO_BOTTOM_RATIO = 0.72

TORSO_PERSON_OCCLUSION_MAX = 0.35


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track-csv",
        type=Path,
        default=DEFAULT_TRACK_CSV,
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def parse_optional_float(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        rows = []

        for row in reader:
            rows.append(
                {
                    "frame_index": int(
                        float(
                            row["frame_index"]
                        )
                    ),
                    "timestamp_s": float(
                        row["timestamp_s"]
                    ),
                    "track_id": int(
                        float(
                            row["track_id"]
                        )
                    ),
                    "person_conf": float(
                        row["person_conf"]
                    ),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "head_conf": (
                        parse_optional_float(
                            row.get(
                                "head_conf"
                            )
                        )
                    ),
                    "helmet_conf": (
                        parse_optional_float(
                            row.get(
                                "helmet_conf"
                            )
                        )
                    ),
                    "vest_conf": (
                        parse_optional_float(
                            row.get(
                                "vest_conf"
                            )
                        )
                    ),
                    "glass_conf": (
                        parse_optional_float(
                            row.get(
                                "glass_conf"
                            )
                        )
                    ),
                }
            )

    rows.sort(
        key=lambda row: (
            row["frame_index"],
            row["track_id"],
        )
    )

    return rows


def is_present(confidence):
    return (
        confidence is not None
        and confidence >= PPE_CONF_MIN
    )


def person_quality(row):
    width = (
        row["x2"]
        - row["x1"]
    )

    height = (
        row["y2"]
        - row["y1"]
    )

    return (
        row["person_conf"]
        >= PERSON_CONF_MIN
        and width
        >= PERSON_WIDTH_MIN
        and height
        >= PERSON_HEIGHT_MIN
    )


def box_intersection_area(
    box_a,
    box_b,
):
    x1 = max(
        box_a[0],
        box_b[0],
    )

    y1 = max(
        box_a[1],
        box_b[1],
    )

    x2 = min(
        box_a[2],
        box_b[2],
    )

    y2 = min(
        box_a[3],
        box_b[3],
    )

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def torso_box(row):
    x1 = row["x1"]
    y1 = row["y1"]
    x2 = row["x2"]
    y2 = row["y2"]

    width = max(
        1.0,
        x2 - x1,
    )

    height = max(
        1.0,
        y2 - y1,
    )

    return (
        x1
        + TORSO_X_MARGIN_RATIO
        * width,
        y1
        + TORSO_TOP_RATIO
        * height,
        x2
        - TORSO_X_MARGIN_RATIO
        * width,
        y1
        + TORSO_BOTTOM_RATIO
        * height,
    )


def torso_person_occlusion_ratio(
    target_row,
    frame_rows,
):
    torso = torso_box(
        target_row
    )

    torso_area = (
        max(
            0.0,
            torso[2]
            - torso[0],
        )
        * max(
            0.0,
            torso[3]
            - torso[1],
        )
    )

    if torso_area <= 0:
        return 1.0

    total_overlap = 0.0

    for other in frame_rows:
        if (
            other["track_id"]
            == target_row["track_id"]
        ):
            continue

        other_box = (
            other["x1"],
            other["y1"],
            other["x2"],
            other["y2"],
        )

        total_overlap += (
            box_intersection_area(
                torso,
                other_box,
            )
        )

    return min(
        1.0,
        total_overlap
        / torso_area,
    )


def helmet_observation(row):
    helmet = is_present(
        row["helmet_conf"]
    )

    bare_head = is_present(
        row["head_conf"]
    )

    if helmet and not bare_head:
        return (
            "COMPLIANT",
            "helmet_detected",
        )

    if bare_head and not helmet:
        return (
            "VIOLATION",
            "bare_head_detected",
        )

    if (
        helmet
        and bare_head
    ):
        return (
            "UNKNOWN",
            "contradictory_head_helmet",
        )

    return (
        "UNKNOWN",
        "no_head_evidence",
    )


def vest_observation(
    row,
    frame_rows,
    last_vest_seen_s,
):
    if is_present(
        row["vest_conf"]
    ):
        return (
            "COMPLIANT",
            "vest_detected",
            0.0,
            True,
        )

    if not person_quality(row):
        return (
            "UNKNOWN",
            "person_quality_low",
            None,
            False,
        )

    occlusion_ratio = (
        torso_person_occlusion_ratio(
            row,
            frame_rows,
        )
    )

    if (
        occlusion_ratio
        > TORSO_PERSON_OCCLUSION_MAX
    ):
        return (
            "UNKNOWN",
            "torso_occluded_by_person",
            occlusion_ratio,
            False,
        )

    if (
        last_vest_seen_s
        is not None
    ):
        age = (
            row["timestamp_s"]
            - last_vest_seen_s
        )

        if (
            age
            <= VEST_RECENT_POSITIVE_GRACE_S
        ):
            return (
                "UNKNOWN",
                "recent_positive_vest_evidence",
                occlusion_ratio,
                True,
            )

    # Important:
    # this is NOT a confirmed violation.
    # It is only absence evidence under a limited
    # torso-visibility proxy.
    return (
        "ABSENT_EVIDENCE",
        "vest_not_detected_visible_proxy",
        occlusion_ratio,
        True,
    )


class HelmetTemporalRule:
    def __init__(self):
        self.history = deque()

        self.state = "UNKNOWN"

        self.event_id = None

        self.recovery_start_s = None

    def prune(
        self,
        timestamp_s,
    ):
        cutoff = (
            timestamp_s
            - WINDOW_S
        )

        while (
            self.history
            and self.history[0][0]
            < cutoff
        ):
            self.history.popleft()

    def add(
        self,
        timestamp_s,
        observation,
    ):
        self.history.append(
            (
                timestamp_s,
                observation,
            )
        )

        self.prune(
            timestamp_s
        )

    def metrics(self):
        valid = [
            item
            for item
            in self.history
            if item[1]
            != "UNKNOWN"
        ]

        count = len(valid)

        if count == 0:
            return {
                "valid_count": 0,
                "violation_count": 0,
                "compliant_count": 0,
                "violation_ratio": None,
                "evidence_span_s": 0.0,
            }

        violations = sum(
            1
            for item
            in valid
            if item[1]
            == "VIOLATION"
        )

        ratio = (
            violations
            / count
        )

        span = (
            valid[-1][0]
            - valid[0][0]
            if count >= 2
            else 0.0
        )

        return {
            "valid_count": count,
            "violation_count": (
                violations
            ),
            "compliant_count": (
                count
                - violations
            ),
            "violation_ratio": (
                ratio
            ),
            "evidence_span_s": (
                span
            ),
        }


class VestTemporalRule:
    def __init__(self):
        self.history = deque()

        self.state = "UNKNOWN"

        self.event_id = None

        self.recovery_start_s = None

    def prune(
        self,
        timestamp_s,
    ):
        cutoff = (
            timestamp_s
            - WINDOW_S
        )

        while (
            self.history
            and self.history[0][0]
            < cutoff
        ):
            self.history.popleft()

    def add(
        self,
        timestamp_s,
        observation,
    ):
        self.history.append(
            (
                timestamp_s,
                observation,
            )
        )

        self.prune(
            timestamp_s
        )

    def metrics(self):
        valid = [
            item
            for item
            in self.history
            if item[1]
            in {
                "COMPLIANT",
                "ABSENT_EVIDENCE",
            }
        ]

        count = len(valid)

        if count == 0:
            return {
                "valid_count": 0,
                "absent_count": 0,
                "compliant_count": 0,
                "absent_ratio": None,
                "evidence_span_s": 0.0,
            }

        absent = sum(
            1
            for item
            in valid
            if item[1]
            == "ABSENT_EVIDENCE"
        )

        ratio = (
            absent
            / count
        )

        span = (
            valid[-1][0]
            - valid[0][0]
            if count >= 2
            else 0.0
        )

        return {
            "valid_count": count,
            "absent_count": absent,
            "compliant_count": (
                count
                - absent
            ),
            "absent_ratio": ratio,
            "evidence_span_s": span,
        }


def make_event_id(
    event_type,
    track_id,
    frame_index,
):
    return (
        f"{event_type}_"
        f"T{track_id}_"
        f"F{frame_index}"
    )


def append_event(
    events,
    *,
    event_id,
    track_id,
    ppe_type,
    event_type,
    event,
    status,
    frame_index,
    timestamp_s,
    evidence_ratio,
    valid_samples,
    reason,
):
    events.append(
        {
            "event_id": event_id,
            "track_id": track_id,
            "ppe_type": ppe_type,
            "event_type": event_type,
            "event": event,
            "status": status,
            "frame_index": frame_index,
            "timestamp_s": timestamp_s,
            "evidence_ratio": evidence_ratio,
            "valid_samples": valid_samples,
            "reason": reason,
        }
    )


def update_helmet_rule(
    rule,
    track_id,
    frame_index,
    timestamp_s,
    observation,
    events,
):
    rule.add(
        timestamp_s,
        observation,
    )

    metrics = rule.metrics()

    valid_count = (
        metrics["valid_count"]
    )

    ratio = (
        metrics[
            "violation_ratio"
        ]
    )

    span = (
        metrics[
            "evidence_span_s"
        ]
    )

    if valid_count == 0:
        if rule.state not in {
            "ACTIVE",
            "RECOVERING",
        }:
            rule.state = "UNKNOWN"

        return metrics

    can_open = (
        valid_count
        >= HELMET_MIN_VALID_OPEN
        and span
        >= HELMET_MIN_SPAN_S
        and ratio is not None
        and ratio
        >= HELMET_OPEN_RATIO
    )

    can_close = (
        valid_count
        >= HELMET_MIN_VALID_CLOSE
        and ratio is not None
        and ratio
        <= HELMET_CLOSE_RATIO
    )

    if rule.state in {
        "UNKNOWN",
        "SAFE",
        "CLOSED",
    }:
        if can_open:
            rule.state = "ACTIVE"

            rule.event_id = (
                make_event_id(
                    "no_helmet",
                    track_id,
                    frame_index,
                )
            )

            append_event(
                events,
                event_id=rule.event_id,
                track_id=track_id,
                ppe_type="helmet",
                event_type="NO_HELMET",
                event="OPEN",
                status="CONFIRMED",
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                evidence_ratio=ratio,
                valid_samples=valid_count,
                reason=(
                    "bare_head_temporal_threshold"
                ),
            )

        else:
            rule.state = "SAFE"

    elif rule.state == "ACTIVE":
        if can_close:
            rule.state = "RECOVERING"

            rule.recovery_start_s = (
                timestamp_s
            )

    elif rule.state == "RECOVERING":
        if (
            ratio is not None
            and ratio
            >= HELMET_OPEN_RATIO
        ):
            rule.state = "ACTIVE"

            rule.recovery_start_s = None

        elif can_close:
            recovery_age = (
                timestamp_s
                - rule.recovery_start_s
            )

            if (
                recovery_age
                >= HELMET_RECOVERY_HOLD_S
            ):
                append_event(
                    events,
                    event_id=rule.event_id,
                    track_id=track_id,
                    ppe_type="helmet",
                    event_type="NO_HELMET",
                    event="CLOSE",
                    status="CONFIRMED",
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    evidence_ratio=ratio,
                    valid_samples=valid_count,
                    reason="helmet_recovered",
                )

                rule.state = "CLOSED"
                rule.event_id = None
                rule.recovery_start_s = None

        else:
            rule.state = "ACTIVE"
            rule.recovery_start_s = None

    return metrics


def update_vest_rule(
    rule,
    track_id,
    frame_index,
    timestamp_s,
    observation,
    events,
):
    rule.add(
        timestamp_s,
        observation,
    )

    metrics = rule.metrics()

    valid_count = (
        metrics["valid_count"]
    )

    ratio = (
        metrics[
            "absent_ratio"
        ]
    )

    span = (
        metrics[
            "evidence_span_s"
        ]
    )

    if valid_count == 0:
        if rule.state not in {
            "SUSPECTED",
            "RECOVERING",
        }:
            rule.state = "UNKNOWN"

        return metrics

    can_suspect = (
        valid_count
        >= VEST_MIN_VALID_SUSPECT
        and span
        >= VEST_MIN_SPAN_S
        and ratio is not None
        and ratio
        >= VEST_SUSPECT_RATIO
    )

    can_recover = (
        valid_count
        >= 10
        and ratio is not None
        and ratio
        <= VEST_RECOVER_RATIO
    )

    if rule.state in {
        "UNKNOWN",
        "SAFE",
        "CLOSED",
    }:
        if can_suspect:
            rule.state = "SUSPECTED"

            rule.event_id = (
                make_event_id(
                    "suspected_no_vest",
                    track_id,
                    frame_index,
                )
            )

            append_event(
                events,
                event_id=rule.event_id,
                track_id=track_id,
                ppe_type="vest",
                event_type=(
                    "SUSPECTED_NO_VEST"
                ),
                event="OPEN",
                status="SUSPECTED",
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                evidence_ratio=ratio,
                valid_samples=valid_count,
                reason=(
                    "absence_evidence_temporal_threshold"
                ),
            )

        else:
            rule.state = "SAFE"

    elif rule.state == "SUSPECTED":
        if can_recover:
            rule.state = "RECOVERING"

            rule.recovery_start_s = (
                timestamp_s
            )

    elif rule.state == "RECOVERING":
        if (
            ratio is not None
            and ratio
            >= VEST_SUSPECT_RATIO
        ):
            rule.state = "SUSPECTED"
            rule.recovery_start_s = None

        elif can_recover:
            recovery_age = (
                timestamp_s
                - rule.recovery_start_s
            )

            if (
                recovery_age
                >= VEST_RECOVERY_HOLD_S
            ):
                append_event(
                    events,
                    event_id=rule.event_id,
                    track_id=track_id,
                    ppe_type="vest",
                    event_type=(
                        "SUSPECTED_NO_VEST"
                    ),
                    event="CLOSE",
                    status="SUSPECTED",
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    evidence_ratio=ratio,
                    valid_samples=valid_count,
                    reason=(
                        "vest_detected_recovered"
                    ),
                )

                rule.state = "CLOSED"
                rule.event_id = None
                rule.recovery_start_s = None

        else:
            rule.state = "SUSPECTED"
            rule.recovery_start_s = None

    return metrics


def close_lost_event(
    rule,
    track_id,
    ppe_type,
    event_type,
    status,
    frame_index,
    timestamp_s,
    events,
):
    active_states = {
        "ACTIVE",
        "RECOVERING",
        "SUSPECTED",
    }

    if rule.state not in active_states:
        return

    if rule.event_id is None:
        return

    append_event(
        events,
        event_id=rule.event_id,
        track_id=track_id,
        ppe_type=ppe_type,
        event_type=event_type,
        event="CLOSE",
        status=status,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        evidence_ratio=None,
        valid_samples=None,
        reason="track_lost",
    )

    rule.state = "CLOSED"
    rule.event_id = None
    rule.recovery_start_s = None


def main():
    args = parse_args()

    track_csv = (
        args.track_csv.resolve()
    )

    summary_json = (
        args.summary_json.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    states_csv = (
        output_dir
        / "ppe_temporal_states.csv"
    )

    events_csv = (
        output_dir
        / "events.csv"
    )

    events_json = (
        output_dir
        / "events.json"
    )

    output_summary = (
        output_dir
        / "summary.json"
    )

    if not track_csv.is_file():
        raise FileNotFoundError(
            track_csv
        )

    rows = load_rows(
        track_csv
    )

    source_summary = {}

    if summary_json.is_file():
        with summary_json.open(
            "r",
            encoding="utf-8",
        ) as f:
            source_summary = json.load(f)

    frames = defaultdict(list)

    for row in rows:
        frames[
            row["frame_index"]
        ].append(row)

    helmet_rules = {}
    vest_rules = {}

    last_seen_s = {}
    last_vest_seen_s = {}

    events = []
    state_rows = []

    for frame_index in sorted(
        frames.keys()
    ):
        frame_rows = (
            frames[frame_index]
        )

        timestamp_s = min(
            row["timestamp_s"]
            for row in frame_rows
        )

        current_ids = {
            row["track_id"]
            for row in frame_rows
        }

        # Close open events only after track-lost grace.
        for track_id in list(
            last_seen_s.keys()
        ):
            if track_id in current_ids:
                continue

            lost_age = (
                timestamp_s
                - last_seen_s[
                    track_id
                ]
            )

            if (
                lost_age
                <= TRACK_LOST_GRACE_S
            ):
                continue

            helmet_rule = (
                helmet_rules.get(
                    track_id
                )
            )

            if helmet_rule is not None:
                close_lost_event(
                    rule=helmet_rule,
                    track_id=track_id,
                    ppe_type="helmet",
                    event_type="NO_HELMET",
                    status="CONFIRMED",
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    events=events,
                )

            vest_rule = (
                vest_rules.get(
                    track_id
                )
            )

            if vest_rule is not None:
                close_lost_event(
                    rule=vest_rule,
                    track_id=track_id,
                    ppe_type="vest",
                    event_type=(
                        "SUSPECTED_NO_VEST"
                    ),
                    status="SUSPECTED",
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    events=events,
                )

        for row in frame_rows:
            track_id = (
                row["track_id"]
            )

            last_seen_s[
                track_id
            ] = timestamp_s

            (
                helmet_obs,
                helmet_reason,
            ) = helmet_observation(
                row
            )

            previous_vest_seen = (
                last_vest_seen_s.get(
                    track_id
                )
            )

            (
                vest_obs,
                vest_reason,
                torso_occlusion_ratio,
                torso_visible_proxy,
            ) = vest_observation(
                row=row,
                frame_rows=frame_rows,
                last_vest_seen_s=(
                    previous_vest_seen
                ),
            )

            if is_present(
                row["vest_conf"]
            ):
                last_vest_seen_s[
                    track_id
                ] = timestamp_s

            if (
                track_id
                not in helmet_rules
            ):
                helmet_rules[
                    track_id
                ] = HelmetTemporalRule()

            if (
                track_id
                not in vest_rules
            ):
                vest_rules[
                    track_id
                ] = VestTemporalRule()

            helmet_metrics = (
                update_helmet_rule(
                    rule=helmet_rules[
                        track_id
                    ],
                    track_id=track_id,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    observation=helmet_obs,
                    events=events,
                )
            )

            vest_metrics = (
                update_vest_rule(
                    rule=vest_rules[
                        track_id
                    ],
                    track_id=track_id,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    observation=vest_obs,
                    events=events,
                )
            )

            state_rows.append(
                {
                    "frame_index": (
                        frame_index
                    ),
                    "timestamp_s": (
                        timestamp_s
                    ),
                    "track_id": track_id,
                    "person_conf": (
                        row[
                            "person_conf"
                        ]
                    ),
                    "person_width": (
                        row["x2"]
                        - row["x1"]
                    ),
                    "person_height": (
                        row["y2"]
                        - row["y1"]
                    ),
                    "quality_valid": int(
                        person_quality(
                            row
                        )
                    ),
                    "head_conf": (
                        ""
                        if row[
                            "head_conf"
                        ]
                        is None
                        else row[
                            "head_conf"
                        ]
                    ),
                    "helmet_conf": (
                        ""
                        if row[
                            "helmet_conf"
                        ]
                        is None
                        else row[
                            "helmet_conf"
                        ]
                    ),
                    "vest_conf": (
                        ""
                        if row[
                            "vest_conf"
                        ]
                        is None
                        else row[
                            "vest_conf"
                        ]
                    ),
                    "torso_person_occlusion_ratio": (
                        ""
                        if torso_occlusion_ratio
                        is None
                        else torso_occlusion_ratio
                    ),
                    "torso_visible_proxy": int(
                        torso_visible_proxy
                    ),
                    "helmet_observation": (
                        helmet_obs
                    ),
                    "helmet_reason": (
                        helmet_reason
                    ),
                    "helmet_state": (
                        helmet_rules[
                            track_id
                        ].state
                    ),
                    "helmet_valid_samples": (
                        helmet_metrics[
                            "valid_count"
                        ]
                    ),
                    "helmet_violation_ratio": (
                        ""
                        if helmet_metrics[
                            "violation_ratio"
                        ]
                        is None
                        else helmet_metrics[
                            "violation_ratio"
                        ]
                    ),
                    "vest_observation": (
                        vest_obs
                    ),
                    "vest_reason": (
                        vest_reason
                    ),
                    "vest_state": (
                        vest_rules[
                            track_id
                        ].state
                    ),
                    "vest_valid_samples": (
                        vest_metrics[
                            "valid_count"
                        ]
                    ),
                    "vest_absent_ratio": (
                        ""
                        if vest_metrics[
                            "absent_ratio"
                        ]
                        is None
                        else vest_metrics[
                            "absent_ratio"
                        ]
                    ),
                }
            )

    # Close events at clip end.
    if rows:
        final_frame = max(
            row["frame_index"]
            for row in rows
        )

        final_time = max(
            row["timestamp_s"]
            for row in rows
        )

        for (
            track_id,
            rule,
        ) in helmet_rules.items():
            if (
                rule.event_id
                is not None
            ):
                append_event(
                    events,
                    event_id=rule.event_id,
                    track_id=track_id,
                    ppe_type="helmet",
                    event_type="NO_HELMET",
                    event="CLOSE",
                    status="CONFIRMED",
                    frame_index=final_frame,
                    timestamp_s=final_time,
                    evidence_ratio=None,
                    valid_samples=None,
                    reason="clip_end",
                )

        for (
            track_id,
            rule,
        ) in vest_rules.items():
            if (
                rule.event_id
                is not None
            ):
                append_event(
                    events,
                    event_id=rule.event_id,
                    track_id=track_id,
                    ppe_type="vest",
                    event_type=(
                        "SUSPECTED_NO_VEST"
                    ),
                    event="CLOSE",
                    status="SUSPECTED",
                    frame_index=final_frame,
                    timestamp_s=final_time,
                    evidence_ratio=None,
                    valid_samples=None,
                    reason="clip_end",
                )

    state_fields = [
        "frame_index",
        "timestamp_s",
        "track_id",
        "person_conf",
        "person_width",
        "person_height",
        "quality_valid",
        "head_conf",
        "helmet_conf",
        "vest_conf",
        "torso_person_occlusion_ratio",
        "torso_visible_proxy",
        "helmet_observation",
        "helmet_reason",
        "helmet_state",
        "helmet_valid_samples",
        "helmet_violation_ratio",
        "vest_observation",
        "vest_reason",
        "vest_state",
        "vest_valid_samples",
        "vest_absent_ratio",
    ]

    with states_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=state_fields,
        )

        writer.writeheader()
        writer.writerows(
            state_rows
        )

    event_fields = [
        "event_id",
        "track_id",
        "ppe_type",
        "event_type",
        "event",
        "status",
        "frame_index",
        "timestamp_s",
        "evidence_ratio",
        "valid_samples",
        "reason",
    ]

    with events_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=event_fields,
        )

        writer.writeheader()
        writer.writerows(
            events
        )

    with events_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            events,
            f,
            indent=2,
        )

    helmet_observations = (
        defaultdict(int)
    )

    vest_observations = (
        defaultdict(int)
    )

    vest_reasons = (
        defaultdict(int)
    )

    for row in state_rows:
        helmet_observations[
            row[
                "helmet_observation"
            ]
        ] += 1

        vest_observations[
            row[
                "vest_observation"
            ]
        ] += 1

        vest_reasons[
            row["vest_reason"]
        ] += 1

    open_events = [
        event
        for event in events
        if event["event"]
        == "OPEN"
    ]

    confirmed_open = [
        event
        for event in open_events
        if event["status"]
        == "CONFIRMED"
    ]

    suspected_open = [
        event
        for event in open_events
        if event["status"]
        == "SUSPECTED"
    ]

    result_summary = {
        "source": {
            "track_csv": (
                str(track_csv)
            ),
            "source_summary": (
                source_summary
            ),
        },
        "policy": {
            "helmet": (
                "Positive bare-head evidence "
                "can create CONFIRMED NO_HELMET."
            ),
            "vest": (
                "Absence of vest detection "
                "creates SUSPECTED_NO_VEST only. "
                "V2 does not label absence-only "
                "evidence as confirmed violation."
            ),
            "torso_visibility": (
                "Proxy uses person box quality "
                "and overlap with other person "
                "boxes. It does not detect "
                "occlusion by static objects."
            ),
        },
        "configuration": {
            "window_s": WINDOW_S,
            "track_lost_grace_s": (
                TRACK_LOST_GRACE_S
            ),
            "person_conf_min": (
                PERSON_CONF_MIN
            ),
            "person_height_min": (
                PERSON_HEIGHT_MIN
            ),
            "person_width_min": (
                PERSON_WIDTH_MIN
            ),
            "ppe_conf_min": (
                PPE_CONF_MIN
            ),
            "vest_recent_positive_grace_s": (
                VEST_RECENT_POSITIVE_GRACE_S
            ),
            "vest_suspect_ratio": (
                VEST_SUSPECT_RATIO
            ),
            "vest_min_valid_suspect": (
                VEST_MIN_VALID_SUSPECT
            ),
            "vest_min_span_s": (
                VEST_MIN_SPAN_S
            ),
            "torso_person_occlusion_max": (
                TORSO_PERSON_OCCLUSION_MAX
            ),
        },
        "rows": {
            "track_rows": len(rows),
            "state_rows": (
                len(state_rows)
            ),
        },
        "observations": {
            "helmet": dict(
                helmet_observations
            ),
            "vest": dict(
                vest_observations
            ),
            "vest_reasons": dict(
                vest_reasons
            ),
        },
        "events": {
            "open_event_count": (
                len(open_events)
            ),
            "confirmed_open_events": (
                len(
                    confirmed_open
                )
            ),
            "suspected_open_events": (
                len(
                    suspected_open
                )
            ),
            "confirmed_no_helmet": sum(
                1
                for event
                in confirmed_open
                if event[
                    "event_type"
                ]
                == "NO_HELMET"
            ),
            "suspected_no_vest": sum(
                1
                for event
                in suspected_open
                if event[
                    "event_type"
                ]
                == "SUSPECTED_NO_VEST"
            ),
        },
    }

    with output_summary.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result_summary,
            f,
            indent=2,
        )

    print("DONE")
    print("States:", states_csv)
    print("Events CSV:", events_csv)
    print("Events JSON:", events_json)
    print("Summary:", output_summary)
    print()
    print(
        "Confirmed open events:",
        len(
            confirmed_open
        ),
    )
    print(
        "Suspected open events:",
        len(
            suspected_open
        ),
    )


if __name__ == "__main__":
    main()
