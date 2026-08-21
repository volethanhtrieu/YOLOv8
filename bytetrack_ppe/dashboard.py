from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="PPE Safety Dashboard",
    page_icon="🦺",
    layout="wide",
)

API_DEFAULT = "http://127.0.0.1:5000"
TIMEOUT_S = 120


def request_json(
    method: str,
    api_url: str,
    path: str,
    **kwargs,
):
    response = requests.request(
        method,
        api_url.rstrip("/") + path,
        timeout=TIMEOUT_S,
        **kwargs,
    )

    if response.status_code not in {
        200,
        202,
    }:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"API {response.status_code}: {detail}"
        )

    return response.json()


def request_bytes(
    api_url: str,
    path: str,
    params: dict[str, Any] | None = None,
):
    response = requests.get(
        api_url.rstrip("/") + path,
        params=params,
        timeout=TIMEOUT_S,
    )

    if response.status_code != 200:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"API {response.status_code}: {detail}"
        )

    return response.content, dict(response.headers)


def safe_json(
    method: str,
    api_url: str,
    path: str,
    **kwargs,
):
    try:
        return request_json(
            method,
            api_url,
            path,
            **kwargs,
        )
    except (
        requests.RequestException,
        RuntimeError,
    ) as exc:
        st.error(str(exc))
        return None


def list_jobs(
    api_url: str,
):
    payload = safe_json(
        "GET",
        api_url,
        "/api/jobs",
        params={
            "limit": 30
        },
    )

    if not payload:
        return []

    return payload.get(
        "items",
        [],
    )


def active_job(
    jobs,
):
    for job in jobs:
        if job.get(
            "status"
        ) in {
            "QUEUED",
            "RUNNING",
            "CANCELLING",
        }:
            return job

    return None


def compact_event_table(
    items,
):
    return pd.DataFrame(
        [
            {
                "event_id": item.get(
                    "event_id"
                ),
                "status": item.get(
                    "status"
                ),
                "event_type": item.get(
                    "event_type"
                ),
                "track_id": item.get(
                    "track_id"
                ),
                "state": item.get(
                    "state"
                ),
                "start_s": item.get(
                    "start_s"
                ),
                "duration_s": item.get(
                    "duration_s"
                ),
                "human_decision": (
                    item.get(
                        "human_review_state",
                        "UNREVIEWED",
                    )
                ),
                "final_disposition": (
                    item.get(
                        "final_disposition",
                        "PENDING_REVIEW",
                    )
                ),
                "reviewer": item.get(
                    "human_reviewer"
                ),
                "close_reason": item.get(
                    "close_reason"
                ),
            }
            for item in items
        ]
    )


REVIEW_DECISIONS = [
    "NEEDS_REVIEW",
    "CONFIRMED_VIOLATION",
    "FALSE_ALARM",
]


def show_human_review(
    api_url: str,
    *,
    scope: str,
    event: dict[str, Any],
    key_prefix: str,
):
    event_id = event.get(
        "event_id"
    )

    if not event_id:
        return

    st.subheader(
        "Human review"
    )

    current_payload = safe_json(
        "GET",
        api_url,
        (
            f"/api/reviews/"
            f"{event_id}"
        ),
        params={
            "scope": scope
        },
    )

    current = (
        current_payload.get(
            "review"
        )
        if current_payload
        else None
    )

    if current:
        decision = current.get(
            "decision",
            "NEEDS_REVIEW",
        )

        if (
            decision
            == "CONFIRMED_VIOLATION"
        ):
            st.error(
                "Human decision: "
                "CONFIRMED VIOLATION"
            )
        elif (
            decision
            == "FALSE_ALARM"
        ):
            st.success(
                "Human decision: "
                "FALSE ALARM"
            )
        else:
            st.warning(
                "Human decision: "
                "NEEDS REVIEW"
            )

        st.caption(
            "Reviewed by: "
            f"{current.get('reviewer') or 'unspecified'}"
            " | Updated: "
            f"{current.get('updated_at_utc', '-')}"
        )

    else:
        st.info(
            "No human decision has been saved "
            "for this event."
        )

    default_decision = (
        current.get(
            "decision",
            "NEEDS_REVIEW",
        )
        if current
        else "NEEDS_REVIEW"
    )

    try:
        default_index = (
            REVIEW_DECISIONS.index(
                default_decision
            )
        )
    except ValueError:
        default_index = 0

    form_key = (
        f"{key_prefix}_review_"
        f"{scope}_{event_id}"
    ).replace(
        ":",
        "_",
    )

    with st.form(
        form_key
    ):
        decision = st.radio(
            "Decision",
            REVIEW_DECISIONS,
            index=default_index,
            horizontal=True,
        )

        reviewer = st.text_input(
            "Reviewer",
            value=(
                current.get(
                    "reviewer",
                    "",
                )
                if current
                else ""
            ),
            placeholder=(
                "Optional name or operator ID"
            ),
        )

        note = st.text_area(
            "Review note",
            value=(
                current.get(
                    "note",
                    "",
                )
                if current
                else ""
            ),
            placeholder=(
                "Why is this confirmed, "
                "a false alarm, or still uncertain?"
            ),
        )

        submitted = (
            st.form_submit_button(
                "Save review",
                type="primary",
            )
        )

    if submitted:
        saved = safe_json(
            "POST",
            api_url,
            "/api/reviews",
            json={
                "scope": scope,
                "event_id": event_id,
                "decision": decision,
                "reviewer": reviewer,
                "note": note,
            },
        )

        if saved is not None:
            st.success(
                "Review saved."
            )
            st.rerun()


def show_job_event_evidence(
    api_url: str,
    job_id: str,
    event_id: str,
):
    st.caption(
        "Evidence belongs to this isolated run. "
        "Nothing is published by viewing it."
    )

    try:
        clip, _headers = request_bytes(
            api_url,
            (
                f"/api/jobs/{job_id}"
                f"/preview/events/{event_id}/clip"
            ),
        )

        st.video(
            clip,
            format="video/mp4",
        )
    except (
        requests.RequestException,
        RuntimeError,
    ) as exc:
        st.warning(
            f"Event clip unavailable: {exc}"
        )

    phases = [
        ("pre", "Before"),
        ("open", "Open"),
        ("post", "After"),
    ]

    tabs = st.tabs(
        [
            label
            for _, label
            in phases
        ]
    )

    for tab, (
        phase,
        label,
    ) in zip(
        tabs,
        phases,
    ):
        with tab:
            try:
                crop, crop_headers = request_bytes(
                    api_url,
                    (
                        f"/api/jobs/{job_id}"
                        f"/preview/events/{event_id}"
                        "/evidence"
                    ),
                    params={
                        "phase": phase,
                        "view": "crop",
                    },
                )

                context, context_headers = request_bytes(
                    api_url,
                    (
                        f"/api/jobs/{job_id}"
                        f"/preview/events/{event_id}"
                        "/evidence"
                    ),
                    params={
                        "phase": phase,
                        "view": "context",
                    },
                )
            except (
                requests.RequestException,
                RuntimeError,
            ) as exc:
                st.warning(str(exc))
                continue

            left, right = st.columns(
                [1, 1.6]
            )

            with left:
                st.image(
                    crop,
                    caption=(
                        f"{label}, crop, frame "
                        f"{crop_headers.get('X-Evidence-Frame', '?')}"
                    ),
                    width="stretch",
                )

            with right:
                st.image(
                    context,
                    caption=(
                        f"{label}, context, frame "
                        f"{context_headers.get('X-Evidence-Frame', '?')}"
                    ),
                    width="stretch",
                )


def show_completed_job_preview(
    api_url: str,
    job: dict[str, Any],
):
    job_id = job["job_id"]

    st.subheader(
        "Run preview"
    )

    st.caption(
        "Review this isolated run before publishing."
    )

    try:
        video_bytes, _headers = request_bytes(
            api_url,
            (
                f"/api/jobs/{job_id}"
                "/preview/video"
            ),
        )

        st.video(
            video_bytes,
            format="video/mp4",
            width="stretch",
        )
    except (
        requests.RequestException,
        RuntimeError,
    ) as exc:
        st.warning(
            f"Processed video preview unavailable: {exc}"
        )

    event_payload = safe_json(
        "GET",
        api_url,
        (
            f"/api/jobs/{job_id}"
            "/preview/events"
        ),
    )

    if event_payload is None:
        return

    items = event_payload.get(
        "items",
        [],
    )

    st.subheader(
        "Run events"
    )

    if not items:
        st.info(
            "No events were opened in this run."
        )
        return

    st.dataframe(
        compact_event_table(
            items
        ),
        width="stretch",
        hide_index=True,
    )

    event_id = st.selectbox(
        "Inspect run event",
        [
            item["event_id"]
            for item in items
        ],
        key=(
            "run_event_"
            + job_id
        ),
    )

    event = safe_json(
        "GET",
        api_url,
        (
            f"/api/jobs/{job_id}"
            f"/preview/events/{event_id}"
        ),
    )

    if event is not None:
        st.json(event)

        show_job_event_evidence(
            api_url,
            job_id,
            event_id,
        )

        show_human_review(
            api_url,
            scope=(
                f"job:{job_id}"
            ),
            event=event,
            key_prefix=(
                "job_preview"
            ),
        )


st.title(
    "PPE Safety Monitoring"
)

with st.sidebar:
    api_url = st.text_input(
        "API URL",
        value=API_DEFAULT,
    )

    if st.button(
        "Refresh",
        width="stretch",
    ):
        st.rerun()


jobs_initial = list_jobs(
    api_url
)

running = active_job(
    jobs_initial
)

# Load shared statistics before rendering tabs.
stats = safe_json(
    "GET",
    api_url,
    "/api/stats",
)

if stats is None:
    stats = {}


process_tab, review_tab, events_tab, track_tab, system_tab = st.tabs(
    [
        "Process Video",
        "Review Queue",
        "Events",
        "Track Inspector",
        "System",
    ]
)


with process_tab:
    st.subheader(
        "Process a video"
    )

    if running is not None:
        st.info(
            "A video job is already active. "
            f"Job: {running['job_id']}. "
            f"Status: {running['status']}."
        )

    uploaded = st.file_uploader(
        "Video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
            "m4v",
        ],
        disabled=(
            running is not None
        ),
    )

    max_frames = st.number_input(
        "Maximum frames",
        min_value=0,
        value=60,
        step=1,
        disabled=(
            running is not None
        ),
        help=(
            "0 = full video."
        ),
    )

    if st.button(
        "Upload and Process",
        type="primary",
        disabled=(
            uploaded is None
            or running is not None
        ),
    ):
        result = safe_json(
            "POST",
            api_url,
            "/api/jobs",
            files={
                "video": (
                    uploaded.name,
                    uploaded,
                    (
                        uploaded.type
                        or "application/octet-stream"
                    ),
                )
            },
            data={
                "max_frames": str(
                    int(
                        max_frames
                    )
                )
            },
        )

        if result is not None:
            st.session_state[
                "job_inspector"
            ] = result[
                "job_id"
            ]

            st.rerun()

    st.divider()

    job_container = st.container()

    def render_jobs():
        jobs = list_jobs(
            api_url
        )

        if not jobs:
            job_container.info(
                "No jobs yet."
            )
            return

        with job_container:
            st.subheader(
                "Recent jobs"
            )

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "job_id": j.get(
                                "job_id"
                            ),
                            "video": j.get(
                                "original_name"
                            ),
                            "status": j.get(
                                "status"
                            ),
                            "frames": j.get(
                                "max_frames"
                            ),
                            "published": j.get(
                                "published"
                            ),
                            "created": j.get(
                                "created_at_utc"
                            ),
                        }
                        for j in jobs
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

            job_ids = [
                j["job_id"]
                for j in jobs
            ]

            active = active_job(
                jobs
            )

            current_selected = (
                st.session_state.get(
                    "job_inspector"
                )
            )

            if (
                active is not None
                and current_selected
                != active["job_id"]
            ):
                st.session_state[
                    "job_inspector"
                ] = active[
                    "job_id"
                ]

            elif (
                current_selected
                not in job_ids
            ):
                st.session_state[
                    "job_inspector"
                ] = job_ids[0]

            selected = st.selectbox(
                "Inspect job",
                job_ids,
                key="job_inspector",
            )

            job = safe_json(
                "GET",
                api_url,
                (
                    f"/api/jobs/"
                    f"{selected}"
                ),
            )

            if job is None:
                return

            progress = job.get(
                "progress",
                {},
            )

            a, b, c, d = (
                st.columns(4)
            )

            a.metric(
                "Status",
                job.get(
                    "status"
                ),
            )
            b.metric(
                "Progress",
                (
                    f"{progress.get('percent', 0):.1f}%"
                ),
            )
            c.metric(
                "Current frame",
                (
                    f"{progress.get('current_frame', 0)}"
                    + (
                        f"/{progress.get('target_frames')}"
                        if progress.get(
                            "target_frames"
                        )
                        is not None
                        else ""
                    )
                ),
            )
            d.metric(
                "Published",
                (
                    "YES"
                    if job.get(
                        "published"
                    )
                    else "NO"
                ),
            )

            percent = float(
                progress.get(
                    "percent",
                    0.0,
                )
                or 0.0
            )

            st.progress(
                min(
                    1.0,
                    max(
                        0.0,
                        percent
                        / 100.0,
                    ),
                ),
                text=(
                    f"Pipeline progress: "
                    f"{percent:.1f}%"
                ),
            )

            if job.get(
                "status"
            ) in {
                "QUEUED",
                "RUNNING",
                "CANCELLING",
            }:
                if st.button(
                    "Cancel job",
                    key=(
                        "cancel_"
                        + selected
                    ),
                    disabled=(
                        job.get(
                            "status"
                        )
                        == "CANCELLING"
                    ),
                ):
                    cancelled = safe_json(
                        "POST",
                        api_url,
                        (
                            f"/api/jobs/"
                            f"{selected}"
                            "/cancel"
                        ),
                    )

                    if cancelled:
                        st.rerun()

            result = job.get(
                "result_summary"
            )

            if result:
                st.subheader(
                    "Run result"
                )

                tracking = result.get(
                    "tracking",
                    {},
                )

                events = result.get(
                    "events",
                    {},
                )

                r1, r2, r3, r4 = (
                    st.columns(4)
                )

                r1.metric(
                    "Processed frames",
                    result.get(
                        "processed_frames",
                        0,
                    ),
                )
                r2.metric(
                    "Unique tracks",
                    tracking.get(
                        "unique_track_ids",
                        0,
                    ),
                )
                r3.metric(
                    "Confirmed events",
                    events.get(
                        "confirmed_open_events",
                        0,
                    ),
                )
                r4.metric(
                    "Suspected events",
                    events.get(
                        "suspected_open_events",
                        0,
                    ),
                )

            if (
                job.get("status")
                == "COMPLETED"
            ):
                show_completed_job_preview(
                    api_url,
                    job,
                )

            if (
                job.get("status")
                == "COMPLETED"
                and not job.get(
                    "published"
                )
            ):
                st.warning(
                    "Publishing changes the "
                    "dashboard dataset. The current "
                    "published dataset is backed up."
                )

                if st.button(
                    "Publish this run",
                    key=(
                        "publish_"
                        + selected
                    ),
                ):
                    published = safe_json(
                        "POST",
                        api_url,
                        (
                            f"/api/jobs/"
                            f"{selected}"
                            "/publish"
                        ),
                    )

                    if published:
                        st.rerun()

            logs = job.get(
                "log_tail",
                [],
            )

            if logs:
                with st.expander(
                    "Pipeline log"
                ):
                    st.code(
                        "\n".join(
                            logs
                        )
                    )

    if hasattr(
        st,
        "fragment",
    ):
        auto_render = st.fragment(
            render_jobs,
            run_every="2s",
        )
        auto_render()
    else:
        render_jobs()


with review_tab:
    st.subheader(
        "Human review queue"
    )

    st.caption(
        "Only UNREVIEWED and NEEDS_REVIEW events "
        "appear here. Reviewed false alarms and "
        "confirmed violations leave the queue."
    )

    queue_payload = safe_json(
        "GET",
        api_url,
        "/api/review-queue",
        params={
            "scope": "published",
            "limit": 500,
        },
    )

    queue_items = (
        queue_payload.get(
            "items",
            [],
        )
        if queue_payload
        else []
    )

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Pending queue",
        len(queue_items),
    )

    q2.metric(
        "Unreviewed",
        (
            stats.get(
                "human_review",
                {},
            )
            .get(
                "by_decision",
                {},
            )
            .get(
                "UNREVIEWED",
                0,
            )
        ),
    )

    q3.metric(
        "Needs review",
        (
            stats.get(
                "human_review",
                {},
            )
            .get(
                "by_decision",
                {},
            )
            .get(
                "NEEDS_REVIEW",
                0,
            )
        ),
    )

    if not queue_items:
        st.success(
            "Review queue is clear."
        )
    else:
        st.dataframe(
            compact_event_table(
                queue_items
            ),
            width="stretch",
            hide_index=True,
        )

        queue_event_id = (
            st.selectbox(
                "Review event",
                [
                    item[
                        "event_id"
                    ]
                    for item
                    in queue_items
                ],
                key=(
                    "review_queue_event"
                ),
            )
        )

        queue_event = safe_json(
            "GET",
            api_url,
            (
                f"/api/events/"
                f"{queue_event_id}"
            ),
        )

        if queue_event:
            st.json(
                queue_event
            )

            show_human_review(
                api_url,
                scope="published",
                event=queue_event,
                key_prefix=(
                    "queue"
                ),
            )


with events_tab:
    payload = safe_json(
        "GET",
        api_url,
        "/api/events",
        params={
            "limit": 500
        },
    )

    items = (
        payload.get(
            "items",
            [],
        )
        if payload
        else []
    )

    st.subheader(
        "Published event log"
    )

    review_counts = (
        stats.get(
            "human_review",
            {},
        )
        .get(
            "by_decision",
            {},
        )
    )

    review_a, review_b, review_c, review_d = (
        st.columns(4)
    )

    review_a.metric(
        "Human confirmed",
        review_counts.get(
            "CONFIRMED_VIOLATION",
            0,
        ),
    )

    review_b.metric(
        "False alarms",
        review_counts.get(
            "FALSE_ALARM",
            0,
        ),
    )

    review_c.metric(
        "Needs review",
        review_counts.get(
            "NEEDS_REVIEW",
            0,
        ),
    )

    review_d.metric(
        "Unreviewed",
        review_counts.get(
            "UNREVIEWED",
            0,
        ),
    )

    if items:
        st.dataframe(
            compact_event_table(
                items
            ),
            width="stretch",
            hide_index=True,
        )

        published_event_id = (
            st.selectbox(
                "Inspect published event",
                [
                    item["event_id"]
                    for item in items
                ],
                key=(
                    "published_event_inspector"
                ),
            )
        )

        published_event = safe_json(
            "GET",
            api_url,
            (
                f"/api/events/"
                f"{published_event_id}"
            ),
        )

        if published_event:
            st.json(
                published_event
            )

            show_human_review(
                api_url,
                scope="published",
                event=(
                    published_event
                ),
                key_prefix=(
                    "published"
                ),
            )
    else:
        st.info(
            "No published events."
        )


with track_tab:
    track_id = st.number_input(
        "Track ID",
        min_value=1,
        value=176,
        step=1,
    )

    track = safe_json(
        "GET",
        api_url,
        (
            f"/api/tracks/"
            f"{int(track_id)}"
        ),
    )

    if track is not None:
        st.json(track)


with system_tab:
    health = safe_json(
        "GET",
        api_url,
        "/api/health",
    )

    if health:
        st.json(health)

    if stats:
        st.json(stats)

    review_stats = safe_json(
        "GET",
        api_url,
        "/api/reviews",
    )

    if review_stats:
        st.subheader(
            "Human review statistics"
        )

        st.caption(
            "AI event status and human decision are "
            "stored separately. A FALSE_ALARM review "
            "does not rewrite the Event Engine event."
        )
        st.json(
            review_stats.get(
                "stats",
                {},
            )
        )
