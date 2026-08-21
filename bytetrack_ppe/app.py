from __future__ import annotations

from pathlib import Path
import uuid

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_file,
)
from werkzeug.utils import secure_filename

from clip_service import ClipService
from event_store import EventStore
from evidence_service import EvidenceService
from job_manager import (
    ALLOWED_VIDEO_SUFFIXES,
    UPLOAD_ROOT,
    VideoJobManager,
)
from job_preview import JobPreviewService
from human_review_store import (
    ALLOWED_DECISIONS,
    HumanReviewStore,
)


MAX_UPLOAD_BYTES = (
    5
    * 1024
    * 1024
    * 1024
)


def create_app() -> Flask:
    app = Flask(__name__)

    app.config[
        "MAX_CONTENT_LENGTH"
    ] = MAX_UPLOAD_BYTES

    store = EventStore()
    evidence = EvidenceService(
        store
    )
    clips = ClipService(
        store
    )
    jobs = VideoJobManager()
    preview = JobPreviewService(
        jobs
    )
    reviews = HumanReviewStore()

    def final_disposition(
        event: dict,
    ) -> str:
        review_state = str(
            event.get(
                "human_review_state",
                "UNREVIEWED",
            )
        ).upper()

        if review_state == "CONFIRMED_VIOLATION":
            return "CONFIRMED_VIOLATION"

        if review_state == "FALSE_ALARM":
            return "FALSE_ALARM"

        if review_state == "NEEDS_REVIEW":
            return "NEEDS_REVIEW"

        return "PENDING_REVIEW"

    def review_priority(
        event: dict,
    ) -> int:
        # Lower number means higher review priority.
        ai_status = str(
            event.get(
                "status",
                "",
            )
        ).upper()

        event_type = str(
            event.get(
                "event_type",
                "",
            )
        ).upper()

        if ai_status == "CONFIRMED":
            return 1

        if event_type == "NO_HELMET":
            return 2

        return 3

    def review_overlay(
        scope: str,
        event: dict,
    ) -> dict:
        enriched = dict(event)

        event_id = str(
            enriched.get(
                "event_id",
                "",
            )
        )

        review = (
            reviews.get(
                scope,
                event_id,
            )
            if event_id
            else None
        )

        enriched[
            "human_decision"
        ] = (
            review.get(
                "decision"
            )
            if review
            else None
        )

        enriched[
            "human_reviewer"
        ] = (
            review.get(
                "reviewer"
            )
            if review
            else None
        )

        enriched[
            "human_review_note"
        ] = (
            review.get(
                "note"
            )
            if review
            else None
        )

        enriched[
            "human_review_updated_at"
        ] = (
            review.get(
                "updated_at_utc"
            )
            if review
            else None
        )

        enriched[
            "human_review_state"
        ] = (
            review.get(
                "decision"
            )
            if review
            else "UNREVIEWED"
        )

        enriched[
            "final_disposition"
        ] = final_disposition(
            enriched
        )

        enriched[
            "review_priority"
        ] = review_priority(
            enriched
        )

        return enriched

    def overlay_event_page(
        scope: str,
        payload: dict,
    ) -> dict:
        enriched = dict(payload)

        items = payload.get(
            "items",
            [],
        )

        enriched["items"] = [
            review_overlay(
                scope,
                item,
            )
            for item in items
        ]

        return enriched

    def published_review_stats() -> dict:
        event_page = store.list_events(
            limit=500,
            offset=0,
        )

        items = event_page.get(
            "items",
            [],
        )

        counts = {
            "CONFIRMED_VIOLATION": 0,
            "FALSE_ALARM": 0,
            "NEEDS_REVIEW": 0,
            "UNREVIEWED": 0,
        }

        for event in items:
            enriched = review_overlay(
                "published",
                event,
            )

            state = enriched.get(
                "human_review_state",
                "UNREVIEWED",
            )

            if state not in counts:
                state = "UNREVIEWED"

            counts[state] += 1

        final_counts = {
            "CONFIRMED_VIOLATION": 0,
            "FALSE_ALARM": 0,
            "NEEDS_REVIEW": 0,
            "PENDING_REVIEW": 0,
        }

        for event in items:
            enriched = review_overlay(
                "published",
                event,
            )

            disposition = enriched.get(
                "final_disposition",
                "PENDING_REVIEW",
            )

            if disposition not in final_counts:
                disposition = "PENDING_REVIEW"

            final_counts[
                disposition
            ] += 1

        return {
            "published_event_count": len(
                items
            ),
            "by_decision": counts,
            "by_final_disposition": (
                final_counts
            ),
            "pending_review": (
                counts["UNREVIEWED"]
                + counts["NEEDS_REVIEW"]
            ),
        }

    @app.get("/api/health")
    def health():
        payload = store.health()
        payload["evidence"] = (
            evidence.status()
        )
        payload["clips"] = (
            clips.status()
        )

        return jsonify(payload), (
            200
            if payload["status"]
            == "ok"
            else 503
        )

    @app.get("/api/events")
    def list_events():
        track_id_raw = request.args.get(
            "track_id"
        )

        try:
            track_id = (
                int(track_id_raw)
                if track_id_raw
                is not None
                else None
            )
            limit = int(
                request.args.get(
                    "limit",
                    100,
                )
            )
            offset = int(
                request.args.get(
                    "offset",
                    0,
                )
            )
        except ValueError:
            return jsonify(
                {
                    "error": (
                        "track_id, limit and "
                        "offset must be integers"
                    )
                }
            ), 400

        page = store.list_events(
            status=request.args.get(
                "status"
            ),
            ppe_type=request.args.get(
                "ppe_type"
            ),
            event_type=request.args.get(
                "event_type"
            ),
            state=request.args.get(
                "state"
            ),
            track_id=track_id,
            limit=limit,
            offset=offset,
        )

        return jsonify(
            overlay_event_page(
                "published",
                page,
            )
        )

    @app.get(
        "/api/events/<event_id>"
    )
    def get_event(event_id: str):
        event = store.get_event(
            event_id
        )

        if event is None:
            return jsonify(
                {
                    "error": (
                        "event_not_found"
                    )
                }
            ), 404

        return jsonify(
            review_overlay(
                "published",
                event,
            )
        )

    @app.get(
        "/api/events/<event_id>/evidence"
    )
    def get_event_evidence(
        event_id: str,
    ):
        try:
            image = evidence.get_image(
                event_id,
                phase=request.args.get(
                    "phase",
                    "open",
                ),
                view=request.args.get(
                    "view",
                    "crop",
                ),
            )
        except Exception as exc:
            return jsonify(
                {
                    "error": (
                        "evidence_unavailable"
                    ),
                    "detail": str(exc),
                }
            ), 500

        response = Response(
            image.jpeg_bytes,
            mimetype="image/jpeg",
        )
        response.headers[
            "X-Evidence-Frame"
        ] = str(
            image.source_frame
        )

        return response

    @app.get(
        "/api/events/<event_id>/clip"
    )
    def get_event_clip(
        event_id: str,
    ):
        try:
            clip = clips.get_clip(
                event_id
            )
        except Exception as exc:
            return jsonify(
                {
                    "error": (
                        "clip_unavailable"
                    ),
                    "detail": str(exc),
                }
            ), 500

        return send_file(
            clip.path,
            mimetype="video/mp4",
            conditional=True,
            max_age=0,
        )

    @app.get(
        "/api/tracks/<int:track_id>"
    )
    def get_track(track_id: int):
        track = store.get_track(
            track_id
        )

        if track is None:
            return jsonify(
                {
                    "error": (
                        "track_not_found"
                    )
                }
            ), 404

        return jsonify(track)

    @app.get("/api/stats")
    def stats():
        payload = store.stats()

        payload[
            "human_review"
        ] = published_review_stats()

        payload[
            "human_review_store"
        ] = reviews.stats()

        return jsonify(
            payload
        )

    @app.get("/api/jobs")
    def list_jobs():
        try:
            limit = int(
                request.args.get(
                    "limit",
                    50,
                )
            )
        except ValueError:
            return jsonify(
                {
                    "error": (
                        "limit must be integer"
                    )
                }
            ), 400

        return jsonify(
            {
                "items": (
                    jobs.list_jobs(
                        limit=limit
                    )
                )
            }
        )

    @app.get(
        "/api/jobs/<job_id>"
    )
    def get_job(job_id: str):
        job = jobs.get_job(
            job_id
        )

        if job is None:
            return jsonify(
                {
                    "error": (
                        "job_not_found"
                    )
                }
            ), 404

        return jsonify(job)

    @app.post("/api/jobs")
    def create_job():
        if "video" not in request.files:
            return jsonify(
                {
                    "error": (
                        "video_file_required"
                    )
                }
            ), 400

        upload = request.files[
            "video"
        ]

        filename = secure_filename(
            upload.filename or ""
        )

        if filename == "":
            return jsonify(
                {
                    "error": (
                        "invalid_filename"
                    )
                }
            ), 400

        suffix = Path(
            filename
        ).suffix.lower()

        if (
            suffix
            not in ALLOWED_VIDEO_SUFFIXES
        ):
            return jsonify(
                {
                    "error": (
                        "unsupported_video_type"
                    )
                }
            ), 400

        try:
            max_frames = int(
                request.form.get(
                    "max_frames",
                    "0",
                )
            )
        except ValueError:
            return jsonify(
                {
                    "error": (
                        "max_frames_must_be_integer"
                    )
                }
            ), 400

        if max_frames < 0:
            return jsonify(
                {
                    "error": (
                        "max_frames_must_be_nonnegative"
                    )
                }
            ), 400

        stored_name = (
            uuid.uuid4().hex[:10]
            + "_"
            + filename
        )

        path = (
            UPLOAD_ROOT
            / stored_name
        )

        upload.save(path)

        try:
            job = jobs.create_job(
                video_path=path,
                original_name=filename,
                max_frames=max_frames,
            )
        except Exception as exc:
            try:
                path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

            return jsonify(
                {
                    "error": (
                        "job_creation_failed"
                    ),
                    "detail": str(exc),
                }
            ), 409

        return jsonify(job), 202

    @app.post(
        "/api/jobs/<job_id>/cancel"
    )
    def cancel_job(job_id: str):
        try:
            job = jobs.cancel_job(
                job_id
            )
        except LookupError:
            return jsonify(
                {
                    "error": (
                        "job_not_found"
                    )
                }
            ), 404
        except RuntimeError as exc:
            return jsonify(
                {
                    "error": (
                        "cancel_failed"
                    ),
                    "detail": str(exc),
                }
            ), 409

        return jsonify(job)

    @app.post(
        "/api/jobs/<job_id>/publish"
    )
    def publish_job(job_id: str):
        try:
            job = jobs.publish_job(
                job_id
            )
        except LookupError:
            return jsonify(
                {
                    "error": (
                        "job_not_found"
                    )
                }
            ), 404
        except Exception as exc:
            return jsonify(
                {
                    "error": (
                        "publish_failed"
                    ),
                    "detail": str(exc),
                }
            ), 409

        return jsonify(job)

    # -----------------------------------------------------
    # Isolated run preview. These endpoints never publish.
    # -----------------------------------------------------

    @app.get(
        "/api/jobs/<job_id>/preview/video"
    )
    def preview_video(job_id: str):
        try:
            video = (
                preview.processed_video(
                    job_id
                )
            )
        except LookupError:
            return jsonify(
                {
                    "error": (
                        "job_not_found"
                    )
                }
            ), 404
        except RuntimeError as exc:
            return jsonify(
                {
                    "error": (
                        "preview_unavailable"
                    ),
                    "detail": str(exc),
                }
            ), 409
        except FileNotFoundError as exc:
            return jsonify(
                {
                    "error": (
                        "preview_video_missing"
                    ),
                    "detail": str(exc),
                }
            ), 404

        return send_file(
            video,
            mimetype="video/mp4",
            conditional=True,
            max_age=0,
        )

    @app.get(
        "/api/jobs/<job_id>/preview/events"
    )
    def preview_events(job_id: str):
        try:
            page = preview.list_events(
                job_id
            )

            return jsonify(
                overlay_event_page(
                    f"job:{job_id}",
                    page,
                )
            )
        except LookupError:
            return jsonify(
                {
                    "error": (
                        "job_not_found"
                    )
                }
            ), 404
        except RuntimeError as exc:
            return jsonify(
                {
                    "error": (
                        "preview_unavailable"
                    ),
                    "detail": str(exc),
                }
            ), 409

    @app.get(
        "/api/jobs/<job_id>/preview/events/<event_id>"
    )
    def preview_event(
        job_id: str,
        event_id: str,
    ):
        try:
            event = preview.get_event(
                job_id,
                event_id,
            )
        except (
            LookupError,
            RuntimeError,
        ) as exc:
            return jsonify(
                {
                    "error": str(exc)
                }
            ), 409

        if event is None:
            return jsonify(
                {
                    "error": (
                        "event_not_found"
                    )
                }
            ), 404

        return jsonify(
            review_overlay(
                f"job:{job_id}",
                event,
            )
        )

    @app.get(
        "/api/jobs/<job_id>/preview/events/<event_id>/evidence"
    )
    def preview_event_evidence(
        job_id: str,
        event_id: str,
    ):
        try:
            image = (
                preview.evidence_image(
                    job_id,
                    event_id,
                    phase=(
                        request.args.get(
                            "phase",
                            "open",
                        )
                    ),
                    view=(
                        request.args.get(
                            "view",
                            "crop",
                        )
                    ),
                )
            )
        except Exception as exc:
            return jsonify(
                {
                    "error": (
                        "preview_evidence_unavailable"
                    ),
                    "detail": str(exc),
                }
            ), 409

        response = Response(
            image.jpeg_bytes,
            mimetype="image/jpeg",
        )
        response.headers[
            "X-Evidence-Frame"
        ] = str(
            image.source_frame
        )

        return response

    @app.get(
        "/api/jobs/<job_id>/preview/events/<event_id>/clip"
    )
    def preview_event_clip(
        job_id: str,
        event_id: str,
    ):
        try:
            clip = preview.event_clip(
                job_id,
                event_id,
            )
        except Exception as exc:
            return jsonify(
                {
                    "error": (
                        "preview_clip_unavailable"
                    ),
                    "detail": str(exc),
                }
            ), 409

        return send_file(
            clip.path,
            mimetype="video/mp4",
            conditional=True,
            max_age=0,
        )

    @app.get(
        "/api/jobs/<job_id>/preview/tracks/<int:track_id>"
    )
    def preview_track(
        job_id: str,
        track_id: int,
    ):
        try:
            track = preview.get_track(
                job_id,
                track_id,
            )
        except (
            LookupError,
            RuntimeError,
        ) as exc:
            return jsonify(
                {
                    "error": str(exc)
                }
            ), 409

        if track is None:
            return jsonify(
                {
                    "error": (
                        "track_not_found"
                    )
                }
            ), 404

        return jsonify(track)


    @app.get(
        "/api/review-queue"
    )
    def review_queue():
        scope = request.args.get(
            "scope",
            "published",
        )

        try:
            limit = int(
                request.args.get(
                    "limit",
                    100,
                )
            )
        except ValueError:
            return jsonify(
                {
                    "error": (
                        "limit_must_be_integer"
                    )
                }
            ), 400

        limit = max(
            1,
            min(
                limit,
                500,
            ),
        )

        if scope == "published":
            page = store.list_events(
                limit=500,
                offset=0,
            )
        elif scope.startswith(
            "job:"
        ):
            job_id = scope[
                len("job:"):
            ].strip()

            if job_id == "":
                return jsonify(
                    {
                        "error": (
                            "invalid_job_scope"
                        )
                    }
                ), 400

            try:
                page = preview.list_events(
                    job_id
                )
            except LookupError:
                return jsonify(
                    {
                        "error": (
                            "job_not_found"
                        )
                    }
                ), 404
            except RuntimeError as exc:
                return jsonify(
                    {
                        "error": (
                            "job_preview_unavailable"
                        ),
                        "detail": str(exc),
                    }
                ), 409
        else:
            return jsonify(
                {
                    "error": (
                        "invalid_scope"
                    )
                }
            ), 400

        items = [
            review_overlay(
                scope,
                event,
            )
            for event
            in page.get(
                "items",
                [],
            )
        ]

        queue = [
            event
            for event in items
            if event.get(
                "human_review_state"
            )
            in {
                "UNREVIEWED",
                "NEEDS_REVIEW",
            }
        ]

        queue.sort(
            key=lambda event: (
                int(
                    event.get(
                        "review_priority",
                        99,
                    )
                ),
                float(
                    event.get(
                        "start_s",
                        0.0,
                    )
                    or 0.0
                ),
            )
        )

        return jsonify(
            {
                "scope": scope,
                "total": len(
                    queue
                ),
                "items": queue[
                    :limit
                ],
            }
        )

    # -----------------------------------------------------
    # Human review overlay.
    #
    # Scope values:
    #   published
    #   job:<job_id>
    #
    # Reviews are stored separately from Event Engine output.
    # -----------------------------------------------------

    def resolve_review_event(
        scope: str,
        event_id: str,
    ):
        if scope == "published":
            return store.get_event(
                event_id
            )

        if scope.startswith(
            "job:"
        ):
            job_id = scope[
                len("job:"):
            ].strip()

            if job_id == "":
                return None

            try:
                return preview.get_event(
                    job_id,
                    event_id,
                )
            except (
                LookupError,
                RuntimeError,
            ):
                return None

        return None

    @app.get(
        "/api/reviews"
    )
    def list_reviews():
        scope = request.args.get(
            "scope"
        )

        return jsonify(
            {
                "items": (
                    reviews.list(
                        scope=scope
                    )
                ),
                "stats": (
                    reviews.stats()
                ),
            }
        )

    @app.get(
        "/api/reviews/<event_id>"
    )
    def get_review(
        event_id: str,
    ):
        scope = request.args.get(
            "scope",
            "published",
        )

        review = reviews.get(
            scope,
            event_id,
        )

        if review is None:
            return jsonify(
                {
                    "review": None,
                    "scope": scope,
                    "event_id": event_id,
                }
            )

        return jsonify(
            {
                "review": review
            }
        )

    @app.post(
        "/api/reviews"
    )
    def save_review():
        payload = (
            request.get_json(
                silent=True
            )
            or {}
        )

        scope = str(
            payload.get(
                "scope",
                "published",
            )
        ).strip()

        event_id = str(
            payload.get(
                "event_id",
                "",
            )
        ).strip()

        decision = str(
            payload.get(
                "decision",
                "",
            )
        ).strip().upper()

        reviewer = str(
            payload.get(
                "reviewer",
                "",
            )
        )

        note = str(
            payload.get(
                "note",
                "",
            )
        )

        if event_id == "":
            return jsonify(
                {
                    "error": (
                        "event_id_required"
                    )
                }
            ), 400

        if (
            decision
            not in ALLOWED_DECISIONS
        ):
            return jsonify(
                {
                    "error": (
                        "invalid_decision"
                    ),
                    "allowed": sorted(
                        ALLOWED_DECISIONS
                    ),
                }
            ), 400

        event = resolve_review_event(
            scope,
            event_id,
        )

        if event is None:
            return jsonify(
                {
                    "error": (
                        "event_not_found_in_scope"
                    ),
                    "scope": scope,
                    "event_id": event_id,
                }
            ), 404

        try:
            review = reviews.upsert(
                scope=scope,
                event_id=event_id,
                decision=decision,
                reviewer=reviewer,
                note=note,
                event_type=(
                    event.get(
                        "event_type"
                    )
                ),
                track_id=(
                    event.get(
                        "track_id"
                    )
                ),
            )
        except ValueError as exc:
            return jsonify(
                {
                    "error": (
                        "invalid_review"
                    ),
                    "detail": str(exc),
                }
            ), 400

        return jsonify(
            {
                "review": review
            }
        )

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify(
            {
                "error": (
                    "upload_too_large"
                )
            }
        ), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )
