from __future__ import annotations

import argparse
import csv
import io
import logging

from flask import Flask, Response, jsonify, request

from backend.config import load_config
from backend.pipeline import PPEPipeline
from backend.worker import VideoWorker


def create_app(config_path: str = "config.yaml", profile: str | None = None) -> Flask:
    config = load_config(config_path, profile=profile)
    pipeline = PPEPipeline(config)
    worker = VideoWorker(pipeline, jpeg_quality=config.api.jpeg_quality)
    app = Flask(__name__)
    app.config["PPE_CONFIG"] = config
    app.extensions["ppe_pipeline"] = pipeline
    app.extensions["video_worker"] = worker

    @app.get("/health")
    def health():
        status = worker.status()
        code = 200 if not status["error"] else 503
        return jsonify({"ok": code == 200, "worker": status}), code

    @app.get("/api/config")
    def get_config():
        value = config.to_dict()
        value["model"]["path"] = str(config.resolve_path(config.model.path))
        value["storage"]["database"] = str(
            config.resolve_path(config.storage.database)
        )
        return jsonify(value)

    @app.post("/api/start")
    def start():
        payload = request.get_json(silent=True) or {}
        source = payload.get("source", 0)
        camera_id = str(payload.get("camera_id", "camera-01"))
        try:
            worker.start(source, camera_id)
        except (RuntimeError, FileNotFoundError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        return jsonify({"ok": True, "worker": worker.status()}), 202

    @app.post("/api/stop")
    def stop():
        worker.stop()
        return jsonify({"ok": True, "worker": worker.status()})

    @app.get("/video_feed")
    def video_feed():
        return Response(
            worker.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/stats")
    def stats():
        return jsonify({"worker": worker.status(), "pipeline": pipeline.stats()})

    @app.get("/api/events")
    def events():
        try:
            limit = int(request.args.get("limit", "100"))
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400
        status = request.args.get("status")
        if status not in {None, "active", "resolved"}:
            return jsonify({"error": "status must be active or resolved"}), 400
        rows = pipeline.repository.list_events(
            limit=limit,
            camera_id=request.args.get("camera_id"),
            status=status,
        )
        return jsonify({"items": rows, "count": len(rows)})

    @app.get("/api/events.csv")
    def export_events_csv():
        rows = pipeline.repository.list_events(limit=1000)
        columns = [
            "id",
            "event_key",
            "camera_id",
            "track_id",
            "violation_type",
            "status",
            "started_at",
            "last_seen_at",
            "ended_at",
            "confidence",
            "evidence_path",
            "end_reason",
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=ppe_events.csv"},
        )

    @app.get("/api/evidence/<path:filename>")
    def evidence(filename: str):
        from flask import send_from_directory

        return send_from_directory(pipeline.evidence_dir, filename)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPE ablation Flask backend")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    application = create_app(args.config, profile=args.profile)
    cfg = application.config["PPE_CONFIG"]
    application.run(host=cfg.api.host, port=cfg.api.port, threaded=True)
