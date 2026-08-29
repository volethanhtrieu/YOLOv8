from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import wandb

from src.variant_c.pipeline import VariantCBackend


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def det_to_json(det):
    if det is None:
        return None
    return {"bbox": list(det.bbox), "confidence": float(det.confidence)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.50, help="PPE confidence threshold")
    parser.add_argument("--project", default="YOLOv8-PPE-Association")
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (root / model_path).resolve()

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = (root / source_path).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Video not found: {source_path}")

    out_dir = root / "outputs" / "wandb_live"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = source_path.stem
    output_path = Path(args.output) if args.output else out_dir / f"{stem}_wandb.mp4"
    json_path = Path(args.json) if args.json else out_dir / f"{stem}_wandb.jsonl"

    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()
    if not json_path.is_absolute():
        json_path = (root / json_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    backend = VariantCBackend(
        model_path=str(model_path),
        device=int(args.device) if str(args.device).isdigit() else args.device,
        conf=args.conf,
    )

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0:
        source_fps = 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    run_name = args.name or f"variant-c-{stem}-ppeconf-{args.conf:.2f}"

    run = wandb.init(
        project=args.project,
        name=run_name,
        settings=wandb.Settings(
        x_stats_sampling_interval=5.0
        ),
        config={
            "variant": "C",
            "model": model_path.name,
            "source": source_path.name,
            "tracker": "ByteTrack",
            "association": True,
            "device": args.device,
            "confidence/person": float(backend.person_conf),
            "confidence/ppe": float(backend.ppe_conf),
            "source/fps": source_fps,
            "source/frames": source_frames,
            "source/width": width,
            "source/height": height,
            "logging/every_n_frames": args.log_every,
        },
    )

    print("========== W&B LIVE BENCHMARK ==========")
    print("Model:", model_path)
    print("Video:", source_path)
    print(f"PERSON CONF = {backend.person_conf:.2f}")
    print(f"PPE CONF    = {backend.ppe_conf:.2f}")
    print(f"Source FPS  = {source_fps:.2f}")
    print("W&B run:", run.name)
    print("========================================")

    frame_index = 0
    processing_elapsed = 0.0
    wall_start = time.perf_counter()
    frame_latencies_ms = []
    unique_track_ids = set()

    totals = defaultdict(int)
    conf_sums = defaultdict(float)
    conf_counts = defaultdict(int)
    track_stats = defaultdict(lambda: {
        "frames_seen": 0,
        "head_frames": 0,
        "helmet_frames": 0,
        "vest_frames": 0,
        "person_conf_sum": 0.0,
    })

    with json_path.open("w", encoding="utf-8") as jf:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            t0 = time.perf_counter()
            people = backend.process_frame(frame)

            people_in_frame = len(people)
            head_in_frame = 0
            helmet_in_frame = 0
            vest_in_frame = 0

            person_confs_frame = []
            head_confs_frame = []
            helmet_confs_frame = []
            vest_confs_frame = []
            frame_record = {"frame": frame_index, "people": []}

            for person in people:
                tid = int(person["track_id"])
                unique_track_ids.add(tid)

                has_head = bool(person["has_head"])
                has_helmet = bool(person["has_helmet"])
                has_vest = bool(person["has_vest"])

                head_in_frame += int(has_head)
                helmet_in_frame += int(has_helmet)
                vest_in_frame += int(has_vest)

                pconf = float(person["person_conf"])
                person_confs_frame.append(pconf)
                totals["person_detections"] += 1
                conf_sums["person"] += pconf
                conf_counts["person"] += 1

                ts = track_stats[tid]
                ts["frames_seen"] += 1
                ts["head_frames"] += int(has_head)
                ts["helmet_frames"] += int(has_helmet)
                ts["vest_frames"] += int(has_vest)
                ts["person_conf_sum"] += pconf

                if person["head"] is not None:
                    c = float(person["head"].confidence)
                    head_confs_frame.append(c)
                    totals["head_associations"] += 1
                    conf_sums["head"] += c
                    conf_counts["head"] += 1

                if person["helmet"] is not None:
                    c = float(person["helmet"].confidence)
                    helmet_confs_frame.append(c)
                    totals["helmet_associations"] += 1
                    conf_sums["helmet"] += c
                    conf_counts["helmet"] += 1

                if person["vest"] is not None:
                    c = float(person["vest"].confidence)
                    vest_confs_frame.append(c)
                    totals["vest_associations"] += 1
                    conf_sums["vest"] += c
                    conf_counts["vest"] += 1

                x1, y1, x2, y2 = map(int, person["person_bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                status = f"ID {tid} H:{int(has_helmet)} V:{int(has_vest)}"
                cv2.putText(
                    frame,
                    status,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2,
                )

                frame_record["people"].append({
                    "track_id": tid,
                    "person_bbox": list(person["person_bbox"]),
                    "person_conf": pconf,
                    "has_head": has_head,
                    "has_helmet": has_helmet,
                    "has_vest": has_vest,
                    "head": det_to_json(person["head"]),
                    "helmet": det_to_json(person["helmet"]),
                    "vest": det_to_json(person["vest"]),
                })

            cv2.putText(
                frame,
                f"person_conf={backend.person_conf:.2f}  ppe_conf={backend.ppe_conf:.2f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            writer.write(frame)
            jf.write(json.dumps(frame_record) + "\n")

            frame_processing_s = time.perf_counter() - t0
            processing_elapsed += frame_processing_s
            latency_ms = frame_processing_s * 1000.0
            frame_latencies_ms.append(latency_ms)

            instant_fps = 1.0 / frame_processing_s if frame_processing_s > 0 else 0.0
            average_fps = frame_index / processing_elapsed if processing_elapsed > 0 else 0.0

            no_helmet_in_frame = max(0, people_in_frame - helmet_in_frame)
            no_vest_in_frame = max(0, people_in_frame - vest_in_frame)

            if frame_index == 1 or frame_index % args.log_every == 0:
                wandb.log({
                    "runtime/average_fps": average_fps,
                    "runtime/instant_fps": instant_fps,
                    "runtime/latency_ms": latency_ms,
                    "tracking/people_in_frame": people_in_frame,
                    "tracking/unique_person_tracks_so_far": len(unique_track_ids),
                    "association/head_in_frame": head_in_frame,
                    "association/helmet_in_frame": helmet_in_frame,
                    "association/vest_in_frame": vest_in_frame,
                    "ppe/no_helmet_in_frame": no_helmet_in_frame,
                    "ppe/no_vest_in_frame": no_vest_in_frame,
                    "confidence/person_mean_in_frame": mean(person_confs_frame),
                    "confidence/head_mean_in_frame": mean(head_confs_frame),
                    "confidence/helmet_mean_in_frame": mean(helmet_confs_frame),
                    "confidence/vest_mean_in_frame": mean(vest_confs_frame),
                }, step=frame_index)

    cap.release()
    writer.release()

    wall_elapsed = time.perf_counter() - wall_start
    latency_mean_ms = mean(frame_latencies_ms)
    latency_p95_ms = percentile(frame_latencies_ms, 0.95)
    avg_fps = frame_index / processing_elapsed if processing_elapsed > 0 else 0.0

    track_table = wandb.Table(columns=[
        "track_id", "frames_seen", "head_frames", "helmet_frames", "vest_frames",
        "helmet_rate", "vest_rate", "avg_person_conf",
    ])

    for tid in sorted(track_stats):
        s = track_stats[tid]
        n = s["frames_seen"]
        track_table.add_data(
            tid,
            n,
            s["head_frames"],
            s["helmet_frames"],
            s["vest_frames"],
            s["helmet_frames"] / n if n else 0.0,
            s["vest_frames"] / n if n else 0.0,
            s["person_conf_sum"] / n if n else 0.0,
        )

    summary_values = {
        "performance/latency_mean_ms": latency_mean_ms,
        "performance/latency_p95_ms": latency_p95_ms,
        "performance/elapsed_s": wall_elapsed,
        "performance/processing_elapsed_s": processing_elapsed,
        "performance/average_fps": avg_fps,
        "summary/processed_frames": frame_index,
        "summary/unique_person_tracks": len(unique_track_ids),
        "summary/person_detections": totals["person_detections"],
        "summary/head_associations": totals["head_associations"],
        "summary/helmet_associations": totals["helmet_associations"],
        "summary/vest_associations": totals["vest_associations"],
        "summary/avg_person_conf": conf_sums["person"] / conf_counts["person"] if conf_counts["person"] else 0.0,
        "summary/avg_head_conf": conf_sums["head"] / conf_counts["head"] if conf_counts["head"] else 0.0,
        "summary/avg_helmet_conf": conf_sums["helmet"] / conf_counts["helmet"] if conf_counts["helmet"] else 0.0,
        "summary/avg_vest_conf": conf_sums["vest"] / conf_counts["vest"] if conf_counts["vest"] else 0.0,
        "confidence/person_threshold": float(backend.person_conf),
        "confidence/ppe_threshold": float(backend.ppe_conf),
    }

    for k, v in summary_values.items():
        run.summary[k] = v

    media_payload = {"tracking/table": track_table}
    try:
        media_payload["media/annotated_video"] = wandb.Video(str(output_path), format="mp4")
    except Exception as exc:
        print("Warning: could not attach annotated video to W&B:", exc)

    wandb.log(media_payload, step=frame_index + 1)

    print("\n========== FINAL SUMMARY ==========")
    print(f"Processed frames : {frame_index}")
    print(f"Average FPS      : {avg_fps:.2f}")
    print(f"Mean latency     : {latency_mean_ms:.2f} ms")
    print(f"P95 latency      : {latency_p95_ms:.2f} ms")
    print(f"Elapsed          : {wall_elapsed:.2f} s")
    print(f"Unique tracks    : {len(unique_track_ids)}")
    print(f"PERSON CONF      : {backend.person_conf:.2f}")
    print(f"PPE CONF         : {backend.ppe_conf:.2f}")
    print("Output video     :", output_path)
    print("Output JSONL     :", json_path)
    print("===================================")

    run.finish()


if __name__ == "__main__":
    main()
