from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(item, region):
    a = area(item)
    return intersection(item, region) / a if a > 0 else 0.0


def center_inside(item, region):
    cx = (item[0] + item[2]) / 2
    cy = (item[1] + item[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def yolo_to_xyxy(x, y, w, h):
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def read_people(label_path: Path):
    people = []

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 5:
            continue

        cid = int(parts[0])
        if cid != 0:
            continue

        x, y, w, h = map(float, parts[1:])
        people.append({
            "box": yolo_to_xyxy(x, y, w, h)
        })

    return people


def get_vest_id(names):
    if isinstance(names, dict):
        mapping = {str(v).strip().lower(): int(k) for k, v in names.items()}
    else:
        mapping = {str(v).strip().lower(): i for i, v in enumerate(names)}

    if "vest" not in mapping:
        raise RuntimeError(f"No vest class in model: {names}")

    return mapping["vest"]


def predict_vests(model, image_path, vest_id, conf, device):
    result = model.predict(
        source=str(image_path),
        classes=[vest_id],
        conf=conf,
        device=device,
        verbose=False,
    )[0]

    out = []

    if result.boxes is None or len(result.boxes) == 0:
        return out

    boxes = result.boxes.xyxyn.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()

    for box, score in zip(boxes, confs):
        out.append({
            "box": tuple(map(float, box)),
            "conf": float(score),
        })

    return out


def associate_to_people(cands, people, min_containment, min_area_ratio, max_area_ratio):
    valid = []

    for c in cands:
        best = None

        for person_idx, person in enumerate(people):
            pbox = person["box"]

            if not center_inside(c["box"], pbox):
                continue

            cont = containment(c["box"], pbox)
            if cont < min_containment:
                continue

            pa = area(pbox)
            va = area(c["box"])
            ratio = va / pa if pa > 0 else 0.0

            if not (min_area_ratio <= ratio <= max_area_ratio):
                continue

            score = cont

            if best is None or score > best["score"]:
                best = {
                    "person_idx": person_idx,
                    "containment": cont,
                    "area_ratio": ratio,
                    "score": score,
                }

        if best is not None:
            valid.append({**c, **best})

    return valid


def consensus_matches(coco, scratch, min_iou):
    candidates = []

    for i, a in enumerate(coco):
        for j, b in enumerate(scratch):
            ov = iou(a["box"], b["box"])
            if ov < min_iou:
                continue

            score = ov * min(a["conf"], b["conf"])
            candidates.append((score, ov, i, j))

    candidates.sort(reverse=True)

    used_coco = set()
    used_scratch = set()
    matches = []

    for score, ov, i, j in candidates:
        if i in used_coco or j in used_scratch:
            continue

        used_coco.add(i)
        used_scratch.add(j)

        a = coco[i]
        b = scratch[j]

        # Average the two boxes for the consensus box.
        box = tuple((x + y) / 2.0 for x, y in zip(a["box"], b["box"]))

        matches.append({
            "box": box,
            "coco_conf": a["conf"],
            "scratch_conf": b["conf"],
            "iou": ov,
            "min_conf": min(a["conf"], b["conf"]),
            "mean_conf": (a["conf"] + b["conf"]) / 2.0,
        })

    single_coco = [coco[i] for i in range(len(coco)) if i not in used_coco]
    single_scratch = [scratch[j] for j in range(len(scratch)) if j not in used_scratch]

    return matches, single_coco, single_scratch


def choose_one_per_person(cands):
    cands = sorted(
        cands,
        key=lambda x: (
            x.get("status") == "accepted",
            x.get("min_conf", 0.0),
            x.get("iou", 0.0),
        ),
        reverse=True,
    )

    used = set()
    out = []

    for c in cands:
        pid = c["person_idx"]
        if pid in used:
            continue

        used.add(pid)
        out.append(c)

    return out


def draw_preview(image_path, people, cands, out_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    h, w = image.shape[:2]

    def denorm(box):
        x1, y1, x2, y2 = box
        return (
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        )

    for p in people:
        x1, y1, x2, y2 = denorm(p["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    for c in cands:
        x1, y1, x2, y2 = denorm(c["box"])

        color = (0, 220, 0) if c["status"] == "accepted" else (0, 165, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = (
            f"{c['status']} "
            f"C:{c['coco_conf']:.2f} "
            f"S:{c['scratch_conf']:.2f} "
            f"I:{c['iou']:.2f}"
        )

        cv2.putText(
            image,
            text,
            (max(0, x1), max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def iter_original_images(root, split):
    image_dir = root / "images" / split

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if split == "train":
        images = [p for p in images if not p.stem.startswith("aug_")]

    return sorted(images)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-coco", required=True, type=Path)
    parser.add_argument("--model-scratch", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)

    parser.add_argument("--device", default="0")
    parser.add_argument("--infer-conf", type=float, default=0.25)

    parser.add_argument("--consensus-iou", type=float, default=0.50)
    parser.add_argument("--accept-min-conf", type=float, default=0.45)

    parser.add_argument("--min-containment", type=float, default=0.65)
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-area-ratio", type=float, default=0.85)

    parser.add_argument("--preview-limit", type=int, default=120)

    args = parser.parse_args()

    dataset = args.dataset.resolve()
    report_dir = args.report_dir.resolve()

    model_coco = YOLO(str(args.model_coco.resolve()))
    model_scratch = YOLO(str(args.model_scratch.resolve()))

    vest_coco = get_vest_id(model_coco.names)
    vest_scratch = get_vest_id(model_scratch.names)

    print("COCO classes:", model_coco.names)
    print("Scratch classes:", model_scratch.names)
    print("Vest IDs:", vest_coco, vest_scratch)

    report_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    all_records = []
    preview_counts = Counter()

    for split in ("train", "val", "test"):
        images = iter_original_images(dataset, split)
        label_dir = dataset / "labels" / split

        print(f"\n[{split}] images={len(images)}")

        for idx, image_path in enumerate(images, 1):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)

            people = read_people(label_path)

            coco = predict_vests(
                model_coco, image_path, vest_coco, args.infer_conf, args.device
            )
            scratch = predict_vests(
                model_scratch, image_path, vest_scratch, args.infer_conf, args.device
            )

            matches, single_coco, single_scratch = consensus_matches(
                coco, scratch, args.consensus_iou
            )

            matches = associate_to_people(
                matches,
                people,
                args.min_containment,
                args.min_area_ratio,
                args.max_area_ratio,
            )

            annotated = []

            for c in matches:
                status = (
                    "accepted"
                    if c["min_conf"] >= args.accept_min_conf
                    else "borderline"
                )

                annotated.append({
                    **c,
                    "status": status,
                })

            annotated = choose_one_per_person(annotated)

            for c in annotated:
                counts[c["status"]] += 1

                all_records.append({
                    "split": split,
                    "image": image_path.name,
                    "stem": image_path.stem,
                    "status": c["status"],
                    "coco_conf": c["coco_conf"],
                    "scratch_conf": c["scratch_conf"],
                    "min_conf": c["min_conf"],
                    "mean_conf": c["mean_conf"],
                    "iou": c["iou"],
                    "person_idx": c["person_idx"],
                    "containment": c["containment"],
                    "area_ratio": c["area_ratio"],
                    "box_xyxy_norm": list(c["box"]),
                })

            counts["single_coco"] += len(single_coco)
            counts["single_scratch"] += len(single_scratch)

            for status in ("accepted", "borderline"):
                if (
                    any(c["status"] == status for c in annotated)
                    and preview_counts[status] < args.preview_limit
                ):
                    draw_preview(
                        image_path,
                        people,
                        [c for c in annotated if c["status"] == status],
                        report_dir / "preview" / status / image_path.name,
                    )
                    preview_counts[status] += 1

            if idx % 250 == 0 or idx == len(images):
                print(f"  {idx}/{len(images)}")

    with (report_dir / "dual_vest_candidates.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "dataset": str(dataset),
        "model_coco": str(args.model_coco.resolve()),
        "model_scratch": str(args.model_scratch.resolve()),
        "thresholds": {
            "infer_conf": args.infer_conf,
            "consensus_iou": args.consensus_iou,
            "accept_min_conf": args.accept_min_conf,
            "min_containment": args.min_containment,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
        },
        "counts": dict(counts),
        "note": (
            "accepted = both models agree geometrically and both confidence "
            "scores are at least accept_min_conf. No dataset labels are modified."
        ),
    }

    (report_dir / "dual_vest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== DUAL-MODEL AUDIT DONE ==========")
    print("Counts:", dict(counts))
    print("Candidates:", report_dir / "dual_vest_candidates.jsonl")
    print("Summary:", report_dir / "dual_vest_summary.json")
    print("Accepted preview:", report_dir / "preview" / "accepted")
    print("Borderline preview:", report_dir / "preview" / "borderline")


if __name__ == "__main__":
    main()
