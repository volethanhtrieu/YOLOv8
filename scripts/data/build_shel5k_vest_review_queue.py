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


def read_labels(label_path: Path):
    people = []
    existing_vests = []

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 5:
            continue

        cid = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        box = yolo_to_xyxy(x, y, w, h)

        if cid == 0:
            people.append({"box": box})
        elif cid == 3:
            existing_vests.append({"box": box})

    return people, existing_vests


def people_with_existing_vest(people, existing_vests):
    occupied = set()

    for vest in existing_vests:
        best = None

        for person_idx, person in enumerate(people):
            if not center_inside(vest["box"], person["box"]):
                continue

            cont = containment(vest["box"], person["box"])
            if cont < 0.60:
                continue

            if best is None or cont > best[0]:
                best = (cont, person_idx)

        if best is not None:
            occupied.add(best[1])

    return occupied


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

    for box, score in zip(
        result.boxes.xyxyn.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
    ):
        out.append({
            "box": tuple(map(float, box)),
            "conf": float(score),
        })
    return out


def associate_to_people(cands, people, min_containment, min_area_ratio, max_area_ratio):
    valid = []

    for cand in cands:
        best = None

        for person_idx, person in enumerate(people):
            pbox = person["box"]

            if not center_inside(cand["box"], pbox):
                continue

            cont = containment(cand["box"], pbox)
            if cont < min_containment:
                continue

            pa = area(pbox)
            va = area(cand["box"])
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
            valid.append({**cand, **best})

    return valid


def consensus(coco, scratch, min_iou):
    pairs = []

    for i, a in enumerate(coco):
        for j, b in enumerate(scratch):
            ov = iou(a["box"], b["box"])
            if ov >= min_iou:
                pairs.append((ov * min(a["conf"], b["conf"]), ov, i, j))

    pairs.sort(reverse=True)

    used_c = set()
    used_s = set()
    matched = []

    for _, ov, i, j in pairs:
        if i in used_c or j in used_s:
            continue

        used_c.add(i)
        used_s.add(j)

        a = coco[i]
        b = scratch[j]
        box = tuple((x + y) / 2.0 for x, y in zip(a["box"], b["box"]))

        matched.append({
            "box": box,
            "coco_conf": a["conf"],
            "scratch_conf": b["conf"],
            "iou": ov,
            "min_conf": min(a["conf"], b["conf"]),
            "mean_conf": (a["conf"] + b["conf"]) / 2.0,
        })

    singles_c = [coco[i] for i in range(len(coco)) if i not in used_c]
    singles_s = [scratch[j] for j in range(len(scratch)) if j not in used_s]

    return matched, singles_c, singles_s


def iter_original_images(root: Path, split: str):
    image_dir = root / "images" / split
    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if split == "train":
        images = [p for p in images if not p.stem.startswith("aug_")]
    return sorted(images)


def draw_candidate(image_path, people, cand, out_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    h, w = image.shape[:2]

    def d(box):
        x1, y1, x2, y2 = box
        return int(x1*w), int(y1*h), int(x2*w), int(y2*h)

    for p in people:
        x1, y1, x2, y2 = d(p["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    x1, y1, x2, y2 = d(cand["box_xyxy_norm"])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 3)

    label = cand["source_type"]
    if label == "consensus_borderline":
        txt = (
            f"REVIEW dual C:{cand['coco_conf']:.2f} "
            f"S:{cand['scratch_conf']:.2f} I:{cand['iou']:.2f}"
        )
    else:
        txt = f"REVIEW {label} conf:{cand['confidence']:.2f}"

    cv2.putText(
        image, txt, (max(0, x1), max(22, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-coco", required=True, type=Path)
    parser.add_argument("--model-scratch", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)

    parser.add_argument("--device", default="0")
    parser.add_argument("--infer-conf", type=float, default=0.25)
    parser.add_argument("--consensus-iou", type=float, default=0.50)
    parser.add_argument("--accepted-min-conf", type=float, default=0.45)

    # Strong single-model detections only. Weak singles are too noisy for useful review.
    parser.add_argument("--single-min-conf", type=float, default=0.55)

    parser.add_argument("--min-containment", type=float, default=0.65)
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-area-ratio", type=float, default=0.85)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    coco_model = YOLO(str(args.model_coco.resolve()))
    scratch_model = YOLO(str(args.model_scratch.resolve()))

    coco_vest = get_vest_id(coco_model.names)
    scratch_vest = get_vest_id(scratch_model.names)

    queue = []
    counts = Counter()
    candidate_id = 0

    for split in ("train", "val", "test"):
        images = iter_original_images(dataset, split)
        label_dir = dataset / "labels" / split

        print(f"\n[{split}] {len(images)} original images")

        for idx, image_path in enumerate(images, 1):
            people, existing_vests = read_labels(
                label_dir / f"{image_path.stem}.txt"
            )
            occupied_people = people_with_existing_vest(
                people, existing_vests
            )

            coco = predict_vests(
                coco_model, image_path, coco_vest, args.infer_conf, args.device
            )
            scratch = predict_vests(
                scratch_model, image_path, scratch_vest, args.infer_conf, args.device
            )

            matches, singles_c, singles_s = consensus(
                coco, scratch, args.consensus_iou
            )

            # Only consensus predictions that did NOT pass the previous auto-accept threshold.
            borderline = [
                m for m in matches
                if m["min_conf"] < args.accepted_min_conf
            ]

            borderline = associate_to_people(
                borderline, people,
                args.min_containment, args.min_area_ratio, args.max_area_ratio
            )

            strong_c = [
                {**x, "confidence": x["conf"], "source_type": "single_coco"}
                for x in singles_c if x["conf"] >= args.single_min_conf
            ]
            strong_s = [
                {**x, "confidence": x["conf"], "source_type": "single_scratch"}
                for x in singles_s if x["conf"] >= args.single_min_conf
            ]

            strong_c = associate_to_people(
                strong_c, people,
                args.min_containment, args.min_area_ratio, args.max_area_ratio
            )
            strong_s = associate_to_people(
                strong_s, people,
                args.min_containment, args.min_area_ratio, args.max_area_ratio
            )

            candidates = []

            for c in borderline:
                if c["person_idx"] in occupied_people:
                    continue
                candidates.append({
                    **c,
                    "source_type": "consensus_borderline",
                    "priority": 0,
                })

            for c in strong_c:
                if c["person_idx"] in occupied_people:
                    continue
                candidates.append({
                    **c,
                    "priority": 1,
                })

            for c in strong_s:
                if c["person_idx"] in occupied_people:
                    continue
                candidates.append({
                    **c,
                    "priority": 2,
                })

            # Avoid reviewing almost-identical single candidates twice.
            kept = []
            for c in sorted(
                candidates,
                key=lambda z: (
                    z["priority"],
                    -z.get("min_conf", z.get("confidence", 0.0)),
                )
            ):
                duplicate = False
                for k in kept:
                    if c["person_idx"] == k["person_idx"] and iou(c["box"], k["box"]) >= 0.60:
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(c)

            for c in kept:
                candidate_id += 1
                counts[c["source_type"]] += 1

                rec = {
                    "candidate_id": candidate_id,
                    "split": split,
                    "image": image_path.name,
                    "stem": image_path.stem,
                    "source_type": c["source_type"],
                    "person_idx": c["person_idx"],
                    "containment": c["containment"],
                    "area_ratio": c["area_ratio"],
                    "box_xyxy_norm": list(c["box"]),
                }

                if c["source_type"] == "consensus_borderline":
                    rec.update({
                        "coco_conf": c["coco_conf"],
                        "scratch_conf": c["scratch_conf"],
                        "iou": c["iou"],
                        "min_conf": c["min_conf"],
                    })
                else:
                    rec["confidence"] = c["confidence"]

                preview_name = f"{candidate_id:05d}_{split}_{image_path.stem}.jpg"
                rec["preview"] = str((out / "preview" / preview_name).resolve())

                draw_candidate(
                    image_path, people, rec, out / "preview" / preview_name
                )
                queue.append(rec)

            if idx % 250 == 0 or idx == len(images):
                print(f"  {idx}/{len(images)}")

    queue_path = out / "review_queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as f:
        for rec in queue:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "dataset": str(dataset),
        "queue_size": len(queue),
        "counts": dict(counts),
        "policy": {
            "consensus_borderline": f"dual-model consensus with min_conf < {args.accepted_min_conf}",
            "single_model": f"unmatched single-model vest with confidence >= {args.single_min_conf}",
        },
        "note": (
            "This queue improves recall but still cannot guarantee finding vests "
            "missed by both models. Exhaustive ground truth requires reviewing all images."
        ),
    }

    (out / "review_queue_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== REVIEW QUEUE READY ==========")
    print("Queue size:", len(queue))
    print("Counts:", dict(counts))
    print("Queue:", queue_path)
    print("Preview folder:", out / "preview")
    print("Summary:", out / "review_queue_summary.json")


if __name__ == "__main__":
    main()
