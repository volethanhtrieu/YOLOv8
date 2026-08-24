from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def open_image(path: Path):
    if sys.platform.startswith("win"):
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def load_jsonl(path: Path):
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def load_decisions(path: Path):
    decisions = {}
    if not path.exists():
        return decisions

    for rec in load_jsonl(path):
        decisions[int(rec["candidate_id"])] = rec
    return decisions


def save_decisions(path: Path, decisions):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for cid in sorted(decisions):
            f.write(json.dumps(decisions[cid], ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()

    queue = load_jsonl(args.queue.resolve())
    decisions_path = args.decisions.resolve()
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions = load_decisions(decisions_path)

    total = len(queue)

    print("Controls:")
    print("  y = true vest, approve")
    print("  n = not vest, reject")
    print("  s = skip for later")
    print("  q = save and quit")
    print("  b = go back one candidate")
    print()

    idx = max(0, args.start - 1)

    while 0 <= idx < total:
        rec = queue[idx]
        cid = int(rec["candidate_id"])

        if cid in decisions and decisions[cid]["decision"] in {"yes", "no"}:
            idx += 1
            continue

        preview = Path(rec["preview"])
        if not preview.is_file():
            raise FileNotFoundError(preview)

        open_image(preview)

        print("=" * 80)
        print(f"[{idx+1}/{total}] candidate_id={cid}")
        print(f"split={rec['split']} image={rec['image']}")
        print(f"type={rec['source_type']}")

        if rec["source_type"] == "consensus_borderline":
            print(
                f"COCO={rec['coco_conf']:.3f} "
                f"Scratch={rec['scratch_conf']:.3f} "
                f"IoU={rec['iou']:.3f}"
            )
        else:
            print(f"confidence={rec['confidence']:.3f}")

        while True:
            ans = input("Is the ORANGE box a real safety vest? [y/n/s/q/b]: ").strip().lower()

            if ans in {"y", "yes"}:
                decisions[cid] = {
                    "candidate_id": cid,
                    "decision": "yes",
                    "split": rec["split"],
                    "image": rec["image"],
                    "stem": rec["stem"],
                    "source_type": rec["source_type"],
                    "box_xyxy_norm": rec["box_xyxy_norm"],
                }
                save_decisions(decisions_path, decisions)
                idx += 1
                break

            if ans in {"n", "no"}:
                decisions[cid] = {
                    "candidate_id": cid,
                    "decision": "no",
                    "split": rec["split"],
                    "image": rec["image"],
                    "stem": rec["stem"],
                    "source_type": rec["source_type"],
                    "box_xyxy_norm": rec["box_xyxy_norm"],
                }
                save_decisions(decisions_path, decisions)
                idx += 1
                break

            if ans == "s":
                decisions[cid] = {
                    "candidate_id": cid,
                    "decision": "skip",
                    "split": rec["split"],
                    "image": rec["image"],
                    "stem": rec["stem"],
                    "source_type": rec["source_type"],
                    "box_xyxy_norm": rec["box_xyxy_norm"],
                }
                save_decisions(decisions_path, decisions)
                idx += 1
                break

            if ans == "q":
                save_decisions(decisions_path, decisions)
                yes = sum(1 for d in decisions.values() if d["decision"] == "yes")
                no = sum(1 for d in decisions.values() if d["decision"] == "no")
                skip = sum(1 for d in decisions.values() if d["decision"] == "skip")
                print(f"Saved. yes={yes}, no={no}, skip={skip}")
                return

            if ans == "b":
                idx = max(0, idx - 1)
                break

            print("Please type y, n, s, q, or b.")

    save_decisions(decisions_path, decisions)

    yes = sum(1 for d in decisions.values() if d["decision"] == "yes")
    no = sum(1 for d in decisions.values() if d["decision"] == "no")
    skip = sum(1 for d in decisions.values() if d["decision"] == "skip")

    print("\n========== REVIEW COMPLETE ==========")
    print(f"yes={yes}, no={no}, skip={skip}")
    print("Decisions:", decisions_path)


if __name__ == "__main__":
    main()
