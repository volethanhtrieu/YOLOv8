import argparse
import json
from pathlib import Path

import cv2

from src.variant_c.pipeline import VariantCBackend


def detection_to_json(detection):

    if detection is None:
        return None

    return {
        "bbox": list(detection.bbox),
        "confidence": detection.confidence,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)

    parser.add_argument(
        "--output",
        default="outputs/variant_c.mp4",
    )

    parser.add_argument(
        "--json",
        default="outputs/variant_c.jsonl",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    backend = VariantCBackend(
        model_path=args.model,
        device=0,
        conf=args.conf,
    )

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {args.source}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    output = Path(args.output)
    json_path = Path(args.json)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_index += 1

            people = backend.process_frame(frame)

            frame_output = {
                "frame": frame_index,
                "people": [],
            }

            for person in people:

                x1, y1, x2, y2 = map(
                    int,
                    person["person_bbox"],
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2,
                )

                status = (
                    f"ID {person['track_id']} "
                    f"H:{int(person['has_helmet'])} "
                    f"V:{int(person['has_vest'])}"
                )

                cv2.putText(
                    frame,
                    status,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2,
                )

                record = {
                    "track_id": person["track_id"],
                    "person_bbox": list(
                        person["person_bbox"]
                    ),
                    "person_conf": person["person_conf"],

                    "has_head": person["has_head"],
                    "has_helmet": person["has_helmet"],
                    "has_vest": person["has_vest"],

                    "head": detection_to_json(
                        person["head"]
                    ),
                    "helmet": detection_to_json(
                        person["helmet"]
                    ),
                    "vest": detection_to_json(
                        person["vest"]
                    ),
                }

                frame_output["people"].append(
                    record
                )

            json_file.write(
                json.dumps(frame_output)
                + "\n"
            )

            writer.write(frame)

    cap.release()
    writer.release()

    print("Variant C complete")
    print("Video:", output.resolve())
    print("JSONL:", json_path.resolve())


if __name__ == "__main__":
    main()
