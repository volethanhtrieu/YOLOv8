from typing import List

from ultralytics import YOLO

from src.variant_c.association import (
    Detection,
    TrackedPerson,
    associate_ppe,
)


PPE_CLASSES = {
    "head",
    "helmet",
    "vest",
    "glass",
}


class VariantCBackend:

    def __init__(
        self,
        model_path: str,
        device=0,
        conf: float = 0.25,
    ):
        self.model = YOLO(model_path)

        self.device = device
        self.conf = conf
        self.names = self.model.names

        names = set(self.names.values())

        required = {
            "person",
            "head",
            "helmet",
            "vest",
            "glass",
        }

        missing = required - names

        if missing:
            raise ValueError(
                f"Model missing classes: {missing}"
            )

        print("Model classes:", self.names)

    def process_frame(self, frame):

        result = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]

        persons: List[TrackedPerson] = []
        ppe: List[Detection] = []

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        cls_ids = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
        else:
            ids = [None] * len(xyxy)

        for bbox, cls_id, confidence, track_id in zip(
            xyxy,
            cls_ids,
            confs,
            ids,
        ):

            class_name = self.names[int(cls_id)]

            box = tuple(
                float(value)
                for value in bbox
            )

            if class_name == "person":

                if track_id is None:
                    continue

                persons.append(
                    TrackedPerson(
                        track_id=int(track_id),
                        bbox=box,
                        confidence=float(confidence),
                    )
                )

            elif class_name in PPE_CLASSES:

                ppe.append(
                    Detection(
                        class_name=class_name,
                        bbox=box,
                        confidence=float(confidence),
                    )
                )

        return associate_ppe(
            persons,
            ppe,
        )