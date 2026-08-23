from typing import Dict, List

from ultralytics import YOLO

from src.variant_c.association import (
    Detection,
    TrackedPerson,
    associate_ppe,
)


REQUIRED_CLASSES = {
    "person",
    "head",
    "helmet",
    "vest",
}

PPE_CLASS_NAMES = {
    "head",
    "helmet",
    "vest",
}


class VariantCBackend:
    

    def __init__(
        self,
        model_path: str,
        device=0,
        conf: float = 0.25,
    ):
        self.model_path = model_path
        self.device = device

        
        self.person_conf = 0.45

        
        self.ppe_conf = conf

        
        self.tracker_model = YOLO(model_path)
        self.ppe_model = YOLO(model_path)

        self.names = self.tracker_model.names

        self.class_name_to_id = self._build_class_mapping(
            self.names
        )

        actual_names = (
            tuple(
                self.names[class_id]
                for class_id in sorted(self.names)
            )
            if isinstance(self.names, dict)
            else tuple(self.names)
        )
        expected_names = (
            "person",
            "head",
            "helmet",
            "vest",
        )

        if actual_names != expected_names:
            raise ValueError(
                "Checkpoint must use exactly the four-class schema "
                f"{expected_names}; got {actual_names}"
            )

        missing = REQUIRED_CLASSES - set(
            self.class_name_to_id.keys()
        )

        if missing:
            raise ValueError(
                f"Model missing required classes: "
                f"{sorted(missing)}"
            )

        self.person_class_id = self.class_name_to_id[
            "person"
        ]

        self.ppe_class_ids = [
            self.class_name_to_id[name]
            for name in (
                "head",
                "helmet",
                "vest",
            )
        ]

        print("Model classes:", self.names)
        print(
            "ByteTrack class:",
            self.person_class_id,
            "-> person",
        )
        print(
            "PPE detection classes:",
            self.ppe_class_ids,
        )

    @staticmethod
    def _build_class_mapping(
        names,
    ) -> Dict[str, int]:
        

        if isinstance(names, dict):
            return {
                str(name): int(class_id)
                for class_id, name in names.items()
            }

        return {
            str(name): int(class_id)
            for class_id, name in enumerate(names)
        }

    def _track_persons(
        self,
        frame,
    ) -> List[TrackedPerson]:
        

        result = self.tracker_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[self.person_class_id],
            conf=self.person_conf,
            device=self.device,
            verbose=False,
        )[0]

        boxes = result.boxes

        if (
            boxes is None
            or len(boxes) == 0
            or boxes.id is None
        ):
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist()

        persons: List[TrackedPerson] = []

        for bbox, confidence, track_id in zip(
            xyxy,
            confs,
            track_ids,
        ):
            x1, y1, x2, y2 = map(float, bbox)

            width = x2 - x1
            height = y2 - y1

            # Invalid bounding box
            if width <= 0 or height <= 0:
                continue

            
            aspect_ratio = width / height

            if aspect_ratio < 0.28:
                continue

            persons.append(
                TrackedPerson(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                )
            )

        return persons

    def _detect_ppe(
        self,
        frame,
    ) -> List[Detection]:
        

        result = self.ppe_model.predict(
            source=frame,
            classes=self.ppe_class_ids,
            conf=self.ppe_conf,
            device=self.device,
            verbose=False,
        )[0]

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        cls_ids = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        detections: List[Detection] = []

        for bbox, class_id, confidence in zip(
            xyxy,
            cls_ids,
            confs,
        ):
            class_name = self.names[int(class_id)]

            if class_name not in PPE_CLASS_NAMES:
                continue

            detections.append(
                Detection(
                    class_name=class_name,
                    bbox=tuple(
                        float(value)
                        for value in bbox
                    ),
                    confidence=float(confidence),
                )
            )

        return detections

    def process_frame(
        self,
        frame,
    ) -> List[dict]:
        

        persons = self._track_persons(frame)

        if not persons:
            return []

        ppe_detections = self._detect_ppe(frame)

        return associate_ppe(
            persons,
            ppe_detections,
        )
