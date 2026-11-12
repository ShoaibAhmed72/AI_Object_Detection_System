
"""
YOLO object detection + ByteTrack tracking wrapper (Ultralytics),
using the Open Images V7 model (601 classes).
"""

import logging

from ultralytics import YOLO

import config

logger = logging.getLogger("detector")


class ObjectDetector:
    def __init__(self, model_path=config.YOLO_MODEL_PATH, device=config.YOLO_DEVICE):
        logger.info("Loading YOLO model: %s", model_path)

        self.model = YOLO(model_path)
        self.device = device
        self.class_names = self.model.names  # Open Images V7 class names

    def track(self, frame):
        """Runs YOLO detection + ByteTrack tracking on one BGR frame.

        Returns a list of dictionaries:
            {
                "track_id": int,
                "class_name": str,
                "confidence": float,
                "box": (x1, y1, x2, y2)
            }
        """

        results = self.model.track(
            frame,
            persist=True,
            conf=config.YOLO_CONFIDENCE_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD,
            imgsz=config.YOLO_IMAGE_SIZE,
            tracker=config.YOLO_TRACKER_CONFIG,
            device=self.device,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None:
            return detections

        # -------------------------------------------------
        # RAW YOLO DETECTION CHECK (BEFORE TRACKING FILTER)
        # -------------------------------------------------

        raw_classes = boxes.cls.cpu().numpy()
        raw_confs = boxes.conf.cpu().numpy()

        for cls_id, conf in zip(raw_classes, raw_confs):
            class_name = self.class_names.get(
                int(cls_id), str(int(cls_id))
            )

            if "mobile phone" in class_name.lower():
                logger.debug(
                    "RAW YOLO PHONE: Confidence: %.3f",
                    float(conf)
                )

        # -------------------------------------------------
        # CHECK BYTETRACK TRACKING IDS
        # -------------------------------------------------

        if boxes.id is None:
            logger.debug(
                "YOLO returned detections, but ByteTrack assigned no IDs."
            )
            return detections

        ids = boxes.id.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # -------------------------------------------------
        # PREPARE TRACKED DETECTIONS
        # -------------------------------------------------

        for track_id, cls_id, conf, box in zip(
            ids, classes, confs, xyxy
        ):
            class_name = self.class_names.get(
                int(cls_id), str(int(cls_id))
            )

            if (
                config.TARGET_CLASSES
                and class_name not in config.TARGET_CLASSES
            ):
                continue

            detections.append({
                "track_id": int(track_id),
                "class_name": class_name,
                "confidence": float(conf),
                "box": tuple(float(v) for v in box),
            })

        # -------------------------------------------------
        # TRACKED DETECTION LOGGING
        # -------------------------------------------------

        for det in detections:
            logger.debug(
                "DETECTION: %s | Confidence: %.3f | Track ID: %s",
                det["class_name"],
                det["confidence"],
                det["track_id"]
            )

        return detections
