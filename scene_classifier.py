"""
Scene / environment classification using a pretrained Places365 model
(ResNet18, 365 scene categories -- indoor + outdoor). Also manages "scene
sessions" the same way tracker.py manages object sessions: one continuous
scene = one MongoDB record, not one per frame.

Weights are NOT bundled with this repo (they're ~45 MB). Run
`python download_places365.py` once before starting main.py -- see README.md.
If the weights aren't present, scene classification is simply disabled
(current_scene stays None) and the rest of the system keeps working.
"""

import datetime
import logging
import os

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

import config

logger = logging.getLogger("scene_classifier")


class SceneClassifier:
    def __init__(self, weights_path=config.SCENE_MODEL_WEIGHTS,
                 categories_path=config.SCENE_CATEGORIES_FILE):
        self.available = os.path.exists(weights_path) and os.path.exists(categories_path)
        if not self.available:
            logger.warning(
                "Places365 weights/categories not found at %s / %s. "
                "Run download_places365.py first. Scene classification is DISABLED "
                "until then; object detection/tracking still works normally.",
                weights_path, categories_path,
            )
            self.model = None
            self.classes = []
            return

        self.model = models.__dict__[config.SCENE_MODEL_ARCH](num_classes=365)
        checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.classes = []
        with open(categories_path) as f:
            for line in f:
                # Places365 category file format: "/a/airfield 0"
                raw_name = line.strip().split(" ")[0][3:]
                self.classes.append(raw_name)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, frame_bgr):
        """frame_bgr: OpenCV BGR numpy array.
        Returns (friendly_label, confidence) or (None, 0.0) if unavailable."""
        if not self.available:
            return None, 0.0

        img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
        tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]

        top_prob, top_idx = torch.max(probs, dim=0)
        raw_label = self.classes[top_idx.item()]
        confidence = float(top_prob.item())

        friendly = config.SCENE_CATEGORY_MAP.get(raw_label, self._fallback_label(raw_label))
        return friendly, confidence

    @staticmethod
    def _fallback_label(raw_label):
        # Falls back to a readable version of the raw Places365 tag when it
        # isn't one of the categories explicitly requested in the spec.
        return raw_label.replace("_", " ").title()


class _ActiveScene:
    def __init__(self, scene_name, confidence, timestamp):
        self.scene_name = scene_name
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.max_confidence = confidence
        self.missed_frames = 0

    def update(self, confidence, timestamp):
        self.last_seen = timestamp
        self.max_confidence = max(self.max_confidence, confidence)
        self.missed_frames = 0

    def to_document(self):
        duration = (self.last_seen - self.first_seen).total_seconds()
        return {
            "scene_name": self.scene_name,
            "date": self.first_seen.strftime("%Y-%m-%d"),
            "first_seen": self.first_seen.strftime("%H:%M:%S"),
            "last_seen": self.last_seen.strftime("%H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "confidence": round(self.max_confidence, 4),
        }


class SceneSessionManager:
    def __init__(self, db):
        self.db = db
        self._active = None  # only one scene is "current" at a time
        self.current_scene = None
        self.current_confidence = 0.0

    def update(self, scene_name, confidence, timestamp=None):
        timestamp = timestamp or datetime.datetime.now()
        self.current_scene = scene_name
        self.current_confidence = confidence

        if scene_name is None or confidence < config.SCENE_MIN_CONFIDENCE:
            return

        if self._active is None:
            self._active = _ActiveScene(scene_name, confidence, timestamp)
        elif self._active.scene_name == scene_name:
            self._active.update(confidence, timestamp)
        else:
            self._close_active()  # scene changed -> close old session, start a new one
            self._active = _ActiveScene(scene_name, confidence, timestamp)

    def _close_active(self):
        if self._active is None:
            return
        duration = (self._active.last_seen - self._active.first_seen).total_seconds()
        if duration >= config.SCENE_MIN_DURATION_SECONDS:
            doc = self._active.to_document()
            self.db.save_scene_session(doc)
            logger.info("Saved scene session: %s", doc)
        self._active = None

    def close_all(self):
        self._close_active()
