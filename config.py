"""
Central configuration for the AI Object Detection, Tracking & Scene
Understanding System. Edit the values below to match your environment
before running main.py.
"""

import os

from dotenv import load_dotenv
load_dotenv()  # Loads variables from a .env file if present

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --------------------------------------------------------------------------
# General
# --------------------------------------------------------------------------
PROJECT_NAME = "AI Real-Time Object Detection, Tracking & Scene Understanding System"

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1250
FRAME_HEIGHT = 700

# --------------------------------------------------------------------------
# YOLO Object Detection (pretrained on the COCO dataset, 80 classes)
# --------------------------------------------------------------------------

# YOLO_MODEL_PATH = "yolov8n-oiv7.pt"    # Open Images V7 model (601 classes including body parts)
# ##YOLO_MODEL_PATH = "yolov8n.pt"    # auto-downloaded by ultralytics on first run
# YOLO_CONFIDENCE_THRESHOLD = 0.45
# YOLO_IOU_THRESHOLD = 0.45
# YOLO_TRACKER_CONFIG = "bytetrack.yaml"   # ships with ultralytics; "botsort.yaml" also available
# YOLO_DEVICE = "cpu"               # "cpu", "cuda", "cuda:0", or "mps" (Apple Silicon)


# --------------------------------------------------------------------------
# YOLO Object Detection
# --------------------------------------------------------------------------

YOLO_MODEL_PATH = "yolov8n-oiv7.pt"    # Open Images V7 model (601 classes) -- unchanged

# Detector floor passed to YOLO/ByteTrack. This was 0.45, which silently
# disabled ByteTrack's second ("low-score") association pass: any detection
# below 0.45 never reached the tracker, so a normal confidence dip dropped
# the track and the object came back with a NEW id. The tracker must see
# weak detections so it can keep existing tracks alive. Whether a detection
# is trusted/saved is decided per class in the session layer below
# (DETECTION_MIN_CONFIDENCE / CLASS_RULES), not here.
YOLO_CONFIDENCE_THRESHOLD = 0.10
YOLO_IOU_THRESHOLD = 0.40
YOLO_IMAGE_SIZE = 640   # ultralytics default; try 800/960 for small, distant objects (slower on CPU)
YOLO_TRACKER_CONFIG = os.path.join(BASE_DIR, "bytetrack_custom.yaml")   # ByteTrack, tuned (see file)
YOLO_DEVICE = "cpu"

TARGET_CLASSES = None
# Example to restrict to a subset:
# TARGET_CLASSES = ["person", "laptop", "cell phone", "chair", "book", "bottle", "cup"]

# --------------------------------------------------------------------------
# Object Tracking / Session Logic
# --------------------------------------------------------------------------
# A session is closed (and saved as ONE MongoDB record) only when its track
# has been missing for at least MAX_MISSED_FRAMES frames AND at least
# SESSION_GRACE_SECONDS seconds. Frame count alone depended on camera FPS and
# was shorter (20) than ByteTrack's own lost-track buffer, so a session could
# be closed and a duplicate one opened for the very same tracking id.
MAX_MISSED_FRAMES = 60                # keep >= track_buffer in bytetrack_custom.yaml
SESSION_GRACE_SECONDS = 2.0
MIN_SESSION_DURATION_SECONDS = 1.0    # ignore extremely short false-positive blips

# --- Confirmation: a detection becomes a "real" object only after temporal evidence
# Defaults apply to every class unless overridden in CLASS_RULES.
DETECTION_MIN_CONFIDENCE = 0.45       # per-frame confidence for a frame to count as evidence
KEEP_CONFIDENCE_RATIO = 0.6           # confirmed sessions stay alive on frames >= ratio * min confidence
CONFIRM_MIN_HITS = 3                  # evidence frames needed
CONFIRM_MIN_SECONDS = 0.5             # first -> last evidence frame
CONFIRM_MIN_HIT_RATIO = 0.5           # evidence frames / frames since first evidence (rejects flicker)

# Per-class overrides (lower-case Open Images class names).
#  - "mobile phone" has a LOWER threshold so small/distant phones are not lost.
#  - eyewear is the noisiest Open Images class on a nano model, so it needs
#    stronger, steadier evidence. Eyewear that is not on a detected face
#    ("standalone", e.g. lying in front of the camera) needs even more.
# These are starting values: tune them with diagnose.py on your own camera.
_EYEWEAR_RULES = {
    "min_confidence": 0.55, "min_hits": 8, "min_seconds": 1.0,
    "min_hit_ratio": 0.6, "min_avg_confidence": 0.58,
    "standalone_min_hits": 15, "standalone_min_seconds": 2.0,
    "standalone_min_avg_confidence": 0.62,
}
CLASS_RULES = {
    "mobile phone": {"min_confidence": 0.30, "min_hits": 3, "min_seconds": 0.5},
    "glasses": dict(_EYEWEAR_RULES),
    "sunglasses": dict(_EYEWEAR_RULES),
}

# --- Re-association of a lost track that comes back under a NEW tracker id.
# Conservative on purpose: same class, confirmed session lost <= REID_MAX_GAP_SECONDS
# ago, new box overlaps its last box (IoU >= REID_MIN_IOU), and exactly one
# candidate matches. Anything ambiguous is NOT merged (two people never merge).
REID_MAX_GAP_SECONDS = 2.0
REID_MIN_IOU = 0.30

# --- Duplicate boxes for the same person. Open Images has Person/Man/Woman/Boy/Girl,
# so one body can be reported as two classes with near-identical boxes.
PERSON_DUPLICATE_IOU = 0.70

# --- Eyewear plausibility relative to a detected face/head box
EYEWEAR_FACE_MIN_COVERAGE = 0.50      # share of the glasses box inside the face box
EYEWEAR_STANDALONE_MAX_COVERAGE = 0.15  # below this the glasses are "not on a face"
EYEWEAR_CENTER_Y_RANGE = (0.12, 0.65)   # glasses centre, as fraction of face height from its top
EYEWEAR_WIDTH_RATIO_RANGE = (0.45, 1.5) # glasses width / face width
EYEWEAR_MIN_ASPECT = 1.0                # glasses width / height

# --------------------------------------------------------------------------
# Object Co-occurrence / Relationships
# --------------------------------------------------------------------------
MIN_OBJECTS_FOR_RELATIONSHIP = 2
MAX_MISSED_FRAMES_RELATIONSHIP = 20
MIN_RELATIONSHIP_DURATION_SECONDS = 2.0

# --------------------------------------------------------------------------
# Scene Classification (Places365 -- 365 indoor/outdoor scene categories)
# --------------------------------------------------------------------------
SCENE_MODEL_ARCH = "resnet18"
SCENE_MODEL_WEIGHTS = os.path.join("models", "scene_classification_model", "resnet18_places365.pth.tar")
SCENE_CATEGORIES_FILE = os.path.join("models", "scene_classification_model", "categories_places365.txt")
SCENE_CLASSIFY_EVERY_N_FRAMES = 15    # scene classification is heavier; run periodically, not every frame
SCENE_MIN_CONFIDENCE = 0.15
SCENE_MIN_DURATION_SECONDS = 2.0

# Maps raw Places365 labels to the simplified categories requested in the spec.
# Any Places365 label not listed here falls back to a readable title-cased version
# of the raw label (see scene_classifier.SceneClassifier._fallback_label).
SCENE_CATEGORY_MAP = {
    "living_room": "Living Room",
    "bedroom": "Bedroom",
    "kitchen": "Kitchen",
    "office": "Office",
    "office_cubicles": "Office",
    "computer_room": "Office",
    "home_office": "Office",
    "conference_room": "Office",
    "classroom": "Classroom",
    "shop": "Store",
    "clothing_store": "Store",
    "general_store_indoor": "Store",
    "supermarket": "Store",
    "market_indoor": "Store",
    "street": "Street",
    "highway": "Road",
    "crosswalk": "Road",
    "parking_lot": "Parking Area",
    "park": "Park",
    "courtyard": "Park",
    "train_station_platform": "Station",
    "subway_station_platform": "Station",
    "bus_station_indoor": "Station",
    "corridor": "Room",
    "dining_room": "Room",
}
DEFAULT_SCENE_LABEL = "Room"

# --------------------------------------------------------------------------
# MongoDB
# --------------------------------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = "object_detection_db"

COLLECTION_DETECTIONS = "detections"
COLLECTION_SCENES = "scene_sessions"
COLLECTION_RELATIONSHIPS = "object_relationships"
COLLECTION_LIVE_STATE = "live_state"   # single-document collection the dashboard polls

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
WINDOW_NAME = "REAL-TIME AI CAMERA"
