# AI-Based Real-Time Object Detection, Tracking, Scene Classification & Monitoring System

A real-time computer vision system that watches a live webcam feed and answers:

- **What objects are present?** (YOLO, pretrained on COCO — 80 classes)
- **How long has each object been visible?** (per-object tracking sessions, unique IDs)
- **Which objects are visible together, and for how long?** (co-occurrence, not assumed relationships)
- **What kind of environment is this?** (Places365 scene classification — Room, Office, Street, etc.)
- **How long has that scene lasted?**
- **When did all of this happen?** (everything timestamped and stored in MongoDB)

All of the above is stored in MongoDB and viewable/searchable in a Streamlit dashboard.

```
LIVE WEBCAM → YOLO DETECTION → OBJECT TRACKING → SCENE CLASSIFIER → MONGODB → DASHBOARD
```

---

## 1. Project Structure

```
AI_Object_Detection_System/
│
├── main.py                 # Live camera loop: detection + tracking + scene + DB writes
├── detector.py              # YOLO + ByteTrack wrapper (COCO-pretrained)
├── tracker.py                # Turns per-frame detections into "detection sessions"
├── scene_classifier.py       # Places365 scene classifier + "scene sessions"
├── relationship.py           # Object co-occurrence session manager
├── database.py                # All MongoDB reads/writes (PyMongo), error-tolerant
├── config.py                  # All tunable settings in one place
├── dashboard.py               # Streamlit dashboard (history, stats, search)
├── download_places365.py      # One-time script to fetch scene-classifier weights
│
├── requirements.txt
├── README.md
│
├── models/
│   ├── object_detection_model/       # YOLO weights land here if you point YOLO_MODEL_PATH here
│   └── scene_classification_model/   # Places365 weights + categories file go here
│
├── templates/     # reserved (e.g. if you later add a Flask-based UI)
├── static/        # reserved (e.g. static assets for a Flask-based UI)
└── results/        # optional: exported snapshots/reports
```

---

## 2. Requirements

- Python 3.9–3.11
- A webcam (built-in or USB)
- MongoDB (local install **or** a free MongoDB Atlas cluster)
- ~2 GB free disk space (PyTorch + model weights)
- Works on CPU. A CUDA GPU (or Apple Silicon `mps`) makes it noticeably faster but isn't required.

---

## 3. Setup Instructions

### Step 1 — Get the code and create a virtual environment

```bash
cd AI_Object_Detection_System

python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### Step 2 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs OpenCV, Ultralytics (YOLO), PyTorch/torchvision, PyMongo, Streamlit, pandas, numpy, and Pillow.

> If `pip install torch` is slow or you want GPU support, install the CUDA build of PyTorch
> first using the selector at https://pytorch.org/get-started/locally/ , then run
> `pip install -r requirements.txt` again — it will skip torch/torchvision since they're already installed.

### Step 3 — Install and start MongoDB

**Option A — Local MongoDB (recommended for development)**

- macOS: `brew tap mongodb/brew && brew install mongodb-community && brew services start mongodb-community`
- Ubuntu/Debian: follow MongoDB's official apt instructions, then `sudo systemctl start mongod`
- Windows: install "MongoDB Community Server" from mongodb.com, it installs as a service and starts automatically

Verify it's running:

```bash
mongosh --eval "db.runCommand({ ping: 1 })"
```

**Option B — MongoDB Atlas (free cloud cluster, no local install)**

1. Create a free cluster at https://www.mongodb.com/cloud/atlas
2. Get your connection string (looks like `mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/`)
3. Set it as an environment variable before running the app:

```bash
# macOS/Linux
export MONGO_URI="mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/"
# Windows (PowerShell)
$env:MONGO_URI="mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/"
```

If `MONGO_URI` isn't set, the app defaults to `mongodb://localhost:27017` (see `config.py`).

The database (`object_detection_db`) and its collections (`detections`, `scene_sessions`,
`object_relationships`, `live_state`) are created automatically the first time the app writes to them —
you don't need to create them manually.

### Step 4 — Download the YOLO weights (automatic)

Nothing to do here — the first time you run `main.py`, Ultralytics automatically downloads
`yolov8n.pt` (the small/fast COCO-pretrained model, ~6 MB). To use a larger, more accurate model,
change `YOLO_MODEL_PATH` in `config.py` to `yolov8s.pt`, `yolov8m.pt`, etc. — it'll download on first use too.

### Step 5 — Download the Places365 scene-classification weights (one-time)

```bash
python download_places365.py
```

This downloads `resnet18_places365.pth.tar` (~45 MB) and `categories_places365.txt` into
`models/scene_classification_model/`. If you skip this step, the app still runs — it just won't
show a scene prediction until you run it.

### Step 6 — Run the live camera app

```bash
python main.py
```

A window titled **"REAL-TIME AI CAMERA"** opens showing your webcam feed with bounding boxes,
object labels, confidence scores, tracking IDs, the current scene, currently co-occurring
objects, and FPS.

**Controls (click the video window first so it has focus):**

| Key | Action |
|-----|--------|
| `q` or `Esc` | Stop the camera and exit safely (flushes any open sessions to MongoDB) |
| `c` | Clear all history (asks for `yes`/`no` confirmation in the terminal) |
| `Ctrl+C` (in terminal) | Also stops safely |

### Step 7 — Run the dashboard

In a **second terminal** (keep `main.py` running in the first one):

```bash
source .venv/bin/activate      # activate the same venv
streamlit run dashboard.py
```

This opens a browser tab with tabs for: **Live** status, **Detection History**, **Scene History**,
**Relationships**, **Search**, and **Admin** (clear history).

---

## 4. Configuration

Everything tunable lives in `config.py`:

| Setting | What it controls |
|---|---|
| `CAMERA_INDEX` | Which camera to use (0 = default); can also be a video file path for testing |
| `YOLO_MODEL_PATH` | Which YOLO checkpoint to use (`yolov8n.pt`, `yolov8s.pt`, ...) |
| `YOLO_CONFIDENCE_THRESHOLD` | Minimum confidence to count as a detection |
| `TARGET_CLASSES` | Restrict detection/storage to a subset of COCO classes (or `None` for all 80) |
| `MAX_MISSED_FRAMES` | How many frames an object can vanish for before its session is closed |
| `SCENE_CLASSIFY_EVERY_N_FRAMES` | How often to run the (heavier) scene classifier |
| `MONGO_URI`, `MONGO_DB_NAME` | Where to store data |

---

## 5. Using a Different Dataset / Model

The spec asks for COCO (detection) + Places365 (scene). The code is written so either can be swapped:

- **Detection dataset/model**: any Ultralytics-compatible YOLO checkpoint works — including a
  custom-trained model on **Open Images**, a custom dataset, or a fine-tuned COCO model. Just point
  `YOLO_MODEL_PATH` in `config.py` at your `.pt` file (place it in `models/object_detection_model/`)
  and update `TARGET_CLASSES` if the class names differ.
- **Scene dataset/model**: `scene_classifier.py` is isolated behind a `predict(frame) -> (label, confidence)`
  interface, so Places365 could be swapped for another scene-classification backbone (e.g. a
  custom-trained classifier on your own indoor/outdoor image set) without touching `main.py`.

---

## 6. How Sessions Work (avoiding duplicate records)

A new MongoDB record is **not** created every frame. Instead:

- **Detections**: each tracked object (unique ID from ByteTrack) has an "active session" in memory
  that keeps extending its `last_seen` time and confidence stats every frame it's visible. Only when
  the object disappears for `MAX_MISSED_FRAMES` consecutive frames does the session close and get
  written to MongoDB as one record.
- **Scenes**: the same idea — a scene session stays open as long as the same scene keeps winning
  classification; it closes (and a new one opens) when the scene changes.
- **Relationships**: a session stays open as long as the *same combination* of object classes is
  visible; it closes when that exact combination is no longer fully present.

This keeps the database compact (one row per real-world event) instead of one row per frame
(which at 30 FPS would be enormous and mostly redundant).

---

## 7. Error Handling

- **Camera not found / in use** → `main.py` logs a clear error and exits instead of crashing.
- **MongoDB unreachable** → the app keeps running; writes are skipped and logged until the
  connection comes back (it retries automatically on the next write).
- **Places365 weights missing** → scene classification is disabled, everything else keeps working.
- **Stopping the camera** (`q`/`Esc`/`Ctrl+C`) always flushes any still-open sessions to MongoDB
  before exiting, so you don't lose the last in-progress detection.

---

## 8. Performance Notes

- Default model (`yolov8n.pt`) targets real-time speed on CPU; expect roughly 15–30 FPS on a modern
  laptop CPU, faster on GPU.
- Scene classification runs only every `SCENE_CLASSIFY_EVERY_N_FRAMES` frames (default 15) since
  it's the most expensive step per call — this keeps overall FPS high without needing GPU.
- To evaluate the system formally (as called out in the original spec): log YOLO's own `results.boxes`
  confidences for precision/recall/mAP/IoU studies against a labeled validation set, and log ByteTrack
  ID switches by comparing ground-truth tracks — this typically means running `main.py` against
  recorded video with a corresponding annotation file rather than a live webcam.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera index 0` | Try `CAMERA_INDEX = 1` in `config.py`, or close other apps using the webcam |
| Dashboard shows "Not connected to MongoDB" | Confirm `mongod` is running locally, or that `MONGO_URI` is set correctly |
| No scene shown in the overlay | Run `python download_places365.py` first |
| Low FPS | Use `yolov8n.pt` (default, fastest), lower `FRAME_WIDTH`/`FRAME_HEIGHT`, or use a GPU |
| `ultralytics` download fails on first run | Check your internet connection; YOLO weights download from GitHub on first use |

## Tracking stability & validation (settings in `config.py`)

- `bytetrack_custom.yaml` – ByteTrack settings (low-score rescue pass, 60-frame lost buffer).
- `YOLO_CONFIDENCE_THRESHOLD` is only the detector *floor* (0.10); what counts as a real object is decided by `DETECTION_MIN_CONFIDENCE`, `CLASS_RULES` (per-class thresholds, e.g. lower for phones, stricter for glasses) and the confirmation rules (`CONFIRM_*`).
- Every saved detection session carries `run_id` (one camera run) and `session_uid`; the PDF lists only the current run.
- `python diagnose.py --seconds 60 [--show]` prints raw ids vs confirmed sessions and rejection reasons without touching MongoDB. `python -m unittest discover -s tests` runs the logic tests.
