# Fixes report – tracking, detection, sessions, PDF

**Important:** I had no camera, YOLO/torch/pymongo or network in my environment. Session logic, PDF generation and the DB query were tested here with simulated detection streams. Detection quality on real footage (Tests 1–5) must be confirmed on your machine with `python diagnose.py` – see "Remaining limitations".

## Root causes
1. **Too many tracking IDs**
   - `conf=0.45` was passed to `model.track()`. Detections below 0.45 never reached ByteTrack, so its second "low-score" pass (which keeps tracks alive through confidence dips) never ran. Your own PDF shows Clothing/Glasses at 42–53 %, right at the threshold, so tracks died and returned with new IDs (IDs up to 62).
   - ByteTrack's lost-track buffer was 30 frames (≈1 s on a fast PC).
   - One body is reported as several Open Images classes (Person/Man/Woman…) with near-identical boxes, and each object (person, face, clothing, glasses) has its own ID – so "2 people" legitimately produce many IDs.
2. **False/unstable detections**: the Open Images nano model is noisy on `Glasses`; every frame above 0.45 was trusted, and a single frame created a session. Per-frame INFO logging of every detection also slowed the loop (low FPS hurts tracking).
3. **Wrong sessions/durations**: sessions closed after 20 frames (shorter than the tracker's 30-frame memory and FPS-dependent), so a returning ID opened a duplicate session; class label was taken from the first frame only; nothing required temporal evidence.
4. **PDF**: it filtered by date only, so every run of the day was mixed in (tracker IDs restart at 1 each run → "duplicate" IDs); no first/last seen columns; no duplicate removal.

## Files changed
| File | Change |
|---|---|
| `config.py` | Detector floor 0.10; per-class thresholds (`CLASS_RULES`: phone 0.30, glasses stricter), confirmation, grace, re-association, eyewear settings; `YOLO_IMAGE_SIZE` (640 = unchanged); `MAX_MISSED_FRAMES` 20→60 + `SESSION_GRACE_SECONDS` |
| `bytetrack_custom.yaml` (new) | Same ByteTrack; `track_buffer` 60, high/new thresh 0.30, low 0.10 |
| `detection_filters.py` (new) | Duplicate Person/Man box suppression; glasses on-face / standalone / implausible check |
| `tracker.py` | Candidate→confirmed sessions (hits, time span, hit-ratio, avg conf); credible-frame stats; class voting; time-based grace; conservative re-association of lost tracks (same class, ≤2 s, IoU≥0.3, unambiguous only); `run_id`, `session_uid`, `track_ids`; `update()` now returns confirmed detections for drawing |
| `detector.py` | Passes `imgsz`; per-frame INFO logs → DEBUG |
| `database.py` | `search_detections(run_id=…)`; additive sparse indexes `run_id`, unique `session_uid` (failure can't block connect) |
| `pdf_exporter.py` | Current-run filter, dedupe, min-duration guard, columns: object, track ID, first/last seen, duration, avg & max confidence; honest "IDs are tracks" note |
| `main.py` | Uses returned confirmed detections; PDF scoped to run; PDF errors can't skip camera cleanup; run summary logged |
| `diagnose.py`, `tests/` (new) | Camera diagnostic; 15 logic tests |

Unchanged: YOLO model (`yolov8n-oiv7.pt`), ByteTrack, MongoDB collections, Places365, dashboard, GUI.

## Tests performed (simulated – not camera)
- 15 unit tests pass: ID switch + confidence dips → one session; two people never merged; new person elsewhere not merged; duplicate Man/Person boxes → one session; one-frame/flicker rejected; worn glasses saved, bare-face flicker and misplaced glasses rejected; standalone glasses need more evidence; 0.33-conf phone kept while a 0.33 cup is not; dropout keeps one session with exact first/last-seen duration; PDF contains only the current run, no duplicates/old runs/short blips.
- Synthetic before/after (two people, 60 s, 5 seeds, original `tracker.py` with the 0.45 pre-filter vs new): **45.2 → 6.0 sessions/IDs per run** (6 = 2 people × Man/Face/Clothing, the ideal). This simulates only the session layer; ByteTrack's own improvement from the low-score pass isn't included.

## Remaining limitations
- Real-camera results are not measured. All thresholds are starting values. Run `python diagnose.py --seconds 60 --show` for each test and tune `CLASS_RULES` if needed.
- Glasses on a bare face remain a model weakness; they are now suppressed by temporal evidence, not eliminated. Try `YOLO_IMAGE_SIZE = 800` for distant phones if needed.
- Two genuinely overlapping people (IoU ≥ 0.7) would be treated as one box.
- IDs are still per object, not per person, and restart each run (hence `run_id`).
- Not run against a live MongoDB; the new unique index applies only to new documents.
