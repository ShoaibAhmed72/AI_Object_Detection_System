"""
Camera/video diagnostic -- runs the real detector + session logic WITHOUT
touching MongoDB and prints how stable tracking was. Use it for the test
scenarios (one person, two people, glasses, phone):

    python diagnose.py --seconds 60            # webcam
    python diagnose.py --source clip.mp4 --show
"""

import argparse
import collections
import datetime
import time

import cv2

import config
from detector import ObjectDetector
from tracker import ObjectSessionManager


class _MemoryDB:
    def __init__(self):
        self.docs = []

    def save_detection_session(self, doc):
        self.docs.append(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(config.CAMERA_INDEX))
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source}")
    detector, db = ObjectDetector(), _MemoryDB()
    mgr = ObjectSessionManager(db, run_id="diagnose")
    raw_ids = collections.defaultdict(set)
    frames, start = 0, time.time()

    while time.time() - start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        dets = detector.track(frame)
        for d in dets:
            raw_ids[d["class_name"]].add(d["track_id"])
        shown = mgr.update(dets, timestamp=datetime.datetime.now())
        if args.show:
            for d in shown:
                x1, y1, x2, y2 = map(int, d["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f'{d["class_name"]} {d["confidence"]:.2f} ID:{d["track_id"]}',
                            (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            cv2.imshow("diagnose", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    mgr.close_all()
    elapsed = time.time() - start
    print(f"\n{frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-9):.1f} FPS)")
    print("\nPer class: raw tracker ids seen  ->  confirmed sessions saved")
    sessions = collections.Counter(d["object_name"] for d in db.docs)
    for cls in sorted(set(raw_ids) | set(sessions)):
        print(f"  {cls:<20} {len(raw_ids.get(cls, ())):>4} ids  ->  {sessions.get(cls, 0):>3} sessions")
    print("\nSessions:")
    for d in db.docs:
        print(f"  {d['object_name']:<16} id={d['tracking_id']:<4} ids={d['track_ids']} "
              f"{d['first_seen']}-{d['last_seen']} {d['duration_seconds']}s "
              f"avg={d['avg_confidence']:.2f} max={d['max_confidence']:.2f}")
    print("\nRejected sessions:", dict(mgr.session_rejections))
    print("Rejected detections:", dict(mgr.frame_rejections))


if __name__ == "__main__":
    main()
