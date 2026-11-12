"""
Converts frame-by-frame tracked detections (detector.py assigns ids via
ByteTrack) into discrete "detection sessions": one continuous appearance of
an object = one MongoDB record, never one record per frame.

A detection only becomes a session after temporal confirmation (enough
credible frames, over enough time, with a steady hit ratio), so one-frame or
flickering false positives are never saved. Durations are always
last_seen - first_seen taken from real timestamps.
"""

import collections
import datetime
import logging
import uuid

import config
from detection_filters import (
    EYEWEAR_CLASSES, class_group, eyewear_context, iou, suppress_duplicate_people,
)

logger = logging.getLogger("tracker")


def class_rules(class_name):
    rules = {
        "min_confidence": config.DETECTION_MIN_CONFIDENCE,
        "min_hits": config.CONFIRM_MIN_HITS,
        "min_seconds": config.CONFIRM_MIN_SECONDS,
        "min_hit_ratio": config.CONFIRM_MIN_HIT_RATIO,
        "min_avg_confidence": 0.0,
    }
    rules.update(config.CLASS_RULES.get(class_name.lower(), {}))
    return rules


class _ActiveSession:
    def __init__(self, track_id, det, timestamp, frame_idx):
        self.key = track_id              # canonical (first) tracker id
        self.track_ids = [track_id]      # every tracker id merged into this session
        self.uid = uuid.uuid4().hex
        self.class_votes = collections.defaultdict(float)
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.first_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.hits = 0
        self.standalone_hits = 0
        self.conf_sum = 0.0
        self.max_confidence = 0.0
        self.missed_frames = 0
        self.confirmed = False
        self.last_box = det["box"]

    @property
    def class_name(self):
        return max(self.class_votes, key=self.class_votes.get)

    @property
    def avg_confidence(self):
        return self.conf_sum / self.hits if self.hits else 0.0

    def update(self, det, timestamp, frame_idx, strong, context=None):
        if strong:   # only credible frames vote for the class / enter the statistics
            self.class_votes[det["class_name"]] += det["confidence"]
            self.hits += 1
            self.conf_sum += det["confidence"]
            self.max_confidence = max(self.max_confidence, det["confidence"])
            if context == "standalone":
                self.standalone_hits += 1
        self.last_box = det["box"]
        self.last_seen = timestamp
        self.last_seen_frame = frame_idx
        self.missed_frames = 0

    def _requirements(self):
        r = class_rules(self.class_name)
        if self.standalone_hits * 2 > self.hits:   # mostly not on a face
            for k in ("min_hits", "min_seconds", "min_avg_confidence"):
                r[k] = r.get("standalone_" + k, r[k])
        return r

    def failed_requirement(self, frame_idx):
        """None if confirmed-worthy, else a short reason string."""
        r = self._requirements()
        span = (self.last_seen - self.first_seen).total_seconds()
        ratio = self.hits / max(1, frame_idx - self.first_frame + 1)
        if self.hits < r["min_hits"]:
            return f"only {self.hits}/{r['min_hits']} credible frames"
        if span < r["min_seconds"]:
            return f"visible {span:.2f}s < {r['min_seconds']}s"
        if ratio < r["min_hit_ratio"]:
            return f"unstable (hit ratio {ratio:.2f} < {r['min_hit_ratio']})"
        if self.avg_confidence < r["min_avg_confidence"]:
            return f"avg confidence {self.avg_confidence:.2f} < {r['min_avg_confidence']}"
        return None

    def to_document(self, run_id):
        duration = (self.last_seen - self.first_seen).total_seconds()
        return {
            "object_name": self.class_name,
            "tracking_id": self.key,
            "track_ids": list(self.track_ids),
            "date": self.first_seen.strftime("%Y-%m-%d"),
            "first_seen": self.first_seen.strftime("%H:%M:%S"),
            "last_seen": self.last_seen.strftime("%H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "max_confidence": round(self.max_confidence, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "hits": self.hits,
            "session_uid": self.uid,
            "run_id": run_id,
            "confirmed": True,
        }


class ObjectSessionManager:
    """Key = canonical tracker id (new ids re-associated to a lost session map to it)."""

    def __init__(self, db, run_id=None):
        self.db = db
        self.run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._active = {}          # canonical id -> _ActiveSession
        self._alias = {}           # raw tracker id -> canonical id
        self._frame_idx = 0
        self.saved_sessions = 0
        self.frame_rejections = collections.Counter()    # per-detection reasons
        self.session_rejections = collections.Counter()  # per-session reasons

    # ------------------------------------------------------------------
    def update(self, detections, timestamp=None):
        """Feeds one frame. Returns the detections that belong to CONFIRMED
        objects (ids/classes stabilised) -- use these for drawing."""
        timestamp = timestamp or datetime.datetime.now()
        self._frame_idx += 1
        frame = self._frame_idx

        dets, n_dup = suppress_duplicate_people(detections)
        if n_dup:
            self.frame_rejections["duplicate_person_box"] += n_dup

        seen, matched, new_dets = set(), [], []
        for det in dets:
            ctx = None
            if det["class_name"].lower() in EYEWEAR_CLASSES:
                ctx = eyewear_context(det, dets)
                if ctx == "implausible":
                    self.frame_rejections["eyewear_implausible_geometry"] += 1
                    continue
            raw = det["track_id"]
            key = self._alias.get(raw, raw)
            sess = self._active.get(key)
            conf = det["confidence"]
            min_conf = class_rules(det["class_name"])["min_confidence"]
            strong = conf >= min_conf
            if sess is not None:
                if key in seen:
                    continue   # two tracker ids for the same session in one frame
                if strong or (sess.confirmed and conf >= min_conf * config.KEEP_CONFIDENCE_RATIO):
                    sess.update(det, timestamp, frame, strong, ctx)
                    seen.add(key)
                    matched.append((sess, det))
            elif strong:
                new_dets.append((det, ctx))
            else:
                self.frame_rejections["below_min_confidence"] += 1

        self._start_or_reassociate(new_dets, seen, matched, timestamp, frame)

        # confirmation
        for sess, _ in matched:
            if not sess.confirmed and sess.failed_requirement(frame) is None:
                sess.confirmed = True
                logger.info("Confirmed %s (track %s) after %d credible frames",
                            sess.class_name, sess.key, sess.hits)

        # ageing
        for key, sess in list(self._active.items()):
            if sess.last_seen_frame != frame:
                sess.missed_frames += 1
                gap = (timestamp - sess.last_seen).total_seconds()
                if sess.missed_frames >= config.MAX_MISSED_FRAMES and gap >= config.SESSION_GRACE_SECONDS:
                    self._close_session(key)

        return [
            {**det, "track_id": sess.key, "class_name": sess.class_name}
            for sess, det in matched if sess.confirmed
        ]

    def _start_or_reassociate(self, new_dets, seen, matched, timestamp, frame):
        lost = [s for k, s in self._active.items()
                if s.confirmed and k not in seen
                and (timestamp - s.last_seen).total_seconds() <= config.REID_MAX_GAP_SECONDS]
        eligible = []
        claimants = collections.Counter()
        for det, ctx in new_dets:
            el = [s for s in lost
                  if class_group(s.class_name) == class_group(det["class_name"])
                  and iou(s.last_box, det["box"]) >= config.REID_MIN_IOU]
            eligible.append(el)
            for s in el:
                claimants[s.key] += 1

        for (det, ctx), el in zip(new_dets, eligible):
            raw = det["track_id"]
            if len(el) == 1 and claimants[el[0].key] == 1:
                sess = el[0]
                self._alias[raw] = sess.key
                sess.track_ids.append(raw)
                logger.info("Re-associated new track %s with lost session %s (%s)",
                            raw, sess.key, sess.class_name)
            else:
                sess = _ActiveSession(raw, det, timestamp, frame)
                self._active[raw] = sess
            sess.update(det, timestamp, frame, True, ctx)
            seen.add(sess.key)
            matched.append((sess, det))

    # ------------------------------------------------------------------
    def _close_session(self, key):
        sess = self._active.pop(key, None)
        if sess is None:
            return
        for raw in sess.track_ids:
            self._alias.pop(raw, None)
        duration = (sess.last_seen - sess.first_seen).total_seconds()
        if not sess.confirmed:
            reason = sess.failed_requirement(sess.last_seen_frame) or "not confirmed"
            label = {"only": "too few credible frames", "visible": "too brief",
                     "unstable": "unstable / flickering", "avg": "low average confidence"}
            self.session_rejections[label.get(reason.split(" ")[0], "not confirmed")] += 1
            logger.info("Rejected %s (track %s): %s", sess.class_name, sess.key, reason)
            return
        if duration < config.MIN_SESSION_DURATION_SECONDS:
            self.session_rejections["too short"] += 1
            return
        doc = sess.to_document(self.run_id)
        self.db.save_detection_session(doc)
        self.saved_sessions += 1
        logger.info("Saved detection session: %s", doc)

    def current_objects(self):
        """Confirmed objects visible right now, for the live overlay/dashboard."""
        return [
            {"track_id": s.key, "class_name": s.class_name, "confidence": s.max_confidence}
            for s in self._active.values()
            if s.confirmed and s.missed_frames == 0
        ]

    def close_all(self):
        """Flush any still-open sessions (e.g. when the camera is stopped)."""
        for key in list(self._active.keys()):
            self._close_session(key)
