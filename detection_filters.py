"""
Pure-python helpers that validate a frame's tracked detections before they
become sessions. No heavy imports, so they are easy to test.

* suppress_duplicate_people: Open Images reports one body as Person AND Man
  (etc.) with near-identical boxes -> one person was getting several ids.
* eyewear_context: classifies a "Glasses" box as on_face / standalone /
  implausible using the detected face boxes of the same frame.
"""

import config

PERSON_CLASSES = {"person", "man", "woman", "boy", "girl"}
FACE_CLASSES = {"human face", "human head"}
EYEWEAR_CLASSES = {"glasses", "sunglasses"}
FACE_MIN_CONFIDENCE = 0.30


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def iou(a, b):
    i = _inter(a, b)
    u = box_area(a) + box_area(b) - i
    return i / u if u > 0 else 0.0


def coverage(a, b):
    """Fraction of box a that lies inside box b."""
    area = box_area(a)
    return _inter(a, b) / area if area > 0 else 0.0


def class_group(name):
    n = name.lower()
    if n in PERSON_CLASSES:
        return "person"
    if n in EYEWEAR_CLASSES:
        return "eyewear"
    return n


def suppress_duplicate_people(detections, iou_threshold=None):
    """Drops the younger of two person-class boxes that overlap >= threshold.
    Returns (kept_detections, number_dropped)."""
    thr = config.PERSON_DUPLICATE_IOU if iou_threshold is None else iou_threshold
    people = sorted((d for d in detections if d["class_name"].lower() in PERSON_CLASSES),
                    key=lambda d: d["track_id"])
    kept, dropped = [], set()
    for d in people:
        if any(iou(d["box"], k["box"]) >= thr for k in kept):
            dropped.add(id(d))
        else:
            kept.append(d)
    return [d for d in detections if id(d) not in dropped], len(dropped)


def eyewear_context(glasses, detections):
    """'on_face', 'standalone' or 'implausible' for one eyewear detection."""
    gb = glasses["box"]
    gw, gh = gb[2] - gb[0], gb[3] - gb[1]
    if gw <= 0 or gh <= 0:
        return "implausible"
    overlapping = False
    for f in detections:
        if f["class_name"].lower() not in FACE_CLASSES or f["confidence"] < FACE_MIN_CONFIDENCE:
            continue
        fb = f["box"]
        cov = coverage(gb, fb)
        if cov < config.EYEWEAR_STANDALONE_MAX_COVERAGE:
            continue
        overlapping = True
        fw, fh = fb[2] - fb[0], fb[3] - fb[1]
        if fw <= 0 or fh <= 0 or cov < config.EYEWEAR_FACE_MIN_COVERAGE:
            continue
        cy = ((gb[1] + gb[3]) / 2 - fb[1]) / fh
        lo_y, hi_y = config.EYEWEAR_CENTER_Y_RANGE
        lo_w, hi_w = config.EYEWEAR_WIDTH_RATIO_RANGE
        if (lo_y <= cy <= hi_y and lo_w <= gw / fw <= hi_w
                and gw / gh >= config.EYEWEAR_MIN_ASPECT):
            return "on_face"
    return "implausible" if overlapping else "standalone"
