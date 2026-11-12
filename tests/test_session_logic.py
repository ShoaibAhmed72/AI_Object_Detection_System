import datetime, os, random, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

from tracker import ObjectSessionManager  # noqa: E402

T0 = datetime.datetime(2026, 11, 12, 10, 0, 0)


class FakeDB:
    def __init__(self): self.docs = []
    def save_detection_session(self, d): self.docs.append(d)


def det(tid, cls, conf, box):
    return {"track_id": tid, "class_name": cls, "confidence": conf, "box": tuple(map(float, box))}


def run(frames, fps=10, run_id="r1"):
    """frames: list of detection lists (one per frame)."""
    db = FakeDB(); m = ObjectSessionManager(db, run_id=run_id)
    for i, f in enumerate(frames):
        m.update(f, T0 + datetime.timedelta(seconds=i / fps))
    m.close_all()
    return db.docs, m


FACE = (300, 100, 400, 220)
GLASSES_OK = (305, 130, 395, 170)       # eye region of FACE


class T(unittest.TestCase):
    def test_one_person_id_switch_and_conf_dips_single_session(self):
        random.seed(1); frames = []
        for i in range(200):                       # 20 s @10fps
            tid = 1 if i < 80 else 7               # id switch at 8 s
            if 80 <= i < 84: frames.append([]); continue   # 0.4 s gap
            frames.append([det(tid, "Man", random.uniform(0.40, 0.8), (100, 100, 300, 500))])
        docs, _ = run(frames)
        self.assertEqual(len(docs), 1, docs)
        self.assertEqual(docs[0]["track_ids"], [1, 7]); self.assertEqual(docs[0]["tracking_id"], 1)

    def test_two_people_never_merge(self):
        a, b = (50, 100, 250, 500), (400, 100, 600, 500)
        frames = []
        for i in range(150):
            f = [det(2, "Person", .8, b)]
            if not (60 <= i < 64): f.append(det(1 if i < 60 else 3, "Person", .8, a))
            frames.append(f)
        docs, _ = run(frames)
        self.assertEqual(len(docs), 2)
        self.assertEqual(sorted(d["tracking_id"] for d in docs), [1, 2])

    def test_new_person_elsewhere_is_not_merged(self):
        frames = [[det(1, "Person", .8, (50, 100, 250, 500))] for _ in range(50)]
        frames += [[det(2, "Person", .8, (400, 100, 600, 500))] for _ in range(50)]
        docs, _ = run(frames)
        self.assertEqual(len(docs), 2)

    def test_duplicate_person_boxes_one_session(self):
        frames = [[det(1, "Man", .8, (100, 100, 300, 500)), det(2, "Person", .7, (102, 98, 301, 502))] for _ in range(60)]
        docs, _ = run(frames)
        self.assertEqual(len(docs), 1)

    def test_one_frame_and_flicker_never_saved(self):
        frames = [[] for _ in range(100)]
        frames[10] = [det(5, "Clothing", .9, (0, 0, 50, 50))]
        for i in range(30, 100, 7):                 # sporadic flicker
            frames[i] = [det(6, "Glasses", .6, (100, 100, 150, 130))]
        docs, m = run(frames)
        self.assertEqual(docs, [])

    def test_worn_glasses_saved_bare_face_flicker_not(self):
        ok = [[det(1, "Human face", .9, FACE), det(2, "Glasses", .7, GLASSES_OK)] for _ in range(40)]
        docs, _ = run(ok)
        self.assertEqual([d["object_name"] for d in docs if d["object_name"] == "Glasses"], ["Glasses"])
        random.seed(3); bare = []
        for i in range(100):                        # FP on a bare face: appears ~30% of frames
            f = [det(1, "Human face", .9, FACE)]
            if random.random() < .3: f.append(det(9, "Glasses", .58, GLASSES_OK))
            bare.append(f)
        docs, _ = run(bare)
        self.assertNotIn("Glasses", [d["object_name"] for d in docs])

    def test_glasses_wrong_place_on_face_rejected(self):
        chin = (310, 190, 390, 215)
        docs, m = run([[det(1, "Human face", .9, FACE), det(2, "Glasses", .8, chin)] for _ in range(50)])
        self.assertNotIn("Glasses", [d["object_name"] for d in docs])
        self.assertGreater(m.frame_rejections["eyewear_implausible_geometry"], 0)

    def test_standalone_glasses_need_more_evidence(self):
        box = (500, 400, 580, 430)
        short, _ = run([[det(1, "Glasses", .7, box)] for _ in range(12)])      # 1.2 s
        self.assertEqual(short, [])
        long_, _ = run([[det(1, "Glasses", .7, box)] for _ in range(40)])      # 4 s
        self.assertEqual(len(long_), 1)

    def test_low_conf_distant_phone_kept(self):
        docs, _ = run([[det(1, "Mobile phone", .33, (10, 10, 40, 60))] for _ in range(30)])
        self.assertEqual(len(docs), 1)
        # same confidence for a generic class is NOT trusted
        docs, _ = run([[det(1, "Cup", .33, (10, 10, 40, 60))] for _ in range(30)])
        self.assertEqual(docs, [])

    def test_dropout_same_id_single_session_exact_duration(self):
        frames = [[det(1, "Laptop", .8, (0, 0, 200, 100))] for _ in range(50)]
        frames += [[] for _ in range(15)]                       # 1.5 s missing
        frames += [[det(1, "Laptop", .8, (0, 0, 200, 100))] for _ in range(50)]
        docs, _ = run(frames)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["duration_seconds"], 11.4)     # frame 0 -> frame 114 at 10 fps (real timestamps)
        self.assertEqual(docs[0]["first_seen"], "10:00:00"); self.assertEqual(docs[0]["last_seen"], "10:00:11")

    def test_unique_session_ids_and_run_id(self):
        docs, _ = run([[det(1, "Laptop", .8, (0, 0, 9, 9)), det(2, "Cup", .8, (50, 50, 90, 90))] for _ in range(30)], run_id="runX")
        self.assertEqual(len({d["session_uid"] for d in docs}), 2)
        self.assertTrue(all(d["run_id"] == "runX" for d in docs))

    def test_returned_detections_only_confirmed(self):
        db = FakeDB(); m = ObjectSessionManager(db)
        out = [m.update([det(1, "Laptop", .8, (0, 0, 9, 9))], T0 + datetime.timedelta(seconds=i * .1)) for i in range(6)]
        self.assertEqual(out[0], []); self.assertEqual(len(out[-1]), 1)


if __name__ == "__main__":
    unittest.main()
