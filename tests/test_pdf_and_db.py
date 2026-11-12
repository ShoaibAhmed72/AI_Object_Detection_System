import datetime, os, subprocess, sys, types, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import pdf_exporter  # noqa: E402

TODAY = datetime.date.today().strftime("%Y-%m-%d")

def rec(run, uid, name, tid, fs, ls, dur, mx=.8, av=.7):
    return {"run_id": run, "session_uid": uid, "object_name": name, "tracking_id": tid, "date": TODAY,
            "first_seen": fs, "last_seen": ls, "duration_seconds": dur, "max_confidence": mx, "avg_confidence": av}

class FakeDB:
    def __init__(self, docs): self.docs = docs
    def search_detections(self, run_id=None, date=None, limit=500, **k):
        return [d for d in self.docs if (not run_id or d.get("run_id") == run_id) and (not date or d["date"] == date)][:limit]

class T(unittest.TestCase):
    def test_pdf_only_current_run_deduped(self):
        docs = [rec("A", "u1", "Man", 1, "10:00:00", "10:01:00", 60), rec("A", "u1", "Man", 1, "10:00:00", "10:01:00", 60),
                rec("A", "u2", "Human face", 2, "10:00:05", "10:01:00", 55), rec("A", "u3", "Cup", 3, "10:00:05", "10:00:05.2", .4),
                rec("OLD", "u9", "Glasses", 29, "09:00:00", "09:00:03", 3),
                {"object_name": "Legacy", "tracking_id": 5, "date": TODAY, "duration_seconds": 9, "max_confidence": .5}]
        path = pdf_exporter.generate_objects_pdf(FakeDB(docs), filename_prefix="/tmp/pdftest", run_id="A")
        txt = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True).stdout
        self.assertIn("Confirmed sessions: 2", txt)
        self.assertNotIn("Glasses", txt); self.assertNotIn("Legacy", txt); self.assertNotIn("Cup", txt)
        self.assertEqual(txt.count("Man"), 1)
        for col in ("First Seen", "Last Seen", "Duration", "Avg Conf", "Max Conf"): self.assertIn(col, txt)
        print(txt)

    def test_empty_run_no_pdf(self):
        self.assertIsNone(pdf_exporter.generate_objects_pdf(FakeDB([]), run_id="Z"))

    def test_db_query_has_run_id(self):
        pm = types.ModuleType("pymongo"); pm.MongoClient = object; pm.DESCENDING = -1; pm.ASCENDING = 1
        pe = types.ModuleType("pymongo.errors"); pe.PyMongoError = Exception; pe.ServerSelectionTimeoutError = Exception
        sys.modules.update({"pymongo": pm, "pymongo.errors": pe})
        import database
        class Cur(list):
            def sort(s, *a): return s
            def limit(s, n): return s
        got = {}
        class Col:
            def find(s, q): got["q"] = q; return Cur()
        db = database.Database.__new__(database.Database); db.connected = True
        db.db = {database.config.COLLECTION_DETECTIONS: Col()}
        db.search_detections(run_id="A", date="2026-11-12")
        self.assertEqual(got["q"], {"date": "2026-11-12", "run_id": "A"})

if __name__ == "__main__":
    unittest.main()
