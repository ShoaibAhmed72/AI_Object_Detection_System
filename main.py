"""
Entry point: opens the webcam, runs YOLO detection + ByteTrack tracking on
every frame, runs Places365 scene classification periodically, manages
object/scene/relationship sessions, writes finished sessions to MongoDB,
and shows a live annotated preview window.

    START CAMERA        -> run:  python main.py
    STOP CAMERA / EXIT   -> press "q" or ESC in the preview window, or Ctrl+C
    CLEAR HISTORY        -> press "c" in the preview window (asks to confirm
                             in the terminal)
    VIEW HISTORY /
    SCENE HISTORY /
    RELATIONSHIPS / SEARCH -> run:  streamlit run dashboard.py  (separate terminal)
"""

import datetime
import logging
import time
from pdf_exporter import generate_objects_pdf

import cv2

import config
from database import Database
from detector import ObjectDetector
from tracker import ObjectSessionManager
from relationship import RelationshipManager
from scene_classifier import SceneClassifier, SceneSessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("main")


def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {config.CAMERA_INDEX}. "
            "Check that a webcam is connected and not already in use by another app."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def draw_overlay(frame, detections, scene_name, scene_conf, current_group, fps):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f'{det["class_name"]} {det["confidence"] * 100:.0f}% ID:{det["track_id"]}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    y = 25
    if scene_name:
        cv2.putText(frame, f"Scene: {scene_name} ({scene_conf * 100:.0f}%)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y += 30
    if current_group:
        cv2.putText(frame, "Together: " + " + ".join(current_group), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main():
    logger.info("Starting %s", config.PROJECT_NAME)

    db = Database()
    detector = ObjectDetector()
    scene_classifier = SceneClassifier()

    object_sessions = ObjectSessionManager(db)
    relationship_manager = RelationshipManager(db)
    scene_sessions = SceneSessionManager(db)

    try:
        cap = open_camera()
    except RuntimeError as exc:
        logger.error(str(exc))
        return

    frame_count = 0
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Failed to read frame from camera. Stopping.")
                break

            # Un-mirror the webcam feed so left-to-right movements match naturally
            frame = cv2.flip(frame, 1)

            frame_count += 1
            now = datetime.datetime.now()

            detections = detector.track(frame)
            # Returns only detections of CONFIRMED objects, with stabilised ids/classes
            detections = object_sessions.update(detections, timestamp=now)

            current_objects = object_sessions.current_objects()
            relationship_manager.update(current_objects, timestamp=now)

            if frame_count % config.SCENE_CLASSIFY_EVERY_N_FRAMES == 0:
                scene_name, scene_conf = scene_classifier.predict(frame)
                scene_sessions.update(scene_name, scene_conf, timestamp=now)

            elapsed = time.time() - prev_time
            prev_time = time.time()
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed)

            current_group = relationship_manager.current_group()
            frame = draw_overlay(
                frame, detections, scene_sessions.current_scene,
                scene_sessions.current_confidence, current_group, fps,
            )
            cv2.imshow(config.WINDOW_NAME, frame)

            db.update_live_state({
                "scene": scene_sessions.current_scene,
                "scene_confidence": scene_sessions.current_confidence,
                "objects": [o["class_name"] for o in current_objects],
                "objects_together": current_group,
                "fps": round(fps, 1),
                "timestamp": now.isoformat(),
            })

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                logger.info("Stop requested by user.")
                break
            elif key == ord("c"):
                confirm = input("Clear ALL history from MongoDB? (yes/no): ")
                if confirm.strip().lower() == "yes":
                    db.clear_history()
                    logger.info("History cleared.")

    # except KeyboardInterrupt:
    #     logger.info("Interrupted by user (Ctrl+C).")
    # finally:
    #     object_sessions.close_all()
    #     relationship_manager.close_all()
    #     scene_sessions.close_all()
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     db.close()
    #     logger.info("Camera stopped safely. All open sessions were saved.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        object_sessions.close_all()
        relationship_manager.close_all()
        scene_sessions.close_all()
        
        # --- ADD THIS LINE TO EXPORT PDF AUTOMATICALLY ON EXIT ---
        try:
            pdf_path = generate_objects_pdf(db, run_id=object_sessions.run_id)
        except Exception as exc:  # never let report problems skip camera cleanup
            pdf_path = None
            logger.error("PDF report failed: %s", exc)

        cap.release()
        cv2.destroyAllWindows()
        db.close()
        if pdf_path:
            logger.info("Camera stopped safely. PDF report: %s", pdf_path)
        else:
            logger.info("Camera stopped safely. No confirmed detections for this run -> no PDF.")
        logger.info("Run %s: %d sessions saved, rejected sessions: %s, rejected detections: %s",
                    object_sessions.run_id, object_sessions.saved_sessions,
                    dict(object_sessions.session_rejections), dict(object_sessions.frame_rejections))
    # finally:
    #     object_sessions.close_all()
    #     relationship_manager.close_all()
    #     scene_sessions.close_all()
        
    #     # --- ADD THIS LINE TO EXPORT PDF AUTOMATICALLY ON EXIT ---
    #     generate_objects_pdf(db)
        
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     db.close()
    #     logger.info("Camera stopped safely. PDF report generated in 'results/' folder.")


if __name__ == "__main__":
    main()


    