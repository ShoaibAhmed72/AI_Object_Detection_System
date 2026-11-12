"""
MongoDB access layer. Every read/write in the project goes through this
module so the rest of the app never touches PyMongo directly.

Handles connection errors gracefully: if MongoDB is unreachable, the app
keeps running (the camera loop is never blocked) -- writes are simply
skipped and logged until the connection comes back.
"""

import datetime
import logging

from pymongo import MongoClient, DESCENDING, ASCENDING
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

import config

logger = logging.getLogger("database")


class Database:
    def __init__(self, uri=config.MONGO_URI, db_name=config.MONGO_DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False
        self._connect()

    # ------------------------------------------------------------------
    def _connect(self):
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self._ensure_indexes()
            self.connected = True
            logger.info("Connected to MongoDB at %s (db=%s)", self.uri, self.db_name)
        except (PyMongoError, ServerSelectionTimeoutError) as exc:
            self.connected = False
            logger.warning(
                "Could not connect to MongoDB (%s). The app will keep running, "
                "but nothing will be saved until the connection is available.",
                exc,
            )

    def _ensure_indexes(self):
        self.db[config.COLLECTION_DETECTIONS].create_index(
            [("object_name", ASCENDING), ("date", DESCENDING)]
        )
        self.db[config.COLLECTION_DETECTIONS].create_index([("tracking_id", ASCENDING)])
        self.db[config.COLLECTION_SCENES].create_index([("scene_name", ASCENDING), ("date", DESCENDING)])
        self.db[config.COLLECTION_RELATIONSHIPS].create_index([("date", DESCENDING)])
        # Additive indexes for run-scoped reports / duplicate protection. Old
        # documents without these fields are unaffected (sparse); a failure here
        # must never stop the app from connecting.
        try:
            self.db[config.COLLECTION_DETECTIONS].create_index([("run_id", ASCENDING)], sparse=True)
            self.db[config.COLLECTION_DETECTIONS].create_index(
                [("session_uid", ASCENDING)], unique=True, sparse=True)
        except PyMongoError as exc:
            logger.warning("Could not create run_id/session_uid indexes: %s", exc)

    def _reconnect_if_needed(self):
        if not self.connected:
            self._connect()
        return self.connected

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def save_detection_session(self, session: dict):
        """session keys: object_name, tracking_id, date, first_seen, last_seen,
        duration_seconds, max_confidence, avg_confidence"""
        if not self._reconnect_if_needed():
            logger.warning("Skipped saving detection session, DB offline: %s", session)
            return None
        try:
            doc = dict(session)
            doc["created_at"] = datetime.datetime.utcnow()
            return self.db[config.COLLECTION_DETECTIONS].insert_one(doc).inserted_id
        except PyMongoError as exc:
            logger.error("Failed to save detection session: %s", exc)
            self.connected = False
            return None

    def save_scene_session(self, session: dict):
        if not self._reconnect_if_needed():
            logger.warning("Skipped saving scene session, DB offline: %s", session)
            return None
        try:
            doc = dict(session)
            doc["created_at"] = datetime.datetime.utcnow()
            return self.db[config.COLLECTION_SCENES].insert_one(doc).inserted_id
        except PyMongoError as exc:
            logger.error("Failed to save scene session: %s", exc)
            self.connected = False
            return None

    def save_relationship_session(self, session: dict):
        if not self._reconnect_if_needed():
            logger.warning("Skipped saving relationship session, DB offline: %s", session)
            return None
        try:
            doc = dict(session)
            doc["created_at"] = datetime.datetime.utcnow()
            return self.db[config.COLLECTION_RELATIONSHIPS].insert_one(doc).inserted_id
        except PyMongoError as exc:
            logger.error("Failed to save relationship session: %s", exc)
            self.connected = False
            return None

    def update_live_state(self, state: dict):
        """Upserts a single document that the dashboard polls for 'current' info."""
        if not self._reconnect_if_needed():
            return None
        try:
            doc = dict(state)
            doc["updated_at"] = datetime.datetime.utcnow()
            return self.db[config.COLLECTION_LIVE_STATE].update_one(
                {"_id": "live"}, {"$set": doc}, upsert=True
            )
        except PyMongoError as exc:
            logger.error("Failed to update live state: %s", exc)
            self.connected = False
            return None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_live_state(self):
        if not self._reconnect_if_needed():
            return None
        try:
            return self.db[config.COLLECTION_LIVE_STATE].find_one({"_id": "live"})
        except PyMongoError as exc:
            logger.error("Failed to read live state: %s", exc)
            return None

    def search_detections(self, object_name=None, tracking_id=None, date=None,
                           min_duration=None, min_confidence=None, limit=200, run_id=None):
        if not self._reconnect_if_needed():
            return []
        query = {}
        if object_name:
            query["object_name"] = {"$regex": object_name, "$options": "i"}
        if tracking_id not in (None, ""):
            try:
                query["tracking_id"] = int(tracking_id)
            except (TypeError, ValueError):
                pass
        if date:
            query["date"] = date
        if run_id:
            query["run_id"] = run_id
        if min_duration:
            query["duration_seconds"] = {"$gte": float(min_duration)}
        if min_confidence:
            query["max_confidence"] = {"$gte": float(min_confidence)}
        try:
            cursor = self.db[config.COLLECTION_DETECTIONS].find(query).sort(
                "created_at", DESCENDING
            ).limit(limit)
            return list(cursor)
        except PyMongoError as exc:
            logger.error("Search failed: %s", exc)
            return []

    def search_scenes(self, scene_name=None, date=None, limit=200):
        if not self._reconnect_if_needed():
            return []
        query = {}
        if scene_name:
            query["scene_name"] = {"$regex": scene_name, "$options": "i"}
        if date:
            query["date"] = date
        try:
            return list(
                self.db[config.COLLECTION_SCENES].find(query).sort("created_at", DESCENDING).limit(limit)
            )
        except PyMongoError as exc:
            logger.error("Search failed: %s", exc)
            return []

    def search_relationships(self, objects=None, date=None, limit=200):
        if not self._reconnect_if_needed():
            return []
        query = {}
        if objects:
            query["objects"] = {"$all": objects}
        if date:
            query["date"] = date
        try:
            return list(
                self.db[config.COLLECTION_RELATIONSHIPS].find(query).sort("created_at", DESCENDING).limit(limit)
            )
        except PyMongoError as exc:
            logger.error("Search failed: %s", exc)
            return []

    def get_statistics(self):
        if not self._reconnect_if_needed():
            return {}
        try:
            det = self.db[config.COLLECTION_DETECTIONS]
            scenes = self.db[config.COLLECTION_SCENES]

            total_sessions = det.count_documents({})
            unique_objects = len(det.distinct("object_name"))
            longest_doc = next(det.find().sort("duration_seconds", DESCENDING).limit(1), None)

            most_detected_pipeline = [
                {"$group": {"_id": "$object_name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 1},
            ]
            most_detected = list(det.aggregate(most_detected_pipeline))

            total_scene_sessions = scenes.count_documents({})
            most_common_scene_pipeline = [
                {"$group": {"_id": "$scene_name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 1},
            ]
            most_common_scene = list(scenes.aggregate(most_common_scene_pipeline))
            longest_scene = next(scenes.find().sort("duration_seconds", DESCENDING).limit(1), None)

            return {
                "total_detection_sessions": total_sessions,
                "unique_objects": unique_objects,
                "most_detected_object": most_detected[0]["_id"] if most_detected else None,
                "longest_detection_seconds": longest_doc["duration_seconds"] if longest_doc else 0,
                "total_scene_sessions": total_scene_sessions,
                "most_common_scene": most_common_scene[0]["_id"] if most_common_scene else None,
                "longest_scene_name": longest_scene["scene_name"] if longest_scene else None,
                "longest_scene_seconds": longest_scene["duration_seconds"] if longest_scene else 0,
            }
        except PyMongoError as exc:
            logger.error("Failed to compute statistics: %s", exc)
            return {}

    def get_relationship_totals(self, limit=20):
        """Aggregate total time each object combination has spent together."""
        if not self._reconnect_if_needed():
            return []
        try:
            pipeline = [
                {"$group": {"_id": "$objects", "total_seconds": {"$sum": "$duration_seconds"},
                             "sessions": {"$sum": 1}}},
                {"$sort": {"total_seconds": -1}},
                {"$limit": limit},
            ]
            return list(self.db[config.COLLECTION_RELATIONSHIPS].aggregate(pipeline))
        except PyMongoError as exc:
            logger.error("Failed to aggregate relationships: %s", exc)
            return []

    def clear_history(self):
        if not self._reconnect_if_needed():
            return False
        try:
            self.db[config.COLLECTION_DETECTIONS].delete_many({})
            self.db[config.COLLECTION_SCENES].delete_many({})
            self.db[config.COLLECTION_RELATIONSHIPS].delete_many({})
            return True
        except PyMongoError as exc:
            logger.error("Failed to clear history: %s", exc)
            return False

    def close(self):
        if self.client:
            self.client.close()
