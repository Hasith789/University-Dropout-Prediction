"""
database.py - PyMongo Database Layer for University Dropout Prediction System
Course: CSE494 Intelligent NoSQL Databases | Academic Project

Implements native PyMongo database operations:
- Connection pooling and verification (ping command)
- Collection indexing (unique student_id, Target, prediction_timestamp)
- Student CRUD operations (find, insert_one, update_one, delete_one)
- Prediction audit logging with BSON Date timestamps
- Multi-stage Aggregation pipelines with zero-division safeguards
- Query execution plan analysis (explain)
"""

import os
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List, Optional
from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient

# Load local environment variables from .env
load_dotenv(override=True)


def get_connection_config(secrets: Optional[Any] = None) -> Tuple[str, str]:
    """
    Retrieve MongoDB URI and Database Name supporting Streamlit Cloud secrets and local .env.
    Never exposes credentials in logs or public code.
    """
    uri = None
    db_name = None

    # 1. Streamlit Cloud Secrets (st.secrets)
    if secrets is not None:
        try:
            if "MONGODB_URI" in secrets:
                uri = str(secrets["MONGODB_URI"]).strip()
            if "MONGODB_DATABASE" in secrets:
                db_name = str(secrets["MONGODB_DATABASE"]).strip()
        except Exception:
            pass

    # 2. Local Environment Variables (.env / OS)
    if not uri:
        uri = os.getenv("MONGODB_URI")
    if not db_name:
        db_name = os.getenv("MONGODB_DATABASE")

    # 3. Safe Fallback Default
    if not uri:
        uri = "mongodb://localhost:27017"
    if not db_name:
        db_name = "university_dropout_db"

    return uri, db_name


def create_mongo_client(uri: str, timeout_ms: int = 3000) -> MongoClient:
    """
    Instantiate a PyMongo MongoClient configured with proper connection timeouts
    and TLS certificates for MongoDB Atlas.
    """
    kwargs: Dict[str, Any] = {
        "serverSelectionTimeoutMS": timeout_ms,
        "connectTimeoutMS": timeout_ms,
    }
    # Configure TLS certificates for Atlas SRV clusters
    if "mongodb.net" in uri or "+srv" in uri:
        try:
            import certifi
            kwargs["tlsCAFile"] = certifi.where()
        except ImportError:
            pass

    return MongoClient(uri, **kwargs)


def verify_connection(client: MongoClient, uri: str) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    Verify live MongoDB connection using the administrative ping command.
    Returns: (is_connected, host_type, server_version, error_message)
    """
    is_atlas = "mongodb.net" in uri or "+srv" in uri
    host_type = "MongoDB Atlas" if is_atlas else "Local MongoDB"

    try:
        # Administrative ping command
        client.admin.command("ping")
        server_info = client.server_info()
        version = server_info.get("version", "Unknown")
        return True, host_type, version, None
    except Exception as e:
        err = str(e)
        if "TLSV1_ALERT_INTERNAL_ERROR" in err or "SSL handshake failed" in err:
            err = (
                "Atlas SSL handshake failed. In MongoDB Atlas, this occurs when the connecting IP address "
                "is not whitelisted. In your MongoDB Atlas Console, go to Network Access -> Add IP Address "
                "-> Allow Access from Anywhere (0.0.0.0/0)."
            )
        elif "localhost:27017" in uri and ("[Errno 111]" in err or "Connection refused" in err):
            err = (
                "Cannot connect to local MongoDB. If running on Streamlit Cloud, add your "
                "MongoDB Atlas connection string in Streamlit App Settings > Secrets."
            )
        return False, host_type, None, err


def ensure_database_indexes(db: Any) -> Dict[str, List[str]]:
    """
    Create necessary indexes on students and predictions collections using create_index().
    """
    students_indexes = []
    # 1. Unique index on student_id to prevent duplicates
    i1 = db.students.create_index([("student_id", pymongo.ASCENDING)], unique=True, name="idx_unique_student_id")
    students_indexes.append(i1)

    # 2. Index on Target for outcome grouping
    i2 = db.students.create_index([("Target", pymongo.ASCENDING)], name="idx_students_target")
    students_indexes.append(i2)

    # 3. Index on Course for degree program filtering
    i3 = db.students.create_index([("Course", pymongo.ASCENDING)], name="idx_students_course")
    students_indexes.append(i3)

    predictions_indexes = []
    # 4. Reverse-chronological index for prediction audit retrieval
    p1 = db.predictions.create_index([("prediction_timestamp", pymongo.DESCENDING)], name="idx_pred_timestamp")
    predictions_indexes.append(p1)

    # 5. Index on student_id in predictions collection
    p2 = db.predictions.create_index([("student_id", pymongo.ASCENDING)], name="idx_pred_student_id")
    predictions_indexes.append(p2)

    return {
        "students": students_indexes,
        "predictions": predictions_indexes,
    }


# ==============================================================================
# DASHBOARD METRICS & AGGREGATIONS (PYMONGO)
# ==============================================================================
def count_students(db: Any, query: Optional[Dict[str, Any]] = None) -> int:
    """Return count of student documents using count_documents()."""
    return db.students.count_documents(query or {})


def count_predictions(db: Any, query: Optional[Dict[str, Any]] = None) -> int:
    """Return count of prediction documents using count_documents()."""
    return db.predictions.count_documents(query or {})


def get_academic_outcome_distribution(db: Any) -> List[Dict[str, Any]]:
    """
    Execute PyMongo aggregation pipeline to count students grouped by academic outcome.
    Stages: $group -> $sort
    """
    pipeline = [
        {"$group": {"_id": "$Target", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    raw = list(db.students.aggregate(pipeline))
    return [{"outcome": item["_id"], "count": item["count"]} for item in raw if item.get("_id")]


def get_top_courses_dropout_rate(db: Any, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Execute PyMongo aggregation pipeline to compute dropout rate per course.
    Guards against division-by-zero using $cond.
    Stages: $group -> $project -> $sort -> $limit
    """
    pipeline = [
        {
            "$group": {
                "_id": "$Course",
                "total": {"$sum": 1},
                "dropouts": {
                    "$sum": {"$cond": [{"$eq": ["$Target", "Dropout"]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "course_id": "$_id",
                "total_students": "$total",
                "dropouts": "$dropouts",
                "dropout_rate": {
                    "$cond": [
                        {"$eq": ["$total", 0]},
                        0.0,
                        {
                            "$round": [
                                {"$multiply": [{"$divide": ["$dropouts", "$total"]}, 100]},
                                1,
                            ]
                        },
                    ]
                },
                "_id": 0,
            }
        },
        {"$sort": {"dropout_rate": -1}},
        {"$limit": limit},
    ]
    return list(db.students.aggregate(pipeline))


def get_prediction_outcome_distribution(db: Any) -> List[Dict[str, Any]]:
    """
    Execute PyMongo aggregation pipeline to count predictions grouped by predicted outcome.
    """
    pipeline = [
        {"$group": {"_id": "$predicted_outcome", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    raw = list(db.predictions.aggregate(pipeline))
    return [{"outcome": item["_id"], "count": item["count"]} for item in raw if item.get("_id")]


# ==============================================================================
# STUDENT CRUD OPERATIONS (PYMONGO)
# ==============================================================================
def find_students(db: Any, query: Optional[Dict[str, Any]] = None, limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve students matching query using native find() and limit()."""
    return list(db.students.find(query or {}).limit(limit))


def get_student_by_id(db: Any, student_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single student document using native find_one()."""
    return db.students.find_one({"student_id": student_id})


def insert_student(db: Any, student_data: Dict[str, Any]) -> Tuple[bool, str, Optional[Any]]:
    """
    Validate and insert a student document using insert_one().
    Returns: (success, message, inserted_id)
    """
    student_id = student_data.get("student_id")
    if not student_id or not str(student_id).strip():
        return False, "Student ID cannot be empty.", None

    if db.students.find_one({"student_id": student_id}):
        return False, f"Student record with ID '{student_id}' already exists in database.", None

    try:
        res = db.students.insert_one(student_data)
        return True, "Student record successfully inserted.", res.inserted_id
    except Exception as e:
        return False, f"Failed to insert student record: {e}", None


def update_student(db: Any, student_id: str, update_fields: Dict[str, Any]) -> Tuple[bool, str, int]:
    """
    Update a student record using native update_one() with $set operator.
    Returns: (success, message, modified_count)
    """
    try:
        res = db.students.update_one(
            {"student_id": student_id},
            {"$set": update_fields},
        )
        if res.matched_count == 0:
            return False, f"No student found with ID '{student_id}'.", 0
        return True, f"Updated {res.modified_count} student record(s).", res.modified_count
    except Exception as e:
        return False, f"Failed to update student record: {e}", 0


def delete_student(db: Any, student_id: str) -> Tuple[bool, str, int]:
    """
    Delete a student record using native delete_one().
    Returns: (success, message, deleted_count)
    """
    try:
        res = db.students.delete_one({"student_id": student_id})
        if res.deleted_count == 0:
            return False, f"No student found with ID '{student_id}'.", 0
        return True, f"Deleted {res.deleted_count} student record.", res.deleted_count
    except Exception as e:
        return False, f"Failed to delete student record: {e}", 0


# ==============================================================================
# PREDICTION STORAGE & AUDIT HISTORY (PYMONGO)
# ==============================================================================
def validate_prediction_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate prediction document structure before insertion."""
    required = [
        "student_id",
        "predicted_outcome",
        "class_probabilities",
        "model_name",
        "prediction_timestamp",
        "risk_tier",
    ]
    for field in required:
        if field not in payload or payload[field] is None:
            return False, f"Missing required field: '{field}'"

    if not isinstance(payload["student_id"], str) or not payload["student_id"].strip():
        return False, "student_id must be a non-empty string."

    if payload["predicted_outcome"] not in ["Dropout", "Enrolled", "Graduate"]:
        return False, f"Invalid predicted_outcome: {payload['predicted_outcome']}"

    if not isinstance(payload["class_probabilities"], dict):
        return False, "class_probabilities must be a dictionary."

    for cls in ["Dropout", "Enrolled", "Graduate"]:
        if cls not in payload["class_probabilities"]:
            return False, f"class_probabilities missing probability for '{cls}'."

    if not isinstance(payload["prediction_timestamp"], datetime):
        return False, "prediction_timestamp must be a valid datetime instance."

    if payload["risk_tier"] not in ["High Risk", "Moderate Risk", "Low Risk"]:
        return False, f"Invalid risk_tier: {payload['risk_tier']}"

    return True, "Valid"


def insert_prediction(db: Any, prediction_data: Dict[str, Any]) -> Tuple[bool, str, Optional[Any]]:
    """
    Validate and insert a prediction document using insert_one().
    Returns: (success, message, inserted_id)
    """
    is_valid, validation_msg = validate_prediction_payload(prediction_data)
    if not is_valid:
        return False, f"Payload validation failed: {validation_msg}", None

    try:
        res = db.predictions.insert_one(prediction_data)
        return True, "Prediction recorded successfully.", res.inserted_id
    except Exception as e:
        return False, f"Failed to record prediction in MongoDB: {e}", None


def get_prediction_history(db: Any, outcome_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve prediction history using find() with reverse-chronological sort.
    Safely normalizes both current and legacy documents.
    """
    query: Dict[str, Any] = {}
    if outcome_filter and outcome_filter != "All":
        query["predicted_outcome"] = outcome_filter

    raw_preds = list(db.predictions.find(query).sort("prediction_timestamp", pymongo.DESCENDING).limit(limit))
    normalized = []

    for p in raw_preds:
        raw_ts = p.get("prediction_timestamp") or p.get("timestamp")
        if isinstance(raw_ts, datetime):
            ts_str = raw_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        elif isinstance(raw_ts, str):
            ts_str = raw_ts
        else:
            ts_str = "N/A"

        risk = p.get("risk_tier") or p.get("dropout_risk") or "N/A"
        probs = p.get("class_probabilities") or p.get("probabilities") or {}

        normalized.append({
            "Timestamp": ts_str,
            "Student ID": p.get("student_id", "N/A"),
            "Predicted Outcome": p.get("predicted_outcome", "N/A"),
            "Risk Tier": risk,
            "Model": p.get("model_name", "Random Forest Classifier"),
            "Feature Set": p.get("feature_set") or p.get("prediction_mode") or "Full Features",
            "Document ID": str(p.get("_id")),
            "Probabilities": probs,
            "_raw": p,
        })

    return normalized


# ==============================================================================
# QUERY PLAN EXPLAIN MECHANISM (PYMONGO)
# ==============================================================================
def explain_query(db: Any, collection_name: str, query_filter: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect the query execution plan using PyMongo's supported explain() mechanism.
    Extracts winning stage and index name.
    """
    plan = db[collection_name].find(query_filter).explain()
    winning_plan = plan.get("queryPlanner", {}).get("winningPlan", {})

    stage = winning_plan.get("stage", "UNKNOWN")
    index_name = winning_plan.get("indexName", "None")

    # In sharded or complex execution plans, stage might be nested under inputStage
    if stage in ["FETCH", "PROJECTION_SIMPLE", "PROJECTION_DEFAULT"] and "inputStage" in winning_plan:
        stage = winning_plan["inputStage"].get("stage", stage)
        index_name = winning_plan["inputStage"].get("indexName", index_name)

    return {
        "stage": stage,
        "index_name": index_name,
        "raw_plan": plan,
    }
