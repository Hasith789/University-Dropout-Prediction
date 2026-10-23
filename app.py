"""
app.py - University Dropout Prediction System
Course: CSE494 Intelligent NoSQL Databases | Academic Project

Clean, compact, dark-mode academic dashboard integrating MongoDB with Scikit-learn.
"""

import os
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment configuration
load_dotenv(override=True)

def get_connection_config():
    """Retrieve MongoDB URI and Database Name supporting Streamlit Cloud Secrets and .env."""
    uri = None
    db_name = None

    # 1. Streamlit Cloud Secrets (st.secrets)
    try:
        if hasattr(st, "secrets"):
            if "MONGODB_URI" in st.secrets:
                uri = str(st.secrets["MONGODB_URI"]).strip()
            if "MONGODB_DATABASE" in st.secrets:
                db_name = str(st.secrets["MONGODB_DATABASE"]).strip()
    except Exception:
        pass

    # 2. Environment Variables (.env or OS)
    if not uri:
        uri = os.getenv("MONGODB_URI")
    if not db_name:
        db_name = os.getenv("MONGODB_DATABASE")

    # 3. Fallback default
    if not uri:
        uri = "mongodb://localhost:27017"
    if not db_name:
        db_name = "university_dropout_db"

    return uri, db_name

MONGODB_URI, DATABASE_NAME = get_connection_config()

# Set page configuration
st.set_page_config(
    page_title="University Dropout Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Academic Dark Theme CSS
st.markdown(
    """
    <style>
    /* Dark Theme Core Variables */
    :root {
        --bg-main: #0B0F14;
        --bg-sidebar: #141922;
        --bg-card: #171F2B;
        --border-color: #293445;
        --text-primary: #E8EDF4;
        --text-secondary: #9AA7B8;
        --accent-blue: #4C8DFF;
    }

    .stApp {
        background-color: #0B0F14 !important;
        color: #E8EDF4 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Compact Sidebar (220-250px) */
    [data-testid="stSidebar"] {
        background-color: #141922 !important;
        border-right: 1px solid #293445 !important;
        min-width: 220px !important;
        max-width: 250px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }

    /* Clean Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #E8EDF4 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 0.35rem !important;
    }

    p, span, label {
        color: #E8EDF4;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #171F2B !important;
        border: 1px solid #293445 !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
    }

    div[data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        color: #9AA7B8 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #E8EDF4 !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #171F2B !important;
        border: 1px solid #293445 !important;
        border-radius: 4px !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    div[data-testid="stExpander"] summary {
        font-size: 0.82rem !important;
        color: #9AA7B8 !important;
        padding: 6px 10px !important;
    }

    div[data-testid="stExpander"] summary:hover {
        color: #E8EDF4 !important;
    }

    /* Primary Accent Buttons */
    button[kind="primary"] {
        background-color: #4C8DFF !important;
        border-color: #4C8DFF !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
        font-size: 0.85rem !important;
        padding: 0.35rem 0.8rem !important;
        font-weight: 500 !important;
    }

    button[kind="primary"]:hover {
        background-color: #3b76e1 !important;
        border-color: #3b76e1 !important;
    }

    /* Tables & DataFrames */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid #293445 !important;
        border-radius: 4px !important;
        background-color: #171F2B !important;
    }

    /* Sidebar Radio Navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 2px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 6px 8px !important;
        border-radius: 4px !important;
        font-size: 0.85rem !important;
        color: #9AA7B8 !important;
        cursor: pointer;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #1B2433 !important;
        color: #E8EDF4 !important;
    }

    /* Card Box Utility */
    .dashboard-card {
        background-color: #171F2B;
        border: 1px solid #293445;
        border-radius: 4px;
        padding: 12px 14px;
        margin-bottom: 0.85rem;
    }

    .status-dot-green {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #22C55E;
        border-radius: 50%;
        margin-right: 6px;
    }

    .status-dot-red {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #EF4444;
        border-radius: 50%;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Course ID mapping
COURSE_NAMES = {
    1: "Biofuel Production Technologies",
    2: "Animation and Multimedia Design",
    3: "Social Service (Evening)",
    4: "Agronomy",
    5: "Communication Design",
    6: "Veterinary Nursing",
    7: "Informatics Engineering",
    8: "Equiniculture",
    9: "Management",
    10: "Social Service",
    11: "Tourism",
    12: "Nursing",
    13: "Oral Hygiene",
    14: "Advertising and Marketing",
    15: "Journalism and Communication",
    16: "Basic Education",
    17: "Management (Evening)",
}


# ==============================================================================
# DATABASE CONNECTION
# ==============================================================================
def get_db():
    uri, db_name = get_connection_config()
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    return client[db_name]


def test_db_connection():
    uri, db_name = get_connection_config()
    is_atlas = "mongodb.net" in uri or "+srv" in uri
    host_type = "MongoDB Atlas" if is_atlas else "Local MongoDB"
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        server_info = client.server_info()
        version = server_info.get("version", "Unknown")
        client.close()
        return True, host_type, version, None
    except Exception as e:
        err = str(e)
        if "localhost:27017" in uri and ("[Errno 111]" in err or "Connection refused" in err):
            err = (
                "Cannot connect to local MongoDB. If running on Streamlit Cloud, add your "
                "MongoDB Atlas connection string in Streamlit App Settings > Secrets."
            )
        return False, host_type, None, err


@st.cache_resource
def load_ml_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None


model_package = load_ml_model()


# Helper for creating uniform dark-mode Matplotlib figures
def create_dark_figure(figsize=(6, 3.2)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#171F2B")
    ax.set_facecolor("#171F2B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#293445")
    ax.spines["left"].set_color("#293445")
    ax.tick_params(colors="#9AA7B8", labelsize=8)
    ax.xaxis.label.set_color("#9AA7B8")
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_color("#9AA7B8")
    ax.yaxis.label.set_size(8.5)
    ax.title.set_color("#E8EDF4")
    ax.title.set_size(9.5)
    ax.title.set_weight("600")
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#E8EDF4")
    return fig, ax


# ==============================================================================
# PREDICTION VALIDATION
# ==============================================================================
def validate_prediction_payload(payload: dict) -> tuple[bool, str]:
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
            return False, f"Missing required field: {field}"

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


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 0.65rem;">
            <div style="font-size: 0.95rem; font-weight: 600; color: #E8EDF4;">University Dropout Prediction</div>
            <div style="font-size: 0.75rem; color: #9AA7B8;">CSE494 Academic Project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_connected, host_type, db_version, err_msg = test_db_connection()

    if is_connected:
        st.markdown(
            f"""
            <div style="background-color: #171F2B; border: 1px solid #293445; border-radius: 4px; padding: 6px 8px; margin-bottom: 0.85rem;">
                <div style="font-size: 0.78rem; font-weight: 500; color: #22C55E;">
                    <span class="status-dot-green"></span>Connected
                </div>
                <div style="font-size: 0.72rem; color: #9AA7B8; margin-top: 2px;">
                    Host: {host_type}<br>
                    DB: <code>{DATABASE_NAME}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: #171F2B; border: 1px solid #293445; border-radius: 4px; padding: 6px 8px; margin-bottom: 0.85rem;">
                <div style="font-size: 0.78rem; font-weight: 500; color: #EF4444;">
                    <span class="status-dot-red"></span>Disconnected
                </div>
                <div style="font-size: 0.72rem; color: #9AA7B8; margin-top: 2px;">
                    Host: {host_type}<br>
                    DB: <code>{DATABASE_NAME}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    menu_choice = st.radio(
        label="Select Section",
        options=[
            "Dashboard",
            "Student Prediction",
            "Student Records",
            "Prediction History",
            "Model Evaluation",
            "MongoDB Analytics",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div style="margin-top: 1.5rem; border-top: 1px solid #293445; padding-top: 0.6rem; font-size: 0.72rem; color: #9AA7B8;">
            Intelligent NoSQL Databases
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 1: DASHBOARD
# ==============================================================================
if menu_choice == "Dashboard":
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 1.45rem; font-weight: 600; color: #E8EDF4;">University Dropout Prediction System</div>
            <div style="font-size: 0.85rem; color: #9AA7B8; margin-top: 2px;">CSE494 — Intelligent NoSQL Databases | Academic Project</div>
            <div style="font-size: 0.82rem; color: #E8EDF4; margin-top: 4px;">
                This application stores student records in MongoDB, predicts academic outcomes using trained classification models, and records prediction history for analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_connected:
        st.error(f"Database connection unavailable ({host_type}): {err_msg}")
        st.stop()

    db = get_db()

    # Query metrics from MongoDB
    total_students = db.students.count_documents({})
    total_preds = db.predictions.count_documents({})

    pipeline_outcome = [
        {"$group": {"_id": "$Target", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    outcome_counts = list(db.students.aggregate(pipeline_outcome))
    dist_map = {item["_id"]: item["count"] for item in outcome_counts}

    dropouts = dist_map.get("Dropout", 0)
    graduates = dist_map.get("Graduate", 0)
    enrolled = dist_map.get("Enrolled", 0)

    # Metric Cards Row (5 Cards)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Students", f"{total_students:,}")
    m2.metric("Saved Predictions", f"{total_preds:,}")
    m3.metric("Actual Dropouts", f"{dropouts:,}")
    m4.metric("Actual Graduates", f"{graduates:,}")
    m5.metric("Currently Enrolled", f"{enrolled:,}")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Grid Panel Row 1: Outcome Distribution & Top 10 Courses by Dropout Rate
    col_row1_left, col_row1_right = st.columns(2)

    with col_row1_left:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 600; color: #E8EDF4; margin-bottom: 4px;'>Student Academic Outcome Distribution</div>",
            unsafe_allow_html=True,
        )
        if outcome_counts:
            df_dist = pd.DataFrame([{"Outcome": r["_id"], "Count": r["count"]} for r in outcome_counts])
            fig_dist, ax_dist = create_dark_figure(figsize=(5.2, 2.7))
            palette = {"Graduate": "#27AE60", "Dropout": "#EF4444", "Enrolled": "#4C8DFF"}
            bar_colors = [palette.get(o, "#7f8c8d") for o in df_dist["Outcome"]]
            bars = ax_dist.bar(df_dist["Outcome"], df_dist["Count"], color=bar_colors, width=0.48)
            ax_dist.set_ylabel("Document Count")
            for bar in bars:
                h = bar.get_height()
                ax_dist.annotate(
                    f"{int(h):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#E8EDF4",
                )
            plt.tight_layout()
            st.pyplot(fig_dist)
        else:
            st.info("No student outcome records found in MongoDB.")

    with col_row1_right:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 600; color: #E8EDF4; margin-bottom: 4px;'>Top 10 Courses by Dropout Rate</div>",
            unsafe_allow_html=True,
        )
        course_pipeline = [
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
                            0,
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
            {"$limit": 10},
        ]
        course_data = list(db.students.aggregate(course_pipeline))

        if course_data:
            df_course = pd.DataFrame(course_data)
            df_course["Course Name"] = df_course["course_id"].map(
                lambda cid: COURSE_NAMES.get(cid, f"Course {cid}")
            )
            df_course_sorted = df_course.sort_values(by="dropout_rate", ascending=True)

            fig_course, ax_course = create_dark_figure(figsize=(5.4, 2.7))
            ax_course.barh(
                df_course_sorted["Course Name"],
                df_course_sorted["dropout_rate"],
                color="#EF4444",
                height=0.6,
            )
            ax_course.set_xlabel("Dropout Percentage (%)")
            ax_course.grid(axis="x", linestyle="--", alpha=0.15, color="#E8EDF4")
            for i, row in enumerate(df_course_sorted.itertuples()):
                ax_course.annotate(
                    f"{row.dropout_rate}% ({row.dropouts}/{row.total_students})",
                    xy=(row.dropout_rate, i),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=7.2,
                    color="#9AA7B8",
                )
            plt.tight_layout()
            st.pyplot(fig_course)
        else:
            st.info("Course aggregation data unavailable.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Grid Panel Row 2: Predictive Features & Confusion Matrix
    col_row2_left, col_row2_right = st.columns(2)

    with col_row2_left:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 600; color: #E8EDF4; margin-bottom: 4px;'>Top 10 Predictive Features</div>",
            unsafe_allow_html=True,
        )
        if model_package and "feature_importances" in model_package:
            feat_imp = pd.Series(model_package["feature_importances"]).head(10).sort_values(ascending=True)
            fig_feat, ax_feat = create_dark_figure(figsize=(5.2, 2.7))
            ax_feat.barh(feat_imp.index, feat_imp.values, color="#4C8DFF", height=0.6)
            ax_feat.set_xlabel("Gini Importance Score")
            ax_feat.grid(axis="x", linestyle="--", alpha=0.15, color="#E8EDF4")
            plt.tight_layout()
            st.pyplot(fig_feat)
        else:
            st.info("Feature importance data unavailable. Train model in ML_Training.ipynb.")

    with col_row2_right:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 600; color: #E8EDF4; margin-bottom: 4px;'>Random Forest Confusion Matrix</div>",
            unsafe_allow_html=True,
        )
        if model_package and "confusion_matrix" in model_package:
            cm = np.array(model_package["confusion_matrix"])
            labels_eval = ["Dropout", "Enrolled", "Graduate"]
            fig_cm, ax_cm = create_dark_figure(figsize=(5.2, 2.7))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels_eval,
                yticklabels=labels_eval,
                ax=ax_cm,
                cbar=False,
                annot_kws={"size": 9, "color": "#E8EDF4"},
            )
            ax_cm.set_xlabel("Predicted Class")
            ax_cm.set_ylabel("True Class")
            plt.tight_layout()
            st.pyplot(fig_cm)
        else:
            st.info("Confusion matrix unavailable.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Grid Panel Row 3: Model Performance Comparison
    st.markdown(
        "<div style='font-size: 0.92rem; font-weight: 600; color: #E8EDF4; margin-bottom: 4px;'>Model Performance Comparison</div>",
        unsafe_allow_html=True,
    )

    if model_package:
        lr_acc = model_package.get("lr_accuracy", 0.0)
        rf_acc = model_package.get("rf_accuracy", 0.0)
        report = model_package.get("classification_report", {})

        c_cmp1, c_cmp2, c_cmp3, c_cmp4 = st.columns(4)
        c_cmp1.metric("Baseline Logistic Regression", f"{lr_acc*100:.2f}%")
        c_cmp2.metric("Random Forest Accuracy", f"{rf_acc*100:.2f}%")
        c_cmp3.metric("Random Forest Macro F1", f"{report.get('macro avg', {}).get('f1-score', 0.0)*100:.2f}%")
        c_cmp4.metric("Dropout Specific Recall", f"{report.get('Dropout', {}).get('recall', 0.0)*100:.2f}%")
    else:
        st.info("Model evaluation benchmarks unavailable.")


# ==============================================================================
# SECTION 2: STUDENT PREDICTION
# ==============================================================================
elif menu_choice == "Student Prediction":
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 1.35rem; font-weight: 600; color: #E8EDF4;">Student Prediction</div>
            <div style="font-size: 0.82rem; color: #9AA7B8;">
                Inference using trained Random Forest model. Predictions are validated and recorded into MongoDB.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_connected:
        st.error("Database connection unavailable. Predictions cannot be recorded.")
        st.stop()

    if model_package is None:
        st.error("model.pkl not found. Please train and export the model using ML_Training.ipynb.")
        st.stop()

    rf_model = model_package["model"]
    feature_names = model_package["feature_names"]
    classes = model_package["classes"]
    db = get_db()

    input_method = st.radio(
        "Input Mode:",
        ["Retrieve from MongoDB", "Custom Student Profile"],
        horizontal=True,
    )

    student_data = {}
    selected_id = "STU-0001"
    feature_mode = "Full Academic Features (1st & 2nd Sem)"

    if input_method == "Retrieve from MongoDB":
        col_pick, col_search = st.columns([2, 1])
        with col_pick:
            cursor_sample = db.students.find({}, {"student_id": 1, "_id": 0}).limit(50)
            student_ids = [d["student_id"] for d in cursor_sample]
            pick_id = st.selectbox("Select Student ID:", student_ids if student_ids else ["STU-0001"])

        with col_search:
            manual_search_id = st.text_input("Or Enter Student ID:", value="").strip()

        final_id = manual_search_id if manual_search_id else pick_id

        if final_id:
            doc = db.students.find_one({"student_id": final_id})
            if doc:
                selected_id = doc["student_id"]
                student_data = doc
                st.caption(
                    f"Selected: {selected_id} | Recorded Outcome: {doc.get('Target', 'Unknown')} | "
                    f"Course: {COURSE_NAMES.get(doc.get('Course'), doc.get('Course'))}"
                )
            else:
                st.warning(f"Student '{final_id}' not found.")
    else:
        selected_id = st.text_input("Student Identifier:", value="STU-CUSTOM-01").strip()
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age at Enrollment:", min_value=16, max_value=70, value=20)
            gender = st.selectbox("Gender:", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
            course_id = st.selectbox("Course:", list(COURSE_NAMES.items()), format_func=lambda x: f"{x[0]} - {x[1]}")[0]
            tuition = st.selectbox("Tuition Fees Up to Date:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            scholarship = st.selectbox("Scholarship Holder:", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

        with c2:
            sem1_enrolled = st.number_input("1st Sem Enrolled Units:", min_value=0, max_value=25, value=6)
            sem1_approved = st.number_input("1st Sem Approved Units:", min_value=0, max_value=25, value=5)
            sem1_grade = st.number_input("1st Sem Grade Average (0-20):", min_value=0.0, max_value=20.0, value=13.0)
            sem1_evals = st.number_input("1st Sem Evaluations:", min_value=0, max_value=30, value=6)

        with c3:
            sem2_enrolled = st.number_input("2nd Sem Enrolled Units:", min_value=0, max_value=25, value=6)
            sem2_approved = st.number_input("2nd Sem Approved Units:", min_value=0, max_value=25, value=5)
            sem2_grade = st.number_input("2nd Sem Grade Average (0-20):", min_value=0.0, max_value=20.0, value=12.5)
            sem2_evals = st.number_input("2nd Sem Evaluations:", min_value=0, max_value=30, value=6)

        student_data = {
            "Marital status": 1,
            "Application mode": 1,
            "Application order": 1,
            "Course": int(course_id),
            "Daytime/evening attendance": 1,
            "Previous qualification": 1,
            "Nacionality": 1,
            "Mother's qualification": 1,
            "Father's qualification": 1,
            "Mother's occupation": 1,
            "Father's occupation": 1,
            "Displaced": 1,
            "Educational special needs": 0,
            "Debtor": 0,
            "Tuition fees up to date": int(tuition),
            "Gender": int(gender),
            "Scholarship holder": int(scholarship),
            "Age at enrollment": int(age),
            "International": 0,
            "Curricular units 1st sem (credited)": 0,
            "Curricular units 1st sem (enrolled)": int(sem1_enrolled),
            "Curricular units 1st sem (evaluations)": int(sem1_evals),
            "Curricular units 1st sem (approved)": int(sem1_approved),
            "Curricular units 1st sem (grade)": float(sem1_grade),
            "Curricular units 1st sem (without evaluations)": 0,
            "Curricular units 2nd sem (credited)": 0,
            "Curricular units 2nd sem (enrolled)": int(sem2_enrolled),
            "Curricular units 2nd sem (evaluations)": int(sem2_evals),
            "Curricular units 2nd sem (approved)": int(sem2_approved),
            "Curricular units 2nd sem (grade)": float(sem2_grade),
            "Curricular units 2nd sem (without evaluations)": 0,
            "Unemployment rate": 10.8,
            "Inflation rate": 1.4,
            "GDP": 1.74,
        }

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    if st.button("Run Prediction", type="primary"):
        if not student_data:
            st.error("Please provide or retrieve a valid student profile.")
        else:
            input_vector = []
            for col in feature_names:
                val = student_data.get(col, 0.0)
                try:
                    input_vector.append(float(val))
                except (ValueError, TypeError):
                    input_vector.append(0.0)

            df_input = pd.DataFrame([input_vector], columns=feature_names)

            pred_class = rf_model.predict(df_input)[0]
            probs = rf_model.predict_proba(df_input)[0]
            prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)}
            dropout_prob = prob_dict.get("Dropout", 0.0)

            if dropout_prob >= 0.50:
                risk_tier = "High Risk"
            elif dropout_prob >= 0.25:
                risk_tier = "Moderate Risk"
            else:
                risk_tier = "Low Risk"

            # Repeat submission prevention
            sub_sig = f"{selected_id}_{pred_class}_{tuple(input_vector)}"
            last_sig = st.session_state.get("last_pred_sig")

            now_utc = datetime.now(timezone.utc)
            pred_doc = {
                "student_id": selected_id,
                "predicted_outcome": pred_class,
                "class_probabilities": prob_dict,
                "model_name": "Random Forest Classifier",
                "prediction_timestamp": now_utc,
                "feature_set": feature_mode,
                "risk_tier": risk_tier,
            }

            is_valid, validation_msg = validate_prediction_payload(pred_doc)
            if not is_valid:
                st.error(f"Payload validation failed: {validation_msg}")
            else:
                if last_sig == sub_sig:
                    st.caption("Identical prediction already recorded. Showing existing result.")
                else:
                    res_ins = db.predictions.insert_one(pred_doc)
                    st.session_state["last_pred_sig"] = sub_sig
                    st.session_state["last_pred_id"] = str(res_ins.inserted_id)

                doc_id_val = st.session_state.get("last_pred_id", "Recorded")

                # Results Display
                st.markdown(
                    f"""
                    <div class="dashboard-card" style="margin-top: 0.5rem;">
                        <div style="font-size: 0.85rem; color: #9AA7B8;">Predicted Outcome:</div>
                        <div style="font-size: 1.35rem; font-weight: 600; color: #E8EDF4;">{pred_class}</div>
                        <div style="font-size: 0.82rem; color: #9AA7B8; margin-top: 4px;">
                            Risk Tier: <span style="color: {'#EF4444' if risk_tier=='High Risk' else ('#F59E0B' if risk_tier=='Moderate Risk' else '#22C55E')}; font-weight: 600;">{risk_tier}</span> ({dropout_prob*100:.1f}%) | 
                            Student ID: <code>{selected_id}</code> | Record ID: <code>{doc_id_val}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("**Model Class Probabilities:**")
                for c_name, p_val in prob_dict.items():
                    st.progress(float(p_val), text=f"{c_name}: {p_val*100:.1f}%")

                with st.expander("MongoDB Shell Insert Query", expanded=False):
                    st.code(
                        f"""
const db = db.getSiblingDB("university_dropout_db");
db.predictions.insertOne({{
  student_id: "{selected_id}",
  predicted_outcome: "{pred_class}",
  class_probabilities: {prob_dict},
  model_name: "Random Forest Classifier",
  prediction_timestamp: new Date(),
  feature_set: "{feature_mode}",
  risk_tier: "{risk_tier}"
}});
                        """,
                        language="javascript",
                    )

                st.caption(
                    "Statistical estimates for academic guidance. Not clinically or institutionally validated."
                )


# ==============================================================================
# SECTION 3: STUDENT RECORDS
# ==============================================================================
elif menu_choice == "Student Records":
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 1.35rem; font-weight: 600; color: #E8EDF4;">Student Records</div>
            <div style="font-size: 0.82rem; color: #9AA7B8;">
                Real MongoDB CRUD operations executed directly against the <code>students</code> collection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_connected:
        st.error("Database connection unavailable.")
        st.stop()

    db = get_db()

    tab_find, tab_insert, tab_update, tab_delete = st.tabs([
        "Find & Filter",
        "Insert Record",
        "Update Record",
        "Delete Record",
    ])

    with tab_find:
        f_id = st.text_input("Filter by Student ID prefix:", value="").strip()
        f_outcome = st.selectbox("Filter by Academic Outcome:", ["All", "Dropout", "Enrolled", "Graduate"])

        q = {}
        if f_id:
            q["student_id"] = {"$regex": f"^{f_id}", "$options": "i"}
        if f_outcome != "All":
            q["Target"] = f_outcome

        match_count = db.students.count_documents(q)
        st.caption(f"Matching documents: {match_count:,}")

        with st.expander("MongoDB Shell Query", expanded=False):
            if f_id and f_outcome != "All":
                st.code(f'db.students.find({{ student_id: /^"{f_id}"/i, Target: "{f_outcome}" }}).limit(25);', language="javascript")
            elif f_id:
                st.code(f'db.students.find({{ student_id: /^"{f_id}"/i }}).limit(25);', language="javascript")
            elif f_outcome != "All":
                st.code(f'db.students.find({{ Target: "{f_outcome}" }}).limit(25);', language="javascript")
            else:
                st.code("db.students.find({}).limit(25);", language="javascript")

        results = list(db.students.find(q).limit(25))
        if results:
            df_res = pd.DataFrame(results)
            cols = [
                "student_id",
                "Target",
                "Age at enrollment",
                "Gender",
                "Course",
                "Tuition fees up to date",
                "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)",
            ]
            avail = [c for c in cols if c in df_res.columns]
            st.dataframe(df_res[avail], use_container_width=True)
        else:
            st.info("No matching student records found.")

    with tab_insert:
        with st.form("form_insert_stu"):
            ins_id = st.text_input("Unique Student ID:", value="STU-9901").strip()
            ins_target = st.selectbox("Academic Outcome:", ["Enrolled", "Dropout", "Graduate"])
            ins_age = st.number_input("Age at Enrollment:", min_value=16, max_value=80, value=21)
            ins_course = st.selectbox("Course:", list(COURSE_NAMES.items()), format_func=lambda x: f"{x[0]} - {x[1]}")[0]
            ins_tuition = st.selectbox("Tuition Paid:", [1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            ins_sem1_grade = st.number_input("1st Sem Grade (0-20):", min_value=0.0, max_value=20.0, value=12.5)

            with st.expander("MongoDB Shell insertOne Query", expanded=False):
                st.code(
                    f"""
const db = db.getSiblingDB("university_dropout_db");
db.students.insertOne({{
  student_id: "{ins_id}",
  Target: "{ins_target}",
  "Age at enrollment": {ins_age},
  Course: {ins_course},
  "Tuition fees up to date": {ins_tuition},
  "Curricular units 1st sem (grade)": {ins_sem1_grade},
  "Curricular units 1st sem (approved)": 5,
  "Curricular units 1st sem (enrolled)": 6
}});
                    """,
                    language="javascript",
                )

            if st.form_submit_button("Insert Record", type="primary"):
                if not ins_id:
                    st.error("Student ID required.")
                elif db.students.find_one({"student_id": ins_id}):
                    st.error(f"Student ID '{ins_id}' already exists.")
                else:
                    new_doc = {
                        "student_id": ins_id,
                        "Target": ins_target,
                        "Age at enrollment": int(ins_age),
                        "Course": int(ins_course),
                        "Tuition fees up to date": int(ins_tuition),
                        "Curricular units 1st sem (grade)": float(ins_sem1_grade),
                        "Curricular units 1st sem (approved)": 5,
                        "Curricular units 1st sem (enrolled)": 6,
                        "Curricular units 2nd sem (approved)": 5,
                        "Curricular units 2nd sem (enrolled)": 6,
                        "Curricular units 2nd sem (grade)": float(ins_sem1_grade),
                    }
                    res_in = db.students.insert_one(new_doc)
                    st.success(f"Inserted document with ID: {res_in.inserted_id}")

    with tab_update:
        upd_id = st.text_input("Student ID to Update:", value="STU-0001").strip()
        doc_upd = db.students.find_one({"student_id": upd_id})

        if doc_upd:
            st.caption(f"Current Status: {doc_upd.get('Target')} | Tuition Status: {doc_upd.get('Tuition fees up to date')}")
            new_target = st.selectbox("New Academic Outcome:", ["Dropout", "Enrolled", "Graduate"])
            new_tuition = st.selectbox("New Tuition Status:", [1, 0], format_func=lambda x: "Paid (1)" if x == 1 else "In Arrears (0)")

            with st.expander("MongoDB Shell updateOne Query", expanded=False):
                st.code(
                    f"""
const db = db.getSiblingDB("university_dropout_db");
db.students.updateOne(
  {{ student_id: "{upd_id}" }},
  {{
    $set: {{
      Target: "{new_target}",
      "Tuition fees up to date": {new_tuition}
    }}
  }}
);
                    """,
                    language="javascript",
                )

            if st.button("Update Record", type="primary"):
                res_upd = db.students.update_one(
                    {"student_id": upd_id},
                    {"$set": {"Target": new_target, "Tuition fees up to date": int(new_tuition)}},
                )
                st.success(f"Updated {res_upd.modified_count} record.")
        else:
            st.caption("Provide an existing Student ID.")

    with tab_delete:
        del_id = st.text_input("Student ID to Delete:", value="").strip()
        if del_id:
            doc_del = db.students.find_one({"student_id": del_id})
            if doc_del:
                st.warning(f"Document identified: {del_id} ({doc_del.get('Target')}).")

                with st.expander("MongoDB Shell deleteOne Query", expanded=False):
                    st.code(
                        f"""
const db = db.getSiblingDB("university_dropout_db");
db.students.deleteOne({{ student_id: "{del_id}" }});
                        """,
                        language="javascript",
                    )

                confirm = st.checkbox("Confirm permanent deletion of this record.")
                if st.button("Delete Record", type="primary", disabled=not confirm):
                    res_del = db.students.delete_one({"student_id": del_id})
                    st.success(f"Deleted {res_del.deleted_count} document.")
            else:
                st.info("Student ID not found.")


# ==============================================================================
# SECTION 4: PREDICTION HISTORY
# ==============================================================================
elif menu_choice == "Prediction History":
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 1.35rem; font-weight: 600; color: #E8EDF4;">Prediction History</div>
            <div style="font-size: 0.82rem; color: #9AA7B8;">
                Real-time audit records retrieved from <code>university_dropout_db.predictions</code>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_connected:
        st.error("Database connection unavailable.")
        st.stop()

    db = get_db()
    pred_total = db.predictions.count_documents({})
    st.caption(f"Total Prediction Documents: {pred_total:,}")

    outcome_filter = st.selectbox("Filter Outcome:", ["All", "Dropout", "Enrolled", "Graduate"])
    query_p = {}
    if outcome_filter != "All":
        query_p["predicted_outcome"] = outcome_filter

    with st.expander("MongoDB Shell Query", expanded=False):
        if outcome_filter != "All":
            st.code(f'db.predictions.find({{ predicted_outcome: "{outcome_filter}" }}).sort({{ prediction_timestamp: -1 }}).limit(50);', language="javascript")
        else:
            st.code('db.predictions.find().sort({ prediction_timestamp: -1 }).limit(50);', language="javascript")

    preds = list(db.predictions.find(query_p).sort("prediction_timestamp", -1).limit(50))

    if preds:
        normalized = []
        for p in preds:
            raw_ts = p.get("prediction_timestamp") or p.get("timestamp")
            if isinstance(raw_ts, datetime):
                ts_str = raw_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            elif isinstance(raw_ts, str):
                ts_str = raw_ts
            else:
                ts_str = "N/A"

            risk = p.get("risk_tier") or p.get("dropout_risk") or "N/A"
            normalized.append({
                "Timestamp": ts_str,
                "Student ID": p.get("student_id", "N/A"),
                "Predicted Outcome": p.get("predicted_outcome", "N/A"),
                "Risk Tier": risk,
                "Model": p.get("model_name", "Random Forest Classifier"),
                "Feature Set": p.get("feature_set") or p.get("prediction_mode") or "Full Features",
                "Document ID": str(p.get("_id")),
                "_raw": p,
            })

        df_show = pd.DataFrame(normalized)
        cols_display = ["Timestamp", "Student ID", "Predicted Outcome", "Risk Tier", "Model", "Feature Set", "Document ID"]
        st.dataframe(df_show[cols_display], use_container_width=True)

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        sel_pred_id = st.selectbox("Inspect Document BSON:", [r["Document ID"] for r in normalized])
        sel_item = next((r for r in normalized if r["Document ID"] == sel_pred_id), None)
        if sel_item:
            clean_raw = dict(sel_item["_raw"])
            clean_raw["_id"] = str(clean_raw["_id"])
            if isinstance(clean_raw.get("prediction_timestamp"), datetime):
                clean_raw["prediction_timestamp"] = clean_raw["prediction_timestamp"].isoformat()
            st.json(clean_raw)
    else:
        st.info("No prediction history matching criteria.")


# ==============================================================================
# SECTION 5: MODEL EVALUATION
# ==============================================================================
elif menu_choice == "Model Evaluation":
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 1.35rem; font-weight: 600; color: #E8EDF4;">Model Evaluation</div>
            <div style="font-size: 0.82rem; color: #9AA7B8;">
                Measured benchmarks on the stratified 20% test partition (N = 885).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model_package is None:
        st.error("model.pkl not found. Please train the model in ML_Training.ipynb.")
        st.stop()

    lr_acc = model_package.get("lr_accuracy", 0.0)
    rf_acc = model_package.get("rf_accuracy", 0.0)
    report = model_package.get("classification_report", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline Logistic Regression", f"{lr_acc*100:.2f}%")
    m2.metric("Random Forest Accuracy", f"{rf_acc*100:.2f}%")
    m3.metric("Random Forest Macro F1", f"{report.get('macro avg', {}).get('f1-score', 0.0)*100:.2f}%")
    m4.metric("Dropout Recall", f"{report.get('Dropout', {}).get('recall', 0.0)*100:.2f}%")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # Classification Table
    classes_eval = ["Dropout", "Enrolled", "Graduate"]
    rows = []
    for c in classes_eval:
        c_data = report.get(c, {})
        rows.append({
            "Class": c,
            "Precision": f"{c_data.get('precision', 0.0):.4f}",
            "Recall": f"{c_data.get('recall', 0.0):.4f}",
            "F1-Score": f"{c_data.get('f1-score', 0.0):.4f}",
            "Support": int(c_data.get("support", 0)),
        })
    rows.append({
        "Class": "Macro Average",
        "Precision": f"{report.get('macro avg', {}).get('precision', 0.0):.4f}",
        "Recall": f"{report.get('macro avg', {}).get('recall', 0.0):.4f}",
        "F1-Score": f"{report.get('macro avg', {}).get('f1-score', 0.0):.4f}",
        "Support": int(report.get("macro avg", {}).get("support", 0)),
    })
    rows.append({
        "Class": "Weighted Average",
        "Precision": f"{report.get('weighted avg', {}).get('precision', 0.0):.4f}",
        "Recall": f"{report.get('weighted avg', {}).get('recall', 0.0):.4f}",
        "F1-Score": f"{report.get('weighted avg', {}).get('f1-score', 0.0):.4f}",
        "Support": int(report.get("weighted avg", {}).get("support", 0)),
    })

    st.table(pd.DataFrame(rows))

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 600; color: #E8EDF4; margin-bottom: 4px;'>Confusion Matrix (Random Forest)</div>",
            unsafe_allow_html=True,
        )
        cm = np.array(model_package.get("confusion_matrix", []))
        if len(cm) > 0:
            fig_cm, ax_cm = create_dark_figure(figsize=(5, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes_eval,
                yticklabels=classes_eval,
                ax=ax_cm,
                cbar=False,
                annot_kws={"size": 9, "color": "#E8EDF4"},
            )
            ax_cm.set_xlabel("Predicted Class")
            ax_cm.set_ylabel("True Class")
            plt.tight_layout()
            st.pyplot(fig_cm)

    with c_right:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 600; color: #E8EDF4; margin-bottom: 4px;'>Top 10 Feature Importances</div>",
            unsafe_allow_html=True,
        )
        feat_imp = model_package.get("feature_importances", {})
        if feat_imp:
            df_imp = pd.Series(feat_imp).head(10).sort_values(ascending=True)
            fig_imp, ax_imp = create_dark_figure(figsize=(5, 3))
            ax_imp.barh(df_imp.index, df_imp.values, color="#4C8DFF", height=0.6)
            ax_imp.set_xlabel("Gini Importance Score")
            ax_imp.grid(axis="x", linestyle="--", alpha=0.15, color="#E8EDF4")
            plt.tight_layout()
            st.pyplot(fig_imp)

    st.markdown(
        """
        <div class="dashboard-card" style="margin-top: 0.75rem;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #E8EDF4; margin-bottom: 4px;">Methodology & Feature Timing</div>
            <div style="font-size: 0.78rem; color: #9AA7B8; line-height: 1.4;">
                - <b>Scope:</b> Predicts end-of-program academic outcomes using demographic, socioeconomic, and 1st & 2nd semester academic features.<br>
                - <b>Feature Timing:</b> 2nd semester approved units and grades are the strongest predictive signals. For early pre-admission prediction, 2nd semester attributes must be omitted to prevent target leakage.<br>
                - <b>Academic Context:</b> Educational demonstration model for CSE494; not clinically or institutionally validated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 6: MONGODB ANALYTICS
# ==============================================================================
elif menu_choice == "MongoDB Analytics":
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 1.35rem; font-weight: 600; color: #E8EDF4;">MongoDB Analytics</div>
            <div style="font-size: 0.82rem; color: #9AA7B8;">
                Real aggregation pipelines and query optimization execution plans (<code>explain</code>).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_connected:
        st.error("Database connection unavailable.")
        st.stop()

    db = get_db()

    tab_agg, tab_explain, tab_compass = st.tabs([
        "Aggregation Pipelines",
        "Query Optimization (explain)",
        "Compass Verification Guide",
    ])

    with tab_agg:
        st.markdown(
            "<div style='font-size: 0.9rem; font-weight: 600; color: #E8EDF4;'>Program Dropout Rate Pipeline ($group, $project, $sort)</div>",
            unsafe_allow_html=True,
        )

        with st.expander("MongoDB Shell Aggregation Pipeline", expanded=False):
            st.code(
                """
const db = db.getSiblingDB("university_dropout_db");
db.students.aggregate([
  {
    $group: {
      _id: "$Course",
      total_students: { $sum: 1 },
      dropouts: { $sum: { $cond: [{ $eq: ["$Target", "Dropout"] }, 1, 0] } }
    }
  },
  {
    $project: {
      course_id: "$_id",
      total_students: 1,
      dropouts: 1,
      dropout_rate: {
        $cond: [
          { $eq: ["$total_students", 0] },
          0,
          { $round: [{ $multiply: [{ $divide: ["$dropouts", "$total_students"] }, 100] }, 1] }
        ]
      },
      _id: 0
    }
  },
  { $sort: { dropout_rate: -1 } }
]);
                """,
                language="javascript",
            )

        course_pipeline = [
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
                            0,
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
        ]
        course_stats = list(db.students.aggregate(course_pipeline))

        if course_stats:
            df_c = pd.DataFrame(course_stats)
            df_c["Course Name"] = df_c["course_id"].map(lambda cid: COURSE_NAMES.get(cid, f"Course {cid}"))
            df_c = df_c[["Course Name", "total_students", "dropouts", "dropout_rate"]]
            df_c.columns = ["Course Name", "Total Students", "Dropouts", "Dropout Rate (%)"]
            st.dataframe(df_c, use_container_width=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size: 0.9rem; font-weight: 600; color: #E8EDF4;'>Prediction Summary ($group, $sort)</div>",
            unsafe_allow_html=True,
        )

        with st.expander("MongoDB Shell Prediction Aggregation", expanded=False):
            st.code(
                """
const db = db.getSiblingDB("university_dropout_db");
db.predictions.aggregate([
  { $group: { _id: "$predicted_outcome", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
]);
                """,
                language="javascript",
            )

        pred_pipeline = [
            {"$group": {"_id": "$predicted_outcome", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        pred_agg = list(db.predictions.aggregate(pred_pipeline))
        if pred_agg:
            st.dataframe(pd.DataFrame([{"Outcome": p["_id"], "Count": p["count"]} for p in pred_agg]), use_container_width=True)
        else:
            st.caption("No prediction audit records to aggregate.")

    with tab_explain:
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #9AA7B8; margin-bottom: 6px;">
                Comparison between indexed query execution using <code>idx_unique_student_id</code> (IXSCAN) 
                and unindexed field query (COLLSCAN) across documents.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("MongoDB Shell explain() Queries", expanded=False):
            st.code(
                """
const db = db.getSiblingDB("university_dropout_db");

// 1. Indexed lookup (B-Tree scan)
db.students.find({ student_id: "STU-0001" }).explain("executionStats");

// 2. Unindexed scan (Full collection scan)
db.students.find({ "Age at enrollment": 20 }).explain("executionStats");
                """,
                language="javascript",
            )

        plan_indexed = db.students.find({"student_id": "STU-0001"}).explain()
        stage_idx = plan_indexed.get("queryPlanner", {}).get("winningPlan", {}).get("stage", "IXSCAN")
        index_used = plan_indexed.get("queryPlanner", {}).get("winningPlan", {}).get("indexName", "idx_unique_student_id")

        plan_unindexed = db.students.find({"Age at enrollment": 20}).explain()
        stage_unidx = plan_unindexed.get("queryPlanner", {}).get("winningPlan", {}).get("stage", "COLLSCAN")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #22C55E;">Indexed Query: student_id</div>
                    <div style="font-size: 0.78rem; color: #E8EDF4; margin-top: 4px;">
                        Execution Stage: <code>{stage_idx}</code><br>
                        Index Name: <code>{index_used}</code><br>
                        Complexity: Logarithmic B-Tree lookup.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_ex2:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #EF4444;">Unindexed Query: Age at enrollment</div>
                    <div style="font-size: 0.78rem; color: #E8EDF4; margin-top: 4px;">
                        Execution Stage: <code>{stage_unidx}</code><br>
                        Index Name: <code>None</code><br>
                        Complexity: Linear Collection Scan (COLLSCAN).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_compass:
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #9AA7B8; line-height: 1.5;">
                1. <b>Connect:</b> Open MongoDB Compass and connect to your cluster.<br>
                2. <b>Database:</b> Select <code>university_dropout_db</code>.<br>
                3. <b>Collections:</b> Inspect <code>students</code> (4,424 documents) and <code>predictions</code>.<br>
                4. <b>Indexes:</b> Select <code>students</code> &rarr; <i>Indexes</i> tab to verify <code>idx_unique_student_id</code>.<br>
                5. <b>Query Console:</b> Open <code>_mongosh</code> at the bottom to execute verification scripts.
            </div>
            """,
            unsafe_allow_html=True,
        )
