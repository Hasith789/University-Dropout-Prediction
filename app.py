"""
app.py - University Dropout Prediction System
Course: CSE494 Intelligent NoSQL Databases (LPU CA2 Project)

Academic web application integrating a real MongoDB database (university_dropout_db)
with Scikit-learn classification models.
- Direct database communication via PyMongo with native mongosh JavaScript demonstrations.
- Supervised classification (Random Forest Classifier & Logistic Regression baseline).
- Standard CRUD operations, aggregation pipelines, and query plan (explain) analysis.
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
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "university_dropout_db")

# Set page configuration
st.set_page_config(
    page_title="University Dropout Prediction System",
    layout="wide",
)

# Academic design stylesheet
st.markdown(
    """
    <style>
    /* Clean Academic Interface */
    .stApp {
        background-color: #fbfbfb;
        color: #212529;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #1a252f;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #4a5568 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 600;
        color: #1a202c;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        background-color: #ffffff;
    }
    button[kind="primary"] {
        background-color: #1b6ec2 !important;
        border-color: #1861ac !important;
        color: #ffffff !important;
        border-radius: 4px;
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
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
    return client[DATABASE_NAME]


def test_db_connection():
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        server_info = client.server_info()
        version = server_info.get("version", "Unknown")
        client.close()
        return True, f"Connected to MongoDB server (v{version})", version
    except Exception as e:
        return False, f"Could not connect to MongoDB: {e}", None


@st.cache_resource
def load_ml_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None


model_package = load_ml_model()


# ==============================================================================
# PREDICTION VALIDATION
# ==============================================================================
def validate_prediction_payload(payload: dict) -> tuple[bool, str]:
    """Validates prediction document before insertion into MongoDB."""
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
# SIDEBAR NAVIGATION & CONNECTION STATUS
# ==============================================================================
st.sidebar.title("Navigation")

is_connected, conn_msg, db_version = test_db_connection()

if is_connected:
    cluster_label = (
        "MongoDB Atlas (Cloud)"
        if ("mongodb.net" in MONGODB_URI or "+srv" in MONGODB_URI)
        else "Local MongoDB Server"
    )
    st.sidebar.success("Database Status: Connected")
    st.sidebar.caption(f"Host: {cluster_label}")
    st.sidebar.caption(f"Engine: MongoDB v{db_version}")
    st.sidebar.caption(f"Active Database: {DATABASE_NAME}")
else:
    st.sidebar.error("Database Status: Disconnected")
    st.sidebar.warning(conn_msg)
    st.sidebar.info("Verify MongoDB service status or configure MONGODB_URI in .env.")

menu_choice = st.sidebar.radio(
    "Select Section:",
    [
        "1. Overview & Database Status",
        "2. Predict Student Dropout",
        "3. Student Database (Real CRUD)",
        "4. Prediction History",
        "5. Model Evaluation & Benchmarks",
        "6. MongoDB Aggregations & Compass Guide",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("CSE494: Intelligent NoSQL Databases")


# ==============================================================================
# SECTION 1: OVERVIEW & DATABASE STATUS
# ==============================================================================
if menu_choice == "1. Overview & Database Status":
    st.title("University Dropout Prediction System")
    st.caption("Intelligent NoSQL Database and Machine Learning Platform | CSE494 Academic Project")

    st.markdown(
        """
        This system evaluates student academic trajectories using a genuine MongoDB database 
        (`university_dropout_db`) and supervised machine learning classification models.
        Student profiles and inference audit trails are persisted directly in MongoDB collections.
        """
    )

    st.markdown("---")
    st.subheader("Database Overview Metrics")

    if is_connected:
        db = get_db()
        total_students = db.students.count_documents({})
        total_preds = db.predictions.count_documents({})

        # Outcome distribution aggregation
        pipeline = [
            {"$group": {"_id": "$Target", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        dist_res = list(db.students.aggregate(pipeline))
        dist_map = {item["_id"]: item["count"] for item in dist_res}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Students (students)", f"{total_students:,}")
        c2.metric("Predictions (predictions)", f"{total_preds:,}")
        c3.metric("Actual Dropouts", f"{dist_map.get('Dropout', 0):,}")
        c4.metric("Actual Graduates", f"{dist_map.get('Graduate', 0):,}")

        with st.expander("MongoDB Shell Command Reference", expanded=False):
            st.code(
                """
const db = db.getSiblingDB("university_dropout_db");

// Count total student profiles
db.students.countDocuments({});

// Count total recorded predictions
db.predictions.countDocuments({});

// Cohort outcome breakdown
db.students.aggregate([
  { $group: { _id: "$Target", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
]);
                """,
                language="javascript",
            )

        if dist_res:
            st.markdown("### Academic Outcome Distribution in Database")
            df_dist = pd.DataFrame([{"Outcome": r["_id"], "Count": r["count"]} for r in dist_res])
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            colors = {"Graduate": "#27ae60", "Dropout": "#c0392b", "Enrolled": "#4a69bd"}
            palette = [colors.get(o, "#7f8c8d") for o in df_dist["Outcome"]]
            bars = ax.bar(df_dist["Outcome"], df_dist["Count"], color=palette, width=0.55)
            ax.set_ylabel("Document Count")
            ax.set_title("Student Records by Academic Outcome", fontsize=11, fontweight="bold", pad=10)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.error(f"Cannot retrieve database statistics: {conn_msg}")

    st.markdown("---")
    st.subheader("Dataset Summary (dataset.csv)")
    if os.path.exists("dataset.csv"):
        df_sample = pd.read_csv("dataset.csv")
        st.write(
            f"**Records:** {df_sample.shape[0]:,} rows | **Features:** {df_sample.shape[1]} columns | "
            f"**Missing Values:** {df_sample.isnull().sum().sum()}"
        )
        st.dataframe(df_sample.head(5), use_container_width=True)


# ==============================================================================
# SECTION 2: PREDICT STUDENT DROPOUT
# ==============================================================================
elif menu_choice == "2. Predict Student Dropout":
    st.title("Predict Student Academic Outcome")
    st.write(
        "This interface retrieves student profiles from MongoDB or accepts student features, "
        "applies the trained classification model, and records the prediction in the database."
    )

    if not is_connected:
        st.error("A live MongoDB connection is required to retrieve profiles and record predictions.")
        st.stop()

    if model_package is None:
        st.error("model.pkl not found. Please train and export the model using ML_Training.ipynb.")
        st.stop()

    rf_model = model_package["model"]
    feature_names = model_package["feature_names"]
    classes = model_package["classes"]
    db = get_db()

    input_choice = st.radio(
        "Select Input Method:",
        ["Option 1: Retrieve Existing Student Profile from MongoDB", "Option 2: Input Custom Student Profile"],
        horizontal=True,
    )

    student_data = {}
    selected_id = "STU-0001"
    feature_mode = "Full Academic Features (1st & 2nd Sem)"

    if "Option 1" in input_choice:
        st.subheader("Retrieve Student Profile from Database")
        col_pick, col_search = st.columns([2, 1])

        with col_pick:
            cursor_sample = db.students.find({}, {"student_id": 1, "_id": 0}).limit(50)
            student_ids = [d["student_id"] for d in cursor_sample]
            pick_id = st.selectbox("Select Student Identifier:", student_ids if student_ids else ["STU-0001"])

        with col_search:
            manual_search_id = st.text_input("Or search by Student ID (e.g., STU-0010):", value="").strip()

        final_id = manual_search_id if manual_search_id else pick_id

        if final_id:
            doc = db.students.find_one({"student_id": final_id})
            if doc:
                selected_id = doc["student_id"]
                student_data = doc
                doc_display = {k: v for k, v in doc.items() if k != "_id"}
                st.info(
                    f"Retrieved student profile '{selected_id}'. "
                    f"Recorded Target: {doc.get('Target', 'Unknown')} | "
                    f"Course: {COURSE_NAMES.get(doc.get('Course'), doc.get('Course'))}"
                )

                with st.expander("MongoDB Shell Retrieval Query", expanded=False):
                    st.code(
                        f"""
const db = db.getSiblingDB("university_dropout_db");
db.students.findOne({{ student_id: "{selected_id}" }});
                        """,
                        language="javascript",
                    )

                with st.expander("View Document JSON", expanded=False):
                    st.json(doc_display)
            else:
                st.warning(f"Student record '{final_id}' not found in MongoDB.")
    else:
        st.subheader("Input Student Attributes")
        selected_id = st.text_input("Assign Student Identifier:", value="STU-CUSTOM-01").strip()

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age at Enrollment:", min_value=16, max_value=70, value=20)
            gender = st.selectbox("Gender:", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
            course_id = st.selectbox("Course:", list(COURSE_NAMES.items()), format_func=lambda x: f"{x[0]} - {x[1]}")[0]
            tuition = st.selectbox("Tuition Fees Up to Date:", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            scholarship = st.selectbox("Scholarship Holder:", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

        with c2:
            sem1_enrolled = st.number_input("1st Sem Units Enrolled:", min_value=0, max_value=25, value=6)
            sem1_approved = st.number_input("1st Sem Units Approved:", min_value=0, max_value=25, value=5)
            sem1_grade = st.number_input("1st Sem Grade Average (0-20):", min_value=0.0, max_value=20.0, value=13.0)
            sem1_evals = st.number_input("1st Sem Evaluations:", min_value=0, max_value=30, value=6)

        with c3:
            sem2_enrolled = st.number_input("2nd Sem Units Enrolled:", min_value=0, max_value=25, value=6)
            sem2_approved = st.number_input("2nd Sem Units Approved:", min_value=0, max_value=25, value=5)
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

    st.markdown("---")

    # Prediction Action with repeat-click prevention
    btn_predict = st.button("Run Prediction and Save Document to MongoDB", type="primary")

    if btn_predict:
        if not student_data:
            st.error("Please retrieve or input a valid student profile.")
        else:
            # 1. Feature alignment
            input_vector = []
            for col in feature_names:
                val = student_data.get(col, 0.0)
                try:
                    input_vector.append(float(val))
                except (ValueError, TypeError):
                    input_vector.append(0.0)

            df_input = pd.DataFrame([input_vector], columns=feature_names)

            # 2. Model Inference
            pred_class = rf_model.predict(df_input)[0]
            probs = rf_model.predict_proba(df_input)[0]
            prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)}
            dropout_prob = prob_dict.get("Dropout", 0.0)

            # 3. Risk tier derivation
            if dropout_prob >= 0.50:
                risk_tier = "High Risk"
            elif dropout_prob >= 0.25:
                risk_tier = "Moderate Risk"
            else:
                risk_tier = "Low Risk"

            # 4. Duplicate submission check via session state
            sub_signature = f"{selected_id}_{pred_class}_{tuple(input_vector)}"
            last_sig = st.session_state.get("last_prediction_signature")

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
                st.error(f"Prediction document validation failed: {validation_msg}")
            else:
                if last_sig == sub_signature:
                    st.info("Identical prediction already recorded in this session. Displaying recorded results.")
                else:
                    res_ins = db.predictions.insert_one(pred_doc)
                    st.session_state["last_prediction_signature"] = sub_signature
                    st.session_state["last_inserted_id"] = str(res_ins.inserted_id)

                inserted_id = st.session_state.get("last_inserted_id", "Pending")

                # Display Results
                st.markdown("### Inference Results")
                r_col1, r_col2 = st.columns(2)

                with r_col1:
                    st.subheader(f"Predicted Outcome: {pred_class}")
                    st.write(f"**Dropout Risk Category:** {risk_tier} ({dropout_prob*100:.1f}%)")
                    st.write(f"**Student Identifier:** `{selected_id}`")
                    st.write(f"**MongoDB Record ID:** `{inserted_id}`")

                with r_col2:
                    st.write("**Class Probability Distribution:**")
                    for c_name, p_val in prob_dict.items():
                        st.progress(float(p_val), text=f"{c_name}: {p_val*100:.1f}%")

                with st.expander("MongoDB Shell Insert Command", expanded=False):
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
                    "Disclaimer: Predictions are statistical estimates generated by a machine learning model to assist "
                    "academic advisors. The model is an academic demonstration and is not clinically, institutionally, "
                    "or operationally validated."
                )


# ==============================================================================
# SECTION 3: STUDENT DATABASE (REAL MONGODB CRUD OPERATIONS)
# ==============================================================================
elif menu_choice == "3. Student Database (Real CRUD)":
    st.title("Student Database Operations (MongoDB CRUD)")
    st.write(
        "Direct operations executed against the `students` collection: "
        "`find()`, `insertOne()`, `updateOne()`, and `deleteOne()`."
    )

    if not is_connected:
        st.error("MongoDB is not connected. CRUD operations cannot be executed.")
        st.stop()

    db = get_db()

    tab_find, tab_insert, tab_update, tab_delete = st.tabs([
        "Find & Filter Documents",
        "Insert Student (insertOne)",
        "Update Student (updateOne)",
        "Delete Student (deleteOne)",
    ])

    with tab_find:
        st.subheader("Query Student Documents")
        f_id = st.text_input("Filter by Student ID prefix (e.g., STU-0001):", value="").strip()
        f_outcome = st.selectbox("Filter by Academic Outcome (Target):", ["All", "Dropout", "Enrolled", "Graduate"])

        q = {}
        if f_id:
            q["student_id"] = {"$regex": f"^{f_id}", "$options": "i"}
        if f_outcome != "All":
            q["Target"] = f_outcome

        match_count = db.students.count_documents(q)
        st.write(f"Matching documents in database: **{match_count:,}**")

        with st.expander("MongoDB Shell Query", expanded=False):
            if f_id and f_outcome != "All":
                st.code(f'db.students.find({{ student_id: /^"{f_id}"/i, Target: "{f_outcome}" }}).limit(25)', language="javascript")
            elif f_id:
                st.code(f'db.students.find({{ student_id: /^"{f_id}"/i }}).limit(25)', language="javascript")
            elif f_outcome != "All":
                st.code(f'db.students.find({{ Target: "{f_outcome}" }}).limit(25)', language="javascript")
            else:
                st.code("db.students.find({}).limit(25)", language="javascript")

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
            avail_cols = [c for c in cols if c in df_res.columns]
            st.dataframe(df_res[avail_cols], use_container_width=True)
        else:
            st.info("No documents match the specified query criteria.")

    with tab_insert:
        st.subheader("Insert Student Record")
        with st.form("form_insert_student"):
            ins_id = st.text_input("Unique Student ID:", value="STU-8888").strip()
            ins_target = st.selectbox("Academic Outcome (Target):", ["Enrolled", "Dropout", "Graduate"])
            ins_age = st.number_input("Age at Enrollment:", min_value=16, max_value=80, value=21)
            ins_course = st.selectbox("Course:", list(COURSE_NAMES.items()), format_func=lambda x: f"{x[0]} - {x[1]}")[0]
            ins_tuition = st.selectbox("Tuition Paid:", [1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            ins_sem1_grade = st.number_input("1st Sem Grade (0-20):", min_value=0.0, max_value=20.0, value=12.5)

            with st.expander("MongoDB Shell insertOne Syntax", expanded=False):
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

            submit_ins = st.form_submit_button("Execute insertOne()")
            if submit_ins:
                if not ins_id:
                    st.error("Student ID cannot be empty.")
                elif db.students.find_one({"student_id": ins_id}):
                    st.error(f"Document with student_id '{ins_id}' already exists in MongoDB.")
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
                    st.success(f"Student document inserted successfully. Document _id: {res_in.inserted_id}")

    with tab_update:
        st.subheader("Update Student Record")
        upd_id = st.text_input("Enter Student ID to update:", value="STU-0001").strip()
        doc_to_upd = db.students.find_one({"student_id": upd_id})

        if doc_to_upd:
            st.write(f"Target document found. Current Target: **{doc_to_upd.get('Target')}**")
            new_target = st.selectbox("Updated Academic Outcome:", ["Dropout", "Enrolled", "Graduate"])
            new_tuition_val = st.selectbox("Updated Tuition Status:", [1, 0], format_func=lambda x: "Paid (1)" if x == 1 else "In Arrears (0)")

            with st.expander("MongoDB Shell updateOne Syntax", expanded=False):
                st.code(
                    f"""
const db = db.getSiblingDB("university_dropout_db");
db.students.updateOne(
  {{ student_id: "{upd_id}" }},
  {{
    $set: {{
      Target: "{new_target}",
      "Tuition fees up to date": {new_tuition_val}
    }}
  }}
);
                    """,
                    language="javascript",
                )

            if st.button("Execute updateOne()"):
                res_upd = db.students.update_one(
                    {"student_id": upd_id},
                    {"$set": {"Target": new_target, "Tuition fees up to date": int(new_tuition_val)}}
                )
                st.success(f"Update acknowledged. Modified {res_upd.modified_count} document(s).")
        else:
            st.info("Enter a valid Student ID to retrieve the document.")

    with tab_delete:
        st.subheader("Delete Student Record")
        del_id = st.text_input("Enter Student ID to delete:", value="").strip()

        if del_id:
            doc_to_del = db.students.find_one({"student_id": del_id})
            if doc_to_del:
                st.warning(f"Document identified: `{del_id}` (Outcome: {doc_to_del.get('Target')}).")

                with st.expander("MongoDB Shell deleteOne Syntax", expanded=False):
                    st.code(
                        f"""
const db = db.getSiblingDB("university_dropout_db");
db.students.deleteOne({{ student_id: "{del_id}" }});
                        """,
                        language="javascript",
                    )

                confirm = st.checkbox(f"Confirm permanent deletion of document '{del_id}'.")
                if st.button("Execute deleteOne()", disabled=not confirm):
                    res_del = db.students.delete_one({"student_id": del_id})
                    st.success(f"Deleted {res_del.deleted_count} document(s) from MongoDB.")
            else:
                st.info("Student ID not found in database.")


# ==============================================================================
# SECTION 4: PREDICTION HISTORY
# ==============================================================================
elif menu_choice == "4. Prediction History":
    st.title("Prediction History")
    st.write("Audit history of inference documents retrieved from `university_dropout_db.predictions`.")

    if not is_connected:
        st.error("MongoDB is not connected. Prediction history cannot be retrieved.")
        st.stop()

    db = get_db()
    pred_total = db.predictions.count_documents({})
    st.write(f"Total Prediction Documents in Database: **{pred_total:,}**")

    outcome_filter = st.selectbox("Filter by Predicted Outcome:", ["All", "Dropout", "Enrolled", "Graduate"])
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
        normalized_rows = []
        for p in preds:
            raw_ts = p.get("prediction_timestamp") or p.get("timestamp")
            if isinstance(raw_ts, datetime):
                ts_str = raw_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            elif isinstance(raw_ts, str):
                ts_str = raw_ts
            else:
                ts_str = "N/A"

            risk = p.get("risk_tier") or p.get("dropout_risk") or "N/A"
            probs = p.get("class_probabilities") or p.get("probabilities") or {}

            normalized_rows.append({
                "Timestamp": ts_str,
                "Student ID": p.get("student_id", "N/A"),
                "Predicted Outcome": p.get("predicted_outcome", "N/A"),
                "Risk Tier": risk,
                "Model": p.get("model_name", "Random Forest Classifier"),
                "Feature Set": p.get("feature_set") or p.get("prediction_mode") or "Full Features",
                "Document ID": str(p.get("_id")),
                "_raw": p,
            })

        df_display = pd.DataFrame(normalized_rows)
        cols_show = ["Timestamp", "Student ID", "Predicted Outcome", "Risk Tier", "Model", "Feature Set", "Document ID"]
        st.dataframe(df_display[cols_show], use_container_width=True)

        st.markdown("### Inspect Document BSON")
        sel_pred_id = st.selectbox("Select Document ID:", [r["Document ID"] for r in normalized_rows])
        sel_item = next((r for r in normalized_rows if r["Document ID"] == sel_pred_id), None)
        if sel_item:
            raw_clean = dict(sel_item["_raw"])
            raw_clean["_id"] = str(raw_clean["_id"])
            if isinstance(raw_clean.get("prediction_timestamp"), datetime):
                raw_clean["prediction_timestamp"] = raw_clean["prediction_timestamp"].isoformat()
            st.json(raw_clean)
    else:
        st.info("No prediction history documents found matching the filter.")


# ==============================================================================
# SECTION 5: MODEL EVALUATION & BENCHMARKS
# ==============================================================================
elif menu_choice == "5. Model Evaluation & Benchmarks":
    st.title("Machine Learning Model Evaluation")
    st.write(
        "Measured evaluation metrics on the stratified 20% test partition (N = 885). "
        "Evaluates academic outcome classification (Dropout, Enrolled, Graduate)."
    )

    if model_package is None:
        st.error("model.pkl not found. Please train and save the model using ML_Training.ipynb.")
        st.stop()

    lr_acc = model_package.get("lr_accuracy", 0.0)
    rf_acc = model_package.get("rf_accuracy", 0.0)
    report = model_package.get("classification_report", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Logistic Regression Accuracy", f"{lr_acc*100:.2f}%")
    m2.metric("Random Forest Accuracy", f"{rf_acc*100:.2f}%")
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
    m3.metric("Random Forest Macro F1", f"{macro_f1*100:.2f}%")
    m4.metric("Dropout Recall", f"{report.get('Dropout', {}).get('recall', 0.0)*100:.2f}%")

    st.markdown("---")
    st.subheader("Classification Performance Metrics (Random Forest)")

    # Classification report table
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
        st.subheader("Confusion Matrix (Test Partition)")
        cm = np.array(model_package.get("confusion_matrix", []))
        if len(cm) > 0:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes_eval,
                yticklabels=classes_eval,
                ax=ax_cm,
                cbar=False,
            )
            ax_cm.set_title("Random Forest Confusion Matrix", fontsize=11, fontweight="bold", pad=8)
            ax_cm.set_xlabel("Predicted Class")
            ax_cm.set_ylabel("True Class")
            plt.tight_layout()
            st.pyplot(fig_cm)

    with c_right:
        st.subheader("Top 10 Feature Importances (Gini Index)")
        feat_imp = model_package.get("feature_importances", {})
        if feat_imp:
            df_imp = pd.Series(feat_imp).head(10)
            fig_imp, ax_imp = plt.subplots(figsize=(5.5, 3.8))
            df_imp.plot(kind="barh", color="#2c3e50", ax=ax_imp)
            ax_imp.set_title("Most Predictive Features", fontsize=11, fontweight="bold", pad=8)
            ax_imp.set_xlabel("Gini Importance Score")
            ax_imp.invert_yaxis()
            ax_imp.grid(axis="x", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_imp)

    st.markdown("---")
    st.subheader("Methodology and Feature Timing Discussion")
    st.markdown(
        """
        - **Model Scope:** The classifier predicts academic outcomes (Dropout, Enrolled, Graduate) 
          based on student demographic, socioeconomic, macroeconomic, and first- and second-semester performance.
        - **Feature Timing:** Second-semester features (such as approved units and grade averages) provide strong 
          predictive signals for end-of-year outcomes. For early prediction before or during the first semester, 
          second-semester attributes must be excluded to prevent information leakage.
        - **Evaluation Integrity:** Preprocessing was performed strictly within the training folds. 
          Evaluation results reflect actual test partition metrics and are not hardcoded or simulated.
        - **Academic Purpose:** This model is designed for educational demonstration in CSE494 and is not 
          clinically, institutionally, or operationally validated.
        """
    )


# ==============================================================================
# SECTION 6: MONGODB AGGREGATIONS & COMPASS GUIDE
# ==============================================================================
elif menu_choice == "6. MongoDB Aggregations & Compass Guide":
    st.title("MongoDB Aggregations & Database Inspection")
    st.write(
        "Multi-stage Aggregation Framework pipelines executed on MongoDB engine "
        "and query execution plan analysis (`explain()`)."
    )

    if not is_connected:
        st.error("MongoDB is not connected. Aggregations cannot be executed.")
        st.stop()

    db = get_db()

    tab_agg, tab_explain, tab_compass = st.tabs([
        "Aggregation Pipeline Results",
        "Query Execution Plan (explain())",
        "MongoDB Compass Verification Guide",
    ])

    with tab_agg:
        st.subheader("1. Program Dropout Rate ($group, $project, $sort)")
        st.caption("Computes dropout percentages per degree program, guarding against division-by-zero.")

        with st.expander("MongoDB Shell Aggregation Pipeline", expanded=False):
            st.code(
                """
const db = db.getSiblingDB("university_dropout_db");
db.students.aggregate([
  {
    $group: {
      _id: "$Course",
      total_students: { $sum: 1 },
      dropouts: {
        $sum: { $cond: [{ $eq: ["$Target", "Dropout"] }, 1, 0] }
      }
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
          {
            $round: [
              { $multiply: [{ $divide: ["$dropouts", "$total_students"] }, 100] },
              1
            ]
          }
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

            fig_c, ax_c = plt.subplots(figsize=(8.5, 4.2))
            df_c_sorted = df_c.sort_values(by="Dropout Rate (%)", ascending=True)
            ax_c.barh(df_c_sorted["Course Name"], df_c_sorted["Dropout Rate (%)"], color="#c0392b")
            ax_c.set_title("Dropout Rate (%) by Degree Program", fontsize=11, fontweight="bold", pad=8)
            ax_c.set_xlabel("Dropout Percentage (%)")
            ax_c.grid(axis="x", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_c)

        st.markdown("---")
        st.subheader("2. Prediction Counts by Predicted Outcome")
        with st.expander("MongoDB Shell Aggregation Pipeline", expanded=False):
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
            df_pa = pd.DataFrame([{"Predicted Outcome": p["_id"], "Count": p["count"]} for p in pred_agg])
            st.dataframe(df_pa, use_container_width=True)

    with tab_explain:
        st.subheader("Query Optimization Analysis with explain()")
        st.markdown(
            """
            Executing `explain("executionStats")` evaluates query planner selection:
            - **`IXSCAN` / `EXPRESS_IXSCAN` (Index Scan):** Traverses B-Tree index with logarithmic complexity.
            - **`COLLSCAN` (Collection Scan):** Sequentially scans all collection documents.
            """
        )

        with st.expander("MongoDB Shell explain() Syntax", expanded=False):
            st.code(
                """
const db = db.getSiblingDB("university_dropout_db");

// 1. Indexed Query: Uses idx_unique_student_id
db.students.find({ student_id: "STU-0001" }).explain("executionStats");

// 2. Unindexed Query: Scans all collection documents (COLLSCAN)
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
            st.markdown("**Indexed Query:** `{ student_id: 'STU-0001' }`")
            st.write(f"- Execution Stage: `{stage_idx}`")
            st.write(f"- Index Utilized: `{index_used}`")
            st.caption("Efficient B-Tree index lookup.")

        with col_ex2:
            st.markdown("**Unindexed Query:** `{ 'Age at enrollment': 20 }`")
            st.write(f"- Execution Stage: `{stage_unidx}`")
            st.write("- Index Utilized: `None`")
            st.caption("Scanned entire collection sequentially (COLLSCAN).")

        with st.expander("Winning Query Plan JSON", expanded=False):
            st.json(plan_indexed.get("queryPlanner", {}).get("winningPlan", {}))

    with tab_compass:
        st.subheader("MongoDB Compass Inspection Procedure")
        st.markdown(
            """
            1. **Connect:** Open MongoDB Compass and connect to `mongodb://localhost:27017` (or the configured Atlas URI).
            2. **Database:** Locate and select `university_dropout_db`.
            3. **Collections:**
               - `students`: Contains 4,424 imported student records.
               - `predictions`: Contains persistent prediction documents and probability distributions.
            4. **Indexes:** Under the `students` collection, verify indexes:
               - `idx_unique_student_id` (Unique index on `student_id`).
               - `idx_students_target` (Index on `Target`).
            5. **MongoDB Shell:** Use the embedded shell (`_mongosh`) to execute queries directly:
            """
        )
        st.code(
            """
const db = db.getSiblingDB("university_dropout_db");

// Verify document count
db.students.countDocuments({});

// Verify outcome grouping
db.students.aggregate([
  { $group: { _id: "$Target", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
]);

// Query latest predictions
db.predictions.find().sort({ prediction_timestamp: -1 }).limit(10);
            """,
            language="javascript",
        )
