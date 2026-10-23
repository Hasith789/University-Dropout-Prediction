# University Dropout Prediction System

**Course:** CSE494: Intelligent NoSQL Databases | Academic Project  
**Institution:** Lovely Professional University (LPU)  
**Database Interface:** Real MongoDB via PyMongo (`database.py`, `app.py`)  
**Database Engine:** MongoDB Community Server / MongoDB Atlas (`university_dropout_db`)  
**Collections:** `students` (4,424 documents), `predictions` (audit logs)  
**Machine Learning:** Scikit-learn (Random Forest Classifier & Logistic Regression baseline)  
**User Interface:** Streamlit Academic Dark-Mode Dashboard (`app.py`)  

---

## 1. Project Architecture & Database Layer

All application-level database operations are implemented using **PyMongo** inside [`database.py`](file:///c:/Users/HP/OneDrive/Documents/Desktop/New%20folder/Project/database.py). 

- **Connection Management:** `MongoClient` configured with connection pooling (`@st.cache_resource`), timeouts, and `certifi` TLS root certificates for MongoDB Atlas.
- **Connection Verification:** Live connection verified with administrative `ping` command before reporting connected status.
- **Data Integrity:** B-Tree unique index on `student_id` prevents duplicate student entries; compound and reverse-chronological indexes optimize prediction queries.
- **Pure PyMongo API:** All queries, CRUD operations, aggregations, and query plan inspections use native PyMongo methods:
  - `find()`, `find_one()`
  - `insert_one()`
  - `update_one()` with `$set`
  - `delete_one()`
  - `count_documents()`
  - `aggregate()` with multi-stage pipelines
  - `create_index()`
  - `explain()`
- **MongoDB Shell Scripts:** Standalone JavaScript scripts in [`mongodb/`](file:///c:/Users/HP/OneDrive/Documents/Desktop/New%20folder/Project/mongodb) are maintained for CA2 viva demonstrations in `mongosh` and MongoDB Compass.

---

## 2. Directory Structure

```text
university-dropout-prediction/
├── dataset.csv                  # Original Kaggle/UCI student dataset (4,424 records)
├── students_prepared.csv        # Prepared CSV with stable student_id for import
├── ML_Training.ipynb            # Standard college Jupyter Notebook (10 sequential ML sections)
├── model.pkl                    # Exported model package (RF & LR models, scaler, metrics)
├── database.py                  # PyMongo database layer (CRUD, aggregations, indexes, explain)
├── app.py                       # Streamlit web application (compact dark-mode dashboard)
├── requirements.txt             # Python dependencies (PyMongo, Scikit-learn, Streamlit, etc.)
├── .env.example                 # Environment variable template
├── .env                         # Active local or cloud connection string (ignored by git)
├── .gitignore                   # Version control exclusions
├── README.md                    # Project documentation and setup guide
├── .streamlit/
│   ├── config.toml              # Dark theme configuration
│   └── secrets.toml.example     # Streamlit Cloud secrets template
└── mongodb/                     # 100% Native MongoDB Shell (mongosh) JavaScript layer
    ├── setup.js                 # Database & collection initialization (ping, list databases)
    ├── import_students.js       # Native mongosh CSV data importer (bulkWrite with upsert)
    ├── crud.js                  # Standard CRUD demonstration
    ├── predictions.js           # Prediction logging demonstration
    ├── aggregation.js           # Multi-stage pipelines ($group, $project, $sort, $match)
    ├── indexes.js               # B-Tree index creation and query optimization (explain())
    └── verify.js                # Automated 7-point verification suite (7/7 checks passed)
```

---

## 3. Installation & Local Setup

### Step 1: Install Python Dependencies
Ensure Python 3.10 or compatible is installed, then run:
```powershell
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Create or edit your [`.env`](file:///c:/Users/HP/OneDrive/Documents/Desktop/New%20folder/Project/.env) file:

```env
# For MongoDB Atlas (Cloud):
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority

# Or for Local MongoDB Community Server:
# MONGODB_URI=mongodb://localhost:27017

MONGODB_DATABASE=university_dropout_db
```

### Step 3: Import the Dataset
If running against a fresh MongoDB instance, import all 4,424 records from `students_prepared.csv`:
```powershell
# Using mongosh:
mongosh "YOUR_MONGODB_URI" mongodb/import_students.js

# Or using PyMongo:
python -c "import database; db, _, _ = database.get_active_db(); print('Students:', database.count_students(db))"
```

### Step 4: Run the Streamlit Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 4. Machine Learning Models & Evaluation Methodology

The machine learning workflow uses Scikit-learn on a held-out stratified 20% test partition ($N = 885$) from `dataset.csv`:

### Models Trained:
1. **Logistic Regression (Baseline):** Preprocessing with `StandardScaler` fitted strictly on training data inside a `Pipeline` to prevent target leakage.
2. **Random Forest Classifier (Selected):** Ensemble model with 100 decision trees. Selected based on superior Macro F1 and Dropout recall.

### Measured Evaluation Benchmarks (Test Set $N = 885$):

| Metric | Logistic Regression (Baseline) | Random Forest Classifier (Selected) |
| :--- | :---: | :---: |
| **Accuracy** | 75.82% | **77.85%** |
| **Macro Precision** | 69.60% | **73.29%** |
| **Macro Recall** | 66.52% | **68.95%** |
| **Macro F1-Score** | 67.25% | **70.14%** |
| **Weighted F1-Score** | 74.32% | **76.46%** |
| **Dropout Precision** | 79.18% | **81.92%** |
| **Dropout Recall** | 75.00% | **75.00%** |
| **Dropout F1-Score** | 77.03% | **78.31%** |

### Target Classes:
- `Dropout`
- `Enrolled`
- `Graduate`

### Feature Timing & Data Leakage Discussion:
- Second-semester curricular units approved and grade averages are the highest-ranked predictive features in Random Forest Gini importance.
- If designing an **early pre-enrollment screening model**, second-semester attributes must be excluded to prevent information leakage.
- Predictions are statistical estimates intended to assist academic advisors and are not clinically, institutionally, or operationally validated.

---

## 5. Deployment to Streamlit Community Cloud

### Step 1: Configure Secrets
In your Streamlit Cloud app dashboard:
1. Go to **Settings** ➔ **Secrets**.
2. Add your Atlas credentials:
```toml
MONGODB_URI = "mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
MONGODB_DATABASE = "university_dropout_db"
```
3. Click **Save**.

### Step 2: Whitelist Streamlit Cloud IP in MongoDB Atlas
1. Log in to [cloud.mongodb.com](https://cloud.mongodb.com/).
2. Navigate to **Network Access** under Security.
3. Add IP Address: `0.0.0.0/0` (Allow access from anywhere).  
   *(Required because Streamlit Cloud runs on dynamic cloud IP addresses).*

---

## 6. Academic Viva Reference (CSE494)

### Q1: Why use PyMongo instead of raw shell commands in production?
* **Answer:** PyMongo provides connection pooling (`MongoClient`), automatic type translation between Python and BSON, parameter injection protection, and native integration with Pandas DataFrames and web frameworks like Streamlit.

### Q2: What is the difference between Embedding and Referencing in this project?
* **Answer:** Class probabilities and student snapshots are **embedded** inside individual prediction documents for atomic retrieval. However, predictions **reference** the `students` collection via `student_id` to allow infinite audit growth without exceeding the 16MB document size limit.

### Q3: How do indexes improve query speed and how does `explain()` prove it?
* **Answer:** B-Tree indexes create sorted pointer structures in memory. Running `db.students.find({ student_id: "STU-0001" }).explain()` demonstrates that MongoDB uses `EXPRESS_IXSCAN` / `IXSCAN` ($O(\log N)$) rather than a full collection scan `COLLSCAN` ($O(N)$).

### Q4: Explain the Aggregation Pipeline stages used in course dropout calculation.
* **Answer:**
  - `$group`: Groups documents by `Course`, computes total students, and conditionally accumulates dropouts using `$cond`.
  - `$project`: Calculates derived percentage fields while guarding against division-by-zero using `$cond: [ { $eq: ["$total", 0] }, 0, ... ]`.
  - `$sort`: Orders documents descending by dropout rate (`-1`).
  - `$limit`: Restricts output to the top programs.
