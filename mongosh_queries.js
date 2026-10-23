/**
 * mongosh_queries.js - MongoDB Shell & Compass Command Reference
 * Course: CSE494: Intelligent NoSQL Databases (CA2 Academic Evaluation)
 * 
 * Demonstrates native MongoDB Shell commands for:
 * 1. Database selection & indexing
 * 2. CRUD operations
 * 3. Aggregation pipelines
 * 4. Query plan inspection with explain()
 */

// ==============================================================================
// 1. SELECT DATABASE
// ==============================================================================
// Select the project database (Valid MongoDB Shell JavaScript syntax):
const db = db.getSiblingDB("university_dropout_db");

// (Tip for interactive mongosh or MongoDB Compass shell, you can also type:
//   use university_dropout_db
//   show collections)

// Show current active database
print("Active Database:", db.getName());

// Show all collections in this database
print("Active Collections:", db.getCollectionNames());


// ==============================================================================
// 2. CREATE BENEFICIAL INDEXES
// ==============================================================================
// (A) Unique Index on 'student_id':
// Purpose: Guarantees no two students have the same ID, and gives instant O(1) lookups
db.students.createIndex(
  { student_id: 1 },
  { unique: true, name: "idx_unique_student_id" }
);

// (B) Single-field Index on 'Target':
// Purpose: Optimizes queries that filter or group by academic status (Dropout, Enrolled, Graduate)
db.students.createIndex(
  { Target: 1 },
  { name: "idx_students_target" }
);

// (C) Index on 'prediction_timestamp':
// Purpose: Enables fast reverse-chronological retrieval of predictions (latest first)
db.predictions.createIndex(
  { prediction_timestamp: -1 },
  { name: "idx_pred_timestamp" }
);

// (D) Index on 'student_id' in predictions:
// Purpose: Allows fast lookups when joining student records with their prediction audit history
db.predictions.createIndex(
  { student_id: 1 },
  { name: "idx_pred_student_id" }
);

// View all active indexes on both collections:
db.students.getIndexes();
db.predictions.getIndexes();


// ==============================================================================
// 3. READ QUERIES (FIND & COUNT)
// ==============================================================================
// Count total students (verifies exactly 4,424 documents imported)
db.students.countDocuments({});

// Count total predictions saved
db.predictions.countDocuments({});

// Find one sample student document
db.students.findOne();

// Search for a specific student by stable student_id (Uses Index: EXPRESS_IXSCAN)
db.students.findOne({ student_id: "STU-0001" });

// Find students who have dropped out (Limit 5)
db.students.find({ Target: "Dropout" }).limit(5);

// Find students with tuition in arrears (Tuition fees up to date = 0) who dropped out
db.students.find(
  { Target: "Dropout", "Tuition fees up to date": 0 },
  { student_id: 1, Target: 1, "Tuition fees up to date": 1, Course: 1, _id: 0 }
).limit(5);

// Sort students by Age at enrollment descending
db.students.find({}, { student_id: 1, "Age at enrollment": 1, Target: 1, _id: 0 })
  .sort({ "Age at enrollment": -1 })
  .limit(5);


// ==============================================================================
// 4. CREATE / INSERT OPERATIONS (insertOne)
// ==============================================================================
// Insert a new student document
db.students.insertOne({
  student_id: "STU-9999",
  Target: "Enrolled",
  "Age at enrollment": 20,
  Gender: 1,
  Course: 7, // Informatics Engineering
  "Tuition fees up to date": 1,
  "Scholarship holder": 0,
  "Curricular units 1st sem (approved)": 5,
  "Curricular units 1st sem (grade)": 14.2,
  "Curricular units 2nd sem (approved)": 5,
  "Curricular units 2nd sem (grade)": 13.8,
  "Unemployment rate": 10.8,
  "Inflation rate": 1.4,
  GDP: 1.74
});

// Insert an ML prediction document into 'predictions'
db.predictions.insertOne({
  student_id: "STU-9999",
  predicted_outcome: "Graduate",
  class_probabilities: {
    Dropout: 0.08,
    Enrolled: 0.12,
    Graduate: 0.80
  },
  model_name: "Random Forest Classifier",
  prediction_timestamp: "2026-10-23 11:45:00",
  feature_set: "Full Features (1st & 2nd Sem)",
  risk_tier: "Low Risk"
});


// ==============================================================================
// 5. UPDATE OPERATIONS (updateOne)
// ==============================================================================
// Update a student's status and tuition compliance using $set
db.students.updateOne(
  { student_id: "STU-9999" },
  {
    $set: {
      Target: "Graduate",
      "Tuition fees up to date": 1
    }
  }
);

// Verify the update
db.students.findOne({ student_id: "STU-9999" });


// ==============================================================================
// 6. DELETE OPERATIONS (deleteOne)
// ==============================================================================
// Delete the test student document
db.students.deleteOne({ student_id: "STU-9999" });

// Verify deletion (should return null)
db.students.findOne({ student_id: "STU-9999" });


// ==============================================================================
// 7. PREDICTION HISTORY QUERIES
// ==============================================================================
// Retrieve the 10 most recent saved predictions (sorted by timestamp descending)
db.predictions.find()
  .sort({ prediction_timestamp: -1 })
  .limit(10);

// Find predictions filtered by predicted outcome 'Dropout'
db.predictions.find({ predicted_outcome: "Dropout" })
  .sort({ prediction_timestamp: -1 });

// Find all predictions for a specific student
db.predictions.find({ student_id: "STU-0002" });


// ==============================================================================
// 8. MONGODB AGGREGATION PIPELINES ($match, $group, $sort, $project)
// ==============================================================================

// --- PIPELINE 1: Count Students by Actual Academic Outcome ($group, $sort) ---
// Pipeline Explanation:
// Stage 1: $group groups every document by the "Target" field and computes the total count.
// Stage 2: $sort orders the results from highest count to lowest (-1).
db.students.aggregate([
  {
    $group: {
      _id: "$Target",
      count: { $sum: 1 }
    }
  },
  {
    $sort: { count: -1 }
  }
]);


// --- PIPELINE 2: Dropout Rate per Degree Program ($group, $project, $sort) ---
// Pipeline Explanation:
// Stage 1: $group by Course, computing total students and counting dropouts using conditional $cond.
// Stage 2: $project derives the 'dropout_rate' percentage, guarding against division-by-zero.
// Stage 3: $sort lists the courses with the highest dropout rates first.
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
  {
    $sort: { dropout_rate: -1 }
  }
]);


// --- PIPELINE 3: Cross-Analysis ($match + $group) ---
// Pipeline Explanation:
// Stage 1: $match filters students who have unpaid tuition fees ("Tuition fees up to date" = 0).
// Stage 2: $group computes how many in this indebted cohort dropped out vs graduated.
db.students.aggregate([
  {
    $match: {
      "Tuition fees up to date": 0
    }
  },
  {
    $group: {
      _id: "$Target",
      indebted_students_count: { $sum: 1 }
    }
  },
  {
    $sort: { indebted_students_count: -1 }
  }
]);


// --- PIPELINE 4: Aggregated Predictions Breakdown ---
// Pipeline Explanation:
// Counts the frequency of each predicted academic outcome category.
db.predictions.aggregate([
  {
    $group: {
      _id: "$predicted_outcome",
      prediction_count: { $sum: 1 }
    }
  },
  {
    $sort: { prediction_count: -1 }
  }
]);


// ==============================================================================
// 9. QUERY OPTIMIZATION PROOF USING explain()
// ==============================================================================

// (A) INDEXED QUERY: Searching by 'student_id'
// Query Explanation:
// The query engine uses the B-Tree index on student_id ('EXPRESS_IXSCAN' / 'IXSCAN').
// It looks up the key in logarithmic O(log N) time without scanning the collection.
db.students.find({ student_id: "STU-0001" }).explain("executionStats");

// (B) UNINDEXED QUERY: Searching by 'Age at enrollment'
// Query Explanation:
// Because there is no index on 'Age at enrollment', the query optimizer performs a COLLSCAN (Collection Scan).
// MongoDB scans documents sequentially across the entire collection (O(N) time complexity).
db.students.find({ "Age at enrollment": 20 }).explain("executionStats");
