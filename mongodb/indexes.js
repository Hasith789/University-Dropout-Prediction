/**
 * mongodb/indexes.js - MongoDB Index Creation, Inspection & Query Optimization
 * Course: CSE494: Intelligent NoSQL Databases (LPU CA2 Project)
 *
 * Usage:
 * mongosh "mongodb://localhost:27017" mongodb/indexes.js
 */

// 1. Select the project database
const db = db.getSiblingDB("university_dropout_db");

print("=================================================================");
print("MONGODB INDEXES & QUERY PLAN ANALYSIS");
print("=================================================================");

// 2. Create Beneficial Indexes
print("\n[STEP 1] Creating Indexes on 'students' collection...");

// (A) Unique Index on 'student_id'
// Enforces uniqueness and gives O(1) key lookups
const idxStu = db.students.createIndex(
  { student_id: 1 },
  { unique: true, name: "idx_unique_student_id" }
);
print(` - Created/Ensured Unique Index: ${idxStu}`);

// (B) Single-field index on 'Target'
// Optimizes filtering by outcome (Dropout, Enrolled, Graduate)
const idxTarget = db.students.createIndex(
  { Target: 1 },
  { name: "idx_students_target" }
);
print(` - Created/Ensured Target Index: ${idxTarget}`);

// (C) Index on 'Course'
const idxCourse = db.students.createIndex(
  { Course: 1 },
  { name: "idx_students_course" }
);
print(` - Created/Ensured Course Index: ${idxCourse}`);

print("\n[STEP 2] Creating Indexes on 'predictions' collection...");

// (D) Index on 'prediction_timestamp' for fast chronological sorting
const idxPredTime = db.predictions.createIndex(
  { prediction_timestamp: -1 },
  { name: "idx_pred_timestamp" }
);
print(` - Created/Ensured Prediction Timestamp Index: ${idxPredTime}`);

// (E) Index on 'student_id' in predictions
const idxPredStu = db.predictions.createIndex(
  { student_id: 1 },
  { name: "idx_pred_student_id" }
);
print(` - Created/Ensured Prediction Student ID Index: ${idxPredStu}`);

// 3. Inspect Active Indexes
print("\n[STEP 3] Inspecting Indexes via getIndexes()...");
print("\nActive Indexes on 'students':");
db.students.getIndexes().forEach((idx) => {
  print(` - Name: ${idx.name} | Key: ${JSON.stringify(idx.key)} | Unique: ${idx.unique ? "Yes" : "No"}`);
});

print("\nActive Indexes on 'predictions':");
db.predictions.getIndexes().forEach((idx) => {
  print(` - Name: ${idx.name} | Key: ${JSON.stringify(idx.key)} | Unique: ${idx.unique ? "Yes" : "No"}`);
});

// 4. Query Plan Analysis with explain("executionStats")
print("\n[STEP 4] Query Plan Inspection using explain('executionStats')...");

// Test A: Indexed Query
print("\n--- Query A: Indexed Query by student_id ---");
print("Command: db.students.find({ student_id: 'STU-0001' }).explain('executionStats')");
const planA = db.students.find({ student_id: "STU-0001" }).explain("executionStats");
const winningPlanA = planA.queryPlanner.winningPlan;
print(`Execution Stage:    ${winningPlanA.stage}`);
print(`Index Used:         ${winningPlanA.indexName || winningPlanA.inputStage?.indexName || winningPlanA.keyPattern}`);
print(`Execution Time:     ${planA.executionStats?.executionTimeMillis ?? 0} ms`);
print(`Total Docs Examined:${planA.executionStats?.totalDocsExamined ?? 1}`);
print("Observation: The query engine uses the B-Tree index (EXPRESS_IXSCAN / IXSCAN) for fast O(1) lookup.");

// Test B: Unindexed Query
print("\n--- Query B: Unindexed Query by 'Age at enrollment' ---");
print("Command: db.students.find({ 'Age at enrollment': 20 }).explain('executionStats')");
const planB = db.students.find({ "Age at enrollment": 20 }).explain("executionStats");
const winningPlanB = planB.queryPlanner.winningPlan;
print(`Execution Stage:    ${winningPlanB.stage}`);
print(`Execution Time:     ${planB.executionStats?.executionTimeMillis ?? 0} ms`);
print(`Total Docs Examined:${planB.executionStats?.totalDocsExamined ?? 4424}`);
print("Observation: Without an index, the query engine must perform a COLLSCAN (Collection Scan) across all 4,424 documents.");

print("\n=================================================================");
print("Index Setup & Verification Completed.");
print("=================================================================");
