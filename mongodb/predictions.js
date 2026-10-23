/**
 * mongodb/predictions.js - Native MongoDB Shell Prediction Management
 * Course: CSE494: Intelligent NoSQL Databases (LPU CA2 Project)
 *
 * Demonstrates:
 * 1. Inserting model prediction results into 'predictions' collection
 * 2. Exact required schema:
 *    - student_id
 *    - predicted_outcome
 *    - class_probabilities
 *    - model_name
 *    - prediction_timestamp (new Date())
 *    - feature_set
 * 3. Retrieving predictions sorted by timestamp descending
 * 4. Finding predictions for a specific student record
 *
 * Usage:
 * mongosh "mongodb://localhost:27017" mongodb/predictions.js
 */

// 1. Select the project database
const db = db.getSiblingDB("university_dropout_db");

print("=================================================================");
print("NATIVE MONGODB PREDICTIONS MANAGEMENT");
print("=================================================================");

// ------------------------------------------------------------------------------
// OPERATION 1: INSERT A REAL ML PREDICTION RECORD (insertOne)
// ------------------------------------------------------------------------------
print("\n[PREDICTION 1] Inserting New Model Prediction Document...");

const newPrediction = {
  student_id: "STU-0001",
  predicted_outcome: "Dropout",
  class_probabilities: {
    Dropout: 0.77,
    Enrolled: 0.03,
    Graduate: 0.20,
  },
  model_name: "Random Forest Classifier",
  prediction_timestamp: new Date(),
  feature_set: "Full Features (1st & 2nd Sem)",
  risk_tier: "High Risk",
  record_type: "Demonstration / Real Inference",
};

const insertRes = db.predictions.insertOne(newPrediction);
print("Prediction Write Acknowledged:", insertRes.acknowledged);
print("Generated Prediction _id:", insertRes.insertedId);

// ------------------------------------------------------------------------------
// OPERATION 2: RETRIEVE PREDICTION AUDIT TRAIL (find + sort + limit)
// ------------------------------------------------------------------------------
print("\n[PREDICTION 2] Retrieving 5 Most Recent Predictions (Latest First)...");
const recentPredictions = db.predictions
  .find()
  .sort({ prediction_timestamp: -1 })
  .limit(5)
  .toArray();

recentPredictions.forEach((p, idx) => {
  print(
    `#${idx + 1} | Time: ${p.prediction_timestamp} | Student: ${p.student_id} | ` +
    `Outcome: ${p.predicted_outcome} (${p.risk_tier || "N/A"}) | Model: ${p.model_name}`
  );
});

// ------------------------------------------------------------------------------
// OPERATION 3: QUERY PREDICTIONS BY STUDENT ID
// ------------------------------------------------------------------------------
print("\n[PREDICTION 3] Querying Prediction History for Student 'STU-0001'...");
const studentPreds = db.predictions.find({ student_id: "STU-0001" }).toArray();
print(`Total predictions recorded for STU-0001: ${studentPreds.length}`);
studentPreds.forEach((p) => {
  print(` - Outcome: ${p.predicted_outcome} | Probabilities: ${JSON.stringify(p.class_probabilities)}`);
});

// ------------------------------------------------------------------------------
// OPERATION 4: PREDICTION DOCUMENT COUNT
// ------------------------------------------------------------------------------
const totalPredictions = db.predictions.countDocuments({});
print(`\nTotal Prediction Documents in 'predictions' collection: ${totalPredictions}`);

print("\n=================================================================");
print("Predictions Operations Completed Successfully.");
print("=================================================================");
