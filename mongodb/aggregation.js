/**
 * mongodb/aggregation.js - Native MongoDB Shell Aggregation Pipelines
 * Course: CSE494: Intelligent NoSQL Databases (LPU CA2 Project)
 *
 * Demonstrates real multi-stage aggregation pipelines:
 * 1. Pipeline 1: Group & count students by actual outcome ($group, $sort)
 * 2. Pipeline 2: Compute dropout rate per degree program ($group, $project, $sort)
 * 3. Pipeline 3: Indebted student vulnerability ($match, $group, $sort)
 * 4. Pipeline 4: Prediction distribution on 'predictions' collection ($match, $group, $sort)
 *
 * Usage:
 * mongosh "mongodb://localhost:27017" mongodb/aggregation.js
 */

// 1. Select the project database
const db = db.getSiblingDB("university_dropout_db");

print("=================================================================");
print("NATIVE MONGODB AGGREGATION PIPELINES");
print("=================================================================");

// ------------------------------------------------------------------------------
// PIPELINE 1: COUNT STUDENTS BY ACTUAL OUTCOME ($group, $sort)
// ------------------------------------------------------------------------------
print("\n--- Pipeline 1: Student Count by Academic Outcome ($group, $sort) ---");
const outcomeDistribution = db.students
  .aggregate([
    {
      $group: {
        _id: "$Target",
        student_count: { $sum: 1 },
      },
    },
    {
      $sort: { student_count: -1 },
    },
  ])
  .toArray();

print("Results from MongoDB Engine:");
outcomeDistribution.forEach((item) => {
  print(` - Outcome: ${item._id.padEnd(10)} | Count: ${item.student_count.toLocaleString()}`);
});

// ------------------------------------------------------------------------------
// PIPELINE 2: DROPOUT RATE PER DEGREE PROGRAM ($group, $project, $sort)
// ------------------------------------------------------------------------------
print("\n--- Pipeline 2: Dropout Rate (%) by Course Degree Program ($group, $project, $sort) ---");
const courseStats = db.students
  .aggregate([
    // Stage 1: Group by Course, compute total students and conditional dropouts
    {
      $group: {
        _id: "$Course",
        total_students: { $sum: 1 },
        dropouts: {
          $sum: { $cond: [{ $eq: ["$Target", "Dropout"] }, 1, 0] },
        },
      },
    },
    // Stage 2: Project derived percentage field
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
                1,
              ],
            },
          ],
        },
        _id: 0,
      },
    },
    // Stage 3: Sort descending by dropout rate
    {
      $sort: { dropout_rate: -1 },
    },
  ])
  .toArray();

print("Top 5 Courses with Highest Dropout Rate:");
courseStats.slice(0, 5).forEach((c) => {
  print(
    ` - Course ID: ${String(c.course_id).padEnd(3)} | Total: ${String(c.total_students).padEnd(4)} | ` +
    `Dropouts: ${String(c.dropouts).padEnd(4)} | Dropout Rate: ${c.dropout_rate}%`
  );
});

// ------------------------------------------------------------------------------
// PIPELINE 3: INDEBTED STUDENTS VULNERABILITY ($match, $group, $sort)
// ------------------------------------------------------------------------------
print("\n--- Pipeline 3: Outcome of Students with Tuition In Arrears ($match, $group, $sort) ---");
const indebtedOutcomes = db.students
  .aggregate([
    // Stage 1: $match filters ONLY students who owe tuition (fees not up to date = 0)
    {
      $match: {
        "Tuition fees up to date": 0,
      },
    },
    // Stage 2: $group by Target outcome
    {
      $group: {
        _id: "$Target",
        count: { $sum: 1 },
      },
    },
    // Stage 3: $sort descending
    {
      $sort: { count: -1 },
    },
  ])
  .toArray();

print("Academic Outcomes for Students with Unpaid Tuition:");
indebtedOutcomes.forEach((r) => {
  print(` - ${r._id.padEnd(10)}: ${r.count} students`);
});

// ------------------------------------------------------------------------------
// PIPELINE 4: PREDICTIONS REGISTRY BREAKDOWN ($match, $group, $sort)
// ------------------------------------------------------------------------------
print("\n--- Pipeline 4: Aggregate Inferences in 'predictions' collection ($match, $group, $sort) ---");
const predictionSummary = db.predictions
  .aggregate([
    // Stage 1: Optional $match (filter for Random Forest model)
    {
      $match: {
        model_name: { $regex: "Random Forest", $options: "i" },
      },
    },
    // Stage 2: $group by predicted_outcome
    {
      $group: {
        _id: "$predicted_outcome",
        count: { $sum: 1 },
      },
    },
    // Stage 3: $sort descending
    {
      $sort: { count: -1 },
    },
  ])
  .toArray();

print("Predictions Grouped by Predicted Outcome:");
predictionSummary.forEach((p) => {
  print(` - Predicted ${p._id.padEnd(10)}: ${p.count} prediction(s)`);
});

print("\n=================================================================");
print("Aggregation Pipelines Executed Successfully.");
print("=================================================================");
