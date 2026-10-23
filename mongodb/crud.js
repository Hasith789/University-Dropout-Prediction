/**
 * mongodb/crud.js - Native MongoDB Shell CRUD Operations
 * Course: CSE494: Intelligent NoSQL Databases (LPU CA2 Project)
 *
 * Demonstrates:
 * 1. Insert: db.students.insertOne()
 * 2. Find One: db.students.findOne()
 * 3. Find Multiple: db.students.find().limit()
 * 4. Update: db.students.updateOne() with $set
 * 5. Delete: db.students.deleteOne() (protected by explicit confirmation flag)
 *
 * Usage:
 * mongosh "mongodb://localhost:27017" mongodb/crud.js
 */

// 1. Select the project database
const db = db.getSiblingDB("university_dropout_db");

print("=================================================================");
print("NATIVE MONGODB CRUD DEMONSTRATION");
print("=================================================================");

const TEST_STUDENT_ID = "STU-TEST-9999";

// Ensure cleanup of any previous test run
db.students.deleteOne({ student_id: TEST_STUDENT_ID });

// ------------------------------------------------------------------------------
// OPERATION 1: INSERT A STUDENT (insertOne)
// ------------------------------------------------------------------------------
print("\n[CRUD 1: CREATE / INSERT] Executing db.students.insertOne()...");
const insertResult = db.students.insertOne({
  student_id: TEST_STUDENT_ID,
  Target: "Enrolled",
  "Age at enrollment": 21,
  Gender: 1, // Male
  Course: 7, // Informatics Engineering
  "Tuition fees up to date": 1,
  "Scholarship holder": 0,
  "Curricular units 1st sem (approved)": 5,
  "Curricular units 1st sem (grade)": 14.5,
  "Curricular units 2nd sem (approved)": 5,
  "Curricular units 2nd sem (grade)": 14.0,
  "Unemployment rate": 10.8,
  "Inflation rate": 1.4,
  GDP: 1.74,
});

print("Insert Acknowledged:", insertResult.acknowledged);
print("Generated Document _id:", insertResult.insertedId);

// ------------------------------------------------------------------------------
// OPERATION 2: READ / FIND (findOne & find)
// ------------------------------------------------------------------------------
print("\n[CRUD 2: READ / FIND] Executing db.students.findOne()...");
const fetchedStudent = db.students.findOne({ student_id: TEST_STUDENT_ID });
print(`Found Student: ID = ${fetchedStudent.student_id} | Target = ${fetchedStudent.Target} | Age = ${fetchedStudent["Age at enrollment"]}`);

print("\nExecuting db.students.find({ Target: 'Dropout' }).limit(3)...");
const sampleDropouts = db.students
  .find({ Target: "Dropout" }, { student_id: 1, Target: 1, Course: 1, "Tuition fees up to date": 1, _id: 0 })
  .limit(3)
  .toArray();

sampleDropouts.forEach((s) => {
  print(` - Student: ${s.student_id} | Status: ${s.Target} | Course ID: ${s.Course} | Tuition Paid: ${s["Tuition fees up to date"]}`);
});

// ------------------------------------------------------------------------------
// OPERATION 3: UPDATE (updateOne with $set)
// ------------------------------------------------------------------------------
print("\n[CRUD 3: UPDATE] Executing db.students.updateOne()...");
const updateResult = db.students.updateOne(
  { student_id: TEST_STUDENT_ID },
  {
    $set: {
      Target: "Graduate",
      "Curricular units 2nd sem (grade)": 16.0,
      "Tuition fees up to date": 1,
    },
  }
);

print("Matched Count:", updateResult.matchedCount);
print("Modified Count:", updateResult.modifiedCount);

const updatedDoc = db.students.findOne({ student_id: TEST_STUDENT_ID });
print(`Verified Updated Target for ${TEST_STUDENT_ID}: ${updatedDoc.Target} (Grade: ${updatedDoc["Curricular units 2nd sem (grade)"]})`);

// ------------------------------------------------------------------------------
// OPERATION 4: DELETE (deleteOne with confirmation flag)
// ------------------------------------------------------------------------------
print("\n[CRUD 4: DELETE] Executing db.students.deleteOne()...");

// Explicit confirmation flag to protect against accidental deletion
const CONFIRM_DELETE = true;

if (!CONFIRM_DELETE) {
  print("Deletion skipped: CONFIRM_DELETE flag is false.");
} else {
  const deleteResult = db.students.deleteOne({ student_id: TEST_STUDENT_ID });
  print("Delete Acknowledged:", deleteResult.acknowledged);
  print("Deleted Count:", deleteResult.deletedCount);

  // Verify deletion
  const verifyDeleted = db.students.findOne({ student_id: TEST_STUDENT_ID });
  if (!verifyDeleted) {
    print(`Verified: Document '${TEST_STUDENT_ID}' no longer exists in MongoDB.`);
  } else {
    print("Warning: Document still present.");
  }
}

print("\n=================================================================");
print("CRUD Operations Completed Successfully.");
print("=================================================================");
