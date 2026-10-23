/**
 * mongodb/import_students.js - Native MongoDB Shell Student Data Importer
 * Course: CSE494: Intelligent NoSQL Databases (LPU CA2 Project)
 *
 * This script imports student records from CSV into the 'students' collection
 * using native MongoDB Shell (mongosh) JavaScript and bulkWrite with $setOnInsert.
 *
 * It is completely safe to run repeatedly: it will NEVER create duplicate documents.
 *
 * Usage:
 * mongosh "mongodb://localhost:27017" mongodb/import_students.js
 *
 * Alternatively, using MongoDB Database Tools (mongoimport):
 * mongoimport --uri="mongodb://localhost:27017/university_dropout_db" \
 *             --collection=students \
 *             --type=csv \
 *             --headerline \
 *             --file=students_prepared.csv \
 *             --mode=upsert \
 *             --upsertFields=student_id
 */

const fs = require("fs");
const path = require("path");

// 1. Select the project database
const db = db.getSiblingDB("university_dropout_db");

print("=================================================================");
print("NATIVE MONGODB SHELL IMPORT: students collection");
print("=================================================================");

// 2. Ensure Unique Index on student_id exists before import
db.students.createIndex(
  { student_id: 1 },
  { unique: true, name: "idx_unique_student_id" }
);

// 3. Locate CSV file (prefers students_prepared.csv, falls back to dataset.csv)
let csvFilePath = "students_prepared.csv";
if (!fs.existsSync(csvFilePath)) {
  csvFilePath = "dataset.csv";
}

if (!fs.existsSync(csvFilePath)) {
  print(`ERROR: CSV file not found at ${csvFilePath}. Place dataset.csv in project root.`);
} else {
  print(`Reading records from: ${csvFilePath}...`);
  const rawData = fs.readFileSync(csvFilePath, "utf8");
  const lines = rawData.trim().split(/\r?\n/);

  if (lines.length < 2) {
    print("ERROR: CSV file contains insufficient data.");
  } else {
    // Parse CSV header
    const headers = lines[0].split(",").map((h) => h.trim());
    print(`Detected ${headers.length} attributes in header.`);

    const hasStudentId = headers.includes("student_id");
    const operations = [];

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      const values = line.split(",").map((v) => v.trim());
      const doc = {};

      // If student_id was not in CSV header, auto-assign STU-0001 format
      if (!hasStudentId) {
        doc["student_id"] = `STU-${String(i).padStart(4, "0")}`;
      }

      headers.forEach((header, idx) => {
        const rawVal = values[idx];
        if (header === "Target" || header === "student_id") {
          doc[header] = rawVal;
        } else {
          // Parse numeric features
          const num = Number(rawVal);
          doc[header] = isNaN(num) ? rawVal : num;
        }
      });

      // Upsert using $setOnInsert: if document with student_id exists, do nothing; if new, insert
      operations.push({
        updateOne: {
          filter: { student_id: doc.student_id },
          update: { $setOnInsert: doc },
          upsert: true,
        },
      });
    }

    print(`Prepared ${operations.length} bulk upsert operations. Executing against MongoDB...`);

    const BATCH_SIZE = 500;
    let newlyInserted = 0;
    let alreadyExisting = 0;

    for (let j = 0; j < operations.length; j += BATCH_SIZE) {
      const batch = operations.slice(j, j + BATCH_SIZE);
      const res = db.students.bulkWrite(batch, { ordered: false });
      newlyInserted += res.upsertedCount || 0;
      alreadyExisting += res.matchedCount || 0;
    }

    const totalInDb = db.students.countDocuments({});

    print("\n-----------------------------------------------------------------");
    print("IMPORT RESULTS (VERIFIED AGAINST MONGODB):");
    print("-----------------------------------------------------------------");
    print(`Total Records in CSV:         ${operations.length}`);
    print(`Newly Inserted Documents:     ${newlyInserted}`);
    print(`Already Existing (Duplicates): ${alreadyExisting}`);
    print(`Total Documents in MongoDB:   ${totalInDb}`);
    print("-----------------------------------------------------------------");
  }
}

print("\n=================================================================");
print("Student Data Import Completed.");
print("=================================================================");
