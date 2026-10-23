/**
 * mongodb/setup.js - Real MongoDB Database & Collection Initialization
 * Course: CSE494: Intelligent NoSQL Databases (LPU CA2 Project)
 *
 * Usage:
 * mongosh "mongodb://localhost:27017" mongodb/setup.js
 */

// 1. Select the project database
const db = db.getSiblingDB("university_dropout_db");

print("=================================================================");
print("MONGODB SETUP: university_dropout_db");
print("=================================================================");

// 2. Connection verification via ping command
print("\n[STEP 1] Testing Connection to MongoDB Server...");
const pingResult = db.runCommand({ ping: 1 });
if (pingResult.ok === 1) {
  print("SUCCESS: Connected to MongoDB server (ping ok: 1)");
} else {
  print("ERROR: Connection failed:", JSON.stringify(pingResult));
}

// 3. List available databases via administrative command
print("\n[STEP 2] Listing Databases on Server...");
const adminDb = db.getSiblingDB("admin");
const dbsResult = adminDb.runCommand({ listDatabases: 1 });
if (dbsResult.ok === 1) {
  print("Available Databases on Server:");
  dbsResult.databases.forEach((d) => {
    print(` - ${d.name} (${(d.sizeOnDisk / 1024 / 1024).toFixed(2)} MB)`);
  });
}

// 4. Ensure target collections exist without dropping existing data
print("\n[STEP 3] Ensuring Target Collections Exist...");
const existingCollections = db.getCollectionNames();

if (!existingCollections.includes("students")) {
  db.createCollection("students");
  print("Created collection: students");
} else {
  print("Collection already exists: students (retained existing data)");
}

if (!existingCollections.includes("predictions")) {
  db.createCollection("predictions");
  print("Created collection: predictions");
} else {
  print("Collection already exists: predictions (retained existing data)");
}

// 5. Display collection status
print("\n[STEP 4] Current Collections in university_dropout_db:");
db.getCollectionNames().forEach((collName) => {
  const count = db.getCollection(collName).countDocuments({});
  print(` - ${collName}: ${count} document(s)`);
});

print("\n=================================================================");
print("MongoDB Setup Completed Successfully.");
print("=================================================================");
