from database import execute

execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    name TEXT,
    company TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
print("Users table created successfully")

execute("""
CREATE TABLE IF NOT EXISTS workers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    image_url TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, worker_id)
)
""")
print("Workers table created successfully")

execute("""
CREATE TABLE IF NOT EXISTS worker_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_db_id TEXT NOT NULL,
    image_url TEXT NOT NULL,
    face_embedding TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
print("Worker images table created successfully")

execute("""
CREATE TABLE IF NOT EXISTS attendance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event TEXT NOT NULL,
    ppe_score INTEGER DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    date TEXT NOT NULL
)
""")
print("Attendance log table created successfully")

# Run these once to add new columns (safe to run multiple times)
try:
    execute("ALTER TABLE workers ADD COLUMN checkin_time TEXT DEFAULT '--:--'")
    print("Added checkin_time column")
except: pass

try:
    execute("ALTER TABLE workers ADD COLUMN ppe_score INTEGER DEFAULT 0")
    print("Added ppe_score column")
except: pass

try:
    execute("ALTER TABLE workers ADD COLUMN status TEXT DEFAULT 'Off-Site'")
    print("Added status column")
except: pass

try:
    execute("ALTER TABLE users ADD COLUMN esp32_ip TEXT DEFAULT ''")
    print("Added esp32_ip column")
except: pass

try:
    execute("ALTER TABLE workers ADD COLUMN helmet_id TEXT DEFAULT ''")
    print("Added helmet_id column")
except: pass

try:
    execute("ALTER TABLE attendance_log ADD COLUMN helmet_id TEXT DEFAULT ''")
    print("Added helmet_id to attendance_log")
except: pass

execute("""
CREATE TABLE IF NOT EXISTS helmets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    helmet_id TEXT NOT NULL,
    status TEXT DEFAULT 'Available',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, helmet_id)
)
""")
print("Helmets table created successfully")