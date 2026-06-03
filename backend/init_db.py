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