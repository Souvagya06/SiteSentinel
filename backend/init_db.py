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