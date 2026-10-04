"""
Creates or resets the admin@gmail.com account with password uem@kolkata2026
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bcrypt
from database import execute, query_one

EMAIL = "admin@gmail.com"
PASSWORD = "uem@kolkata2026"
NAME = "Site Admin"
COMPANY = "SiteSentinel"

hashed = bcrypt.hashpw(PASSWORD.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
assert len(hashed) == 60, "Hash invalid!"
print(f"Generated hash (len={len(hashed)}): {hashed[:25]}...")

# Check if user exists
existing = query_one(
    "SELECT id, email FROM users WHERE email = ?",
    [{"type": "text", "value": EMAIL}]
)

if existing:
    print(f"User exists (id={existing['id']}). Updating password...")
    execute(
        "UPDATE users SET password = ?, name = ? WHERE email = ?",
        [
            {"type": "text", "value": hashed},
            {"type": "text", "value": NAME},
            {"type": "text", "value": EMAIL}
        ]
    )
else:
    print(f"User does not exist. Creating {EMAIL}...")
    execute(
        "INSERT INTO users (email, password, name, company, face_data_version) VALUES (?, ?, ?, ?, 1)",
        [
            {"type": "text", "value": EMAIL},
            {"type": "text", "value": hashed},
            {"type": "text", "value": NAME},
            {"type": "text", "value": COMPANY}
        ]
    )

# Verify round-trip
user = query_one(
    "SELECT id, email, password FROM users WHERE email = ?",
    [{"type": "text", "value": EMAIL}]
)
if not user:
    print("[FAILED] User still not found after insert!")
    sys.exit(1)

stored = user["password"]
print(f"Stored hash len: {len(stored)}")
ok = bcrypt.checkpw(PASSWORD.encode("utf-8"), stored.encode("utf-8"))
print(f"bcrypt.checkpw: {ok}")

if ok:
    print(f"\n[SUCCESS] Login credentials:")
    print(f"  Email   : {EMAIL}")
    print(f"  Password: {PASSWORD}")
else:
    print("[FAILED] Hash mismatch after writing to DB!")
