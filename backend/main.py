from database import execute, query_one, query_all
from flask import Flask, send_from_directory, request, jsonify, session, redirect
from threading import Thread, Timer
import webbrowser
import os
import sys
import subprocess
import bcrypt
import secrets
import cloudinary
import cloudinary.uploader
import json
from face__utils import get_embedding_from_url
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../frontend/pages")
)

# ─────────────────────────────────────────
# Pages
# ─────────────────────────────────────────
@app.route("/")
def landing():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/login.html")
def login():
    return send_from_directory(FRONTEND_DIR, "login.html")

@app.route("/dashboard.html")
def dashboard():
    if "user_id" not in session:
        return redirect("/login.html")
    return send_from_directory(FRONTEND_DIR, "dashboard.html")

# ─────────────────────────────────────────
# Health
# ─────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "running", "service": "SiteSentinel Backend"})

# ─────────────────────────────────────────
# Auth
# ─────────────────────────────────────────
@app.route("/api/me")
def me():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    user = query_one(
        "SELECT name, email FROM users WHERE id = ?",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"name": user["name"], "email": user["email"]})

@app.route("/api/session-user-id")
def session_user_id():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    return jsonify({"user_id": str(session["user_id"])})

@app.route("/api/signup", methods=["POST"])
def signup():
    data     = request.get_json()
    name     = data.get("name", "").strip()
    company  = data.get("company", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400
    if len(password) < 12:
        return jsonify({"error": "Password must be at least 12 characters."}), 400

    existing = query_one(
        "SELECT id FROM users WHERE email = ?",
        [{"type": "text", "value": email}]
    )
    if existing:
        return jsonify({"error": "An account with this email already exists."}), 409

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    execute(
        "INSERT INTO users (email, password, name, company) VALUES (?, ?, ?, ?)",
        [
            {"type": "text", "value": email},
            {"type": "text", "value": hashed},
            {"type": "text", "value": name},
            {"type": "text", "value": company},
        ]
    )
    return jsonify({"message": "Account created successfully."}), 201

@app.route("/api/login", methods=["POST"])
def login_api():
    data     = request.get_json()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = query_one(
        "SELECT * FROM users WHERE email = ?",
        [{"type": "text", "value": email}]
    )
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"] = user["id"]
    session["email"]   = user["email"]
    from threading import Thread

    Thread(
        target=start_webcam_detection,
        args=(user["id"],),
        daemon=True
    ).start()
    return jsonify({"message": "Login successful."}), 200

@app.route("/api/logout")
def logout():

    global webcam_process

    if webcam_process and webcam_process.poll() is None:
        webcam_process.kill()

    session.clear()

    return redirect("/login.html")

# ─────────────────────────────────────────
# Workers
# ─────────────────────────────────────────
@app.route("/api/workers", methods=["GET"])
def get_workers():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    result = execute(
        "SELECT * FROM workers WHERE user_id = ? ORDER BY created_at DESC",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    try:
        cols    = [c["name"] for c in result["results"][0]["response"]["result"]["cols"]]
        rows    = result["results"][0]["response"]["result"]["rows"]
        workers = [dict(zip(cols, [v["value"] for v in row])) for row in rows]
    except (KeyError, IndexError):
        workers = []
    return jsonify({"workers": workers})

@app.route("/api/workers", methods=["POST"])
def add_worker():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data       = request.get_json()
    first_name = data.get("first_name", "").strip()
    last_name  = data.get("last_name",  "").strip()
    worker_id  = data.get("worker_id",  "").strip()
    images     = data.get("images", [])

    if not first_name or not last_name or not worker_id:
        return jsonify({"error": "First name, last name and worker ID are required."}), 400

    existing = query_one(
        "SELECT id FROM workers WHERE worker_id = ? AND user_id = ?",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )
    if existing:
        return jsonify({"error": "A worker with this ID already exists."}), 409

    image_urls = []
    for i, img_b64 in enumerate(images):
        try:
            upload_result = cloudinary.uploader.upload(
                img_b64,
                folder=f"sitesentinel/{session['user_id']}",
                public_id=f"{worker_id}_{i}",
                overwrite=True
            )
            image_urls.append(upload_result["secure_url"])
        except Exception as e:
            return jsonify({"error": f"Image upload failed: {str(e)}"}), 500

    image_url = image_urls[0] if image_urls else ""

    execute(
        "INSERT INTO workers (user_id, worker_id, first_name, last_name, image_url) VALUES (?, ?, ?, ?, ?)",
        [
            {"type": "text", "value": str(session["user_id"])},
            {"type": "text", "value": worker_id},
            {"type": "text", "value": first_name},
            {"type": "text", "value": last_name},
            {"type": "text", "value": image_url},
        ]
    )

    for url in image_urls:
        embedding      = get_embedding_from_url(url)
        embedding_json = json.dumps(embedding) if embedding else None
        execute(
            "INSERT INTO worker_images (worker_db_id, image_url, face_embedding) VALUES (?, ?, ?)",
            [
                {"type": "text", "value": worker_id},
                {"type": "text", "value": url},
                {"type": "text", "value": embedding_json or ""},
            ]
        )

    return jsonify({"message": "Worker registered successfully.", "image_url": image_url}), 201

@app.route("/api/workers/<int:worker_db_id>", methods=["DELETE"])
def delete_worker(worker_db_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    try:
        worker = query_one(
            "SELECT worker_id FROM workers WHERE id = ? AND user_id = ?",
            [
                {"type": "text", "value": str(worker_db_id)},
                {"type": "text", "value": str(session["user_id"])}
            ]
        )
        if not worker:
            return jsonify({"error": "Worker not found."}), 404
        execute(
            "DELETE FROM worker_images WHERE worker_db_id = ?",
            [{"type": "text", "value": worker["worker_id"]}]
        )
        execute(
            "DELETE FROM workers WHERE id = ? AND user_id = ?",
            [
                {"type": "text", "value": str(worker_db_id)},
                {"type": "text", "value": str(session["user_id"])}
            ]
        )
        return jsonify({"message": "Worker deleted."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/workers/<worker_id>/images", methods=["GET"])
def get_worker_images(worker_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    result = execute(
        "SELECT image_url FROM worker_images WHERE worker_db_id = ?",
        [{"type": "text", "value": worker_id}]
    )
    try:
        rows = result["results"][0]["response"]["result"]["rows"]
        urls = [row[0]["value"] for row in rows]
    except (KeyError, IndexError):
        urls = []
    return jsonify({"images": urls})

# ─────────────────────────────────────────
# Helmets
# ─────────────────────────────────────────
@app.route("/api/helmets", methods=["GET"])
def get_helmets():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    result = execute(
        "SELECT * FROM helmets WHERE user_id = ? ORDER BY created_at DESC",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    try:
        cols    = [c["name"] for c in result["results"][0]["response"]["result"]["cols"]]
        rows    = result["results"][0]["response"]["result"]["rows"]
        helmets = [dict(zip(cols, [v["value"] for v in row])) for row in rows]
    except (KeyError, IndexError):
        helmets = []
    return jsonify({"helmets": helmets})

@app.route("/api/helmets", methods=["POST"])
def add_helmet():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    helmet_id = request.get_json().get("helmet_id", "").strip().upper()
    if not helmet_id:
        return jsonify({"error": "Helmet ID required"}), 400
    existing = query_one(
        "SELECT id FROM helmets WHERE helmet_id = ? AND user_id = ?",
        [{"type": "text", "value": helmet_id},
         {"type": "text", "value": str(session["user_id"])}]
    )
    if existing:
        return jsonify({"error": "Helmet ID already registered"}), 409
    execute(
        "INSERT INTO helmets (user_id, helmet_id, status) VALUES (?, ?, 'Available')",
        [{"type": "text", "value": str(session["user_id"])},
         {"type": "text", "value": helmet_id}]
    )
    return jsonify({"message": f"Helmet {helmet_id} registered"}), 201

@app.route("/api/helmets/<helmet_id>", methods=["DELETE"])
def delete_helmet(helmet_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    execute(
        "DELETE FROM helmets WHERE helmet_id = ? AND user_id = ?",
        [{"type": "text", "value": helmet_id},
         {"type": "text", "value": str(session["user_id"])}]
    )
    return jsonify({"message": "Helmet deleted"})

@app.route("/api/helmets/assign", methods=["POST"])
def assign_helmet():
    """Called by webcam_detection when OCR reads a helmet ID."""
    data       = request.get_json()
    helmet_id  = data.get("helmet_id", "").strip().upper()
    worker_id  = data.get("worker_id", "").strip()

    if not helmet_id or not worker_id:
        return jsonify({"error": "helmet_id and worker_id required"}), 400

    worker = query_one(
        "SELECT user_id FROM workers WHERE worker_id = ?",
        [{"type": "text", "value": worker_id}]
    )
    if not worker:
        return jsonify({"error": "Worker not found"}), 404

    # Check helmet is registered for the same manager
    helmet = query_one(
        "SELECT * FROM helmets WHERE helmet_id = ? AND user_id = ?",
        [
            {"type": "text", "value": helmet_id},
            {"type": "text", "value": str(worker["user_id"])}
        ]
    )
    if not helmet:
        return jsonify({"error": "Helmet not registered"}), 404

    if helmet.get("status") == "Occupied":
        current_owner = query_one(
            "SELECT worker_id FROM workers WHERE helmet_id = ?",
            [{"type": "text", "value": helmet_id}]
        )
        if not current_owner or current_owner.get("worker_id") != worker_id:
            return jsonify({"error": "Helmet already occupied"}), 409

    # Mark helmet as Occupied
    execute(
        "UPDATE helmets SET status = 'Occupied' WHERE helmet_id = ? AND user_id = ?",
        [
            {"type": "text", "value": helmet_id},
            {"type": "text", "value": str(worker["user_id"])}
        ]
    )

    # Assign to worker
    execute(
        "UPDATE workers SET helmet_id = ? WHERE worker_id = ? AND user_id = ?",
        [
            {"type": "text", "value": helmet_id},
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(worker["user_id"])}
        ]
    )
    return jsonify({"message": f"Helmet {helmet_id} assigned to worker {worker_id}"})

@app.route("/api/helmets/release", methods=["POST"])
def release_helmet():
    """Called when worker checks out."""
    helmet_id = request.get_json().get("helmet_id", "").strip().upper()
    if not helmet_id:
        return jsonify({"error": "helmet_id required"}), 400
    execute(
        "UPDATE helmets SET status = 'Available' WHERE helmet_id = ?",
        [{"type": "text", "value": helmet_id}]
    )
    return jsonify({"message": f"Helmet {helmet_id} released"})

@app.route("/api/helmets/check/<helmet_id>")
def check_helmet(helmet_id):
    """Check if helmet is registered and available."""
    helmet = query_one(
        "SELECT * FROM helmets WHERE helmet_id = ?",
        [{"type": "text", "value": helmet_id.upper()}]
    )
    if not helmet:
        return jsonify({"registered": False})
    return jsonify({"registered": True, "status": helmet["status"],
                    "helmet_id": helmet["helmet_id"]})
# ─────────────────────────────────────────
# Check-in / Attendance
# ─────────────────────────────────────────
@app.route("/api/workers/checkin", methods=["POST"])
def worker_checkin():
    data         = request.get_json()
    worker_id    = data.get("worker_id")
    checkin_time = data.get("checkin_time")
    status       = data.get("status", "Active")
    helmet_id    = data.get("helmet_id", "")

    current_worker = query_one(
        "SELECT ppe_score FROM workers WHERE worker_id = ?",
        [{"type": "text", "value": worker_id}]
    )
    ppe_score    = data.get("ppe_score")
    if ppe_score in [None, ""]:
        ppe_score = current_worker["ppe_score"] if current_worker and current_worker.get("ppe_score") is not None else 0

    event = "CHECK-IN" if status == "Active" else "CHECK-OUT"

    from datetime import datetime
    now       = datetime.now()
    date_str  = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    worker_update_sql = "UPDATE workers SET checkin_time = ?, ppe_score = ?, status = ?"
    worker_update_args = [
        {"type": "text", "value": checkin_time},
        {"type": "text", "value": str(ppe_score)},
        {"type": "text", "value": status},
    ]

    if helmet_id:
        worker_update_sql += ", helmet_id = ?"
        worker_update_args.append({"type": "text", "value": helmet_id})

    worker_update_sql += " WHERE worker_id = ?"
    worker_update_args.append({"type": "text", "value": worker_id})

    execute(worker_update_sql, worker_update_args)

    worker      = query_one(
        "SELECT user_id FROM workers WHERE worker_id = ?",
        [{"type": "text", "value": worker_id}]
    )
    user_id_val = str(worker["user_id"]) if worker else ""

    execute(
        """INSERT INTO attendance_log
           (worker_id, user_id, event, ppe_score, timestamp, date, helmet_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": user_id_val},
            {"type": "text", "value": event},
            {"type": "text", "value": str(ppe_score)},
            {"type": "text", "value": timestamp},
            {"type": "text", "value": date_str},
            {"type": "text", "value": helmet_id},
        ]
    )
    return jsonify({"message": f"{event} logged for worker {worker_id}."})

@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    date_filter = request.args.get("date", "")
    if date_filter:
        rows = query_all(
            "SELECT * FROM attendance_log WHERE user_id = ? AND date = ? ORDER BY timestamp DESC",
            [
                {"type": "text", "value": str(session["user_id"])},
                {"type": "text", "value": date_filter}
            ]
        )
    else:
        rows = query_all(
            "SELECT * FROM attendance_log WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100",
            [{"type": "text", "value": str(session["user_id"])}]
        )
    return jsonify({"logs": rows})

@app.route("/api/pi-ip", methods=["GET"])
def get_pi_ip():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [
            {
                "type": "text",
                "value": str(session["user_id"])
            }
        ]
    )

    return jsonify({
        "pi_ip": user["pi_ip"] if user else ""
    })

@app.route("/api/pi-ip", methods=["POST"])
def save_pi_ip():

    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()

    pi_ip = data.get("pi_ip", "").strip()

    execute(
        "UPDATE users SET pi_ip = ? WHERE id = ?",
        [
            {
                "type": "text",
                "value": pi_ip
            },
            {
                "type": "text",
                "value": str(session["user_id"])
            }
        ]
    )

    return jsonify({
        "message": "Pi IP saved successfully"
    })
# ─────────────────────────────────────────
# Auto-start webcam_detection.py
# ─────────────────────────────────────────
webcam_process = None

def start_webcam_detection(user_id):
    global webcam_process
    user_id = str(user_id)

    script = Path(__file__).resolve().parent.parent / "interface" / "webcam_detection.py"
    if not script.exists():
        print(f"WARNING: webcam_detection.py not found at {script}")
        return

    if webcam_process and webcam_process.poll() is None:
        webcam_process.kill()

    webcam_process = subprocess.Popen(
        [sys.executable, str(script), "--user-id", user_id],
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    )
    print(f"webcam_detection.py started for user_id={user_id}")

# ─────────────────────────────────────────
# Login hook — start webcam after login
# ─────────────────────────────────────────
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    if os.getenv("RENDER") is None:
        Timer(1, open_browser).start()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)