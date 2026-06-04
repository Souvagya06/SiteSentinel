from database import execute, query_one, query_all
from flask import Flask, send_from_directory, request, jsonify, session, redirect
from threading import Timer
import webbrowser
import os
import bcrypt
from database import execute, query_one
import secrets
import cloudinary
import cloudinary.uploader
import base64
from face__utils import get_embedding_from_url
import json

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# -----------------------------
# Frontend Path
# -----------------------------
FRONTEND_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../frontend/pages"
    )
)

# -----------------------------
# Landing Page
# -----------------------------
@app.route("/")
def landing():
    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )

# -----------------------------
# Login Page
# -----------------------------
@app.route("/login.html")
def login():
    return send_from_directory(
        FRONTEND_DIR,
        "login.html"
    )

# -----------------------------
# Dashboard Page
# -----------------------------
@app.route("/dashboard.html")
def dashboard():
    if "user_id" not in session:
        return redirect("/login.html")
    return send_from_directory(FRONTEND_DIR, "dashboard.html")

@app.route("/api/me")
def me():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    user = query_one("SELECT name, email FROM users WHERE id = ?", [{"type": "text", "value": str(session["user_id"])}])
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"name": user["name"], "email": user["email"]})

# -----------------------------
# Health Check
# -----------------------------
@app.route("/api/health")
def health():
    return {
        "status": "running",
        "service": "SiteSentinel Backend"
    }

# -----------------------------
# Auto Open Browser
# -----------------------------
def open_browser():
    webbrowser.open(
        "http://127.0.0.1:5000"
    )

# -----------------------------
# Run Server
# -----------------------------
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json()
    name = data.get("name", "").strip()
    company = data.get("company", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400
    if len(password) < 12:
        return jsonify({"error": "Password must be at least 12 characters."}), 400

    existing = query_one("SELECT id FROM users WHERE email = ?", [{"type": "text", "value": email}])
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
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = query_one("SELECT * FROM users WHERE email = ?", [{"type": "text", "value": email}])
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"] = user["id"]
    session["email"] = user["email"]
    return jsonify({"message": "Login successful."}), 200


@app.route("/api/logout")
def logout():
    session.clear()
    return redirect("/login.html")

@app.route("/api/workers", methods=["GET"])
def get_workers():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    result = execute(
        "SELECT * FROM workers WHERE user_id = ? ORDER BY created_at DESC",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    try:
        cols = [c["name"] for c in result["results"][0]["response"]["result"]["cols"]]
        rows = result["results"][0]["response"]["result"]["rows"]
        workers = [dict(zip(cols, [v["value"] for v in row])) for row in rows]
    except (KeyError, IndexError):
        workers = []
    return jsonify({"workers": workers})


@app.route("/api/workers", methods=["POST"])
def add_worker():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    first_name = data.get("first_name", "").strip()
    last_name  = data.get("last_name", "").strip()
    worker_id  = data.get("worker_id", "").strip()
    images     = data.get("images", [])

    if not first_name or not last_name or not worker_id:
        return jsonify({"error": "First name, last name and worker ID are required."}), 400

    existing = query_one(
        "SELECT id FROM workers WHERE worker_id = ? AND user_id = ?",
        [{"type": "text", "value": worker_id}, {"type": "text", "value": str(session["user_id"])}]
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

    # Save all image URLs to worker_images table
    for url in image_urls:
        embedding = get_embedding_from_url(url)
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
        # Get worker_id text value first
        worker = query_one(
            "SELECT worker_id FROM workers WHERE id = ? AND user_id = ?",
            [{"type": "text", "value": str(worker_db_id)},
             {"type": "text", "value": str(session["user_id"])}]
        )
        if not worker:
            return jsonify({"error": "Worker not found."}), 404

        # Delete from worker_images
        execute(
            "DELETE FROM worker_images WHERE worker_db_id = ?",
            [{"type": "text", "value": worker["worker_id"]}]
        )
        # Delete worker
        execute(
            "DELETE FROM workers WHERE id = ? AND user_id = ?",
            [{"type": "text", "value": str(worker_db_id)},
             {"type": "text", "value": str(session["user_id"])}]
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

@app.route("/api/workers/checkin", methods=["POST"])
def worker_checkin():
    data         = request.get_json()
    worker_id    = data.get("worker_id")
    ppe_score    = data.get("ppe_score", 0)
    checkin_time = data.get("checkin_time")
    status       = data.get("status", "Active")

    # Map status to event label
    event = "CHECK-IN" if status == "Active" else "CHECK-OUT"

    from datetime import datetime
    now       = datetime.now()
    date_str  = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Update worker's current status
    execute(
        "UPDATE workers SET checkin_time = ?, ppe_score = ?, status = ? WHERE worker_id = ?",
        [
            {"type": "text", "value": checkin_time},
            {"type": "text", "value": str(ppe_score)},
            {"type": "text", "value": status},
            {"type": "text", "value": worker_id},
        ]
    )

    # Get user_id for this worker
    worker = query_one(
        "SELECT user_id FROM workers WHERE worker_id = ?",
        [{"type": "text", "value": worker_id}]
    )
    user_id_val = str(worker["user_id"]) if worker else ""

    # Log the event
    execute(
        "INSERT INTO attendance_log (worker_id, user_id, event, ppe_score, timestamp, date) VALUES (?, ?, ?, ?, ?, ?)",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": user_id_val},
            {"type": "text", "value": event},
            {"type": "text", "value": str(ppe_score)},
            {"type": "text", "value": timestamp},
            {"type": "text", "value": date_str},
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
            [{"type": "text", "value": str(session["user_id"])},
             {"type": "text", "value": date_filter}]
        )
    else:
        rows = query_all(
            "SELECT * FROM attendance_log WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100",
            [{"type": "text", "value": str(session["user_id"])}]
        )
    return jsonify({"logs": rows})

if __name__ == "__main__":

    Timer(
        1,
        open_browser
    ).start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )