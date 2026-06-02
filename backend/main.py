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
    user = query_one("SELECT name, email FROM users WHERE id = ?", [{"type": "integer", "value": session["user_id"]}])
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
        [{"type": "integer", "value": session["user_id"]}]
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
            {"type": "integer", "value": session["user_id"]},
            {"type": "text",    "value": worker_id},
            {"type": "text",    "value": first_name},
            {"type": "text",    "value": last_name},
            {"type": "text",    "value": image_url},
        ]
    )
    return jsonify({"message": "Worker registered successfully.", "image_url": image_url}), 201


@app.route("/api/workers/<int:worker_db_id>", methods=["DELETE"])
def delete_worker(worker_db_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    execute(
        "DELETE FROM workers WHERE id = ? AND user_id = ?",
        [
            {"type": "integer", "value": worker_db_id},
            {"type": "integer", "value": session["user_id"]},
        ]
    )
    return jsonify({"message": "Worker deleted."})

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