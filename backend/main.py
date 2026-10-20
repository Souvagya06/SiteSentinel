import os
import sys
from pathlib import Path

# Ensure backend folder is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import execute, query_one, query_all
from flask import Flask, send_from_directory, request, jsonify, session, redirect, Response
import io
from threading import Thread, Timer, Lock
import webbrowser
import subprocess
import bcrypt
import secrets
import cloudinary
import cloudinary.uploader
import json
import time
from datetime import datetime, timedelta
from face__utils import get_embedding_from_url
from report_service import generate_csv_report, generate_pdf_report
import urllib.parse
import requests
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ── Shared In-Process Frame Buffer (for dashboard MJPEG stream) ──
import tempfile
_latest_frame_jpeg: bytes = b""
_latest_cv_frame: bytes = b""
_frame_lock = Lock()
_frame_updated_at: float = 0.0
_cv_frame_updated_at: float = 0.0
_stream_active: bool = True
_SHARED_FRAME_PATH = Path(tempfile.gettempdir()) / "sitesentinel_live_frame.jpg"

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../frontend/pages")
)
ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))
IMAGES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../frontend/images")
)

@app.route("/assets/<path:filename>")
def frontend_asset(filename):
    return send_from_directory(ASSET_DIR, filename)

@app.route("/frontend/images/<path:filename>")
@app.route("/images/<path:filename>")
def frontend_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(IMAGES_DIR, "favicon.png", mimetype="image/png")

def bump_face_data_version(user_id):
    """Increments the face_data_version for the user so live CV processes hot-reload embeddings."""
    try:
        execute(
            "UPDATE users SET face_data_version = COALESCE(face_data_version, 1) + 1 WHERE id = ?",
            [{"type": "text", "value": str(user_id)}]
        )
    except Exception as e:
        print(f"Error bumping face data version: {e}")

# ─────────────────────────────────────────
# Pages
# ─────────────────────────────────────────
@app.route("/")
@app.route("/index.html")
@app.route("/frontend/pages/index.html")
@app.route("/frontend/pages/index")
@app.route("/frontend/pages")
def landing():
    if "user_id" in session:
        return redirect("/dashboard")
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/login")
@app.route("/login.html")
@app.route("/frontend/pages/login.html")
@app.route("/frontend/pages/login")
@app.route("/frontend/login.html")
@app.route("/frontend/login")
def login():
    if "user_id" in session:
        return redirect("/dashboard")
    return send_from_directory(FRONTEND_DIR, "login.html")

@app.route("/dashboard")
@app.route("/dashboard.html")
@app.route("/frontend/pages/dashboard.html")
@app.route("/frontend/pages/dashboard")
@app.route("/frontend/dashboard.html")
@app.route("/frontend/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")
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
    return jsonify({
        "name": user["name"],
        "email": user["email"],
        "login_time": session.get("login_time", time.time())
    })

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
        "INSERT INTO users (email, password, name, company, face_data_version) VALUES (?, ?, ?, ?, 1)",
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
    data     = request.get_json() or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = query_one(
        "SELECT * FROM users WHERE email = ?",
        [{"type": "text", "value": email}]
    )
    
    if not user:
        print(f"[Auth] Login failed: User '{email}' not found.")
        return jsonify({"error": "Invalid email or password."}), 401

    if not user.get("password"):
        print(f"[Auth] Login failed: User '{email}' has no password set (registered via Google).")
        return jsonify({"error": "This account was registered with Google. Please use 'Continue with Google'."}), 401

    if not bcrypt.checkpw(password.encode(), user["password"].encode()):
        print(f"[Auth] Login failed: Password mismatch for user '{email}'.")
        return jsonify({"error": "Invalid email or password."}), 401

    session.permanent = True
    session["user_id"] = user["id"]
    session["email"]   = user["email"]
    session["login_time"] = time.time()
    print(f"[Auth] Login successful for: {email} (ID: {user['id']})")
    return jsonify({"message": "Login successful.", "login_time": session["login_time"]}), 200

@app.route("/api/logout")
def logout():
    global webcam_process
    if webcam_process and webcam_process.poll() is None:
        webcam_process.kill()
        webcam_process = None
    session.clear()
    return redirect("/login")

# ─────────────────────────────────────────
# Google OAuth
# ─────────────────────────────────────────
@app.route("/auth/google")
@app.route("/api/auth/google")
def google_login():
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        return jsonify({"error": "Google OAuth is not configured. GOOGLE_CLIENT_ID is missing."}), 500

    forwarded_host = request.headers.get("X-Forwarded-Host", request.host)
    if "sitesentinel.site" in forwarded_host:
        redirect_uri = "https://sitesentinel.site/auth/google/callback"
    elif "127.0.0.1" in forwarded_host:
        redirect_uri = "http://127.0.0.1:5000/auth/google/callback"
    elif "localhost" in forwarded_host:
        redirect_uri = "http://localhost:5000/auth/google/callback"
    else:
        scheme = "https" if request.is_secure or request.headers.get("X-Forwarded-Proto") == "https" else "http"
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", f"{scheme}://{forwarded_host}/auth/google/callback")

    session["oauth_state"] = secrets.token_urlsafe(16)
    session["oauth_redirect_uri"] = redirect_uri

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": session["oauth_state"],
        "prompt": "select_account"
    }
    google_auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
    return redirect(google_auth_url)

@app.route("/auth/google/callback")
@app.route("/api/auth/google/callback")
def google_callback():
    error = request.args.get("error")
    if error:
        return redirect(f"/login?error={urllib.parse.quote(error)}")

    code = request.args.get("code")
    state = request.args.get("state")
    stored_state = session.get("oauth_state")

    if not code:
        return redirect("/login?error=missing_auth_code")
    if stored_state and state and state != stored_state:
        print(f"[Google OAuth] State mismatch: {state} != {stored_state}")
        return redirect("/login?error=invalid_oauth_state")

    redirect_uri = session.get("oauth_redirect_uri")
    if not redirect_uri:
        forwarded_host = request.headers.get("X-Forwarded-Host", request.host)
        if "sitesentinel.site" in forwarded_host:
            redirect_uri = "https://sitesentinel.site/auth/google/callback"
        elif "127.0.0.1" in forwarded_host:
            redirect_uri = "http://127.0.0.1:5000/auth/google/callback"
        else:
            redirect_uri = "http://localhost:5000/auth/google/callback"

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    # Exchange authorization code for tokens
    try:
        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri
            },
            timeout=10
        )
        if not token_resp.ok:
            print(f"[Google OAuth] Token exchange error: {token_resp.text}")
            return redirect("/login?error=token_exchange_failed")

        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            return redirect("/login?error=missing_access_token")

        userinfo_resp = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10
        )
        if not userinfo_resp.ok:
            print(f"[Google OAuth] Userinfo fetch error: {userinfo_resp.text}")
            return redirect("/login?error=userinfo_fetch_failed")

        userinfo = userinfo_resp.json()
        google_sub = userinfo.get("sub")
        email = userinfo.get("email", "").strip().lower()
        name = userinfo.get("name", "") or "Google User"
        picture = userinfo.get("picture", "")

        if not google_sub or not email:
            return redirect("/login?error=invalid_google_profile")

        # 1. Check if user already exists by google_sub or email
        user = query_one("SELECT * FROM users WHERE google_sub = ?", [{"type": "text", "value": google_sub}])
        if not user:
            user = query_one("SELECT * FROM users WHERE email = ?", [{"type": "text", "value": email}])

        if user:
            # Link/update Google sub and profile picture
            execute(
                "UPDATE users SET google_sub = ?, profile_picture = COALESCE(NULLIF(?, ''), profile_picture) WHERE id = ?",
                [{"type": "text", "value": google_sub}, {"type": "text", "value": picture}, {"type": "text", "value": str(user["id"])}]
            )
            user["google_sub"] = google_sub
            if picture:
                user["profile_picture"] = picture
        else:
            # 2. Create new user record
            execute(
                "INSERT INTO users (email, password, name, company, google_sub, profile_picture, face_data_version) VALUES (?, '', ?, 'Google User', ?, ?, 1)",
                [{"type": "text", "value": email}, {"type": "text", "value": name}, {"type": "text", "value": google_sub}, {"type": "text", "value": picture}]
            )
            user = query_one("SELECT * FROM users WHERE email = ?", [{"type": "text", "value": email}])

        if not user:
            print(f"[Google OAuth] Failed to retrieve user for {email}")
            return redirect("/login?error=user_creation_failed")

        session.permanent = True
        session["user_id"] = user["id"]
        session["email"] = user["email"]
        session["login_time"] = time.time()
        print(f"[Google OAuth] Authenticated user_id={user['id']}, email={user['email']}")
        return redirect("/dashboard")

    except Exception as e:
        print(f"[Google OAuth] Exception: {e}")
        return redirect("/login?error=oauth_internal_error")

# ─────────────────────────────────────────
# Workers CRUD
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
        workers = [dict(zip(cols, [v.get("value") for v in row])) for row in rows]
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
    helmet_id  = data.get("helmet_id",  "").strip().upper()

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

    # Validate pre-assigned helmet
    if helmet_id:
        helmet_row = query_one(
            "SELECT status FROM helmets WHERE helmet_id = ? AND user_id = ?",
            [
                {"type": "text", "value": helmet_id},
                {"type": "text", "value": str(session["user_id"])}
            ]
        )
        if not helmet_row:
            return jsonify({"error": f"Helmet {helmet_id} is not registered in your inventory."}), 400
        if helmet_row["status"] == "Occupied":
            return jsonify({"error": f"Helmet {helmet_id} is already assigned to another worker."}), 409

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
    occupation = data.get("occupation", "Labourer / Construction").strip() or "Labourer / Construction"

    execute(
        "INSERT INTO workers (user_id, worker_id, first_name, last_name, image_url, helmet_id, occupation) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            {"type": "text", "value": str(session["user_id"])},
            {"type": "text", "value": worker_id},
            {"type": "text", "value": first_name},
            {"type": "text", "value": last_name},
            {"type": "text", "value": image_url},
            {"type": "text", "value": helmet_id},
            {"type": "text", "value": occupation},
        ]
    )

    if helmet_id:
        execute(
            "UPDATE helmets SET status = 'Available' WHERE helmet_id = ? AND user_id = ?",
            [
                {"type": "text", "value": helmet_id},
                {"type": "text", "value": str(session["user_id"])}
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

    # Hot reload trigger
    bump_face_data_version(session["user_id"])

    return jsonify({"message": "Worker registered successfully.", "image_url": image_url}), 201

@app.route("/api/workers/<int:worker_db_id>", methods=["DELETE"])
def delete_worker(worker_db_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    try:
        worker = query_one(
            "SELECT worker_id, helmet_id FROM workers WHERE id = ? AND user_id = ?",
            [
                {"type": "text", "value": str(worker_db_id)},
                {"type": "text", "value": str(session["user_id"])}
            ]
        )
        if not worker:
            return jsonify({"error": "Worker not found."}), 404
        
        # Release helmet if assigned
        if worker.get("helmet_id"):
            execute(
                "UPDATE helmets SET status = 'Available' WHERE helmet_id = ? AND user_id = ?",
                [
                    {"type": "text", "value": worker["helmet_id"]},
                    {"type": "text", "value": str(session["user_id"])}
                ]
            )

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

        bump_face_data_version(session["user_id"])
        return jsonify({"message": "Worker deleted."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/workers/<int:worker_db_id>", methods=["PUT"])
def update_worker(worker_db_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    first_name = (data.get("first_name") or "").strip()
    last_name  = (data.get("last_name")  or "").strip()
    worker_id  = (data.get("worker_id")  or "").strip()
    if not first_name or not last_name or not worker_id:
        return jsonify({"error": "First name, last name and worker ID are required."}), 400
    try:
        existing = query_one(
            "SELECT id, worker_id FROM workers WHERE id = ? AND user_id = ?",
            [
                {"type": "text", "value": str(worker_db_id)},
                {"type": "text", "value": str(session["user_id"])}
            ]
        )
        if not existing:
            return jsonify({"error": "Worker not found."}), 404

        conflict = query_one(
            "SELECT id FROM workers WHERE worker_id = ? AND user_id = ? AND id != ?",
            [
                {"type": "text", "value": worker_id},
                {"type": "text", "value": str(session["user_id"])},
                {"type": "text", "value": str(worker_db_id)}
            ]
        )
        if conflict:
            return jsonify({"error": "Worker ID already in use by another worker."}), 409

        occupation = (data.get("occupation") or "").strip()
        if occupation:
            execute(
                "UPDATE workers SET first_name = ?, last_name = ?, worker_id = ?, occupation = ? WHERE id = ? AND user_id = ?",
                [
                    {"type": "text", "value": first_name},
                    {"type": "text", "value": last_name},
                    {"type": "text", "value": worker_id},
                    {"type": "text", "value": occupation},
                    {"type": "text", "value": str(worker_db_id)},
                    {"type": "text", "value": str(session["user_id"])}
                ]
            )
        else:
            execute(
                "UPDATE workers SET first_name = ?, last_name = ?, worker_id = ? WHERE id = ? AND user_id = ?",
                [
                    {"type": "text", "value": first_name},
                    {"type": "text", "value": last_name},
                    {"type": "text", "value": worker_id},
                    {"type": "text", "value": str(worker_db_id)},
                    {"type": "text", "value": str(session["user_id"])}
                ]
            )
        bump_face_data_version(session["user_id"])
        return jsonify({"message": "Worker updated."})
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
# Worker Details & Deep History
# ─────────────────────────────────────────
@app.route("/api/workers/<int:worker_db_id>/details", methods=["GET"])
def get_worker_full_details(worker_db_id):
    """Returns comprehensive worker profile, current status, attendance, and safety history."""
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    worker = query_one(
        "SELECT * FROM workers WHERE id = ? AND user_id = ?",
        [
            {"type": "text", "value": str(worker_db_id)},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )
    if not worker:
        return jsonify({"error": "Worker not found"}), 404

    worker_id = worker["worker_id"]

    # Face embeddings status
    face_rows = query_all(
        "SELECT image_url, face_embedding FROM worker_images WHERE worker_db_id = ?",
        [{"type": "text", "value": worker_id}]
    )
    has_embeddings = any(bool(r.get("face_embedding")) for r in face_rows)
    face_status = "Registered" if has_embeddings else ("Pending" if face_rows else "Needs photo")

    # Recent attendance
    att_logs = query_all(
        "SELECT * FROM attendance_log WHERE worker_id = ? AND user_id = ? ORDER BY timestamp DESC LIMIT 20",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )

    # Recent safety events
    events = query_all(
        "SELECT * FROM safety_events WHERE worker_id = ? AND manager_id = ? ORDER BY created_at DESC LIMIT 20",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )

    return jsonify({
        "worker": worker,
        "face_status": face_status,
        "image_count": len(face_rows),
        "recent_attendance": att_logs,
        "recent_events": events
    })

@app.route("/api/workers/<worker_id>/attendance", methods=["GET"])
def get_worker_attendance_history(worker_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    rows = query_all(
        "SELECT * FROM attendance_log WHERE worker_id = ? AND user_id = ? ORDER BY timestamp DESC LIMIT 50",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )
    return jsonify({"attendance": rows})

@app.route("/api/workers/<worker_id>/events", methods=["GET"])
def get_worker_safety_events(worker_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    rows = query_all(
        "SELECT * FROM safety_events WHERE worker_id = ? AND manager_id = ? ORDER BY created_at DESC LIMIT 50",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )
    return jsonify({"events": rows})

@app.route("/api/workers/<worker_id>/ppe-history", methods=["GET"])
def get_worker_ppe_history(worker_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    rows = query_all(
        "SELECT id, timestamp, event, ppe_score, helmet_id FROM attendance_log WHERE worker_id = ? AND user_id = ? ORDER BY timestamp ASC LIMIT 100",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )
    return jsonify({"history": rows})

# ─────────────────────────────────────────
# Assign / Unassign Helmet per Worker
# ─────────────────────────────────────────
@app.route("/api/workers/<worker_id>/helmet", methods=["PUT"])
def update_worker_helmet(worker_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data      = request.get_json()
    new_hid   = (data.get("helmet_id") or "").strip().upper()

    worker = query_one(
        "SELECT helmet_id FROM workers WHERE worker_id = ? AND user_id = ?",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )
    if not worker:
        return jsonify({"error": "Worker not found."}), 404

    old_hid = (worker.get("helmet_id") or "").strip().upper()

    if new_hid:
        helmet_row = query_one(
            "SELECT status FROM helmets WHERE helmet_id = ? AND user_id = ?",
            [
                {"type": "text", "value": new_hid},
                {"type": "text", "value": str(session["user_id"])}
            ]
        )
        if not helmet_row:
            return jsonify({"error": f"Helmet {new_hid} is not registered in your inventory."}), 400
        if helmet_row["status"] == "Occupied" and new_hid != old_hid:
            return jsonify({"error": f"Helmet {new_hid} is already assigned to another worker."}), 409

    if old_hid and old_hid != new_hid:
        execute(
            "UPDATE helmets SET status = 'Available' WHERE helmet_id = ? AND user_id = ?",
            [
                {"type": "text", "value": old_hid},
                {"type": "text", "value": str(session["user_id"])}
            ]
        )

    execute(
        "UPDATE workers SET helmet_id = ? WHERE worker_id = ? AND user_id = ?",
        [
            {"type": "text", "value": new_hid},
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )

    if new_hid:
        worker_status = worker.get("status", "Off-Site")
        target_status = "Occupied" if worker_status in ["Active", "On-Site"] else "Available"
        execute(
            "UPDATE helmets SET status = ? WHERE helmet_id = ? AND user_id = ?",
            [
                {"type": "text", "value": target_status},
                {"type": "text", "value": new_hid},
                {"type": "text", "value": str(session["user_id"])}
            ]
        )

    bump_face_data_version(session["user_id"])
    action = f"Assigned helmet {new_hid} to worker {worker_id}" if new_hid else f"Unassigned helmet from worker {worker_id}"
    return jsonify({"message": action})

# ─────────────────────────────────────────
# Helmets
# ─────────────────────────────────────────
@app.route("/api/helmets", methods=["GET"])
def get_helmets():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    manager_id = str(session["user_id"])
    result = execute(
        "SELECT * FROM helmets WHERE user_id = ? ORDER BY created_at DESC",
        [{"type": "text", "value": manager_id}]
    )
    try:
        cols    = [c["name"] for c in result["results"][0]["response"]["result"]["cols"]]
        rows    = result["results"][0]["response"]["result"]["rows"]
        helmets = [dict(zip(cols, [v.get("value") for v in row])) for row in rows]
    except (KeyError, IndexError):
        helmets = []

    # Dynamically cross-check with active workers currently on site
    try:
        active_w_res = execute(
            "SELECT active_helmet, helmet_id FROM workers WHERE user_id = ? AND status IN ('Active', 'On-Site')",
            [{"type": "text", "value": manager_id}]
        )
        w_cols = [c["name"] for c in active_w_res["results"][0]["response"]["result"]["cols"]]
        w_rows = active_w_res["results"][0]["response"]["result"]["rows"]
        active_occupied_helmets = set()
        for r in w_rows:
            wd = dict(zip(w_cols, [v.get("value") for v in r]))
            h = (wd.get("active_helmet") or wd.get("helmet_id") or "").strip().upper()
            if h:
                active_occupied_helmets.add(h)

        for h in helmets:
            hid = (h.get("helmet_id") or "").strip().upper()
            if hid in active_occupied_helmets:
                h["status"] = "Occupied"
            elif h.get("status") == "Occupied" and hid not in active_occupied_helmets:
                h["status"] = "Available"
    except Exception:
        pass

    return jsonify({"helmets": helmets})

@app.route("/api/helmets", methods=["POST"])
def add_helmet():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    body = request.get_json() or {}
    helmet_id = body.get("helmet_id", "").strip().upper()
    helmet_type = body.get("helmet_type", "Labourer, Heavy-duty operations, & Construction tasks").strip()
    color = body.get("color", "Yellow").strip()

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
        "INSERT INTO helmets (user_id, helmet_id, status, helmet_type, color) VALUES (?, ?, 'Available', ?, ?)",
        [
            {"type": "text", "value": str(session["user_id"])},
            {"type": "text", "value": helmet_id},
            {"type": "text", "value": helmet_type},
            {"type": "text", "value": color}
        ]
    )
    return jsonify({"message": f"Helmet {helmet_id} registered ({color})"}), 201

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

    w_status = worker.get("status", "Off-Site")
    target_status = "Occupied" if w_status in ["Active", "On-Site"] else "Available"
    execute(
        "UPDATE helmets SET status = ? WHERE helmet_id = ? AND user_id = ?",
        [
            {"type": "text", "value": target_status},
            {"type": "text", "value": helmet_id},
            {"type": "text", "value": str(worker["user_id"])}
        ]
    )
    execute(
        "UPDATE workers SET helmet_id = ? WHERE worker_id = ? AND user_id = ?",
        [
            {"type": "text", "value": helmet_id},
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(worker["user_id"])}
        ]
    )
    bump_face_data_version(worker["user_id"])
    return jsonify({"message": f"Helmet {helmet_id} assigned to worker {worker_id}"})

@app.route("/api/helmets/release", methods=["POST"])
def release_helmet():
    helmet_id = request.get_json().get("helmet_id", "").strip().upper()
    if not helmet_id:
        return jsonify({"error": "helmet_id required"}), 400
    execute(
        "UPDATE helmets SET status = 'Available' WHERE helmet_id = ?",
        [{"type": "text", "value": helmet_id}]
    )
    return jsonify({"message": f"Helmet {helmet_id} released"})

# ─────────────────────────────────────────
# Face Data Hot Reload Endpoints
# ─────────────────────────────────────────
@app.route("/api/face-data/version", methods=["GET"])
def get_face_data_version():
    manager_id = request.args.get("manager_id") or session.get("user_id")
    if not manager_id:
        return jsonify({"error": "manager_id required"}), 400

    user = query_one(
        "SELECT face_data_version FROM users WHERE id = ?",
        [{"type": "text", "value": str(manager_id)}]
    )
    version = user.get("face_data_version", 1) if user else 1
    return jsonify({"version": version, "manager_id": str(manager_id)})

@app.route("/api/face-data", methods=["GET"])
def get_face_data():
    manager_id = request.args.get("manager_id") or session.get("user_id")
    if not manager_id:
        return jsonify({"error": "manager_id required"}), 400

    user = query_one(
        "SELECT face_data_version FROM users WHERE id = ?",
        [{"type": "text", "value": str(manager_id)}]
    )
    version = user.get("face_data_version", 1) if user else 1

    # Fetch all workers and their face embeddings for this manager
    rows = query_all(
        """
        SELECT w.id, w.worker_id, w.first_name, w.last_name, w.helmet_id, w.occupation, w.status, wi.face_embedding,
               h.color AS helmet_color, h.helmet_type
        FROM workers w
        JOIN worker_images wi ON w.worker_id = wi.worker_db_id
        LEFT JOIN helmets h ON w.helmet_id = h.helmet_id AND w.user_id = h.user_id
        WHERE w.user_id = ? AND wi.face_embedding IS NOT NULL AND wi.face_embedding != ''
        """,
        [{"type": "text", "value": str(manager_id)}]
    )

    return jsonify({
        "version": version,
        "manager_id": str(manager_id),
        "faces": rows
    })

@app.route("/api/face-data/refresh", methods=["POST"])
def refresh_face_data():
    manager_id = request.args.get("manager_id") or session.get("user_id")
    if not manager_id:
        return jsonify({"error": "Not authenticated"}), 401

    bump_face_data_version(manager_id)
    return jsonify({"message": "Face data reload signaled successfully.", "reloaded": True})

# ─────────────────────────────────────────
# Structured Safety Events & Real-time Alerts
# ─────────────────────────────────────────
@app.route("/api/events", methods=["POST"])
def create_event():
    data = request.get_json() or {}
    manager_id = str(data.get("manager_id") or session.get("user_id") or "")
    if not manager_id:
        return jsonify({"error": "manager_id required"}), 400

    worker_id    = data.get("worker_id", "")
    event_type   = data.get("event_type", "INFO")
    message      = data.get("message", "")
    ppe_score    = data.get("ppe_score")
    helmet_id    = data.get("helmet_id", "")
    camera_id    = data.get("camera_id", "Pi Camera 01")
    evidence_url = data.get("evidence_url", "")

    ppe_score_val = str(ppe_score) if ppe_score is not None else None

    execute(
        """
        INSERT INTO safety_events (manager_id, worker_id, event_type, message, ppe_score, helmet_id, camera_id, evidence_url, acknowledged)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        [
            {"type": "text", "value": manager_id},
            {"type": "text", "value": worker_id},
            {"type": "text", "value": event_type},
            {"type": "text", "value": message},
            {"type": "text", "value": ppe_score_val or ""},
            {"type": "text", "value": helmet_id},
            {"type": "text", "value": camera_id},
            {"type": "text", "value": evidence_url},
        ]
    )

    return jsonify({"message": "Event recorded", "event_type": event_type}), 201

@app.route("/api/events", methods=["GET"])
def get_events():
    manager_id = request.args.get("manager_id") or session.get("user_id")
    if not manager_id:
        return jsonify({"error": "Not logged in"}), 401

    after_id = request.args.get("after_id", type=int)
    worker_id = request.args.get("worker_id", "")
    limit = request.args.get("limit", default=100, type=int)

    if after_id is not None:
        rows = query_all(
            """SELECT se.*, w.first_name, w.last_name, w.image_url 
               FROM safety_events se
               LEFT JOIN workers w ON (se.worker_id = w.worker_id AND se.manager_id = w.user_id)
               WHERE se.manager_id = ? AND se.id > ? 
               ORDER BY se.id ASC LIMIT ?""",
            [
                {"type": "text", "value": str(manager_id)},
                {"type": "text", "value": str(after_id)},
                {"type": "text", "value": str(limit)}
            ]
        )
    elif worker_id:
        rows = query_all(
            """SELECT se.*, w.first_name, w.last_name, w.image_url 
               FROM safety_events se
               LEFT JOIN workers w ON (se.worker_id = w.worker_id AND se.manager_id = w.user_id)
               WHERE se.manager_id = ? AND se.worker_id = ? 
               ORDER BY se.created_at DESC LIMIT ?""",
            [
                {"type": "text", "value": str(manager_id)},
                {"type": "text", "value": worker_id},
                {"type": "text", "value": str(limit)}
            ]
        )
    else:
        rows = query_all(
            """SELECT se.*, w.first_name, w.last_name, w.image_url 
               FROM safety_events se
               LEFT JOIN workers w ON (se.worker_id = w.worker_id AND se.manager_id = w.user_id)
               WHERE se.manager_id = ? 
               ORDER BY se.id DESC LIMIT ?""",
            [
                {"type": "text", "value": str(manager_id)},
                {"type": "text", "value": str(limit)}
            ]
        )

    return jsonify({"events": rows})

@app.route("/api/events/<int:event_id>/acknowledge", methods=["POST"])
def acknowledge_event(event_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    execute(
        "UPDATE safety_events SET acknowledged = 1 WHERE id = ? AND manager_id = ?",
        [
            {"type": "text", "value": str(event_id)},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )
    return jsonify({"message": "Event acknowledged."})

# ─────────────────────────────────────────
# Check-in / Attendance Evaluation Commit
# ─────────────────────────────────────────
@app.route("/api/workers/checkin", methods=["POST"])
@app.route("/api/attendance", methods=["POST"])
def worker_checkin():
    data         = request.get_json() or {}
    worker_id    = data.get("worker_id", "").strip()
    checkin_time = data.get("checkin_time")
    status_req   = data.get("status")
    helmet_id    = data.get("helmet_id", "")
    user_id_req  = data.get("user_id") or session.get("user_id")

    if not worker_id:
        return jsonify({"error": "worker_id required"}), 400

    if user_id_req:
        worker = query_one(
            "SELECT * FROM workers WHERE worker_id = ? AND user_id = ?",
            [
                {"type": "text", "value": worker_id},
                {"type": "text", "value": str(user_id_req)}
            ]
        )
    else:
        worker = query_one(
            "SELECT * FROM workers WHERE worker_id = ?",
            [
                {"type": "text", "value": worker_id}
            ]
        )
    if not worker:
        return jsonify({"error": f"Worker {worker_id} not found."}), 404

    manager_id = str(worker["user_id"])
    current_status = worker.get("status", "Off-Site")

    # Determine event (Check-in or Check-out)
    if status_req:
        new_status = "Active" if status_req in ["Active", "On-Site"] else "Off-Site"
    else:
        new_status = "Off-Site" if current_status in ["Active", "On-Site"] else "Active"

    event = "CHECK-IN" if new_status == "Active" else "CHECK-OUT"

    # Score handling: preserve 0, 50, 100 explicitly
    raw_ppe = data.get("ppe_score")
    if raw_ppe is not None and str(raw_ppe).isdigit():
        ppe_score = int(raw_ppe)
    else:
        ppe_score = int(worker.get("ppe_score") or 0)

    now       = datetime.now()
    date_str  = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    time_str  = now.strftime("%I:%M %p")

    actual_helmet = (helmet_id or "").strip().upper() if event == "CHECK-IN" else ""
    if not actual_helmet and event == "CHECK-IN" and ppe_score >= 50:
        actual_helmet = (worker.get("active_helmet") or worker.get("helmet_id") or "").strip().upper()

    if event == "CHECK-IN":
        worker_update_sql = "UPDATE workers SET checkin_time = ?, checkout_time = '', ppe_score = ?, status = ?, active_helmet = ?"
        worker_update_args = [
            {"type": "text", "value": checkin_time or time_str},
            {"type": "text", "value": str(ppe_score)},
            {"type": "text", "value": new_status},
            {"type": "text", "value": actual_helmet},
        ]
        if actual_helmet and not (worker.get("helmet_id") or "").strip():
            worker_update_sql += ", helmet_id = ?"
            worker_update_args.append({"type": "text", "value": actual_helmet})
    else:
        worker_update_sql = "UPDATE workers SET checkout_time = ?, ppe_score = ?, status = ?, active_helmet = ''"
        worker_update_args = [
            {"type": "text", "value": time_str},
            {"type": "text", "value": "0"},
            {"type": "text", "value": new_status},
        ]

    worker_update_sql += " WHERE id = ?"
    worker_update_args.append({"type": "text", "value": str(worker["id"])})

    execute(worker_update_sql, worker_update_args)

    # Sync helmets table status to match whether a helmet is ACTUALLY occupied
    if event == "CHECK-IN":
        if actual_helmet:
            # Helmet was actually worn & verified -> mark as Occupied
            execute(
                "UPDATE helmets SET status = 'Occupied' WHERE UPPER(helmet_id) = ? AND user_id = ?",
                [
                    {"type": "text", "value": actual_helmet},
                    {"type": "text", "value": manager_id}
                ]
            )
        else:
            # NO helmet detected -> ensure worker's assigned helmet remains Available (not occupied)
            assigned_h = (worker.get("helmet_id") or "").strip().upper()
            if assigned_h:
                execute(
                    "UPDATE helmets SET status = 'Available' WHERE UPPER(helmet_id) = ? AND user_id = ?",
                    [
                        {"type": "text", "value": assigned_h},
                        {"type": "text", "value": manager_id}
                    ]
                )
    elif event == "CHECK-OUT":
        # Release any helmet associated with this worker back to Available
        prev_helm = (actual_helmet or worker.get("active_helmet") or worker.get("helmet_id") or "").strip().upper()
        if prev_helm:
            execute(
                "UPDATE helmets SET status = 'Available' WHERE UPPER(helmet_id) = ? AND user_id = ?",
                [
                    {"type": "text", "value": prev_helm},
                    {"type": "text", "value": manager_id}
                ]
            )

    # Log in attendance_log (only store helmet if actually detected/worn)
    execute(
        """INSERT INTO attendance_log
           (worker_id, user_id, event, ppe_score, timestamp, date, helmet_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": manager_id},
            {"type": "text", "value": event},
            {"type": "text", "value": str(ppe_score)},
            {"type": "text", "value": timestamp},
            {"type": "text", "value": date_str},
            {"type": "text", "value": actual_helmet},
        ]
    )

    # Automatically generate structured safety event
    worker_full_name = f"{worker.get('first_name', '')} {worker.get('last_name', '')}".strip()
    if event == "CHECK-IN" and ppe_score < 100:
        event_type = "PPE_VIOLATION"
        msg = f"{worker_full_name} ({worker_id}) check-in with non-compliant PPE score {ppe_score}%"
    else:
        event_type = "CHECK_IN" if event == "CHECK-IN" else "CHECK_OUT"
        msg = f"{worker_full_name} ({worker_id}) completed {event.lower()} · PPE: {ppe_score}%"

    execute(
        """INSERT INTO safety_events (manager_id, worker_id, event_type, message, ppe_score, helmet_id, camera_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            {"type": "text", "value": manager_id},
            {"type": "text", "value": worker_id},
            {"type": "text", "value": event_type},
            {"type": "text", "value": msg},
            {"type": "text", "value": str(ppe_score)},
            {"type": "text", "value": actual_helmet},
            {"type": "text", "value": "Pi Camera 01"},
        ]
    )

    return jsonify({
        "message": f"{event} logged for {worker_full_name}.",
        "worker_id": worker_id,
        "event": event,
        "status": new_status,
        "ppe_score": ppe_score,
        "helmet_id": actual_helmet
    })

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

# ─────────────────────────────────────────
# Exportable Dashboard Reports (CSV & PDF)
# ─────────────────────────────────────────
@app.route("/api/reports/export.csv", methods=["GET"])
def export_csv_report():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    manager_id = str(session["user_id"])
    user = query_one("SELECT name FROM users WHERE id = ?", [{"type": "text", "value": manager_id}])
    manager_name: str = str(user.get("name")) if (user and user.get("name")) else "Site Manager"

    from_date = request.args.get("from", "").strip()
    to_date = request.args.get("to", "").strip()
    worker_filter = request.args.get("worker_id", "").strip()

    # Query Workers
    if worker_filter:
        workers = query_all("SELECT * FROM workers WHERE user_id = ? AND worker_id = ?",
                            [{"type": "text", "value": manager_id}, {"type": "text", "value": worker_filter}])
    else:
        workers = query_all("SELECT * FROM workers WHERE user_id = ? ORDER BY worker_id ASC",
                            [{"type": "text", "value": manager_id}])

    # Query Attendance Logs with Date Filtering
    att_sql = "SELECT * FROM attendance_log WHERE user_id = ?"
    att_args = [{"type": "text", "value": manager_id}]
    if worker_filter:
        att_sql += " AND worker_id = ?"
        att_args.append({"type": "text", "value": worker_filter})
    if from_date:
        att_sql += " AND date >= ?"
        att_args.append({"type": "text", "value": from_date})
    if to_date:
        att_sql += " AND date <= ?"
        att_args.append({"type": "text", "value": to_date})
    att_sql += " ORDER BY timestamp DESC"
    attendance_logs = query_all(att_sql, att_args)

    # Query Safety Events
    ev_sql = "SELECT * FROM safety_events WHERE manager_id = ?"
    ev_args = [{"type": "text", "value": manager_id}]
    if worker_filter:
        ev_sql += " AND worker_id = ?"
        ev_args.append({"type": "text", "value": worker_filter})
    if from_date:
        ev_sql += " AND created_at >= ?"
        ev_args.append({"type": "text", "value": f"{from_date} 00:00:00"})
    if to_date:
        ev_sql += " AND created_at <= ?"
        ev_args.append({"type": "text", "value": f"{to_date} 23:59:59"})
    ev_sql += " ORDER BY created_at DESC"
    safety_events = query_all(ev_sql, ev_args)

    csv_content = generate_csv_report(
        manager_name=manager_name,
        workers=workers,
        attendance_logs=attendance_logs,
        safety_events=safety_events,
        from_date=from_date,
        to_date=to_date
    )

    filename = f"SiteSentinel_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.route("/api/reports/export.pdf", methods=["GET"])
def export_pdf_report():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    manager_id = str(session["user_id"])
    user = query_one("SELECT name FROM users WHERE id = ?", [{"type": "text", "value": manager_id}])
    manager_name: str = str(user.get("name")) if (user and user.get("name")) else "Site Manager"

    from_date = request.args.get("from", "").strip()
    to_date = request.args.get("to", "").strip()
    worker_filter = request.args.get("worker_id", "").strip()

    if worker_filter:
        workers = query_all("SELECT * FROM workers WHERE user_id = ? AND worker_id = ?",
                            [{"type": "text", "value": manager_id}, {"type": "text", "value": worker_filter}])
    else:
        workers = query_all("SELECT * FROM workers WHERE user_id = ? ORDER BY worker_id ASC",
                            [{"type": "text", "value": manager_id}])

    att_sql = "SELECT * FROM attendance_log WHERE user_id = ?"
    att_args = [{"type": "text", "value": manager_id}]
    if worker_filter:
        att_sql += " AND worker_id = ?"
        att_args.append({"type": "text", "value": worker_filter})
    if from_date:
        att_sql += " AND date >= ?"
        att_args.append({"type": "text", "value": from_date})
    if to_date:
        att_sql += " AND date <= ?"
        att_args.append({"type": "text", "value": to_date})
    att_sql += " ORDER BY timestamp DESC"
    attendance_logs = query_all(att_sql, att_args)

    ev_sql = "SELECT * FROM safety_events WHERE manager_id = ?"
    ev_args = [{"type": "text", "value": manager_id}]
    if worker_filter:
        ev_sql += " AND worker_id = ?"
        ev_args.append({"type": "text", "value": worker_filter})
    if from_date:
        ev_sql += " AND created_at >= ?"
        ev_args.append({"type": "text", "value": f"{from_date} 00:00:00"})
    if to_date:
        ev_sql += " AND created_at <= ?"
        ev_args.append({"type": "text", "value": f"{to_date} 23:59:59"})
    ev_sql += " ORDER BY created_at DESC"
    safety_events = query_all(ev_sql, ev_args)

    pdf_bytes = generate_pdf_report(
        manager_name=manager_name,
        workers=workers,
        attendance_logs=attendance_logs,
        safety_events=safety_events,
        from_date=from_date,
        to_date=to_date
    )

    filename = f"SiteSentinel_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ─────────────────────────────────────────
# Raspberry Pi Configuration & Controller
# ─────────────────────────────────────────
@app.route("/api/pi-ip", methods=["GET"])
def get_pi_ip():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    return jsonify({"pi_ip": user["pi_ip"] if user else ""})

@app.route("/api/pi-ip", methods=["POST"])
def save_pi_ip():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    pi_ip = data.get("pi_ip", "").strip()
    if pi_ip and not ("." in pi_ip or ":" in pi_ip):
        pi_ip = pi_ip.lower()

    execute(
        "UPDATE users SET pi_ip = ? WHERE id = ?",
        [
            {"type": "text", "value": pi_ip},
            {"type": "text", "value": str(session["user_id"])}
        ]
    )
    return jsonify({"message": "Pi IP saved successfully"})

webcam_process = None
_webcam_log_path = Path(__file__).resolve().parent.parent / "webcam_detection.log"

def start_webcam_detection(user_id):
    global webcam_process
    user_id = str(user_id)

    script = Path(__file__).resolve().parent.parent / "interface" / "webcam_detection.py"
    if not script.exists():
        print(f"WARNING: webcam_detection.py not found at {script}")
        return False

    if webcam_process and webcam_process.poll() is None:
        webcam_process.kill()
        webcam_process = None

    try:
        log_file = open(str(_webcam_log_path), "w", encoding="utf-8", buffering=1)
    except Exception as e:
        print(f"WARNING: Could not open log file: {e}")
        log_file = None

    # Use -u flag (unbuffered) so log writes appear immediately
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_CONSOLE

    webcam_process = subprocess.Popen(
        [sys.executable, "-u", str(script), "--user-id", user_id],
        cwd=str(script.parent),
        stdout=log_file,
        stderr=log_file,
        creationflags=creationflags
    )
    print(f"webcam_detection.py started (PID {webcam_process.pid}) — log: {_webcam_log_path}")
    return True

def _generate_placeholder_jpeg(line1="SiteSentinel Gate Camera", line2="Connecting to camera feed..."):
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (640, 360), color=(15, 17, 23))
        draw = ImageDraw.Draw(img)
        draw.text((180, 150), line1, fill=(200, 210, 230))
        draw.text((170, 180), line2, fill=(120, 130, 150))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()
    except Exception:
        # Minimal 1x1 valid JPEG fallback to ensure stream boundaries never receive empty frames
        return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9'

# ── Dashboard MJPEG Stream Routes ──────────────────────
@app.route("/api/pi/frame/push", methods=["POST"])
def push_frame():
    """Called by Pi_Camera.py or webcam_detection.py to push the latest annotated JPEG frame."""
    global _latest_frame_jpeg, _frame_updated_at, _latest_cv_frame, _cv_frame_updated_at, _stream_active
    data = request.get_data()
    if data and len(data) > 100:
        _stream_active = True
        now = time.time()
        is_cv = (request.headers.get("X-Frame-Source") == "cv" or request.args.get("annotated") == "1")
        with _frame_lock:
            if is_cv:
                _latest_cv_frame = data
                _cv_frame_updated_at = now
                _latest_frame_jpeg = data
                _frame_updated_at = now
            else:
                # Prioritize active CV HUD frames if received within last 3 seconds
                if now - _cv_frame_updated_at > 3.0:
                    _latest_frame_jpeg = data
                    _frame_updated_at = now
        try:
            with open(_SHARED_FRAME_PATH, "wb") as f:
                f.write(data)
        except Exception:
            pass
    return '', 204

def _mjpeg_generator():
    """Yields MJPEG frames from the shared buffer for the dashboard stream."""
    BOUNDARY = b"--frame"
    placeholder_bytes = _generate_placeholder_jpeg()
    try:
        while _stream_active:
            now = time.time()
            with _frame_lock:
                # Prioritize active annotated CV frame with HUD modes (IDLE, VERIFYING, COOLDOWN)
                if _latest_cv_frame and (now - _cv_frame_updated_at < 4.0):
                    frame = _latest_cv_frame
                    updated = _cv_frame_updated_at
                else:
                    frame = _latest_frame_jpeg
                    updated = _frame_updated_at

            # If memory is empty (e.g. separate gunicorn worker), check shared disk buffer
            if (not frame or (now - updated > 5.0)) and _SHARED_FRAME_PATH.is_file():
                try:
                    mtime = _SHARED_FRAME_PATH.stat().st_mtime
                    if now - mtime < 5.0:
                        frame = _SHARED_FRAME_PATH.read_bytes()
                        updated = mtime
                except Exception:
                    pass

            if frame and (now - updated < 8.0):
                yield (
                    BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame + b"\r\n"
                )
                time.sleep(0.06)  # ~16 fps cap: optimal bandwidth & zero lag across internet
            else:
                if placeholder_bytes:
                    yield (
                        BOUNDARY + b"\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" +
                        placeholder_bytes + b"\r\n"
                    )
                time.sleep(0.5)
    except (GeneratorExit, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        pass

@app.route("/api/pi/stream")
def pi_stream():
    """MJPEG stream endpoint consumed by the dashboard <img> tag."""
    resp = Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Accel-Buffering"] = "no"  # Disable buffering in Nginx, Cloudflare, Vercel, Render
    return resp

@app.route("/api/pi/frame/latest")
def pi_frame_latest():
    """Returns the single latest JPEG frame. Instant snapshot / polling fallback."""
    now = time.time()
    with _frame_lock:
        if _latest_cv_frame and (now - _cv_frame_updated_at < 4.0):
            frame = _latest_cv_frame
        else:
            frame = _latest_frame_jpeg
    if not frame and _SHARED_FRAME_PATH.is_file():
        try:
            if now - _SHARED_FRAME_PATH.stat().st_mtime < 6.0:
                frame = _SHARED_FRAME_PATH.read_bytes()
        except Exception:
            pass
    if not frame:
        frame = _generate_placeholder_jpeg()
    resp = Response(frame, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/api/pi/test", methods=["POST"])
def api_test_pi():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json() or {}
    ip = data.get("pi_ip", "").strip()
    if not ip or ip in ["0", "1", "local", "webcam", "usb"]:
        return jsonify({"ok": True, "type": "local", "message": "Local camera mode (no remote Pi required)."})

    is_cloud_device = bool(ip and not ("." in ip or ":" in ip))

    # Check via Firebase Realtime Database
    try:
        from firebase_service import is_firebase_available, get_device_status, get_default_device_id, get_init_error
        if is_cloud_device:
            if not is_firebase_available():
                err = get_init_error() or "FIREBASE_CREDENTIALS_JSON not configured"
                return jsonify({
                    "ok": False,
                    "type": "firebase",
                    "device_id": ip,
                    "message": f"Firebase Cloud is not configured on the hosted server ({err}). Please add the 'FIREBASE_CREDENTIALS_JSON' environment variable in your Render dashboard."
                })

            dev_id = ip.strip().lower()
            status = get_device_status(dev_id)
            if status.get("online"):
                sec = status.get("seconds_ago", 0)
                cam_status = status.get("data", {}).get("camera_server", "online")
                return jsonify({
                    "ok": True,
                    "type": "firebase",
                    "device_id": dev_id,
                    "message": f"Connected to Raspberry Pi ({dev_id}) via Firebase Cloud! Heartbeat received {sec}s ago. Camera server: {cam_status}."
                })
            else:
                sec = status.get("seconds_ago")
                last_msg = f"Last seen {sec}s ago." if sec is not None else "Device has not sent a heartbeat yet."
                return jsonify({
                    "ok": False,
                    "type": "firebase",
                    "device_id": dev_id,
                    "message": f"Device '{dev_id}' is OFFLINE in Firebase Cloud. {last_msg} Make sure 'python3 pi_firebase_bridge.py' is running on the Pi."
                })
    except Exception as e:
        print(f"[API] Firebase test check exception: {e}")

    # Fallback to direct HTTP check (for local development on same Wi-Fi)
    try:
        start_t = time.time()
        r = requests.get(f"http://{ip}:8080/health", timeout=2.5)
        latency = round((time.time() - start_t) * 1000)
        if r.ok:
            return jsonify({"ok": True, "type": "pi", "latency_ms": latency, "message": f"Connected to Raspberry Pi ({latency}ms latency)"})
        else:
            return jsonify({"ok": False, "type": "pi", "message": f"Pi replied with HTTP {r.status_code}"})
    except Exception as e:
        return jsonify({"ok": False, "type": "pi", "message": f"Cannot connect to {ip}:8080. If using cloud mode, enter device ID 'pisite_01'."})

@app.route("/api/pi/start", methods=["POST"])
def api_start_pi():
    global _stream_active
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    _stream_active = True

    user = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    if not user:
        return jsonify({"error": "User not found."}), 404

    pi_ip = (user.get("pi_ip") or "").strip()
    is_render = os.getenv("RENDER") is not None
    is_cloud_device = bool(pi_ip and not ("." in pi_ip or ":" in pi_ip))

    if is_render:
        # On Render / Cloud deployment, dispatch stream command to Pi via Firebase
        try:
            from firebase_service import is_firebase_available, send_device_command, get_default_device_id
            if is_firebase_available():
                dev_id = (pi_ip if is_cloud_device else get_default_device_id()).strip().lower()
                send_device_command(action="stream_start", device_id=dev_id, wait_completion=False)
                return jsonify({
                    "message": f"Stream activated. Raspberry Pi ({dev_id}) is streaming frames to the cloud.",
                    "running": True
                })
        except Exception as e:
            print(f"[API] Cloud stream activation note: {e}")
        return jsonify({
            "message": "Cloud stream active. Waiting for frames from Raspberry Pi or CV engine.",
            "running": True
        })

    success = start_webcam_detection(session["user_id"])
    if not success:
        return jsonify({"error": "Failed to start webcam detection script."}), 500

    return jsonify({"message": "Camera stream and AI safety detection started.", "running": True})

@app.route("/api/pi/cloud_status", methods=["GET"])
def api_pi_cloud_status():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    user = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    pi_ip = (user.get("pi_ip") if user else None) or ""
    try:
        from firebase_service import is_firebase_available, get_device_status, get_default_device_id, get_init_error
        dev_id = (pi_ip if (pi_ip and not ("." in pi_ip or ":" in pi_ip)) else get_default_device_id()).strip().lower()
        if is_firebase_available():
            return jsonify(get_device_status(dev_id))
        return jsonify({"configured": False, "online": False, "message": f"Firebase not initialized ({get_init_error() or 'No credentials'})"})
    except Exception as e:
        return jsonify({"configured": False, "online": False, "error": str(e)})

@app.route("/api/pi/actuate/<action>", methods=["POST"])
def api_pi_actuate(action):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    user = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    pi_ip = (user.get("pi_ip") if user else None) or ""

    # Parse params
    params = {}
    if action == "ppe":
        score_val = 100
        if request.is_json and request.json and "score" in request.json:
            try: score_val = int(request.json["score"])
            except: score_val = 100
        elif request.args.get("score"):
            try: score_val = int(request.args.get("score"))
            except: score_val = 100
        params["score"] = score_val

    # 1. Try Firebase Cloud Route First
    try:
        from firebase_service import is_firebase_available, send_device_command, get_default_device_id
        is_render = os.getenv("RENDER") is not None
        is_cloud_device = bool(pi_ip and not ("." in pi_ip or ":" in pi_ip))

        if is_firebase_available() and (is_render or is_cloud_device or not pi_ip or pi_ip.strip().lower() == get_default_device_id()):
            dev_id = (pi_ip if is_cloud_device else get_default_device_id()).strip().lower()
            res = send_device_command(action=action, params=params, device_id=dev_id, wait_completion=False)
            if res.get("ok"):
                return jsonify({
                    "message": f"Command '{action}' dispatched to Raspberry Pi ({dev_id})",
                    "channel": "firebase",
                    "status": "dispatched"
                })
            else:
                return jsonify({"error": res.get("error", "Failed to dispatch command to Firebase")}), 502
    except Exception as e:
        print(f"[API] Firebase actuation fallback error: {e}")

    # 2. Local Fallback via direct HTTP to PiController
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "interface"))
    from pi_controller import PiController
    controller = PiController(pi_ip or "0")

    if action == "checkin":
        controller.checkin()
        return jsonify({"message": "Green LED (Check-in) command sent to Pi"})
    elif action == "checkout":
        controller.checkout()
        return jsonify({"message": "Red LED (Check-out) command sent to Pi"})
    elif action == "buzzer":
        def _buzz():
            controller.buzzer_on()
            time.sleep(1.0)
            controller.buzzer_off()
        Thread(target=_buzz, daemon=True).start()
        return jsonify({"message": "Buzzer test command sent"})
    elif action == "ppe":
        score_val = params.get("score", 100)
        controller.show_score(score_val)
        return jsonify({"message": f"Matrix test command sent for score {score_val}%"})
    elif action in ["verify", "button"]:
        controller.set_state("ACTIVE")
        return jsonify({"message": "Verification trigger sent to Raspberry Pi"})
    return jsonify({"error": "Invalid action"}), 400

@app.route("/api/pi/trigger-verify", methods=["POST"])
def api_pi_trigger_verify():
    """Triggers gate evaluation (virtual button press) via Firebase or local Pi."""
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    pi_ip = (user.get("pi_ip") if user else None) or ""

    try:
        from firebase_service import is_firebase_available, send_device_command, get_default_device_id
        if is_firebase_available():
            dev_id = (pi_ip if (pi_ip and not ("." in pi_ip or ":" in pi_ip)) else get_default_device_id()).strip().lower()
            send_device_command(action="state", params={"set": "ACTIVE"}, device_id=dev_id, wait_completion=False)
            send_device_command(action="button", device_id=dev_id, wait_completion=False)
            return jsonify({"ok": True, "message": "Gate evaluation triggered via Firebase Cloud."})
    except Exception as e:
        print(f"[API] Firebase trigger-verify note: {e}")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "interface"))
        from pi_controller import PiController
        controller = PiController(pi_ip or "0")
        controller.set_state("ACTIVE")
        return jsonify({"ok": True, "message": "Gate evaluation triggered via local controller."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/pi/status", methods=["GET"])
def api_pi_status():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    if not _stream_active:
        return jsonify({
            "running": False,
            "streaming": False,
            "last_frame_sec": None,
            "exit_code": None
        })

    frames_active = bool(_frame_updated_at and (time.time() - _frame_updated_at < 6.0))
    if not frames_active and _SHARED_FRAME_PATH.is_file():
        try:
            if time.time() - _SHARED_FRAME_PATH.stat().st_mtime < 6.0:
                frames_active = True
        except Exception:
            pass

    is_running = (webcam_process is not None and webcam_process.poll() is None) or frames_active
    exit_code = webcam_process.poll() if webcam_process is not None else None
    return jsonify({
        "running": is_running,
        "streaming": frames_active,
        "last_frame_sec": round(time.time() - _frame_updated_at, 1) if _frame_updated_at else None,
        "exit_code": exit_code
    })

@app.route("/api/pi/log", methods=["GET"])
def api_pi_log():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    try:
        with open(str(_webcam_log_path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return jsonify({"log": "".join(lines[-80:]), "exit_code": webcam_process.poll() if webcam_process else None})
    except FileNotFoundError:
        return jsonify({"log": "No log file yet. Start the Pi stream first.", "exit_code": None})
    except Exception as e:
        return jsonify({"log": str(e), "exit_code": None})

@app.route("/api/pi/stop", methods=["POST"])
def api_pi_stop():
    global webcam_process, _latest_frame_jpeg, _frame_updated_at, _stream_active
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    _stream_active = False
    with _frame_lock:
        _latest_frame_jpeg = b""
        _frame_updated_at = 0.0

    try:
        if _SHARED_FRAME_PATH.is_file():
            _SHARED_FRAME_PATH.unlink()
    except Exception:
        pass

    user = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    pi_ip = (user.get("pi_ip") if user else None) or ""
    is_render = os.getenv("RENDER") is not None
    is_cloud_device = bool(pi_ip and not ("." in pi_ip or ":" in pi_ip))

    # Send stream_stop to Raspberry Pi via Firebase Cloud
    if is_render or is_cloud_device:
        try:
            from firebase_service import is_firebase_available, send_device_command, get_default_device_id
            if is_firebase_available():
                dev_id = (pi_ip if is_cloud_device else get_default_device_id()).strip().lower()
                send_device_command(action="stream_stop", device_id=dev_id, wait_completion=False)
        except Exception as e:
            print(f"[API] Cloud stream stop error: {e}")

    if webcam_process and webcam_process.poll() is None:
        webcam_process.kill()
        webcam_process = None

    return jsonify({"message": "Camera stream stopped.", "running": False})

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    if os.getenv("RENDER") is None:
        Timer(1, open_browser).start()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
