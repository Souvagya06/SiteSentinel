import os
import sys
from pathlib import Path

# Ensure backend folder is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import execute, query_one, query_all
from flask import Flask, send_from_directory, request, jsonify, session, redirect, Response
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
_latest_frame_jpeg: bytes = b""
_frame_lock = Lock()
_frame_updated_at: float = 0.0

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../frontend/pages")
)

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

    execute(
        "INSERT INTO workers (user_id, worker_id, first_name, last_name, image_url, helmet_id) VALUES (?, ?, ?, ?, ?, ?)",
        [
            {"type": "text", "value": str(session["user_id"])},
            {"type": "text", "value": worker_id},
            {"type": "text", "value": first_name},
            {"type": "text", "value": last_name},
            {"type": "text", "value": image_url},
            {"type": "text", "value": helmet_id},
        ]
    )

    if helmet_id:
        execute(
            "UPDATE helmets SET status = 'Occupied' WHERE helmet_id = ? AND user_id = ?",
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
        execute(
            "UPDATE helmets SET status = 'Occupied' WHERE helmet_id = ? AND user_id = ?",
            [
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
    result = execute(
        "SELECT * FROM helmets WHERE user_id = ? ORDER BY created_at DESC",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    try:
        cols    = [c["name"] for c in result["results"][0]["response"]["result"]["cols"]]
        rows    = result["results"][0]["response"]["result"]["rows"]
        helmets = [dict(zip(cols, [v.get("value") for v in row])) for row in rows]
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

    execute(
        "UPDATE helmets SET status = 'Occupied' WHERE helmet_id = ? AND user_id = ?",
        [
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
        SELECT w.id, w.worker_id, w.first_name, w.last_name, w.helmet_id, w.status, wi.face_embedding
        FROM workers w
        JOIN worker_images wi ON w.worker_id = wi.worker_db_id
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

    worker = query_one(
        "SELECT * FROM workers WHERE worker_id = ? AND (user_id = ? OR ? IS NULL)",
        [
            {"type": "text", "value": worker_id},
            {"type": "text", "value": str(user_id_req) if user_id_req else ""},
            {"type": "text", "value": str(user_id_req) if user_id_req else None}
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

    if event == "CHECK-IN":
        worker_update_sql = "UPDATE workers SET checkin_time = ?, checkout_time = '', ppe_score = ?, status = ?"
        worker_update_args = [
            {"type": "text", "value": checkin_time or time_str},
            {"type": "text", "value": str(ppe_score)},
            {"type": "text", "value": new_status},
        ]
    else:
        worker_update_sql = "UPDATE workers SET checkout_time = ?, ppe_score = ?, status = ?"
        worker_update_args = [
            {"type": "text", "value": time_str},
            {"type": "text", "value": str(ppe_score)},
            {"type": "text", "value": new_status},
        ]

    if helmet_id:
        worker_update_sql += ", helmet_id = ?"
        worker_update_args.append({"type": "text", "value": helmet_id})

    worker_update_sql += " WHERE id = ?"
    worker_update_args.append({"type": "text", "value": str(worker["id"])})

    execute(worker_update_sql, worker_update_args)

    # Sync helmets table status to match the worker's assignment
    effective_helmet = helmet_id or (worker.get("helmet_id") or "")
    if event == "CHECK-IN" and effective_helmet:
        # Mark the assigned helmet as Occupied
        execute(
            "UPDATE helmets SET status = 'Occupied' WHERE helmet_id = ? AND user_id = ?",
            [
                {"type": "text", "value": effective_helmet},
                {"type": "text", "value": manager_id}
            ]
        )
    elif event == "CHECK-OUT" and effective_helmet:
        # Release the helmet back to Available when worker checks out
        execute(
            "UPDATE helmets SET status = 'Available' WHERE helmet_id = ? AND user_id = ?",
            [
                {"type": "text", "value": effective_helmet},
                {"type": "text", "value": manager_id}
            ]
        )

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
            {"type": "text", "value": helmet_id or worker.get("helmet_id", "")},
        ]
    )

    # Automatically generate structured safety event
    worker_full_name = f"{worker.get('first_name', '')} {worker.get('last_name', '')}".strip()
    if ppe_score < 100:
        event_type = "PPE_VIOLATION"
        msg = f"{worker_full_name} ({worker_id}) {event.lower()} with non-compliant PPE score {ppe_score}%"
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
            {"type": "text", "value": helmet_id or worker.get("helmet_id", "")},
            {"type": "text", "value": "Pi Camera 01"},
        ]
    )

    return jsonify({
        "message": f"{event} logged for {worker_full_name}.",
        "worker_id": worker_id,
        "event": event,
        "status": new_status,
        "ppe_score": ppe_score,
        "helmet_id": helmet_id or worker.get("helmet_id", "")
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

# ── Dashboard MJPEG Stream Routes ──────────────────────
@app.route("/api/pi/frame/push", methods=["POST"])
def push_frame():
    """Called by webcam_detection.py to push the latest annotated JPEG frame."""
    global _latest_frame_jpeg, _frame_updated_at
    data = request.get_data()
    if data:
        with _frame_lock:
            _latest_frame_jpeg = data
            _frame_updated_at = time.time()
    return '', 204

def _mjpeg_generator():
    """Yields MJPEG frames from the shared buffer for the dashboard stream."""
    BOUNDARY = b"--frame"
    while True:
        with _frame_lock:
            frame = _latest_frame_jpeg
            updated = _frame_updated_at
        if frame:
            yield (
                BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame + b"\r\n"
            )
        time.sleep(0.04)  # ~25 fps cap

@app.route("/api/pi/stream")
def pi_stream():
    """MJPEG stream endpoint consumed by the dashboard <img> tag."""
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/pi/test", methods=["POST"])
def api_test_pi():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json() or {}
    ip = data.get("pi_ip", "").strip()
    if not ip or ip in ["0", "1", "local", "webcam", "usb"]:
        return jsonify({"ok": True, "type": "local", "message": "Local camera mode (no remote Pi required)."})
    try:
        start_t = time.time()
        r = requests.get(f"http://{ip}:8080/health", timeout=2.5)
        latency = round((time.time() - start_t) * 1000)
        if r.ok:
            return jsonify({"ok": True, "type": "pi", "latency_ms": latency, "message": f"Connected to Raspberry Pi ({latency}ms latency)"})
        else:
            return jsonify({"ok": False, "type": "pi", "message": f"Pi replied with HTTP {r.status_code}"})
    except Exception as e:
        return jsonify({"ok": False, "type": "pi", "message": f"Cannot connect to {ip}:8080. Make sure the Pi is running raspberry_Pi_Code.py on this Wi-Fi network."})

@app.route("/api/pi/start", methods=["POST"])
def api_start_pi():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    user = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    if not user:
        return jsonify({"error": "User not found."}), 404

    success = start_webcam_detection(session["user_id"])
    if not success:
        return jsonify({"error": "Failed to start webcam detection script."}), 500

    return jsonify({"message": "Camera stream and AI safety detection started.", "running": True})

@app.route("/api/pi/actuate/<action>", methods=["POST"])
def api_pi_actuate(action):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    user = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [{"type": "text", "value": str(session["user_id"])}]
    )
    pi_ip = (user.get("pi_ip") if user else None) or "0"
    
    # Import PiController
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "interface"))
    from pi_controller import PiController
    controller = PiController(pi_ip)
    
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
        controller.show_score(100)
        return jsonify({"message": "Matrix test command sent"})
    return jsonify({"error": "Invalid action"}), 400

@app.route("/api/pi/status", methods=["GET"])
def api_pi_status():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    is_running = webcam_process is not None and webcam_process.poll() is None
    exit_code = webcam_process.poll() if webcam_process is not None else None
    return jsonify({"running": is_running, "exit_code": exit_code})

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
    global webcam_process, _latest_frame_jpeg
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    if webcam_process and webcam_process.poll() is None:
        webcam_process.kill()
        webcam_process = None
        with _frame_lock:
            _latest_frame_jpeg = b""
        return jsonify({"message": "Pi stream stopped.", "running": False})
    return jsonify({"message": "Pi stream is not currently running.", "running": False})

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    if os.getenv("RENDER") is None:
        Timer(1, open_browser).start()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)