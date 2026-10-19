#!/usr/bin/env python3
"""
SiteSentinel — Raspberry Pi Firebase Cloud Bridge (Outbound Connection)
Runs on the Raspberry Pi alongside Pi_Camera.py.

Role:
1. Connects OUTBOUND to Firebase Realtime Database (no port forwarding or static IP needed).
2. Sends device telemetry and heartbeat every 3 seconds to:
      /devices/{DEVICE_ID}
3. Listens for authorized commands dispatched by the hosted dashboard at:
      /commands/{DEVICE_ID}
4. Forwards commands to the local hardware server (http://127.0.0.1:8080) for GPIO/LED/Matrix/Buzzer.

Run on Raspberry Pi:
    python3 pi_firebase_bridge.py
"""

import os
import sys
import time
import json
import socket
import threading
import requests
from pathlib import Path

# Configuration defaults (can be overridden via environment variables)
FIREBASE_DATABASE_URL = os.getenv(
    "FIREBASE_DATABASE_URL",
    "https://sitesentinel-50bfc-default-rtdb.asia-southeast1.firebasedatabase.app/"
).strip()
DEVICE_ID = os.getenv("FIREBASE_DEVICE_ID", "pisite_01").strip()
PI_LOCAL_SERVER = os.getenv("PI_LOCAL_SERVER", "http://127.0.0.1:8080").rstrip("/")
CREDENTIALS_FILE = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")

def find_credentials_file():
    candidates = [
        Path(CREDENTIALS_FILE),
        Path(__file__).resolve().parent / CREDENTIALS_FILE,
        Path(__file__).resolve().parent / "firebase-credentials.json",
        Path(__file__).resolve().parent / "serviceAccountKey.json",
        Path.home() / "Site" / "firebase-credentials.json",
        Path.home() / "Site" / "serviceAccountKey.json",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

print("=" * 60)
print(" SiteSentinel — Raspberry Pi Firebase Realtime Cloud Bridge")
print("=" * 60)
print(f"Device ID    : {DEVICE_ID}")
print(f"Firebase URL : {FIREBASE_DATABASE_URL}")
print(f"Local Server : {PI_LOCAL_SERVER}")
print(f"Local IP     : {get_local_ip()}")

# Import Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, db
except ImportError:
    print("\n[ERROR] firebase-admin is not installed.")
    print("Please install it on the Raspberry Pi using:")
    print("    pip3 install firebase-admin requests\n")
    sys.exit(1)

# Locate credentials
cred_path = find_credentials_file()
if not cred_path:
    print(f"\n[ERROR] Firebase credentials file '{CREDENTIALS_FILE}' not found.")
    print("Please place your downloaded service account JSON file in this directory as:")
    print(f"    {Path(__file__).resolve().parent / 'firebase-credentials.json'}\n")
    sys.exit(1)

print(f"Credentials  : {cred_path}")

try:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        "databaseURL": FIREBASE_DATABASE_URL
    })
    print("[Firebase] Successfully connected to Realtime Database!")
except Exception as e:
    print(f"\n[FATAL] Failed to initialize Firebase Admin SDK: {e}")
    sys.exit(1)

device_ref = db.reference(f"devices/{DEVICE_ID}")
command_ref = db.reference(f"commands/{DEVICE_ID}")

# ── Command Execution Logic ──────────────────────────────────────────
_last_executed_cmd_id = None
_cmd_lock = threading.Lock()

def execute_action(action: str, params: dict) -> tuple[bool, str]:
    """Forwards command to local Pi_Camera.py server on port 8080."""
    action = (action or "").strip().lower()
    url = None

    if action in ["checkin", "green"]:
        url = f"{PI_LOCAL_SERVER}/checkin"
    elif action in ["checkout", "red"]:
        url = f"{PI_LOCAL_SERVER}/checkout"
    elif action == "buzzer":
        # Turn buzzer on, sleep 1s, turn off
        try:
            requests.get(f"{PI_LOCAL_SERVER}/on", timeout=1.5)
            time.sleep(1.0)
            requests.get(f"{PI_LOCAL_SERVER}/off", timeout=1.5)
            return True, "Buzzer sounded for 1s"
        except Exception as e:
            return False, f"Buzzer actuation failed: {e}"
    elif action in ["buzzer_on", "on"]:
        url = f"{PI_LOCAL_SERVER}/on"
    elif action in ["buzzer_off", "off"]:
        url = f"{PI_LOCAL_SERVER}/off"
    elif action == "ppe":
        score = params.get("score", 100)
        url = f"{PI_LOCAL_SERVER}/ppe?score={score}"
    elif action == "reset":
        url = f"{PI_LOCAL_SERVER}/reset"
    elif action == "state":
        st = params.get("set", "IDLE")
        url = f"{PI_LOCAL_SERVER}/state?set={st}"
    else:
        return False, f"Unknown action: '{action}'"

    try:
        r = requests.get(url, timeout=2.5)
        if r.ok:
            return True, f"Executed {action} (HTTP {r.status_code})"
        return False, f"Local server returned HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return False, f"Failed to contact {url}: {e}"

def handle_command(cmd_data: dict):
    global _last_executed_cmd_id
    if not isinstance(cmd_data, dict):
        return

    cmd_id = cmd_data.get("command_id")
    status = cmd_data.get("status")
    action = cmd_data.get("action")
    params = cmd_data.get("params") or {}

    if not cmd_id or status != "pending":
        return

    with _cmd_lock:
        if cmd_id == _last_executed_cmd_id:
            return
        _last_executed_cmd_id = cmd_id

    print(f"\n[Command] Received: ID={cmd_id} Action='{action}' Params={params}")

    # Mark as in-progress
    try:
        command_ref.update({
            "status": "acknowledged",
            "ack_at": time.time()
        })
    except Exception:
        pass

    # Execute
    success, msg = execute_action(action, params)

    now = time.time()
    try:
        if success:
            print(f"[Command] SUCCESS: {msg}")
            command_ref.update({
                "status": "completed",
                "completed_at": now,
                "result": msg
            })
        else:
            print(f"[Command] ERROR: {msg}")
            command_ref.update({
                "status": "failed",
                "failed_at": now,
                "error": msg
            })
    except Exception as e:
        print(f"[Firebase] Error updating command status: {e}")

# ── Command Listener Thread ──────────────────────────────────────────
def command_stream_listener():
    """Listens for commands via Firebase Realtime Database SSE stream."""
    print("[Listener] Starting real-time command listener on Firebase...")
    
    def on_change(event):
        try:
            # event.data could be the full command dict or a single updated field
            data = command_ref.get()
            if isinstance(data, dict):
                handle_command(data)
        except Exception as e:
            print(f"[Listener] Stream error: {e}")

    while True:
        try:
            stream = command_ref.listen(on_change)
            # Keep stream open
            time.sleep(3600)
        except Exception as e:
            print(f"[Listener] Connection reset, reconnecting in 5s: {e}")
            time.sleep(5)

# ── Polling Fallback for Commands ─────────────────────────────────────
def command_polling_fallback():
    """Safety poller to ensure no command is missed even if SSE stream disconnects."""
    while True:
        try:
            data = command_ref.get()
            if isinstance(data, dict) and data.get("status") == "pending":
                handle_command(data)
        except Exception:
            pass
        time.sleep(1.0)

# ── Heartbeat and Status Reporter Thread ──────────────────────────────
def heartbeat_loop():
    print("[Heartbeat] Starting telemetry reporter (every 3 seconds)...")
    local_ip = get_local_ip()

    while True:
        now = time.time()
        status_payload = {
            "status": "online",
            "last_seen": now,
            "device_id": DEVICE_ID,
            "local_ip": local_ip,
            "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        }

        # Query local Pi_Camera.py server on port 8080
        try:
            r = requests.get(f"{PI_LOCAL_SERVER}/status", timeout=1.2)
            if r.status_code == 200:
                local_data = r.json()
                status_payload.update({
                    "camera_server": "online",
                    "button_pressed": bool(local_data.get("button_pressed", False)),
                    "verification_state": str(local_data.get("state", "IDLE")),
                    "green": bool(local_data.get("green", False)),
                    "red": bool(local_data.get("red", False)),
                    "buzzer": bool(local_data.get("buzzer", False)),
                    "uptime": local_data.get("uptime", 0)
                })
            else:
                status_payload["camera_server"] = f"HTTP {r.status_code}"
        except Exception as e:
            status_payload["camera_server"] = "offline (Pi_Camera.py not running on port 8080)"

        try:
            device_ref.update(status_payload)
        except Exception as e:
            print(f"[Heartbeat] Error reporting to Firebase: {e}")

        time.sleep(3.0)

# ── Cloud Stream Forwarder ─────────────────────────────────────────────
CLOUD_STREAM_URL = os.getenv("CLOUD_STREAM_URL", "https://sitesentinel.site/api/pi/frame/push").strip()
ENABLE_STREAM_FORWARD = os.getenv("ENABLE_CLOUD_STREAM", "true").lower() in ["true", "1", "yes"]

def stream_forwarder():
    """Reads MJPEG stream from local Pi_Camera.py (port 8080) and POSTs frames to hosted site."""
    if not ENABLE_STREAM_FORWARD or not CLOUD_STREAM_URL:
        return
    local_url = f"{PI_LOCAL_SERVER}/stream"
    print(f"[Stream Forwarder] Forwarding {local_url} -> {CLOUD_STREAM_URL}")
    post_session = requests.Session()
    while True:
        try:
            r = requests.get(local_url, stream=True, timeout=5)
            bytes_buf = b""
            for chunk in r.iter_content(chunk_size=4096):
                bytes_buf += chunk
                a = bytes_buf.find(b"\xff\xd8")
                b = bytes_buf.find(b"\xff\xd9")
                if a != -1 and b != -1:
                    jpg = bytes_buf[a:b+2]
                    bytes_buf = bytes_buf[b+2:]
                    try:
                        post_session.post(
                            CLOUD_STREAM_URL,
                            data=jpg,
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=0.6
                        )
                    except Exception:
                        pass
                    time.sleep(0.08)  # ~12 fps
        except Exception:
            time.sleep(2.0)

if __name__ == "__main__":
    t_cmd_stream = threading.Thread(target=command_stream_listener, daemon=True)
    t_cmd_poll = threading.Thread(target=command_polling_fallback, daemon=True)
    t_heartbeat = threading.Thread(target=heartbeat_loop, daemon=True)
    t_stream = threading.Thread(target=stream_forwarder, daemon=True)

    t_cmd_stream.start()
    t_cmd_poll.start()
    t_heartbeat.start()
    t_stream.start()

    print("\n[READY] Raspberry Pi is online and connected to SiteSentinel Cloud!")
    print("Press Ctrl+C to terminate.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping bridge...")
        try:
            device_ref.update({"status": "offline", "last_seen": time.time()})
        except Exception:
            pass
        print("Bridge stopped.")
