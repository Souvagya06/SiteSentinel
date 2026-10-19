import os
import sys
import json
import time
import secrets
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_firebase_app = None
_firebase_initialized = False
_init_error = None

def _init_firebase():
    """Initializes Firebase Admin SDK lazily using service account credentials."""
    global _firebase_app, _firebase_initialized, _init_error
    if _firebase_initialized:
        return _firebase_app is not None

    _firebase_initialized = True
    database_url = os.getenv(
        "FIREBASE_DATABASE_URL",
        "https://sitesentinel-50bfc-default-rtdb.asia-southeast1.firebasedatabase.app/"
    ).strip()

    try:
        import firebase_admin
        from firebase_admin import credentials, db

        # Avoid re-initializing if already done elsewhere
        try:
            _firebase_app = firebase_admin.get_app()
            print("[Firebase] Using existing Firebase Admin app instance.")
            return True
        except ValueError:
            pass

        cred = None

        # 1. Check raw JSON string in environment (ideal for Render cloud hosting)
        cred_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON", "").strip()
        if cred_json_str:
            try:
                cert_dict = json.loads(cred_json_str)
                cred = credentials.Certificate(cert_dict)
                print("[Firebase] Loaded credentials from FIREBASE_CREDENTIALS_JSON env var.")
            except Exception as e:
                print(f"[Firebase] Error parsing FIREBASE_CREDENTIALS_JSON: {e}")

        # 2. Check JSON file path (ideal for local development)
        if not cred:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json").strip()
            # Check relative to backend or project root
            candidates = [
                Path(cred_path),
                Path(__file__).resolve().parent / cred_path,
                Path(__file__).resolve().parent.parent / cred_path,
                Path(__file__).resolve().parent.parent / "firebase-credentials.json",
                Path(__file__).resolve().parent.parent / "serviceAccountKey.json"
            ]
            for c in candidates:
                if c.is_file():
                    try:
                        cred = credentials.Certificate(str(c))
                        print(f"[Firebase] Loaded credentials from file: {c}")
                        break
                    except Exception as e:
                        print(f"[Firebase] Error loading certificate from {c}: {e}")

        if not cred:
            _init_error = "No valid Firebase credentials found (checked FIREBASE_CREDENTIALS_JSON and firebase-credentials.json)."
            print(f"[Firebase] WARNING: {_init_error}")
            return False

        _firebase_app = firebase_admin.initialize_app(cred, {
            "databaseURL": database_url
        })
        print(f"[Firebase] Successfully initialized with Database URL: {database_url}")
        return True

    except ImportError:
        _init_error = "firebase-admin library is not installed."
        print(f"[Firebase] WARNING: {_init_error}")
        return False
    except Exception as e:
        _init_error = str(e)
        print(f"[Firebase] Initialization failed: {e}")
        return False

def get_default_device_id() -> str:
    return os.getenv("FIREBASE_DEVICE_ID", "pisite_01").strip()

def is_firebase_available() -> bool:
    return _init_firebase()

def get_device_status(device_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Reads the device status from Firebase Realtime Database.
    Determines if device is currently online (last_seen within 15 seconds).
    """
    if not _init_firebase():
        return {
            "configured": False,
            "online": False,
            "error": _init_error or "Firebase not initialized"
        }

    dev_id = device_id or get_default_device_id()
    try:
        from firebase_admin import db
        ref = db.reference(f"devices/{dev_id}")
        data = ref.get() or {}

        last_seen = data.get("last_seen", 0)
        now = time.time()
        # Device is considered online if heartbeat was within last 18 seconds
        is_online = bool(last_seen and (now - float(last_seen)) < 18.0)

        return {
            "configured": True,
            "device_id": dev_id,
            "online": is_online,
            "last_seen": last_seen,
            "seconds_ago": round(now - float(last_seen), 1) if last_seen else None,
            "button_pressed": data.get("button_pressed", False),
            "verification_state": data.get("verification_state", "IDLE"),
            "green": data.get("green", False),
            "red": data.get("red", False),
            "buzzer": data.get("buzzer", False),
            "uptime": data.get("uptime", 0),
            "data": data
        }
    except Exception as e:
        return {
            "configured": True,
            "device_id": dev_id,
            "online": False,
            "error": str(e)
        }

def send_device_command(
    action: str,
    params: Optional[Dict[str, Any]] = None,
    device_id: Optional[str] = None,
    wait_completion: bool = True,
    timeout_s: float = 3.5
) -> Dict[str, Any]:
    """
    Writes an authorized command for the Raspberry Pi into Firebase RTDB under /commands/{DEVICE_ID}.
    Optionally polls for the Pi's completion acknowledgment up to timeout_s.
    """
    if not _init_firebase():
        return {
            "ok": False,
            "error": f"Firebase not configured on server: {_init_error}"
        }

    dev_id = device_id or get_default_device_id()
    cmd_id = f"cmd_{int(time.time()*1000)}_{secrets.token_hex(3)}"
    now = time.time()

    payload = {
        "command_id": cmd_id,
        "action": action,
        "params": params or {},
        "status": "pending",
        "created_at": now
    }

    try:
        from firebase_admin import db
        cmd_ref = db.reference(f"commands/{dev_id}")
        cmd_ref.set(payload)

        if not wait_completion:
            return {
                "ok": True,
                "command_id": cmd_id,
                "action": action,
                "status": "pending",
                "message": f"Command '{action}' dispatched to Firebase queue for {dev_id}."
            }

        # Poll briefly for Pi acknowledgment
        start_wait = time.time()
        while time.time() - start_wait < timeout_s:
            time.sleep(0.15)
            curr = cmd_ref.get() or {}
            if curr.get("command_id") == cmd_id:
                curr_status = curr.get("status")
                if curr_status in ["completed", "acknowledged"]:
                    return {
                        "ok": True,
                        "command_id": cmd_id,
                        "action": action,
                        "status": curr_status,
                        "latency_ms": round((time.time() - start_wait) * 1000),
                        "message": f"Command '{action}' executed by Raspberry Pi ({curr_status})."
                    }
                elif curr_status == "failed":
                    return {
                        "ok": False,
                        "command_id": cmd_id,
                        "action": action,
                        "status": "failed",
                        "error": curr.get("error", "Pi reported command failure.")
                    }

        # Timed out waiting for ack, but command is queued in Firebase
        return {
            "ok": True,
            "command_id": cmd_id,
            "action": action,
            "status": "queued",
            "message": f"Command '{action}' delivered to Firebase for {dev_id}. Pi may execute momentarily."
        }

    except Exception as e:
        return {
            "ok": False,
            "action": action,
            "error": f"Failed to send command to Firebase: {e}"
        }

def log_device_event(event_type: str, data: Dict[str, Any], device_id: Optional[str] = None):
    """Optionally records a device telemetry event in /device_events/{dev_id}."""
    if not _init_firebase():
        return
    dev_id = device_id or get_default_device_id()
    try:
        from firebase_admin import db
        ref = db.reference(f"device_events/{dev_id}").push()
        ref.set({
            "event_type": event_type,
            "timestamp": time.time(),
            **data
        })
    except Exception:
        pass
