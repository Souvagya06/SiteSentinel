"""
EventClient — Helper for sending structured safety events and attendance
updates to the Flask backend with error handling and retry support.
"""
import time
import threading
import requests


class EventClient:
    def __init__(self, backend_url: str, manager_id: str):
        self.backend_url = backend_url.rstrip("/")
        self.manager_id = manager_id

    def log_safety_event(self, event_type: str, message: str, worker_id: str = "",
                         ppe_score: int | None = None, helmet_id: str = "", camera_id: str = "Pi Camera 01",
                         evidence_url: str = ""):
        """Sends a safety event to /api/events in a background thread."""
        def _send():
            payload = {
                "manager_id": self.manager_id,
                "worker_id": worker_id,
                "event_type": event_type,
                "message": message,
                "ppe_score": ppe_score,
                "helmet_id": helmet_id,
                "camera_id": camera_id,
                "evidence_url": evidence_url
            }
            try:
                resp = requests.post(
                    f"{self.backend_url}/api/events",
                    json=payload,
                    timeout=5
                )
                if not resp.ok:
                    print(f"[EventClient] Failed to log safety event ({resp.status_code}): {resp.text}")
            except Exception as e:
                print(f"[EventClient] Error posting safety event: {e}")

        threading.Thread(target=_send, daemon=True).start()

    def record_attendance(self, worker_id: str, ppe_score: int, helmet_id: str = ""):
        """
        Sends an attendance action to /api/attendance.
        Returns the parsed JSON response or None on error.
        """
        try:
            payload = {
                "worker_id": worker_id,
                "user_id": self.manager_id,
                "ppe_score": ppe_score,
                "helmet_id": helmet_id
            }
            resp = requests.post(
                f"{self.backend_url}/api/attendance",
                json=payload,
                timeout=5
            )
            if resp.ok:
                return resp.json()
            else:
                print(f"[EventClient] Attendance request failed ({resp.status_code}): {resp.text}")
                return None
        except Exception as e:
            print(f"[EventClient] Error recording attendance: {e}")
            return None
