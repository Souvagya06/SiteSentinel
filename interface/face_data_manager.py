"""
FaceDataManager — Handles loading, caching, and hot-reloading face embeddings
for the active manager without restarting the CV pipeline.
"""
import time
import json
import threading
import requests
import numpy as np
import face_recognition


class FaceDataManager:
    def __init__(self, backend_url: str, manager_id: str, poll_interval: float = 10.0):
        self.backend_url = backend_url.rstrip("/")
        self.manager_id = manager_id
        self.poll_interval = poll_interval

        self._lock = threading.Lock()
        self._known_faces = []  # List of dicts: {"worker_id": str, "name": str, "embedding": np.ndarray}
        self._current_version = 0
        self._last_checked: float = 0.0
        self._last_reloaded: float = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def start_hot_reload(self):
        """Initializes face snapshot and starts background version check thread."""
        self.reload()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                time.sleep(self.poll_interval)
                self.check_and_reload_if_stale()
            except Exception as e:
                print(f"[FaceDataManager] Background poll error: {e}")

    def check_and_reload_if_stale(self) -> bool:
        """Queries the backend for face data version. Reloads if version incremented."""
        try:
            resp = requests.get(
                f"{self.backend_url}/api/face-data/version",
                params={"manager_id": self.manager_id},
                timeout=4
            )
            if resp.ok:
                data = resp.json()
                remote_version = data.get("version", 0)
                if remote_version > self._current_version:
                    print(f"[FaceDataManager] New version detected ({remote_version} > {self._current_version}). Reloading...")
                    return self.reload(new_version=remote_version)
            self._last_checked = float(time.time())
        except Exception as e:
            # Keep existing snapshot on error
            pass
        return False

    def reload(self, new_version: int | None = None) -> bool:
        """Fetches worker face data from backend and atomically updates the snapshot."""
        try:
            resp = requests.get(
                f"{self.backend_url}/api/face-data",
                params={"manager_id": self.manager_id},
                timeout=8
            )
            if not resp.ok:
                print(f"[FaceDataManager] Failed to fetch face data: {resp.status_code} - {resp.text}")
                return False

            payload = resp.json()
            raw_faces = payload.get("faces", [])
            version = new_version if new_version is not None else payload.get("version", self._current_version + 1)

            parsed_faces = []
            for item in raw_faces:
                w_id = item.get("worker_id", "")
                w_name = f"{item.get('first_name', '')} {item.get('last_name', '')}".strip()
                emb_data = item.get("face_embedding")
                
                if not emb_data:
                    continue

                if isinstance(emb_data, str):
                    try:
                        emb_arr = np.array(json.loads(emb_data), dtype=np.float64)
                    except Exception:
                        continue
                elif isinstance(emb_data, list):
                    emb_arr = np.array(emb_data, dtype=np.float64)
                else:
                    continue

                if emb_arr.ndim == 1 and len(emb_arr) == 128:
                    parsed_faces.append({
                        "worker_id": w_id,
                        "name": w_name,
                        "worker_db_id": item.get("id"),
                        "helmet_id": item.get("helmet_id", ""),
                        "status": item.get("status", "Off-Site"),
                        "embedding": emb_arr
                    })

            # Atomic swap under lock
            with self._lock:
                self._known_faces = parsed_faces
                self._current_version = version
                self._last_reloaded = time.time()
                self._last_checked = self._last_reloaded

            print(f"[FaceDataManager] Loaded {len(parsed_faces)} known face(s) (Version {self._current_version})")
            return True

        except Exception as e:
            print(f"[FaceDataManager] Reload error: {e}")
            return False

    def get_snapshot(self):
        """Returns a copy of the active face snapshot."""
        with self._lock:
            return list(self._known_faces)

    def match_face(self, live_embedding, threshold: float = 0.5):
        """
        Matches a live 128-dim embedding against the snapshot.
        Returns: (matched: bool, worker_dict: dict or None, distance: float)
        """
        snapshot = self.get_snapshot()
        if not snapshot or live_embedding is None:
            return False, None, 1.0

        live_arr = np.array(live_embedding, dtype=np.float64)
        if live_arr.ndim != 1 or len(live_arr) != 128:
            return False, None, 1.0

        best_match = None
        best_dist = 1.0

        for face in snapshot:
            dist = float(face_recognition.face_distance([face["embedding"]], live_arr)[0])
            if dist < best_dist:
                best_dist = dist
                if dist < threshold:
                    best_match = face

        if best_match is not None:
            return True, best_match, best_dist
        return False, None, best_dist
