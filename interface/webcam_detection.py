"""
SiteSentinel — High-Performance Parallel AI Computer Vision Gate Monitoring Engine
Features:
- Parallel Event-Driven Single-Worker Evaluation State Machine
- Concurrent Asynchronous Workers for Face Recognition & Helmet OCR
- Spatial PPE Anatomical Attribution (Head -> Hardhat, Torso -> Vest)
- Fast Adaptive 4-Frame Temporal PPE Consensus Voting
- Vectorized In-Memory Face Data Matching & Hot-Reloading
- Real-time Structured Safety Events & Non-blocking Attendance Logging
- Raspberry Pi Hardware Integration (LEDs, Buzzer, 8x8 Matrix)
- Rich Annotated Gate Monitoring HUD
"""
import sys
import os
import argparse
import re
import time
import threading
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import cv2
import requests
import json
import torch

# Fix PyTorch >= 2.6 default weights_only=True breaking Ultralytics YOLO & EasyOCR weights loading
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

import numpy as np

from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Add backend and interface to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from stream_reader import PiStream
from pi_controller import PiController
from face_data_manager import FaceDataManager
from event_client import EventClient
from worker_session import WorkerSessionManager, SessionState
from database import query_one, execute
from face__utils import get_embedding_from_frame
from ultralytics import YOLO
from helmet_color_classifier import classify_helmet_color, get_expected_color_for_occupation

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    easyocr = None
    OCR_AVAILABLE = False

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")
CLOUD_STREAM_URL = os.getenv("CLOUD_STREAM_URL", "https://sitesentinel.site/api/pi/frame/push").strip()
ENABLE_CLOUD_PUSH = os.getenv("ENABLE_CLOUD_PUSH", "true").lower() in ["true", "1", "yes"]

# ── Parse Command Line Arguments ───────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--user-id", required=True, help="Manager User ID")
parser.add_argument("--camera-id", default="Pi Camera 01", help="Camera identifier")
parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference size (320 or 416)")
parser.add_argument("--headless", action="store_true", help="Run without OpenCV display window (for dashboard streaming)")
parser.add_argument("--pi-ip", default=None, help="Explicit Raspberry Pi IP address override")
parser.add_argument("--backend-url", default=None, help="Explicit backend URL override (e.g. https://sitesentinel.site)")
args = parser.parse_args()
MANAGER_USER_ID = str(args.user_id)
CAMERA_ID = str(args.camera_id)
YOLO_IMGSZ = args.imgsz
HEADLESS = args.headless

if args.backend_url:
    BACKEND_URL = args.backend_url.rstrip("/")

print("=" * 65)
print(f"  SiteSentinel Parallel CV Engine | Manager: {MANAGER_USER_ID} | Camera: {CAMERA_ID} | Backend: {BACKEND_URL}")
print("=" * 65)

# ── Verify Raspberry Pi Configuration ───────────────────
if args.pi_ip:
    PI_IP = str(args.pi_ip).strip()
else:
    user_row = query_one(
        "SELECT pi_ip FROM users WHERE id = ?",
        [{"type": "text", "value": MANAGER_USER_ID}]
    )
    PI_IP = (user_row.get("pi_ip") if user_row else None) or "0"

if PI_IP and not ("." in PI_IP or PI_IP.isdigit() or PI_IP in ["0", "1", "local", "webcam"]):
    for candidate in ["192.168.1.25"]:
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            if s.connect_ex((candidate, 8080)) == 0:
                s.close()
                print(f"[Init] Auto-resolved device '{PI_IP}' -> Local Pi IP {candidate}")
                PI_IP = candidate
                break
            s.close()
        except Exception:
            pass

print(f"[Init] Connecting to camera / Pi endpoint: {PI_IP}")

# Initialize Subsystems
stream = PiStream(PI_IP, fallback_to_local=True)
pi = PiController(PI_IP)
event_client = EventClient(BACKEND_URL, MANAGER_USER_ID)
face_manager = FaceDataManager(BACKEND_URL, MANAGER_USER_ID, poll_interval=8.0)
face_manager.start_hot_reload()

session_mgr = WorkerSessionManager(
    required_ppe_frames=4,
    grace_period_sec=2.0,
    max_verify_sec=4.0,
    score_display_sec={100: 6.0, 50: 4.5, 0: 3.5}
)

# ── Load YOLO PPE Model ────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model weights not found at: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"[Init] YOLO PPE Model loaded on device: {device.upper()} (Inference Size: {YOLO_IMGSZ}x{YOLO_IMGSZ})")

# ── Initialize OCR & Barcode / QR Readers ────────────────
ocr_reader = None
if easyocr is not None:
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        print("[Init] EasyOCR initialized for helmet ID verification")
    except Exception as e:
        print(f"[Init] EasyOCR init warning: {e}")

qr_detector = cv2.QRCodeDetector()
try:
    barcode_detector = cv2.barcode.BarcodeDetector()
except Exception:
    barcode_detector = None

HELMET_ID_PATTERN = re.compile(r'\b[A-Za-z]{1,6}[_\-\s]?\d{1,6}\b|\b\d{2,6}\b', re.IGNORECASE)

# ── Thread Pools for Concurrent Verification Workers ──
face_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="FaceWorker")
ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="OCRWorker")
face_future = None
ocr_future = None

# ── State Tracking & Metrics ───────────────────────────
last_buzzer_state = False
fps_counter = 0
fps_start_time = time.time()
current_fps = 0.0
last_ocr_dispatch_time = 0.0
last_face_dispatch_time = 0.0
last_button_active_time = 0.0

_push_session = requests.Session()
_last_push_time = 0.0
_PUSH_INTERVAL = 0.045  # ~22 fps local push rate

# ── Cloud Stream Background Mirror (Non-Blocking) ─────
_cloud_push_lock = threading.Lock()
_latest_cloud_buf = None
_cloud_push_session = requests.Session()

def _cloud_pusher_worker():
    """Asynchronously forwards annotated HUD frames to the hosted dashboard without stalling local CV."""
    global _latest_cloud_buf
    target_url = CLOUD_STREAM_URL
    direct_url = "https://sitesentinel-or0h.onrender.com/api/pi/frame/push" if "sitesentinel.site" in target_url else target_url
    print(f"[Cloud Stream] Mirroring annotated AI HUD frames -> {direct_url}")
    while True:
        buf_to_send = None
        with _cloud_push_lock:
            if _latest_cloud_buf is not None:
                buf_to_send = _latest_cloud_buf
                _latest_cloud_buf = None
        if buf_to_send is not None and direct_url:
            try:
                _cloud_push_session.post(
                    direct_url,
                    data=buf_to_send,
                    headers={
                        'Content-Type': 'application/octet-stream',
                        'X-Frame-Source': 'cv'
                    },
                    timeout=3.5
                )
            except Exception:
                if direct_url != target_url:
                    try:
                        _cloud_push_session.post(
                            target_url,
                            data=buf_to_send,
                            headers={
                                'Content-Type': 'application/octet-stream',
                                'X-Frame-Source': 'cv'
                            },
                            timeout=3.5
                        )
                    except Exception:
                        pass
        time.sleep(0.05)  # optimal upload pacing with zero local CV stall

if ENABLE_CLOUD_PUSH and CLOUD_STREAM_URL:
    threading.Thread(target=_cloud_pusher_worker, daemon=True).start()

def _push_frame_to_dashboard(frame_bgr):
    """JPEG-encode the annotated frame and POST it to local and cloud frame buffers."""
    global _last_push_time, _latest_cloud_buf
    now = time.time()
    if now - _last_push_time < _PUSH_INTERVAL:
        return
    _last_push_time = now
    try:
        ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            raw_data = buf.tobytes()
            # 1. Local backend push (immediate, non-blocking)
            if BACKEND_URL:
                try:
                    _push_session.post(
                        f"{BACKEND_URL}/api/pi/frame/push",
                        data=raw_data,
                        headers={
                            'Content-Type': 'application/octet-stream',
                            'X-Frame-Source': 'cv'
                        },
                        timeout=0.25
                    )
                except Exception:
                    pass
            # 2. Queue latest frame for background cloud push
            if ENABLE_CLOUD_PUSH and CLOUD_STREAM_URL and (not BACKEND_URL or BACKEND_URL not in CLOUD_STREAM_URL):
                with _cloud_push_lock:
                    _latest_cloud_buf = raw_data
    except Exception:
        pass

# ── Centroid Tracker for Worker Lock ───────────────────
class SimpleCentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 1
        self.objects = {}       # object_id -> centroid (cx, cy)
        self.bboxes = {}        # object_id -> [x1, y1, x2, y2]
        self.disappeared = {}   # object_id -> count
        self.max_disappeared = max_disappeared

    def update(self, rects):
        """rects: list of [x1, y1, x2, y2]"""
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return []

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        if len(self.objects) == 0:
            results = []
            for i in range(len(rects)):
                obj_id = self._register(input_centroids[i], rects[i])
                results.append({"track_id": obj_id, "bbox": rects[i], "centroid": input_centroids[i]})
            return results

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Distance matrix
        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        results = []
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > 150:  # Max distance threshold
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = rects[col]
            self.disappeared[obj_id] = 0

            used_rows.add(row)
            used_cols.add(col)
            results.append({"track_id": obj_id, "bbox": rects[col], "centroid": input_centroids[col]})

        unused_cols = set(range(len(input_centroids))).difference(used_cols)
        for col in unused_cols:
            obj_id = self._register(input_centroids[col], rects[col])
            results.append({"track_id": obj_id, "bbox": rects[col], "centroid": input_centroids[col]})

        unused_rows = set(range(len(object_centroids))).difference(used_rows)
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        return results

    def _register(self, centroid, bbox):
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.next_object_id += 1
        return obj_id

    def _deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
            del self.bboxes[object_id]
            del self.disappeared[object_id]

tracker = SimpleCentroidTracker(max_disappeared=20)


# ── Asynchronous Background Tasks ──────────────────────
def async_face_recognition_task(person_crop_rgb):
    """Worker task: Extracts 128-d face embedding and matches against in-memory cache."""
    try:
        if person_crop_rgb is None or person_crop_rgb.size == 0:
            return None
        embeddings = get_embedding_from_frame(person_crop_rgb)
        if embeddings:
            for emb in embeddings:
                matched, matched_worker, dist = face_manager.match_face(emb, threshold=0.52)
                if matched and matched_worker:
                    return matched_worker
    except Exception as e:
        print(f"[AsyncFaceWorker] Error: {e}")
    return None


def async_helmet_ocr_task(head_crop_bgr, known_helmets=None):
    """Worker task: Runs QR code, barcode, and enhanced multi-pass OCR on cropped helmet region."""
    if head_crop_bgr is None or head_crop_bgr.size == 0:
        return ""

    # 1. High-speed QR Code detection (<2ms)
    try:
        val, _, _ = qr_detector.detectAndDecode(head_crop_bgr)
        if val:
            clean = re.sub(r'[^A-Za-z0-9_\-]', '', val).strip().upper()
            if clean:
                print(f"[Helmet Scanner] [QR] QR Code detected: {clean}")
                return clean
    except Exception:
        pass

    # 2. Barcode detection
    if barcode_detector is not None:
        try:
            ok, decoded_info, _, _ = barcode_detector.detectAndDecode(head_crop_bgr)
            if ok and decoded_info:
                val = decoded_info[0] if isinstance(decoded_info, (list, tuple)) else decoded_info
                clean = re.sub(r'[^A-Za-z0-9_\-]', '', str(val)).strip().upper()
                if clean:
                    print(f"[Helmet Scanner] [Barcode] Barcode detected: {clean}")
                    return clean
        except Exception:
            pass

    # 3. EasyOCR Text Recognition
    if ocr_reader is None:
        return ""

    try:
        # Pass 1: Direct color inference (CRAFT works best on BGR/RGB)
        results = ocr_reader.readtext(head_crop_bgr, detail=0)

        # Pass 2: Grayscale with CLAHE contrast enhancement & scaling
        if not results:
            gray = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced = cv2.resize(enhanced, (0, 0), fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            results = ocr_reader.readtext(enhanced, detail=0)

        # Pass 3: Otsu thresholding
        if not results:
            gray = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            results = ocr_reader.readtext(thresh, detail=0)

        for text in results:
            clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
            if not clean:
                continue

            # Prioritize matching against known database helmet IDs (e.g. H1, H2, H3, H4)
            if known_helmets:
                for kh in known_helmets:
                    kh_clean = re.sub(r'[^A-Za-z0-9]', '', str(kh)).upper()
                    if kh_clean and (clean == kh_clean or kh_clean in clean):
                        print(f"[Helmet Scanner] [OCR Match] Matched Known Helmet: {kh.upper()}")
                        return kh.upper()

            # Pattern match
            match = HELMET_ID_PATTERN.search(clean)
            if match:
                matched_id = match.group(0).upper()
                print(f"[Helmet Scanner] [OCR Pattern] Detected: {matched_id}")
                return matched_id
    except Exception as e:
        pass
    return ""


# ── Finalize Session Action ────────────────────────────
def commit_evaluation_session(
    worker_dict: dict,
    final_ppe_score: int,
    scanned_helmet_id: str,
    elapsed_time: float,
    is_mismatch: bool = False,
    has_hardhat: bool = False,
    is_color_mismatch: bool = False,
    detected_color: str = "",
    expected_color: str = ""
):
    """
    Submits finalized attendance and validation to backend and actuates Pi hardware asynchronously.
    """
    def _async_commit():
        worker_id = worker_dict.get("worker_id", "")
        worker_name = worker_dict.get("name", worker_id)
        assigned_helmet = worker_dict.get("helmet_id", "")
        worker_occ = worker_dict.get("occupation", "Worker")

        # Fetch latest status from database for reliable checkin/checkout transition
        w_row = query_one(
            "SELECT status FROM workers WHERE worker_id = ? AND user_id = ?",
            [{"type": "text", "value": worker_id}, {"type": "text", "value": MANAGER_USER_ID}]
        )
        current_status = (w_row.get("status") if w_row else None) or worker_dict.get("status", "Off-Site")

        # Determine attendance transition:
        # Off-Site -> Check-in (Green LED)
        # Active/On-Site -> Check-out (Red LED)
        new_status = "Off-Site" if current_status in ["Active", "On-Site"] else "Active"
        action_type = "Check-in" if new_status == "Active" else "Check-out"

        # A helmet is ONLY registered/occupied if a hardhat was detected or a helmet code was scanned!
        effective_helmet = (scanned_helmet_id or session_mgr.finalized_helmet_id or session_mgr.scanned_helmet_id or assigned_helmet or "").strip().upper()
        registered_helmet = ""
        if action_type == "Check-in" and (has_hardhat or effective_helmet):
            registered_helmet = effective_helmet

        color_info = f" | Color: {detected_color}" if detected_color else ""
        print(f"\n[Session Finalized ({elapsed_time:.2f}s)] Worker: {worker_name} ({worker_occ}) | Action: {action_type} | PPE: {final_ppe_score}%{color_info} | Helmet: {registered_helmet or 'None (Not detected)'}")

        # Check for helmet color mismatch vs occupation role
        if action_type == "Check-in" and is_color_mismatch and worker_id != "UNKNOWN":
            print(f"[Validation] [HELMET COLOR MISMATCH] {worker_name} ({worker_occ}) wore {detected_color} helmet, expected {expected_color}!")
            event_client.log_safety_event(
                event_type="HELMET_COLOR_MISMATCH",
                message=f"Occupation/Helmet Mismatch for {worker_name} ({worker_occ}): Wore {detected_color} helmet, expected {expected_color}",
                worker_id=worker_id,
                ppe_score=final_ppe_score,
                helmet_id=registered_helmet or scanned_helmet_id,
                camera_id=CAMERA_ID
            )
            # Actuate buzzer and abort check-in on color mismatch
            def _color_mismatch_alert():
                pi.buzzer_on()
                time.sleep(1.8)
                pi.buzzer_off()
            Thread(target=_color_mismatch_alert, daemon=True).start()
            return

        # Check for helmet ID mismatch (only strictly enforced on Check-in entry when helmet is worn)
        has_mismatch = (action_type == "Check-in") and (is_mismatch or (
            bool(assigned_helmet) and bool(scanned_helmet_id) and 
            assigned_helmet.upper() != scanned_helmet_id.upper() and 
            worker_id != "UNKNOWN"
        ))

        if has_mismatch:
            print(f"[Validation] [HELMET MISMATCH] {worker_name} assigned {assigned_helmet}, but scanned {scanned_helmet_id}!")
            event_client.log_safety_event(
                event_type="HELMET_MISMATCH",
                message=f"Helmet mismatch for {worker_name}: Assigned {assigned_helmet}, scanned {scanned_helmet_id}",
                worker_id=worker_id,
                ppe_score=final_ppe_score,
                helmet_id=scanned_helmet_id,
                camera_id=CAMERA_ID
            )
            # Actuate buzzer and abort check-in on mismatch
            def _mismatch_alert():
                pi.buzzer_on()
                time.sleep(1.8)
                pi.buzzer_off()
            Thread(target=_mismatch_alert, daemon=True).start()
            return

        # Record attendance in database when valid
        if worker_id != "UNKNOWN":
            event_client.record_attendance(
                worker_id=worker_id,
                ppe_score=final_ppe_score,
                helmet_id=registered_helmet,
                status=new_status
            )

        # Hardware Feedback
        if worker_id != "UNKNOWN":
            if action_type == "Check-in":
                print(f"[Hardware] [Check-in] Verified for {worker_name} (Helmet: {registered_helmet or 'None'}): GREEN LED (GPIO 17)")
                pi.checkin()
            else:
                print(f"[Hardware] [Check-out] Verified for {worker_name}: RED LED (GPIO 27)")
                pi.checkout()
            
            pi.show_score(final_ppe_score)
            
            # If PPE violation, also sound warning buzzer
            if final_ppe_score < 100:
                print(f"[Hardware] [PPE Violation] ({final_ppe_score}%): Actuating warning buzzer")
                def _warn_buzzer():
                    pi.buzzer_on()
                    time.sleep(1.2)
                    pi.buzzer_off()
                Thread(target=_warn_buzzer, daemon=True).start()
        else:
            # Unknown person: warn with buzzer
            print("[Hardware] [Unknown Worker] Gate buzzer triggered")
            pi.show_score(final_ppe_score)
            def _warn_buzzer():
                pi.buzzer_on()
                time.sleep(1.2)
                pi.buzzer_off()
            Thread(target=_warn_buzzer, daemon=True).start()

        # Refresh worker on-site statuses
        face_manager.check_and_reload_if_stale()

    Thread(target=_async_commit, daemon=True).start()


# ── Main Video Processing Loop ──────────────────────────
print("\n[SiteSentinel] Starting live camera parallel processing loop. Press 'q' to exit.\n")

if not HEADLESS:
    try:
        cv2.namedWindow("SiteSentinel - Live Gate Inspection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SiteSentinel - Live Gate Inspection", 960, 540)
    except Exception as e:
        print(f"[OpenCV Window Warning]: {e}")

try:
    while True:
        loop_start = time.time()
        success, frame = stream.read()

        if not success or frame is None:
            time.sleep(0.02)
            continue

        frame_h, frame_w = frame.shape[:2]
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        # 1. Run YOLO PPE Detection & Person Detection continuously (camera stream never stops)
        results = model(frame, imgsz=YOLO_IMGSZ, conf=0.45, verbose=False)[0]
        
        person_rects = []
        ppe_detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            if label == "Person":
                person_rects.append([x1, y1, x2, y2])
            elif "mask" in label.lower():
                # Ignore Mask and NO-Mask detections (do not draw or track)
                continue
            else:
                ppe_detections.append({
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf
                })

        # 2. Update Person Tracking
        tracked_persons = tracker.update(person_rects)

        # ── Physical Button State Trigger ──────────────────
        button_active = pi.is_button_pressed()
        if button_active:
            last_button_active_time = time.time()
        
        # Clicking the button activates verification state window for 10.0 seconds
        is_button_triggered = (time.time() - last_button_active_time < 10.0)
        in_eval_process = session_mgr.state in [
            SessionState.VERIFYING,
            SessionState.CONFIRMING,
            SessionState.COMPLETED,
            SessionState.COOLDOWN,
            SessionState.CROWDED,
            SessionState.LOCKED
        ]

        # Evaluate if button was pressed, or if in progress, or auto-verify when exactly 1 worker is at the gate
        auto_eval = os.getenv("ENABLE_AUTO_EVAL", "true").lower() in ["true", "1", "yes"]
        can_evaluate = is_button_triggered or in_eval_process or (auto_eval and len(tracked_persons) == 1)

        if not can_evaluate:
            session_mgr.state = SessionState.IDLE
            session_mgr.hud_message = "Gate Ready | Press Button to Verify"
        else:
            # 3. Update Single-Worker Evaluation State Machine
            session_mgr.update_frame(tracked_persons, frame_w, frame_h)

            # 4. Check Background Worker Results (Non-Blocking)
            if face_future is not None and face_future.done():
                try:
                    matched_worker = face_future.result()
                    if matched_worker:
                        session_mgr.set_identified_worker(matched_worker)
                        print(f"[Face Matched (Async)] {matched_worker['name']} ({session_mgr.elapsed_verify_time:.2f}s)")
                except Exception as e:
                    print(f"[AsyncFace] Error reading result: {e}")
                face_future = None

            if ocr_future is not None and ocr_future.done():
                try:
                    scanned_hid = ocr_future.result()
                    if scanned_hid:
                        if session_mgr.active_worker is None:
                            w_by_hid = face_manager.get_worker_by_helmet_id(scanned_hid)
                            if w_by_hid:
                                print(f"[Helmet Pre-Match] Detected Helmet {scanned_hid} belongs to {w_by_hid['name']}")
                        session_mgr.record_ocr_reading(scanned_hid)
                        print(f"[Helmet OCR/QR (Async)] Detected: {scanned_hid} ({session_mgr.elapsed_verify_time:.2f}s)")
                except Exception as e:
                    print(f"[AsyncOCR] Error reading result: {e}")
                ocr_future = None

            # 5. Dispatch Parallel Tasks When in VERIFYING State
            if session_mgr.state == SessionState.VERIFYING and session_mgr.active_bbox is not None:
                # A. Calculate Spatial PPE Attribution for current frame
                ppe_obs = session_mgr.attribute_ppe_detections(ppe_detections)

                bx1, by1, bx2, by2 = session_mgr.active_bbox
                px1, py1 = max(0, bx1), max(0, by1)
                px2, py2 = min(frame_w, bx2), min(frame_h, by2)

                # B. Dispatch Face Recognition asynchronously if identity not resolved yet
                if session_mgr.active_worker is None and face_future is None and (time.time() - last_face_dispatch_time) > 0.25:
                    person_crop_bgr = frame[py1:py2, px1:px2]
                    if person_crop_bgr.size > 0:
                        person_crop_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
                        face_future = face_executor.submit(async_face_recognition_task, person_crop_rgb)
                        last_face_dispatch_time = time.time()

                # C. Dispatch Helmet OCR/QR asynchronously
                bh = py2 - py1
                if ocr_future is None and (time.time() - last_ocr_dispatch_time) > 0.25:
                    helmet_crop_bgr = None
                    matched_boxes = ppe_obs.get("matched_boxes", []) if ppe_obs else []
                    hardhat_boxes = [b for b in matched_boxes if b.get("label") == "Hardhat"]
                    if not hardhat_boxes:
                        hardhat_boxes = [b for b in ppe_detections if b.get("label") == "Hardhat" and py1 <= (b["bbox"][1] + b["bbox"][3]) / 2 <= py1 + bh * 0.45]

                    if hardhat_boxes:
                        best_h = max(hardhat_boxes, key=lambda b: b.get("conf", 0))
                        hx1, hy1, hx2, hy2 = best_h["bbox"]
                        hw_pad = int((hx2 - hx1) * 0.15)
                        hh_pad = int((hy2 - hy1) * 0.15)
                        hx1 = max(0, hx1 - hw_pad)
                        hy1 = max(0, hy1 - hh_pad)
                        hx2 = min(frame_w, hx2 + hw_pad)
                        hy2 = min(frame_h, hy2 + hh_pad)
                        if (hy2 - hy1) > 10 and (hx2 - hx1) > 10:
                            helmet_crop_bgr = frame[hy1:hy2, hx1:hx2]

                    if helmet_crop_bgr is None or helmet_crop_bgr.size == 0:
                        helmet_crop_bgr = frame[py1:min(frame_h, int(py1 + bh * 0.40)), px1:px2]

                    if helmet_crop_bgr is not None and helmet_crop_bgr.size > 0:
                        # Classify helmet color in real-time
                        color_det, color_conf = classify_helmet_color(helmet_crop_bgr)
                        if color_det != "Unknown":
                            session_mgr.record_helmet_color(color_det)

                        known_hids = [w.get("helmet_id") for w in face_manager.get_snapshot() if w.get("helmet_id")]
                        ocr_future = ocr_executor.submit(async_helmet_ocr_task, helmet_crop_bgr, known_hids)
                        last_ocr_dispatch_time = time.time()

            # 6. State: COMPLETED / COOLDOWN (Trigger backend & GPIO asynchronously once)
            if session_mgr.state in [SessionState.CONFIRMING, SessionState.COMPLETED, SessionState.COOLDOWN] and session_mgr.finalized_action is None:
                if session_mgr.active_worker is not None:
                    final_score = session_mgr.finalized_ppe_score if session_mgr.finalized_ppe_score is not None else 0
                    session_mgr.finalized_action = "DONE"
                    commit_evaluation_session(
                        worker_dict=session_mgr.active_worker,
                        final_ppe_score=final_score,
                        scanned_helmet_id=session_mgr.finalized_helmet_id or session_mgr.scanned_helmet_id,
                        elapsed_time=session_mgr.elapsed_verify_time,
                        is_mismatch=session_mgr.helmet_mismatch,
                        has_hardhat=session_mgr.has_detected_hardhat(),
                        is_color_mismatch=session_mgr.helmet_color_mismatch,
                        detected_color=session_mgr.detected_helmet_color,
                        expected_color=session_mgr.expected_helmet_color
                    )

            # 7. Real-Time Violation Buzzer
            current_score = session_mgr.get_smoothed_ppe_score()
            has_active_violation = (session_mgr.state == SessionState.VERIFYING and current_score is not None and current_score < 100)
            if has_active_violation != last_buzzer_state:
                if has_active_violation:
                    Thread(target=pi.buzzer_on, daemon=True).start()
                else:
                    Thread(target=pi.buzzer_off, daemon=True).start()
                last_buzzer_state = has_active_violation

        # ───────────────────────────────────────────────────
        # 8. Render Rich HUD Annotations
        # ───────────────────────────────────────────────────
        annotated = frame.copy()
        gx1, gy1, gx2, gy2 = session_mgr.get_gate_roi(frame_w, frame_h)

        # Draw Gate Zone ROI
        roi_overlay = annotated.copy()
        cv2.rectangle(roi_overlay, (gx1, 0), (gx2, frame_h), (255, 107, 43), -1)
        cv2.addWeighted(roi_overlay, 0.08, annotated, 0.92, 0, annotated)
        cv2.rectangle(annotated, (gx1, 0), (gx2, frame_h), (255, 107, 43), 1, cv2.LINE_AA)
        cv2.putText(annotated, "GATE ZONE", (gx1 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 107, 43), 1, cv2.LINE_AA)

        # Draw Person Bounding Boxes
        for p in tracked_persons:
            tid = p["track_id"]
            bx1, by1, bx2, by2 = p["bbox"]
            is_active = (tid == session_mgr.active_track_id)
            box_color = (0, 212, 170) if is_active else (180, 180, 180)
            thickness = 2 if is_active else 1
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, thickness, cv2.LINE_AA)

            tag = f"Track #{tid}" + (" [LOCKED]" if is_active else "")
            cv2.putText(annotated, tag, (bx1, max(15, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

        # Draw PPE Detection Boxes
        for det in ppe_detections:
            lbl = det["label"]
            if "mask" in lbl.lower():
                continue
            dx1, dy1, dx2, dy2 = det["bbox"]
            is_safe = lbl in ["Hardhat", "Safety Vest"]
            d_color = (0, 220, 100) if is_safe else (50, 50, 240)
            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), d_color, 1, cv2.LINE_AA)
            cv2.putText(annotated, lbl, (dx1, max(12, dy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, d_color, 1, cv2.LINE_AA)

        # Top Header Bar Overlay
        cv2.rectangle(annotated, (0, 0), (frame_w, 42), (12, 12, 20), -1)
        cv2.line(annotated, (0, 42), (frame_w, 42), (50, 50, 70), 1)

        cv2.putText(annotated, "SiteSentinel AI", (12, 27), cv2.FONT_HERSHEY_DUPLEX, 0.56, (255, 107, 43), 1, cv2.LINE_AA)
        
        status_text = f"FPS: {current_fps:.1f} | Pi: {PI_IP}"
        cv2.putText(annotated, status_text, (190, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)

        # State Badge (Top Right)
        state_colors = {
            SessionState.IDLE: (140, 140, 140),
            SessionState.CROWDED: (50, 50, 240),
            SessionState.LOCKED: (0, 200, 255),
            SessionState.VERIFYING: (255, 170, 0),
            SessionState.CONFIRMING: (0, 220, 100),
            SessionState.COMPLETED: (0, 220, 100),
            SessionState.COOLDOWN: (0, 212, 170)
        }
        st_color = state_colors.get(session_mgr.state, (200, 200, 200))
        st_label = f"[{session_mgr.state}]"
        cv2.putText(annotated, st_label, (frame_w - 140, 27), cv2.FONT_HERSHEY_DUPLEX, 0.50, st_color, 1, cv2.LINE_AA)

        # Bottom Information Card Overlay
        cv2.rectangle(annotated, (0, frame_h - 70), (frame_w, frame_h), (12, 12, 20), -1)
        cv2.line(annotated, (0, frame_h - 70), (frame_w, frame_h - 70), (50, 50, 70), 1)

        # HUD Message
        clean_msg = session_mgr.hud_message.replace("—", "|").replace("•", "|")
        cv2.putText(annotated, clean_msg, (14, frame_h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Worker Details (Bottom Left)
        if session_mgr.active_worker:
            w_occ = session_mgr.active_worker.get('occupation', 'Worker')
            w_info = f"Worker: {session_mgr.active_worker['name']} ({w_occ})"
            h_color_tag = f"[{session_mgr.detected_helmet_color}]" if session_mgr.detected_helmet_color else ""
            h_info = f"Helmet: {session_mgr.finalized_helmet_id or session_mgr.active_worker.get('helmet_id') or 'ID: Scanning'} {h_color_tag}"
            cv2.putText(annotated, f"{w_info}  |  {h_info}", (14, frame_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 212, 170), 1, cv2.LINE_AA)

        # PPE Score Gauge (Bottom Right)
        score_val = session_mgr.get_smoothed_ppe_score()
        if score_val is not None:
            score_color = (0, 220, 100) if score_val >= 100 else ((0, 200, 255) if score_val >= 50 else (50, 50, 240))
            cv2.putText(annotated, f"PPE: {score_val}%", (frame_w - 170, frame_h - 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, score_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "PPE: --", (frame_w - 170, frame_h - 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (140, 140, 140), 1, cv2.LINE_AA)

        # Crowding warning overlay banner
        if session_mgr.crowding_warning:
            cv2.rectangle(annotated, (gx1, 50), (gx2, 90), (0, 0, 180), -1)
            cv2.putText(annotated, "! CROWDING: ONE WORKER AT A TIME !", (gx1 + 15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Helmet Color Mismatch warning banner
        if session_mgr.helmet_color_mismatch:
            cv2.rectangle(annotated, (gx1, 95), (gx2, 135), (0, 0, 180), -1)
            cv2.putText(annotated, f"! COLOR MISMATCH: {session_mgr.detected_helmet_color} != Exp {session_mgr.expected_helmet_color} !", (gx1 + 10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # Always push annotated frame to the dashboard MJPEG stream
        _push_frame_to_dashboard(annotated)

        # Also show OpenCV window for local monitoring (press q to quit)
        if not HEADLESS:
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in [ord('v'), ord('b')]:
                    last_button_active_time = time.time()
                    print("[Keyboard Trigger] -> Key 'v'/'b' pressed: Activating verification state")
            except Exception:
                pass

except KeyboardInterrupt:
    print("\n[SiteSentinel] Interrupted by user.")
finally:
    print("[SiteSentinel] Shutting down parallel CV engine...")
    try:
        pi.buzzer_off()
    except:
        pass
    face_executor.shutdown(wait=False)
    ocr_executor.shutdown(wait=False)
    face_manager.stop()
    stream.close()
    cv2.destroyAllWindows()
    print("[SiteSentinel] Shutdown clean and complete.")