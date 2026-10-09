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
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import cv2
import requests
import json
import torch
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

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    easyocr = None
    OCR_AVAILABLE = False

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")

# ── Parse Command Line Arguments ───────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--user-id", required=True, help="Manager User ID")
parser.add_argument("--camera-id", default="Pi Camera 01", help="Camera identifier")
parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference size (320 or 416)")
parser.add_argument("--headless", action="store_true", help="Run without OpenCV display window (for dashboard streaming)")
parser.add_argument("--pi-ip", default=None, help="Explicit Raspberry Pi IP address override")
args = parser.parse_args()
MANAGER_USER_ID = str(args.user_id)
CAMERA_ID = str(args.camera_id)
YOLO_IMGSZ = args.imgsz
HEADLESS = args.headless

print("=" * 65)
print(f"  SiteSentinel Parallel CV Engine | Manager: {MANAGER_USER_ID} | Camera: {CAMERA_ID} | imgsz: {YOLO_IMGSZ}")
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

print(f"[Init] Connecting to camera / Pi endpoint: {PI_IP}")

# Initialize Subsystems
stream = PiStream(PI_IP, fallback_to_local=True)
pi = PiController(PI_IP)
event_client = EventClient(BACKEND_URL, MANAGER_USER_ID)
face_manager = FaceDataManager(BACKEND_URL, MANAGER_USER_ID, poll_interval=8.0)
face_manager.start_hot_reload()

session_mgr = WorkerSessionManager(required_ppe_frames=4, grace_period_sec=2.0, max_verify_sec=4.0)

# ── Load YOLO PPE Model ────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model weights not found at: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"[Init] YOLO PPE Model loaded on device: {device.upper()} (Inference Size: {YOLO_IMGSZ}x{YOLO_IMGSZ})")

# ── Initialize OCR ────────────────────────────────────
ocr_reader = None
if easyocr is not None:
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        print("[Init] EasyOCR initialized for helmet ID verification")
    except Exception as e:
        print(f"[Init] EasyOCR init warning: {e}")

HELMET_ID_PATTERN = re.compile(r'[A-Za-z]{1,6}[_\-]?\d{1,6}', re.IGNORECASE)

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

_push_session = requests.Session()
_last_push_time = 0.0
_PUSH_INTERVAL = 0.05  # ~20 fps max push rate

def _push_frame_to_dashboard(frame_bgr):
    """JPEG-encode the annotated frame and POST it to the Flask frame buffer."""
    global _last_push_time
    now = time.time()
    if now - _last_push_time < _PUSH_INTERVAL:
        return
    _last_push_time = now
    try:
        ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            _push_session.post(
                f"{BACKEND_URL}/api/pi/frame/push",
                data=buf.tobytes(),
                headers={'Content-Type': 'application/octet-stream'},
                timeout=0.3
            )
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


def async_helmet_ocr_task(head_crop_bgr):
    """Worker task: Runs OCR on cropped helmet region."""
    if ocr_reader is None or head_crop_bgr is None or head_crop_bgr.size == 0:
        return ""
    try:
        gray = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        results = ocr_reader.readtext(gray, detail=0)
        for text in results:
            clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
            match = HELMET_ID_PATTERN.search(clean)
            if match:
                return match.group(0)
    except Exception as e:
        pass
    return ""


# ── Finalize Session Action ────────────────────────────
def commit_evaluation_session(worker_dict: dict, final_ppe_score: int, scanned_helmet_id: str, elapsed_time: float):
    """
    Submits finalized attendance and validation to backend and actuates Pi hardware asynchronously.
    """
    def _async_commit():
        worker_id = worker_dict.get("worker_id", "")
        worker_name = worker_dict.get("name", worker_id)
        assigned_helmet = worker_dict.get("helmet_id", "")

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

        print(f"\n[Session Finalized ({elapsed_time:.2f}s)] Worker: {worker_name} ({worker_id}) | Action: {action_type} | PPE: {final_ppe_score}% | Helmet: {scanned_helmet_id or assigned_helmet or 'None'}")

        # Validate helmet assignment
        if assigned_helmet and scanned_helmet_id and assigned_helmet.upper() != scanned_helmet_id.upper() and worker_id != "UNKNOWN":
            event_client.log_safety_event(
                event_type="HELMET_MISMATCH",
                message=f"Helmet mismatch for {worker_name}: Assigned {assigned_helmet}, scanned {scanned_helmet_id}",
                worker_id=worker_id,
                ppe_score=final_ppe_score,
                helmet_id=scanned_helmet_id,
                camera_id=CAMERA_ID
            )

        # Record attendance in database
        if worker_id != "UNKNOWN":
            event_client.record_attendance(
                worker_id=worker_id,
                ppe_score=final_ppe_score,
                helmet_id=scanned_helmet_id or assigned_helmet
            )

        # Hardware Feedback
        if worker_id != "UNKNOWN":
            if action_type == "Check-in":
                print(f"[Hardware] 🟢 Check-in verified for {worker_name}: Triggering GREEN LED (GPIO 17)")
                pi.checkin()
            else:
                print(f"[Hardware] 🔴 Check-out verified for {worker_name}: Triggering RED LED (GPIO 27)")
                pi.checkout()
            
            pi.show_score(final_ppe_score)
            
            # If PPE violation, also sound warning buzzer
            if final_ppe_score < 100:
                print(f"[Hardware] ⚠️ PPE Violation ({final_ppe_score}%): Actuating warning buzzer")
                def _warn_buzzer():
                    pi.buzzer_on()
                    time.sleep(1.2)
                    pi.buzzer_off()
                Thread(target=_warn_buzzer, daemon=True).start()
        else:
            # Unknown person: warn with buzzer
            print("[Hardware] ⚠️ Unknown person detected at gate: Actuating buzzer")
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

        # ── Physical Button Gating ─────────────────────────
        button_active = pi.is_button_pressed()
        
        # If in completed/confirming state, allow session feedback to display
        in_completion_state = session_mgr.state in [SessionState.CONFIRMING, SessionState.COMPLETED, SessionState.COOLDOWN]

        if not button_active and not in_completion_state:
            # Button is released -> Pause AI inference and remain in IDLE mode
            if session_mgr.state in [SessionState.VERIFYING, SessionState.LOCKED, SessionState.CROWDED]:
                print("[Session] Button released before verification complete -> Reset to IDLE")
                session_mgr.reset_to_idle()
            
            session_mgr.hud_message = "Gate Ready | Press & Hold Button to Verify"
            tracked_persons = []
            ppe_detections = []
        else:
            # Button is actively PRESSED or finalizing transaction -> Run full parallel AI pipeline
            # 1. Run YOLO PPE Detection & Person Detection with optimized imgsz
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
                else:
                    ppe_detections.append({
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf
                    })

            # 2. Update Person Tracking
            tracked_persons = tracker.update(person_rects)

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
                        session_mgr.record_ocr_reading(scanned_hid)
                        print(f"[Helmet OCR (Async)] Detected: {scanned_hid} ({session_mgr.elapsed_verify_time:.2f}s)")
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

                # C. Dispatch Helmet OCR asynchronously
                bh = py2 - py1
                if ocr_future is None and (time.time() - last_ocr_dispatch_time) > 0.4:
                    head_crop_bgr = frame[py1:min(frame_h, int(py1 + bh * 0.38)), px1:px2]
                    if head_crop_bgr.size > 0:
                        ocr_future = ocr_executor.submit(async_helmet_ocr_task, head_crop_bgr)
                        last_ocr_dispatch_time = time.time()

            # 6. State: COMPLETED (Trigger backend & GPIO asynchronously once)
            if session_mgr.state in [SessionState.CONFIRMING, SessionState.COMPLETED] and session_mgr.finalized_action is None:
                if session_mgr.active_worker is not None:
                    final_score = session_mgr.finalized_ppe_score if session_mgr.finalized_ppe_score is not None else 0
                    session_mgr.finalized_action = "DONE"
                    commit_evaluation_session(
                        worker_dict=session_mgr.active_worker,
                        final_ppe_score=final_score,
                        scanned_helmet_id=session_mgr.finalized_helmet_id,
                        elapsed_time=session_mgr.elapsed_verify_time
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
            SessionState.COOLDOWN: (200, 140, 255)
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
            w_info = f"Worker: {session_mgr.active_worker['name']} ({session_mgr.active_worker['worker_id']})"
            h_info = f"Helmet: {session_mgr.finalized_helmet_id or session_mgr.active_worker.get('helmet_id') or 'Scanning...'}"
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

        # Always push annotated frame to the dashboard MJPEG stream
        _push_frame_to_dashboard(annotated)

        # Also show OpenCV window for local monitoring (press q to quit)
        if not HEADLESS:
            cv2.imshow("SiteSentinel - Live Gate Inspection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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