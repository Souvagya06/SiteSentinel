"""
SiteSentinel — Advanced AI Computer Vision Gate Monitoring Engine
Features:
- Single-worker evaluation state machine & Gate Occupancy Lock
- Spatial PPE anatomical attribution (Head -> Hardhat, Torso -> Vest)
- 8-frame temporal smoothing & locked PPE evaluation (0, 50, 100)
- Face data hot-reloading in background without restarts
- Real-time structured safety events & attendance logging
- Raspberry Pi GPIO hardware integration (LEDs, Buzzer, Matrix)
- Rich annotated monitoring HUD
"""
import sys
import os
import argparse
import re
import time
from threading import Thread
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
args = parser.parse_args()
MANAGER_USER_ID = str(args.user_id)
CAMERA_ID = str(args.camera_id)

print("=" * 65)
print(f"  SiteSentinel CV Engine | Manager ID: {MANAGER_USER_ID} | Camera: {CAMERA_ID}")
print("=" * 65)

# ── Verify Raspberry Pi Configuration ───────────────────
user_row = query_one(
    "SELECT pi_ip FROM users WHERE id = ?",
    [{"type": "text", "value": MANAGER_USER_ID}]
)

if not user_row or not user_row.get("pi_ip"):
    raise RuntimeError(f"No Raspberry Pi IP configured for manager {MANAGER_USER_ID}")

PI_IP = user_row["pi_ip"]
print(f"[Init] Connecting to Raspberry Pi at: {PI_IP}")

# Initialize Subsystems
stream = PiStream(PI_IP)
pi = PiController(PI_IP)
event_client = EventClient(BACKEND_URL, MANAGER_USER_ID)
face_manager = FaceDataManager(BACKEND_URL, MANAGER_USER_ID, poll_interval=8.0)
face_manager.start_hot_reload()

session_mgr = WorkerSessionManager(required_ppe_frames=8, grace_period_sec=2.0)

# ── Load YOLO PPE Model ────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model weights not found at: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"[Init] YOLO PPE Model loaded on device: {device.upper()}")

# ── Initialize OCR ────────────────────────────────────
ocr_reader = None
if easyocr is not None:
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        print("[Init] EasyOCR initialized for helmet ID verification")
    except Exception as e:
        print(f"[Init] EasyOCR init warning: {e}")

HELMET_ID_PATTERN = re.compile(r'[A-Za-z]{1,6}[_\-]?\d{1,6}', re.IGNORECASE)

# ── State Tracking & Metrics ───────────────────────────
last_buzzer_state = False
fps_counter = 0
fps_start_time = time.time()
current_fps = 0.0
last_ocr_time = 0
ocr_cached_helmet = ""

# Track simple centroids across frames for person track IDs
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


def extract_helmet_id_from_crop(head_crop_bgr):
    """Runs OCR on the head crop to detect stenciled/printed helmet IDs."""
    if ocr_reader is None or head_crop_bgr is None or head_crop_bgr.size == 0:
        return ""
    try:
        # Preprocess for contrast
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
def commit_evaluation_session(worker_dict: dict, final_ppe_score: int, scanned_helmet_id: str):
    """
    Submits finalized attendance and validation to Flask backend and actuates Pi hardware.
    """
    worker_id = worker_dict.get("worker_id", "")
    worker_name = worker_dict.get("name", worker_id)
    assigned_helmet = worker_dict.get("helmet_id", "")
    current_status = worker_dict.get("status", "Off-Site")

    print(f"\n[Session Finalized] Worker: {worker_name} ({worker_id}) | PPE: {final_ppe_score}% | Assigned Helmet: {assigned_helmet or 'None'} | Scanned: {scanned_helmet_id or 'None'}")

    # Determine attendance transition
    new_status = "Off-Site" if current_status in ["Active", "On-Site"] else "Active"
    action_type = "Check-in" if new_status == "Active" else "Check-out"

    # Validate helmet assignment
    if assigned_helmet and scanned_helmet_id and assigned_helmet.upper() != scanned_helmet_id.upper():
        event_client.log_safety_event(
            event_type="HELMET_MISMATCH",
            message=f"Helmet mismatch for {worker_name}: Assigned {assigned_helmet}, scanned {scanned_helmet_id}",
            worker_id=worker_id,
            ppe_score=final_ppe_score,
            helmet_id=scanned_helmet_id,
            camera_id=CAMERA_ID
        )

    # Record attendance in database
    resp = event_client.record_attendance(
        worker_id=worker_id,
        ppe_score=final_ppe_score,
        helmet_id=scanned_helmet_id or assigned_helmet
    )

    # Hardware Feedback
    if final_ppe_score >= 100:
        if action_type == "Check-in":
            pi.checkin()
        else:
            pi.checkout()
        pi.show_score(final_ppe_score)
    else:
        # PPE Violation feedback
        pi.show_score(final_ppe_score)
        # Pulse buzzer
        def _warn_buzzer():
            pi.buzzer_on()
            time.sleep(1.2)
            pi.buzzer_off()
        Thread(target=_warn_buzzer, daemon=True).start()

    # Trigger hot-reload check to refresh worker on-site statuses
    Thread(target=face_manager.check_and_reload_if_stale, daemon=True).start()


# ── Main Video Processing Loop ──────────────────────────
print("\n[SiteSentinel] Starting live camera processing loop. Press 'q' to exit.\n")

try:
    while True:
        loop_start = time.time()
        success, frame = stream.read()

        if not success or frame is None:
            # Reconnecting or waiting for stream
            time.sleep(0.05)
            continue

        frame_h, frame_w = frame.shape[:2]
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        # 1. Run YOLO PPE Detection & Person Detection
        results = model(frame, conf=0.35, verbose=False)[0]
        
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

        # 4. State-Specific Operations
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── State: IDENTIFYING ──
        if session_mgr.state == SessionState.IDENTIFYING and session_mgr.active_worker is None:
            # Crop locked worker area to run face recognition
            if session_mgr.active_bbox is not None:
                bx1, by1, bx2, by2 = session_mgr.active_bbox
                # Add margin
                px1 = max(0, bx1)
                py1 = max(0, by1)
                px2 = min(frame_w, bx2)
                py2 = min(frame_h, by2)
                
                person_crop_rgb = frame_rgb[py1:py2, px1:px2]
                if person_crop_rgb.size > 0:
                    live_embeddings = get_embedding_from_frame(person_crop_rgb)
                    if live_embeddings:
                        for emb in live_embeddings:
                            matched, matched_worker, dist = face_manager.match_face(emb, threshold=0.52)
                            if matched and matched_worker:
                                session_mgr.set_identified_worker(matched_worker)
                                print(f"[Face Matched] {matched_worker['name']} (Dist: {dist:.3f})")
                                break

        # ── State: EVALUATING_PPE ──
        if session_mgr.state == SessionState.EVALUATING_PPE:
            # Calculate strict spatial PPE attribution
            ppe_obs = session_mgr.attribute_ppe_detections(ppe_detections)

            # Try helmet OCR once during evaluation if head crop available
            if (time.time() - last_ocr_time) > 2.0 and session_mgr.active_bbox:
                bx1, by1, bx2, by2 = session_mgr.active_bbox
                bh = by2 - by1
                head_crop = frame[max(0, by1):min(frame_h, int(by1 + bh * 0.35)), max(0, bx1):min(frame_w, bx2)]
                if head_crop.size > 0:
                    detected_hid = extract_helmet_id_from_crop(head_crop)
                    if detected_hid:
                        ocr_cached_helmet = detected_hid
                        session_mgr.finalized_helmet_id = detected_hid
                last_ocr_time = time.time()

        # ── State: FINALIZED (Trigger backend & GPIO once) ──
        if session_mgr.state == SessionState.FINALIZED and session_mgr.finalized_action is None:
            if session_mgr.active_worker is not None:
                final_score = session_mgr.finalized_ppe_score if session_mgr.finalized_ppe_score is not None else 0
                session_mgr.finalized_action = "DONE"
                commit_evaluation_session(
                    worker_dict=session_mgr.active_worker,
                    final_ppe_score=final_score,
                    scanned_helmet_id=session_mgr.finalized_helmet_id or ocr_cached_helmet
                )

        # ── Handle Violation Buzzer ──
        current_score = session_mgr.get_smoothed_ppe_score()
        has_active_violation = (session_mgr.state == SessionState.EVALUATING_PPE and current_score is not None and current_score < 100)
        if has_active_violation != last_buzzer_state:
            if has_active_violation:
                Thread(target=pi.buzzer_on, daemon=True).start()
            else:
                Thread(target=pi.buzzer_off, daemon=True).start()
            last_buzzer_state = has_active_violation

        # ───────────────────────────────────────────────────
        # 5. Render Rich HUD Annotations
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
        cv2.rectangle(annotated, (0, 0), (frame_w, 48), (12, 12, 20), -1)
        cv2.line(annotated, (0, 48), (frame_w, 48), (50, 50, 70), 1)

        cv2.putText(annotated, "SiteSentinel AI", (14, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 107, 43), 2, cv2.LINE_AA)
        
        status_text = f"FPS: {current_fps:.1f}  |  Pi: {PI_IP} [CONNECTED]"
        cv2.putText(annotated, status_text, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Evaluation State Badge (Top Right)
        state_colors = {
            SessionState.IDLE: (140, 140, 140),
            SessionState.CROWDED: (50, 50, 240),
            SessionState.IDENTIFYING: (0, 200, 255),
            SessionState.EVALUATING_PPE: (255, 170, 0),
            SessionState.FINALIZED: (0, 220, 100),
            SessionState.COOLDOWN: (200, 140, 255)
        }
        st_color = state_colors.get(session_mgr.state, (200, 200, 200))
        st_label = f"[{session_mgr.state}]"
        cv2.putText(annotated, st_label, (frame_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, st_color, 2, cv2.LINE_AA)

        # Bottom Information Card Overlay
        cv2.rectangle(annotated, (0, frame_h - 75), (frame_w, frame_h), (12, 12, 20), -1)
        cv2.line(annotated, (0, frame_h - 75), (frame_w, frame_h - 75), (50, 50, 70), 1)

        # HUD Message
        cv2.putText(annotated, session_mgr.hud_message, (16, frame_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Worker Details (Bottom Left)
        if session_mgr.active_worker:
            w_info = f"Worker: {session_mgr.active_worker['name']} ({session_mgr.active_worker['worker_id']})"
            h_info = f"Helmet: {session_mgr.finalized_helmet_id or session_mgr.active_worker.get('helmet_id') or 'N/A'}"
            cv2.putText(annotated, f"{w_info}  |  {h_info}", (16, frame_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 212, 170), 1, cv2.LINE_AA)

        # PPE Score Gauge (Bottom Right)
        score_val = session_mgr.get_smoothed_ppe_score()
        if score_val is not None:
            score_color = (0, 220, 100) if score_val >= 100 else ((0, 200, 255) if score_val >= 50 else (50, 50, 240))
            cv2.putText(annotated, f"PPE SCORE: {score_val}%", (frame_w - 220, frame_h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, score_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "PPE SCORE: --", (frame_w - 220, frame_h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (140, 140, 140), 1, cv2.LINE_AA)

        # Crowding warning overlay banner
        if session_mgr.crowding_warning:
            cv2.rectangle(annotated, (gx1, 60), (gx2, 100), (0, 0, 180), -1)
            cv2.putText(annotated, "! CROWDING: ONE WORKER AT A TIME !", (gx1 + 20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Display window
        cv2.imshow("SiteSentinel — Live Gate Inspection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[SiteSentinel] Interrupted by user.")
finally:
    print("[SiteSentinel] Shutting down CV engine...")
    try:
        pi.buzzer_off()
    except:
        pass
    face_manager.stop()
    stream.close()
    cv2.destroyAllWindows()
    print("[SiteSentinel] Shutdown clean and complete.")