import sys
import os
import argparse
import re
import easyocr
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from threading import Thread
from ultralytics import YOLO
import cv2
import requests
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from database import query_all, query_one
from face__utils import get_embedding_from_frame, match_face

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")

# -----------------------------
# Parse user_id passed from main.py at launch
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--user-id", required=True, help="Logged-in manager user_id")
args = parser.parse_args()
MANAGER_USER_ID = str(args.user_id)
print(f"Running as manager user_id={MANAGER_USER_ID}")

# -----------------------------
# Load PPE model
# -----------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
model = YOLO(str(MODEL_PATH))
print("Loaded Model Classes:", model.names)

# -----------------------------
# Load known faces — reads directly from DB, no session needed
# -----------------------------
def load_known_faces():
    rows = query_all("""
        SELECT w.worker_id, w.first_name, w.last_name, wi.face_embedding
        FROM workers w
        JOIN worker_images wi ON w.worker_id = wi.worker_db_id
        WHERE wi.face_embedding IS NOT NULL
          AND wi.face_embedding != ''
          AND w.user_id = ?
    """, [{"type": "text", "value": MANAGER_USER_ID}])

    known = []
    for row in rows:
        try:
            emb = json.loads(row["face_embedding"])
            known.append({
                "worker_id": row["worker_id"],
                "name": f"{row['first_name']} {row['last_name']}",
                "embedding": emb
            })
        except:
            pass
    print(f"Loaded {len(known)} known face(s) for user_id={MANAGER_USER_ID}")
    return known

known_faces = load_known_faces()

# -----------------------------
# Open webcam — try indices 0, 1, 2
# -----------------------------
cap = None
for cam_index in range(3):
    _cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if _cap.isOpened():
        cap = _cap
        print(f"Camera opened at index {cam_index}")
        break
    _cap.release()

if cap is None:
    print("ERROR: No camera found at indices 0, 1, or 2.")
    input("Press Enter to exit...")
    exit()

last_alarm_state  = None
checked_in_today  = set()
face_last_seen    = {}
frame_counter     = 0
ocr_reader        = easyocr.Reader(['en'], gpu=False, verbose=False)
helmet_registered = set()   # tracks which worker_ids already have helmet_id saved this session
HELMET_ID_PATTERN = re.compile(r'[A-Z]{2,6}_?\d{3,6}', re.IGNORECASE)  # e.g. SKC0001, SKC_0001, HELM042

# -----------------------------
# Get ESP32 IP directly from DB — no session needed
# -----------------------------
def get_esp32_ip():
    try:
        user = query_one(
            "SELECT esp32_ip FROM users WHERE id = ?",
            [{"type": "text", "value": MANAGER_USER_ID}]
        )
        return (user["esp32_ip"] or "").strip() if user else ""
    except Exception as e:
        print(f"get_esp32_ip DB error: {e}")
        return ""

# -----------------------------
# Check if helmet is registered in database
# -----------------------------
def is_helmet_registered(helmet_id):
    try:
        res = query_one(
            "SELECT id FROM helmets WHERE helmet_id = ? AND user_id = ?",
            [{"type": "text", "value": helmet_id}, {"type": "text", "value": MANAGER_USER_ID}]
        )
        return res is not None
    except Exception as e:
        print(f"Error checking helmet registration in DB: {e}")
        return False

# -----------------------------
# Check if scanned OCR text matches a registered helmet ID using lookalikes
# -----------------------------
def ocr_matches_registered(ocr_text, registered_id):
    ocr_norm = ocr_text.strip().upper().replace(' ', '').replace('_', '')
    reg_norm = registered_id.strip().upper().replace(' ', '').replace('_', '')
    
    if len(ocr_norm) != len(reg_norm):
        return False
        
    lookalike_groups = [
        {'0', 'O'},
        {'1', 'I', 'L'},
        {'5', 'S'},
        {'6', 'G'},
        {'8', 'B'},
        {'2', 'Z'},
        {'7', 'T'}
    ]
    
    for c1, c2 in zip(ocr_norm, reg_norm):
        if c1 == c2:
            continue
        matched_group = False
        for group in lookalike_groups:
            if c1 in group and c2 in group:
                matched_group = True
                break
        if not matched_group:
            return False
            
    return True

# -----------------------------
# Non-blocking backend + ESP32 notify
# -----------------------------
def notify(wid, score_to_save, checkin_time, new_status):
    # Fetch current helmet_id for this worker from DB
    helmet_id = ""
    try:
        worker = query_one(
            "SELECT helmet_id FROM workers WHERE worker_id = ?",
            [{"type": "text", "value": wid}]
        )
        helmet_id = (worker["helmet_id"] or "") if worker else ""
    except:
        pass

    try:
        requests.post(f"{BACKEND_URL}/api/workers/checkin",
            json={
                "worker_id":    wid,
                "ppe_score":    score_to_save,
                "checkin_time": checkin_time,
                "status":       new_status,
                "helmet_id":    helmet_id
            },
            timeout=5)
    except Exception as e:
        print("Check-in API error:", e)

    ip = get_esp32_ip()
    if ip:
        try:
            endpoint = "/checkin" if new_status == "Active" else "/checkout"
            requests.get(f"http://{ip}{endpoint}", timeout=5)
            print(f"{'Green' if new_status == 'Active' else 'Red'} light triggered")
        except Exception as e:
            print("ESP32 LED error:", e)

# -----------------------------
# Save helmet ID to backend
# -----------------------------
def register_helmet(worker_id, helmet_id):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/workers/{worker_id}/helmet",
            json={"helmet_id": helmet_id, "user_id": MANAGER_USER_ID},
            timeout=5
        )
        if resp.ok:
            print(f"Helmet {helmet_id} registered to worker {worker_id}")
        else:
            print(f"Helmet register failed: {resp.text}")
            helmet_registered.discard(worker_id)  # allow retry with different helmet on failure
    except Exception as e:
        print(f"Helmet register error: {e}")
        helmet_registered.discard(worker_id)

print("SiteSentinel Started — Press Q to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        annotated       = frame.copy()
        detected_labels = []

        # --- PPE Detection ---
        results = model(frame, conf=0.7, verbose=False)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                class_name = model.names[cls_id]
                if class_name in ["Mask", "NO-Mask"]:
                    continue
                detected_labels.append(class_name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if class_name in ["NO-Hardhat", "NO-Safety Vest"] else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Helmet ID OCR (every 20 frames to reduce CPU load) ---
        if frame_counter % 20 == 0:
            ocr_results = ocr_reader.readtext(frame, detail=1, paragraph=False)
            if ocr_results:
                print(f"[OCR DEBUG] Found {len(ocr_results)} text region(s) in frame {frame_counter}")
                try:
                    db_helmets = query_all(
                        "SELECT helmet_id FROM helmets WHERE user_id = ?",
                        [{"type": "text", "value": MANAGER_USER_ID}]
                    )
                    registered_helmet_ids = [h["helmet_id"] for h in db_helmets]
                    print(f"[OCR DEBUG] Registered helmets in inventory: {registered_helmet_ids}")
                except Exception as e:
                    print(f"Error fetching helmets from DB: {e}")
                    registered_helmet_ids = []
            else:
                registered_helmet_ids = []

            for (bbox, text, conf) in ocr_results:
                text_clean = text.strip().upper().replace(' ', '').replace('_', '')
                conf = float(conf)
                print(f"[OCR DEBUG] Raw text: '{text}' | Cleaned: '{text_clean}' | Conf: {conf:.2f}")
                if conf < 0.5:
                    continue

                matched_registered_id = None
                for reg_id in registered_helmet_ids:
                    if ocr_matches_registered(text_clean, reg_id):
                        matched_registered_id = reg_id
                        break

                if not matched_registered_id:
                    print(f"[OCR DEBUG] Text '{text_clean}' does not match any registered helmet in inventory.")
                    continue

                print(f"[OCR DEBUG] Match found! Scanned: '{text_clean}' matches Registered: '{matched_registered_id}'")

                # Draw bounding box around detected text
                pts = [tuple(map(int, pt)) for pt in bbox]
                for i in range(4):
                    cv2.line(annotated, pts[i], pts[(i+1)%4], (255, 165, 0), 2)
                cv2.putText(annotated, f"ID: {matched_registered_id}", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                # Link to most-recently-seen checked-in worker not yet registered
                best_wid  = None
                best_name = None
                best_time = 0
                for kf in known_faces:
                    wid = kf["worker_id"]
                    if wid in checked_in_today and wid not in helmet_registered:
                        t = face_last_seen.get(wid, 0)
                        if t > best_time:
                            best_time  = t
                            best_wid   = wid
                            best_name  = kf["name"]
                if best_wid:
                    helmet_registered.add(best_wid)
                    Thread(
                        target=register_helmet,
                        args=(best_wid, matched_registered_id),
                        daemon=True
                    ).start()
                    cv2.putText(annotated, f"Helmet linked: {best_name}", (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # --- PPE Score ---
        helmet_ok = "Hardhat"     in detected_labels and "NO-Hardhat"     not in detected_labels
        vest_ok   = "Safety Vest" in detected_labels and "NO-Safety Vest" not in detected_labels
        ppe_score = (50 if helmet_ok else 0) + (50 if vest_ok else 0)

        # --- Face Recognition (every 15 frames) ---
        if frame_counter % 15 == 0 and known_faces:
            rgb             = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_embeddings = get_embedding_from_frame(rgb)

            for live_emb in live_embeddings:
                best_match = None
                best_dist  = 1.0
                for kf in known_faces:
                    matched, dist = match_face(live_emb, json.dumps(kf["embedding"]))
                    if matched and dist < best_dist:
                        best_dist  = dist
                        best_match = kf

                if not best_match:
                    continue

                wid = best_match["worker_id"]
                now = datetime.now()

                # 10-second cooldown to avoid rapid toggling
                if (now.timestamp() - face_last_seen.get(wid, 0)) < 10:
                    cv2.putText(annotated, best_match["name"], (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
                    continue

                face_last_seen[wid] = now.timestamp()

                # Toggle check-in / check-out
                if wid not in checked_in_today:
                    checked_in_today.add(wid)
                    checkin_time  = now.strftime("%I:%M %p")
                    new_status    = "Active"
                    score_to_save = ppe_score
                    print(f"Checked IN:  {best_match['name']} | PPE: {ppe_score}")
                else:
                    checked_in_today.discard(wid)
                    helmet_registered.discard(wid)   # allow re-scan next shift
                    checkin_time  = "--:--"
                    new_status    = "Off-Site"
                    score_to_save = 0
                    print(f"Checked OUT: {best_match['name']}")

                Thread(
                    target=notify,
                    args=(wid, score_to_save, checkin_time, new_status),
                    daemon=True
                ).start()

                cv2.putText(annotated, f"{best_match['name']} ({new_status})", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)

        # --- HUD overlays ---
        person_count = detected_labels.count("Person")
        violation    = "NO-Hardhat" in detected_labels or "NO-Safety Vest" in detected_labels

        cv2.putText(annotated, f"Persons: {person_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated, f"PPE Score: {ppe_score}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if violation:
            if last_alarm_state != "ON":
                ip = get_esp32_ip()
                if ip:
                    try: requests.get(f"http://{ip}/on", timeout=1)
                    except: pass
                last_alarm_state = "ON"
            cv2.putText(annotated, "PPE VIOLATION DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            if last_alarm_state != "OFF":
                ip = get_esp32_ip()
                if ip:
                    try: requests.get(f"http://{ip}/off", timeout=1)
                    except: pass
                last_alarm_state = "OFF"

        cv2.imshow("SiteSentinel PPE Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    try:
        off_ip = get_esp32_ip()
        if off_ip:
            requests.get(f"http://{off_ip}/off", timeout=2)
            print("Alarm/buzzer turned OFF on exit")
    except Exception as e:
        print(f"Could not turn off alarm on exit: {e}")
    cap.release()
    cv2.destroyAllWindows()
    print("System safely closed")