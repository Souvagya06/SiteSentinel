import sys
import os
import argparse
import re
import easyocr
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
import time 

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
def get_cam_ip():
    try:
        user = query_one(
            "SELECT esp32_cam_ip FROM users WHERE id = ?",
            [{"type": "text", "value": MANAGER_USER_ID}]
        )
        return (user.get("esp32_cam_ip") or "").strip() if user else ""
    except:
        return ""

cam_ip = get_cam_ip()
if cam_ip:
    stream_url = f"http://{cam_ip}:81/stream?rand={time.time()}"
    print(f"Connecting to ESP32-CAM: {stream_url}")
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)   # 5s timeout
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)              # reduce latency
    if not cap.isOpened():
        print("ESP32-CAM stream failed — falling back to USB webcam")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        print("ESP32-CAM stream connected!")

        try:
            requests.get(f"http://{cam_ip}:81/flash/on", timeout=2)
            print("ESP32-CAM flash ON")
        except Exception as e:
            print("Could not turn flash ON:", e)
else:
    print("No ESP32-CAM IP set — using USB webcam")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: No camera available")
    input("Press Enter to exit...")
    exit()

last_alarm_state  = None
face_last_seen    = {}
frame_counter     = 0
ocr_reader        = easyocr.Reader(['en'], gpu=False, verbose=False)
HELMET_ID_PATTERN = re.compile(r'[A-Z]{2,6}_?\d{3,6}', re.IGNORECASE)  # e.g. SKC0001, SKC_0001, HELM042

# Cache for OCR helmet detection to avoid flickering
last_ocr_helmet_seen_frame = -999
last_ocr_helmet_id = None

# Overlap helper function
def is_inside_or_overlaps(box_b, box_p, threshold=0.5):
    xb1, yb1, xb2, yb2 = box_b
    xp1, yp1, xp2, yp2 = box_p
    
    # Calculate intersection
    xi1 = max(xb1, xp1)
    yi1 = max(yb1, yp1)
    xi2 = min(xb2, xp2)
    yi2 = min(yb2, yp2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return False
        
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    b_area = (xb2 - xb1) * (yb2 - yb1)
    
    center_x = (xb1 + xb2) / 2
    center_y = (yb1 + yb2) / 2
    
    center_inside = (xp1 <= center_x <= xp2) and (yp1 <= center_y <= yp2)
    overlap_ratio = inter_area / b_area
    
    return center_inside or (overlap_ratio > threshold)

# -----------------------------
# Pre-load checked-in workers and registered helmets from DB
# -----------------------------
def load_checked_in_workers():
    try:
        rows = query_all(
            "SELECT worker_id, helmet_id FROM workers WHERE status = 'Active' AND user_id = ?",
            [{"type": "text", "value": MANAGER_USER_ID}]
        )
        checked_in = set()
        registered = set()
        for row in rows:
            wid = row["worker_id"]
            checked_in.add(wid)
            if row["helmet_id"] and row["helmet_id"].strip():
                registered.add(wid)
        print(f"Pre-loaded {len(checked_in)} checked-in worker(s) and {len(registered)} with helmet(s).")
        return checked_in, registered
    except Exception as e:
        print(f"Error loading checked-in workers: {e}")
        return set(), set()

checked_in_today, helmet_registered = load_checked_in_workers()

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

    # Post check-in to backend
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
        # LED trigger
        try:
            if new_status == "Active":
                requests.get(f"http://{ip}/checkin", timeout=5)
                print("Green light triggered")
            else:
                requests.get(f"http://{ip}/checkout", timeout=5)
                print("Red light triggered")
        except Exception as e:
            print("ESP32 LED error:", e)

        # Matrix: show PPE score on check-in, clear on check-out
        try:
            if new_status == "Active":
                requests.get(
                    f"http://{ip}/ppe",
                    params={"score": score_to_save},
                    timeout=5
                )
                print(f"Matrix showing PPE score: {score_to_save}")
            else:
                # checkout — matrix already clears in /checkout handler
                pass
        except Exception as e:
            print("Matrix error:", e)

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
            ip = get_esp32_ip()
            if ip:
                try:
                    requests.get(f"http://{ip}/checkin", timeout=5)
                    print("Green light triggered (helmet assigned)")
                except Exception as e:
                    print("ESP32 LED error on helmet assignment:", e)
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
            print("ESP32 stream lost. Reconnecting...")

            cap.release()
            cv2.destroyAllWindows()

            time.sleep(1)

            cap = cv2.VideoCapture(
                f"http://{cam_ip}:81/stream",
                cv2.CAP_FFMPEG
            )

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            continue

        frame_counter += 1
        annotated       = frame.copy()
        # --- PPE Detection ---
        results = model(frame, conf=0.5, verbose=False)
        persons = []
        hardhats = []
        no_hardhats = []
        vests = []
        no_vests = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                class_name = model.names[cls_id]
                if class_name in ["Mask", "NO-Mask"]:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_data = (x1, y1, x2, y2, conf)
                if class_name == "Person":
                    persons.append(box_data)
                elif class_name == "Hardhat":
                    hardhats.append(box_data)
                elif class_name == "NO-Hardhat":
                    no_hardhats.append(box_data)
                elif class_name == "Safety Vest":
                    vests.append(box_data)
                elif class_name == "NO-Safety Vest":
                    no_vests.append(box_data)

                # Draw bounding box
                color = (0, 0, 255) if class_name in ["NO-Hardhat", "NO-Safety Vest"] else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Find active person (largest person box)
        active_person = None
        max_area = 0
        for box in persons:
            x1, y1, x2, y2, conf = box
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                active_person = box

        # --- PPE Score ---
        if active_person:
            has_hardhat = any(is_inside_or_overlaps(h[:4], active_person[:4]) for h in hardhats)
            has_no_hardhat = any(is_inside_or_overlaps(nh[:4], active_person[:4]) for nh in no_hardhats)
            has_vest = any(is_inside_or_overlaps(v[:4], active_person[:4]) for v in vests)
            has_no_vest = any(is_inside_or_overlaps(nv[:4], active_person[:4]) for nv in no_vests)
        else:
            has_hardhat = len(hardhats) > 0
            has_no_hardhat = len(no_hardhats) > 0
            has_vest = len(vests) > 0
            has_no_vest = len(no_vests) > 0

        # OCR fallback for helmet
        if last_ocr_helmet_id and (frame_counter - last_ocr_helmet_seen_frame < 60):
            has_hardhat = True
            has_no_hardhat = False

        helmet_ok = has_hardhat and not has_no_hardhat
        vest_ok   = has_vest and not has_no_vest
        ppe_score = (50 if helmet_ok else 0) + (50 if vest_ok else 0)

        # --- Helmet ID OCR (every 20 frames to reduce CPU load) ---
        if active_person and frame_counter % 20 == 0:
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

                # Cache detected helmet
                last_ocr_helmet_seen_frame = frame_counter
                last_ocr_helmet_id = matched_registered_id
                helmet_ok = True
                ppe_score = (50 if helmet_ok else 0) + (50 if vest_ok else 0)

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
                    try:
                        worker_db = query_one(
                            "SELECT ppe_score, helmet_id FROM workers WHERE worker_id = ? AND user_id = ?",
                            [{"type": "text", "value": wid}, {"type": "text", "value": MANAGER_USER_ID}]
                        )
                    except Exception as db_err:
                        print(f"Error querying worker DB for checkout: {db_err}")
                        worker_db = None

                    if worker_db:
                        db_ppe = int(worker_db.get("ppe_score") or 0)
                        db_helmet = (worker_db.get("helmet_id") or "").strip()
                        if db_ppe > 0 and not db_helmet:
                            print(f"Checkout block checking for helmet ID for {best_match['name']}...")
                            ocr_results = ocr_reader.readtext(frame, detail=1, paragraph=False)
                            matched_registered_id = None
                            if ocr_results:
                                try:
                                    db_helmets = query_all(
                                        "SELECT helmet_id FROM helmets WHERE user_id = ?",
                                        [{"type": "text", "value": MANAGER_USER_ID}]
                                    )
                                    registered_helmet_ids = [h["helmet_id"] for h in db_helmets]
                                except:
                                    registered_helmet_ids = []

                                for (bbox, text, conf) in ocr_results:
                                    text_clean = text.strip().upper().replace(' ', '').replace('_', '')
                                    if float(conf) < 0.5:
                                        continue
                                    for reg_id in registered_helmet_ids:
                                        if ocr_matches_registered(text_clean, reg_id):
                                            matched_registered_id = reg_id
                                            break
                                    if matched_registered_id:
                                        break

                            if matched_registered_id:
                                print(f"Found helmet {matched_registered_id} during checkout block. Registering...")
                                try:
                                    resp = requests.post(
                                        f"{BACKEND_URL}/api/workers/{wid}/helmet",
                                        json={"helmet_id": matched_registered_id, "user_id": MANAGER_USER_ID},
                                        timeout=5
                                    )
                                    if resp.ok:
                                        print(f"Helmet {matched_registered_id} registered to worker {wid} during checkout block.")
                                        helmet_registered.add(wid)
                                        db_helmet = matched_registered_id
                                        ip = get_esp32_ip()
                                        if ip:
                                            try:
                                                requests.get(f"http://{ip}/checkin", timeout=5)
                                                print("Green light triggered (helmet assigned on checkout block)")
                                            except Exception as e:
                                                print("ESP32 LED error on helmet assignment:", e)
                                    else:
                                        print(f"Helmet register failed: {resp.text}")
                                except Exception as e:
                                    print(f"Helmet register error: {e}")

                            if not db_helmet:
                                print(f"Checkout BLOCKED for {best_match['name']}: PPE score > 0 but helmet not assigned.")
                                cv2.putText(annotated, "CHECKOUT BLOCKED: NO HELMET", (20, 260),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                continue

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
        person_count = len(persons)
        if person_count > 0:
            if active_person:
                violation = (not helmet_ok) or (not vest_ok)
            else:
                violation = len(no_hardhats) > 0 or len(no_vests) > 0
        else:
            violation = False

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
            try:
                if cam_ip:
                    requests.get(f"http://{cam_ip}:81/flash/off", timeout=2)
                    print("ESP32-CAM flash OFF")
            except Exception as e:
                print("Could not turn flash OFF:", e)

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
    try:
        if cam_ip:
            requests.get(f"http://{cam_ip}:81/flash/off", timeout=2)
            print("ESP32-CAM flash OFF on exit")
    except:
        pass
    cap.release()
    cv2.destroyAllWindows()
    print("System safely closed")