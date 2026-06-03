import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ultralytics import YOLO
import cv2
import requests
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from database import query_all
from face__utils import get_embedding_from_frame, match_face

load_dotenv()
ESP32_IP     = os.getenv("ESP32_IP")
ALARM_ON_URL  = f"http://{ESP32_IP}/on"
ALARM_OFF_URL = f"http://{ESP32_IP}/off"
BACKEND_URL   = "http://127.0.0.1:5000"

# Load PPE model
model = YOLO("models/best.pt")
print("Loaded Model Classes:", model.names)

# Load all workers + embeddings from DB
def load_known_faces():
    rows = query_all("""
        SELECT w.worker_id, w.first_name, w.last_name, wi.face_embedding
        FROM workers w
        JOIN worker_images wi ON w.worker_id = wi.worker_db_id
        WHERE wi.face_embedding IS NOT NULL AND wi.face_embedding != ''
    """)
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
    print(f"Loaded {len(known)} known face(s)")
    return known

known_faces = load_known_faces()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam")
    exit()

last_alarm_state = None
checked_in_today = set()   # tracks who is currently Active
face_last_seen   = {}      # tracks last toggle timestamp per worker
frame_counter    = 0

print("SiteSentinel Started — Press Q to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        annotated      = frame.copy()
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

        # --- PPE Score ---
        helmet_ok = "Hardhat" in detected_labels and "NO-Hardhat" not in detected_labels
        vest_ok   = "Safety Vest" in detected_labels and "NO-Safety Vest" not in detected_labels
        ppe_score = (50 if helmet_ok else 0) + (50 if vest_ok else 0)

        # --- Face Recognition (every 15 frames for performance) ---
        if frame_counter % 15 == 0 and known_faces:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_embeddings = get_embedding_from_frame(rgb)

            for live_emb in live_embeddings:
                # Find best matching known face
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

                # Cooldown: ignore same face for 10 seconds to avoid rapid toggling
                last_seen = face_last_seen.get(wid, 0)
                if (now.timestamp() - last_seen) < 10:
                    cv2.putText(annotated, best_match["name"], (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
                    continue

                # Update last seen timestamp
                face_last_seen[wid] = now.timestamp()

                # Toggle check-in / check-out
                if wid not in checked_in_today:
                    checked_in_today.add(wid)
                    checkin_time = now.strftime("%I:%M %p")
                    new_status   = "Active"
                    score_to_save = ppe_score
                    print(f"Checked IN:  {best_match['name']} | PPE: {ppe_score}")
                else:
                    checked_in_today.discard(wid)
                    checkin_time  = "--:--"
                    new_status    = "Off-Site"
                    score_to_save = 0
                    print(f"Checked OUT: {best_match['name']}")

                # Send to backend
                try:
                    requests.post(
                        f"{BACKEND_URL}/api/workers/checkin",
                        json={
                            "worker_id":    wid,
                            "ppe_score":    score_to_save,
                            "checkin_time": checkin_time,
                            "status":       new_status
                        },
                        timeout=3
                    )
                except Exception as e:
                    print("Check-in API error:", e)

                # Show name + status on frame
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
                try: requests.get(ALARM_ON_URL, timeout=1)
                except: pass
                last_alarm_state = "ON"
            cv2.putText(annotated, "PPE VIOLATION DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            if last_alarm_state != "OFF":
                try: requests.get(ALARM_OFF_URL, timeout=1)
                except: pass
                last_alarm_state = "OFF"

        cv2.imshow("SiteSentinel PPE Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    try: requests.get(ALARM_OFF_URL, timeout=1)
    except: pass
    cap.release()
    cv2.destroyAllWindows()
    print("System safely closed")