import sys
import os
import argparse
import re
import time
import easyocr
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from stream_reader import PiStream
from threading import Thread
from ultralytics import YOLO
import cv2
import requests
import json
import torch
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from database import execute, query_all, query_one
from face__utils import get_embedding_from_frame, match_face


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")

last_buzzer_state = None
last_ppe_score = -1
ppe_score_buffer = []
PPE_SMOOTH_FRAMES = 8

# ── Args ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--user-id", required=True)
args = parser.parse_args()
MANAGER_USER_ID = str(args.user_id)
print(f"Running as manager user_id={MANAGER_USER_ID}")

PI = query_one(
    "SELECT pi_ip FROM users WHERE id = ?",
    [{"type": "text", "value": MANAGER_USER_ID}]
)

if PI is None or not PI.get("pi_ip"):
    raise RuntimeError(
        f"No Raspberry Pi IP configured for manager {MANAGER_USER_ID}"
    )

PI_IP = PI["pi_ip"]
stream = PiStream(PI_IP)
from pi_controller import PiController

pi = PiController(PI_IP)
print("=" * 60)
print(f"Connected Raspberry Pi : {PI_IP}")
print("=" * 60)

# ── Load PPE model ────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
model = YOLO(str(MODEL_PATH))
if torch.cuda.is_available():
    model.to("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")
print("Loaded Model Classes:", model.names)

# ── Load known faces ──────────────────────────────────
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
                "name":      f"{row['first_name']} {row['last_name']}",
                "embedding": emb
            })
        except:
            pass
    print(f"Loaded {len(known)} known face(s)")
    return known

known_faces = load_known_faces()

# ── OCR for helmet IDs ────────────────────────────────
ocr_reader        = easyocr.Reader(['en'], gpu=False, verbose=False)
matched_helmet_ids = set()
HELMET_ID_PATTERN = re.compile(r'[A-Za-z]{1,6}[_\-]?\d{1,6}', re.IGNORECASE)


def normalize_helmet_id(helmet_id):
    return re.sub(r'[^A-Z0-9]', '', (helmet_id or '').upper())


def load_available_helmets():
    try:
        rows = query_all(
            """
            SELECT helmet_id
            FROM helmets
            WHERE user_id = ? AND status = 'Available'
            """,
            [{"type": "text", "value": MANAGER_USER_ID}]
        )
    except Exception as e:
        print(f"Available helmet lookup error: {e}")
        return {}

    return {
        normalize_helmet_id(row.get("helmet_id")): row.get("helmet_id")
        for row in rows
        if row.get("helmet_id")
    }

# ── State ─────────────────────────────────────────────
checked_in_today = set()
face_last_seen   = {}
frame_counter    = 0
current_detected_worker_id = None
current_worker_ppe_score = None

# ── Backend notify (non-blocking) ─────────────────────
def notify(wid, score_to_save, checkin_time, new_status):
    helmet_id = ""
    try:
        worker    = query_one(
            "SELECT helmet_id FROM workers WHERE worker_id = ?",
            [{"type": "text", "value": wid}]
        )
        helmet_id = (worker.get("helmet_id") or "") if worker else ""
    except:
        pass

    try:
        requests.post(f"{BACKEND_URL}/api/workers/checkin",
            json={"worker_id": wid, "ppe_score": score_to_save,
                  "checkin_time": checkin_time, "status": new_status,
                  "helmet_id": helmet_id},
            timeout=5)
        print(f"Backend updated: {wid} → {new_status} | PPE: {score_to_save}")
    except Exception as e:
        print("Check-in API error:", e)

    # Release helmet on checkout
    if new_status == "Active" and helmet_id:
        # Mark helmet as Occupied on check-in
        try:
            requests.post(f"{BACKEND_URL}/api/helmets/assign",
                json={"helmet_id": helmet_id, "worker_id": wid},
                timeout=3)
            print(f"Helmet {helmet_id} marked Occupied for worker {wid}")
        except Exception as e:
            print(f"Helmet occupy error: {e}")

    elif new_status == "Off-Site" and helmet_id:
        # Release helmet on check-out
        try:
            requests.post(f"{BACKEND_URL}/api/helmets/release",
                json={"helmet_id": helmet_id}, timeout=3)
            print(f"Helmet {helmet_id} released")
            execute(
                "UPDATE workers SET helmet_id = '' WHERE worker_id = ?",
                [{"type": "text", "value": wid}]
            )
        except Exception as e:
            print(f"Helmet release error: {e}")            

def sync_active_workers_ppe_score(score):
    """Persist the latest stable PPE score for every worker currently on site."""
    for wid in list(checked_in_today):
        try:
            execute(
                "UPDATE workers SET ppe_score = ? WHERE worker_id = ?",
                [
                    {"type": "text", "value": str(score)},
                    {"type": "text", "value": wid}
                ]
            )
        except Exception as e:
            print(f"PPE score sync error for {wid}: {e}")

def store_helmet_for_worker(worker_id, helmet_id, ppe_score_at_checkin=0):
    """Assign helmet only if registered AND Available. Then show PPE score on matrix."""
    if not worker_id or not helmet_id:
        return

    try:
        helmet = query_one(
            "SELECT helmet_id, status FROM helmets WHERE helmet_id = ? AND user_id = ?",
            [
                {"type": "text", "value": helmet_id},
                {"type": "text", "value": MANAGER_USER_ID}
            ]
        )
    except Exception as e:
        print(f"Helmet lookup error: {e}")
        return

    if not helmet:
        print(f"Helmet {helmet_id} not in registered inventory — ignoring")
        return

    if helmet.get("status") == "Occupied":
        # Check if already assigned to THIS worker
        current_worker = query_one(
            "SELECT helmet_id FROM workers WHERE worker_id = ?",
            [{"type": "text", "value": worker_id}]
        )
        if current_worker and current_worker.get("helmet_id") == helmet_id:
            print(f"Helmet {helmet_id} already assigned to this worker")
            return
        print(f"Helmet {helmet_id} already occupied by another worker — ignoring")
        return

    # Helmet is Available — assign it
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/helmets/assign",
            json={"helmet_id": helmet_id, "worker_id": worker_id},
            timeout=5
        )
        if resp.ok:
            matched_helmet_ids.add(helmet_id)
            print(f"Helmet {helmet_id} assigned to worker {worker_id}")
            # Show PPE score on matrix now that helmet is confirmed
            Thread(
                target=pi.show_score,
                args=(ppe_score_at_checkin,),
                daemon=True
            ).start()
            print(f"Matrix showing check-in PPE score: {ppe_score_at_checkin}")
        else:
            print(f"Helmet assign failed: {resp.text}")
    except Exception as e:
        print(f"Helmet store error: {e}")

def register_helmet(worker_id, helmet_id):
    """Assign helmet to worker via backend — only if helmet is registered."""
    try:
        available_helmets = load_available_helmets()
        resolved_helmet_id = available_helmets.get(normalize_helmet_id(helmet_id))
        if not resolved_helmet_id:
            print(f"Helmet {helmet_id} NOT available — ignoring")
            return

        # Assign helmet
        resp = requests.post(
            f"{BACKEND_URL}/api/helmets/assign",
            json={"helmet_id": resolved_helmet_id, "worker_id": worker_id},
            timeout=5
        )
        if resp.ok:
            print(f"Helmet {resolved_helmet_id} assigned to worker {worker_id}")
        else:
            print(f"Helmet assign failed: {resp.text}")
    except Exception as e:
        print(f"Helmet register error: {e}")

# ── Main loop ─────────────────────────────────────────
cam_label = "Pi Camera Stream"
print(f"SiteSentinel Started [{cam_label}] — Press Q to quit")

try:
    while True:
        ret, frame = stream.read()
        if not ret:
            print ("Pi stream lost. Reconnecting...")
            stream.close()
            time.sleep(2)
            stream = PiStream(PI_IP)
            continue
        if frame is None:
            time.sleep(0.1)
            continue

        frame_counter += 1
        annotated       = frame.copy()
        detected_labels = []

        # ── PPE Detection ──────────────────────────────
        results = model(frame, conf=0.5, verbose=False)
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

# ── PPE Score ──────────────────────────────────
        any_ppe = any(l in detected_labels for l in [
            "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"
        ])

        if not any_ppe:
            ppe_score = -1
        else:
            helmet_ok = (
                "Hardhat" in detected_labels
                and "NO-Hardhat" not in detected_labels
            )
            vest_ok = (
                "Safety Vest" in detected_labels
                and "NO-Safety Vest" not in detected_labels
            )
            ppe_score = (50 if helmet_ok else 0) + (50 if vest_ok else 0)

        # Smooth score — only update after PPE_SMOOTH_FRAMES consistent readings
        if ppe_score >= 0:
            ppe_score_buffer.append(ppe_score)
            if len(ppe_score_buffer) > PPE_SMOOTH_FRAMES:
                ppe_score_buffer.pop(0)

            # Only trigger if all recent frames agree on same score
            if (len(ppe_score_buffer) == PPE_SMOOTH_FRAMES
                    and len(set(ppe_score_buffer)) == 1
                    and ppe_score_buffer[0] != last_ppe_score):

                stable_score   = ppe_score_buffer[0]
                last_ppe_score = stable_score
                
                sync_active_workers_ppe_score(stable_score)
                print(f"PPE Score Stable: {stable_score}")
        else:
            # No person — clear buffer
            ppe_score_buffer.clear()

        effective_ppe_score = ppe_score if ppe_score > 0 else 0

        # ── Face Recognition (every 15 frames) ─────────
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
                current_detected_worker_id = wid
                now = datetime.now()

                if (now.timestamp() - face_last_seen.get(wid, 0)) < 10:
                    cv2.putText(annotated, best_match["name"], (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
                    continue

                face_last_seen[wid] = now.timestamp()

                if wid not in checked_in_today:
                    checked_in_today.add(wid)
                    checkin_time  = now.strftime("%I:%M %p")
                    new_status    = "Active"
                    score_to_save = ppe_score if ppe_score >= 0 else last_ppe_score if last_ppe_score >= 0 else None
                    current_worker_ppe_score = score_to_save
                    current_detected_worker_id = wid
                    print(f"Checked IN:  {best_match['name']} | PPE: {ppe_score}")
                else:
                    checked_in_today.discard(wid)
                    worker_row = query_one(
                        "SELECT helmet_id FROM workers WHERE worker_id = ?",
                        [{"type": "text", "value": wid}]
                    )
                    if worker_row and worker_row.get("helmet_id"):
                        matched_helmet_ids.discard(worker_row["helmet_id"])
                    if current_detected_worker_id == wid:
                        current_detected_worker_id = None
                    current_worker_ppe_score = None
                    checkin_time  = "--:--"
                    new_status    = "Off-Site"
                    score_to_save = 0
                    print(f"Checked OUT: {best_match['name']}")

                Thread(
                    target=notify,
                    args=(wid, score_to_save, checkin_time, new_status),
                    daemon=True
                ).start()   

                if new_status == "Active":
                    Thread(
                        target=pi.checkin,
                        daemon=True
                    ).start()
                else:
                    Thread(
                        target=pi.checkout,
                        daemon=True
                    ).start()

                cv2.putText(annotated,
                            f"{best_match['name']} ({new_status})", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)

        # ── Helmet OCR (every 10 frames, always scan) ──
        if current_detected_worker_id and frame_counter % 10 == 0:
            ocr_results = ocr_reader.readtext(frame, detail=1, paragraph=False)
            for (bbox, text_ocr, conf) in ocr_results:
                text_clean = text_ocr.strip().upper().replace(' ', '_')
                if float(conf) < 0.4:
                    continue
                m = HELMET_ID_PATTERN.search(text_clean)
                if not m:
                    continue

                helmet_id_found = m.group(0)
                pts = [tuple(map(int, pt)) for pt in bbox]
                for i in range(4):
                    cv2.line(annotated, pts[i], pts[(i+1)%4], (255, 165, 0), 2)
                cv2.putText(annotated, f"HELMET: {helmet_id_found}", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                if helmet_id_found in matched_helmet_ids:
                    continue

                # Pass current PPE score so matrix shows it after helmet confirmed
                score_for_matrix = ppe_score if ppe_score >= 0 else last_ppe_score if last_ppe_score >= 0 else 0
                Thread(
                    target=store_helmet_for_worker,
                    args=(current_detected_worker_id, helmet_id_found, score_for_matrix),
                    daemon=True
                ).start()

        # ── HUD ────────────────────────────────────────
        person_count = detected_labels.count("Person")
        violation    = "NO-Hardhat" in detected_labels or "NO-Safety Vest" in detected_labels
        if violation != last_buzzer_state:
            if violation:
                Thread(
                    target=pi.buzzer_on,
                    daemon=True
                ).start()
            else:
                Thread(
                    target=pi.buzzer_off,
                    daemon=True
                ).start()

            last_buzzer_state = violation
        cv2.putText(annotated, f"Persons: {person_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        display_score = current_worker_ppe_score if current_worker_ppe_score is not None else (ppe_score if ppe_score >= 0 else "--")
        cv2.putText(annotated, f"PPE Score: {display_score}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if violation:
            cv2.putText(annotated, "PPE VIOLATION DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Camera source label bottom-left
        cv2.putText(annotated, cam_label,
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("SiteSentinel PPE Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    try:
        pi.buzzer_off()
    except:
        pass
    try:
        pi.checkout()
    except:
        pass
    stream.close()
    cv2.destroyAllWindows()
    print("System safely closed")