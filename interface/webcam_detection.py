from ultralytics import YOLO
import cv2
import requests
import time
from dotenv import load_dotenv
import os

# -----------------------------
# Configuration
# -----------------------------
load_dotenv()
ESP32_IP = os.getenv("ESP32_IP")

ALARM_ON_URL = f"http://{ESP32_IP}/on"
ALARM_OFF_URL = f"http://{ESP32_IP}/off"

# -----------------------------
# Load PPE Model
# -----------------------------
model = YOLO("models/best.pt")

print("\nLoaded Model Classes:")
print(model.names)
print("-" * 50)

# -----------------------------
# Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open webcam")
    exit()

# -----------------------------
# Alarm State Tracking
# -----------------------------
last_state = None

print("SiteSentinel Started")
print("Press Q to quit")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to access webcam")
            break

        # -----------------------------
        # Run Detection
        # -----------------------------
        results = model(
            frame,
            conf=0.7,
            verbose=False
        )

        # -----------------------------
        # Create Custom Annotated Frame
        # -----------------------------
        annotated_frame = frame.copy()

        detected_labels = []

        if results[0].boxes is not None:

            for box in results[0].boxes:

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                class_name = model.names[cls_id]

                # Ignore Mask Classes Completely
                if class_name in ["Mask", "NO-Mask"]:
                    continue

                detected_labels.append(class_name)

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Red for violations
                if class_name in [
                    "NO-Hardhat",
                    "NO-Safety Vest"
                ]:
                    color = (0, 0, 255)

                else:
                    color = (0, 255, 0)

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # -----------------------------
        # Debug Output Every 30 Frames
        # -----------------------------
        if not hasattr(model, "frame_counter"):
            model.frame_counter = 0

        model.frame_counter += 1

        if model.frame_counter % 30 == 0:

            print("\nDetections:")

            for label in detected_labels:
                print(label)

        # -----------------------------
        # Person Count
        # -----------------------------
        person_count = detected_labels.count("Person")

        cv2.putText(
            annotated_frame,
            f"Persons: {person_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # -----------------------------
        # PPE Violation Logic
        # -----------------------------
        violation = (
            "NO-Hardhat" in detected_labels or
            "NO-Safety Vest" in detected_labels
        )

        # -----------------------------
        # Alarm Control
        # -----------------------------
        if violation:

            if last_state != "ON":

                try:
                    requests.get(
                        ALARM_ON_URL,
                        timeout=1
                    )

                    print("Alarm ON")

                except Exception as e:
                    print(
                        "ESP32 Connection Error:",
                        e
                    )

                last_state = "ON"

            cv2.putText(
                annotated_frame,
                "PPE VIOLATION DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        else:

            if last_state != "OFF":

                try:
                    requests.get(
                        ALARM_OFF_URL,
                        timeout=1
                    )

                    print("Alarm OFF")

                except Exception as e:
                    print(
                        "ESP32 Connection Error:",
                        e
                    )

                last_state = "OFF"

        # -----------------------------
        # Display Output
        # -----------------------------
        cv2.imshow(
            "SiteSentinel PPE Detection",
            annotated_frame
        )

        # -----------------------------
        # Exit on Q
        # -----------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exit requested")
            break

except KeyboardInterrupt:

    print("\nStopped by user")

finally:

    print("Shutting down...")

    try:
        requests.get(
            ALARM_OFF_URL,
            timeout=1
        )

    except:
        pass

    time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

    print("System safely closed")