from ultralytics import YOLO
import cv2
import requests
import time

# -----------------------------
# Configuration
# -----------------------------
ESP32_IP = "10.36.176.180"

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

        # DEBUG (prints once every 30 frames)
        if not hasattr(model, "frame_counter"):
            model.frame_counter = 0

        model.frame_counter += 1

        if model.frame_counter % 30 == 0:

            print("\nDetections:")

            for box in results[0].boxes:

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                print(
                    f"{model.names[cls_id]} : {conf:.2f}"
                )

        annotated_frame = results[0].plot()

        # -----------------------------
        # Extract Detected Classes
        # -----------------------------
        detected_labels = []

        if results[0].boxes is not None:

            detections = results[0].boxes.cls.tolist()

            detected_labels = [
                model.names[int(cls)]
                for cls in detections
            ]
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
            "NO-Mask" in detected_labels or
            "NO-Safety Vest" in detected_labels
        )

        # -----------------------------
        # Alarm Control
        # -----------------------------
        if violation:

            if last_state != "ON":

                try:
                    requests.get(ALARM_ON_URL, timeout=1)
                    print("Alarm ON")
                except Exception as e:
                    print("ESP32 Connection Error:", e)

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
                    requests.get(ALARM_OFF_URL, timeout=1)
                    print("Alarm OFF")
                except Exception as e:
                    print("ESP32 Connection Error:", e)

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
        requests.get(ALARM_OFF_URL, timeout=1)
    except:
        pass

    time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

    print("System safely closed")