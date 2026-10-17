# SiteSentinel

## AI-Powered Construction Site Safety Monitoring & Smart Attendance System

**SiteSentinel** is an AI, Computer Vision, and IoT-based construction site gate-monitoring platform designed to enforce worker safety and automate workforce attendance in real time.

The system combines a **custom-trained YOLOv8 PPE detection model**, **face recognition**, **helmet OCR/QR/barcode scanning**, **helmet color classification**, **Raspberry Pi IoT hardware integration**, **GPU-accelerated inference**, a **Turso cloud database**, and a **Flask web dashboard** -- all unified into a single parallel, event-driven computer vision pipeline operating at the construction site gate.

---

# Key Features

## Parallel Event-Driven CV Engine

The heart of SiteSentinel is the **High-Performance Parallel AI Computer Vision Gate Monitoring Engine** (`interface/webcam_detection.py`).

Core capabilities:

- **Parallel Execution** -- Face recognition, helmet OCR/QR scanning, and helmet color classification run concurrently in background thread pools (ThreadPoolExecutor) without blocking the main YOLO inference loop.
- **Single-Worker Gate Evaluation State Machine** -- A 7-state FSM (WorkerSessionManager) tracks one worker at a time through the gate: IDLE -> VERIFYING -> CONFIRMING -> COMPLETED -> COOLDOWN. Also handles CROWDED and LOCKED states.
- **Physical Button Trigger** -- The Raspberry Pi has a physical push button. Pressing it activates a 6-second verification window, locking onto the first worker entering the gate zone.
- **Centroid-Based Person Tracker** (SimpleCentroidTracker) -- Tracks multiple persons across frames using centroid distance matching and a disappearance timeout, maintaining stable per-person track IDs.
- **4-Frame Temporal PPE Consensus Voting** -- PPE scores are smoothed over a sliding window of 4 observations; 3 consecutive identical readings trigger early convergence.
- **Vectorized In-Memory Face Matching** -- Face embeddings are cached as a NumPy matrix for fast batch distance comparisons without repeated DB calls.
- **Hot-Reloading Face Data** (FaceDataManager) -- Watches a backend version counter every 8 seconds and reloads face embeddings on-the-fly whenever a manager adds or removes a worker, without restarting the CV engine.
- **Dashboard MJPEG Streaming** -- Annotated frames are JPEG-encoded and pushed to the Flask backend at up to 20 FPS via /api/pi/frame/push, powering the live dashboard feed.

---

## YOLOv8 PPE Detection

SiteSentinel uses a **custom-trained YOLOv8 model** (`models/best.pt`) for real-time object detection.

| Class | Description |
|---|---|
| Person | Worker detection trigger |
| Hardhat | Helmet worn (PASS) |
| NO-Hardhat | Helmet missing (FAIL) |
| Safety Vest | Vest worn (PASS) |
| NO-Safety Vest | Vest missing (FAIL) |
| Safety Cone | Site equipment |
| Machinery | Heavy machinery |
| Vehicle | Vehicles on site |

> Note: `Mask` and `NO-Mask` classes are intentionally filtered out from scoring and display.

### Spatial PPE Anatomical Attribution

PPE detections are not blindly applied. The system performs strict anatomical zone attribution:

- Hardhat / NO-Hardhat detections must have their centroid fall within the head zone (top ~40% of person bounding box).
- Safety Vest / NO-Safety Vest detections must fall within the torso zone (20%-80% of person height).
- Only PPE attributed to the locked worker bounding box contributes to the score.

### PPE Scoring

| Score | Meaning |
|---|---|
| 100% | Hardhat + Safety Vest both present |
| 50% | One PPE item detected |
| 0% | No PPE detected |

---

## Helmet ID Verification (QR / Barcode / OCR)

Each worker can be assigned a helmet ID in the database (e.g. H1, H2). During gate evaluation the system scans the helmet in 3 passes:

1. **QR Code** -- high-speed (<2ms) using cv2.QRCodeDetector.
2. **Barcode** -- using cv2.barcode.BarcodeDetector.
3. **Multi-Pass EasyOCR** (GPU-accelerated when available):
   - Pass 1: Direct color inference.
   - Pass 2: Grayscale + CLAHE contrast enhancement + 1.8x upscaling.
   - Pass 3: Otsu thresholding.

Scanned helmet IDs are cross-checked against the worker assigned helmet ID. On mismatch, check-in is aborted and the buzzer sounds.

---

## Helmet Color Classification & Occupation Cross-Check

SiteSentinel enforces industry-standard helmet color coding based on worker occupation using real-time HSV color analysis.

| Helmet Color | Occupation Role |
|---|---|
| Yellow | Labourer / Construction / Heavy-duty |
| White + Grey | Manager / Engineer / Supervisor / Visitor |
| Red | Firefighter |
| Brown | Welder / High Heat |
| Blue | Electrician / Technical |
| Green | Safety Officer |
| Pink | Female Worker / Additional |

The `classify_helmet_color()` function analyzes the upper-central dome region of the helmet crop in HSV space. The dominant color is selected by pixel count with a minimum 20% coverage threshold. Color observations are buffered over 6 frames and the most frequent result is selected via Counter.

If a color mismatch is detected between the worn helmet color and the worker occupation, check-in is aborted and a safety event is logged.

---

## Face Recognition & Smart Attendance

### Worker Registration Flow

```
Manager Adds Worker (Dashboard)
         |
Worker Info + Photo Uploaded to Cloudinary
         |
128-d Face Embedding Generated (dlib / face_recognition)
         |
Embedding Stored in Turso Database
         |
face_data_version Incremented (triggers hot-reload in CV engine)
         |
Worker Ready for Recognition
```

### Live Recognition Flow

```
Camera Frame
         |
YOLO: Person Detected -> Locked onto Track ID
         |
Face Crop Sent to Async Face Worker (ThreadPoolExecutor)
         |
128-d Embedding Extracted (dlib)
         |
Vectorized Distance Match Against Cached Embeddings (threshold: 0.52)
         |
Worker Identified -> set_identified_worker() callback
```

### Attendance Transition Logic

| Current DB Status | Action |
|---|---|
| Off-Site | Check-in (Green LED) |
| Active / On-Site | Check-out (Red LED) |

- A 10-second worker cooldown prevents immediate re-triggering of the same worker.
- Unknown workers (face not matched within max_verify_sec) trigger a buzzer alert without recording attendance.

---

## Raspberry Pi Camera Streaming & IoT Integration

The Raspberry Pi serves as both a camera streaming server and an IoT actuator endpoint.

### Pi Server API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /stream | GET | Live MJPEG camera stream |
| /health | GET | Server health check |
| /status | GET | Button state + verification state (~5 Hz poll) |
| /checkin | GET | Green LED ON for 3s |
| /checkout | GET | Red LED ON for 3s |
| /on | GET | Buzzer ON |
| /off | GET | Buzzer OFF |
| /ppe?score=N | GET | Scroll PPE score on 8x8 LED matrix |
| /state?set=STATE | GET | Set verification state |

All hardware commands from PiController are dispatched non-blocking via background threads. Simulated GPIO mode activates automatically when no valid Pi IP is configured, using a local webcam as fallback.

### IoT Hardware

| Component | GPIO | Role |
|---|---|---|
| Green LED | BCM 17 | Check-in confirmation |
| Red LED | BCM 27 | Check-out confirmation |
| Buzzer | BCM 18 | PPE violation / helmet mismatch alert |
| 8x8 LED Matrix (MAX7219) | SPI0 CE0 | Scrolling PPE score display |
| Physical Push Button | Pi GPIO | Activates 6-second gate verification window |

---

## Web Dashboard & Backend

The Flask backend (`backend/main.py`) serves the full web application and all REST API endpoints.

### Authentication

- Email/Password signup and login with bcrypt password hashing.
- Google OAuth 2.0 (sign in with Google) -- full authorization code flow with state CSRF protection, profile picture sync, and automatic account linking by email.
- 7-day persistent sessions via Flask session cookies.

### Dashboard Features

- **Live camera feed** -- MJPEG stream pushed from the CV engine via /api/pi/frame/push and served at /api/pi/frame/stream.
- **Start / Stop CV Engine** -- Manager can launch or terminate the webcam_detection.py subprocess from the dashboard with one click.
- **Worker Management** -- Add, edit, delete workers; upload face photos; assign helmet IDs and occupations.
- **Helmet Management** -- Dedicated helmet registry tracking status (Available / In Use / Damaged), color, and type metadata.
- **Real-time Attendance Table** -- Live view of worker statuses, check-in/check-out times, and PPE scores.
- **Attendance Log** -- Filterable full history of all check-in / check-out events.
- **Safety Events Log** -- Structured log of AI-detected violations: HELMET_MISMATCH, HELMET_COLOR_MISMATCH, PPE_VIOLATION.
- **Compliance Reports** -- Export as CSV or PDF (ReportLab) with date range filtering.
- **Raspberry Pi Configuration** -- Manager can configure and save the Pi IP from dashboard settings.

---

## Database Schema

SiteSentinel uses **Turso Database** (cloud SQLite / LibSQL).

| Table | Purpose |
|---|---|
| users | Manager accounts (email, password hash, company, Google OAuth sub, Pi IP, face_data_version) |
| workers | Worker roster (worker_id, name, occupation, helmet_id, status, check-in/out times, PPE score) |
| worker_images | Face photo URLs and 128-d dlib face embeddings per worker |
| attendance_log | Full log of all check-in / check-out events with PPE scores, helmet IDs, and timestamps |
| helmets | Helmet registry (helmet_id, color, type, status) per manager |
| safety_events | Safety violation events (type, message, worker_id, ppe_score, helmet_id, camera_id, evidence_url, acknowledged flag) |

---

## Cloud Storage

Worker face photos are stored on **Cloudinary**.

```
Worker Photo Upload (Dashboard)
         |
Cloudinary Upload -> Public Image URL Stored
         |
Face Embedding Generated from URL (dlib)
         |
Embedding Stored in worker_images Table
         |
face_data_version Incremented -> CV Hot-Reload Triggered
```

---

## GPU-Accelerated Inference

SiteSentinel uses PyTorch CUDA for GPU-accelerated YOLO inference and EasyOCR. Automatically falls back to CPU when no GPU is available.

---

# Full System Architecture

```
+----------------------------------------------------------------------+
|                        Raspberry Pi 4                               |
|                                                                      |
|  Pi Camera -> MJPEG Stream Server (Port 8080)                       |
|  GPIO: Green LED | Red LED | Buzzer | 8x8 LED Matrix | Push Button  |
+----------------------------------+-----------------------------------+
                                   | MJPEG + HTTP API
                                   v
+----------------------------------------------------------------------+
|          SiteSentinel Parallel CV Engine (Main Computer)             |
|                                                                      |
|  PiStream -> Frame Buffer                                            |
|                    |                                                 |
|                    v                                                 |
|          YOLOv8 Inference (GPU/CPU)                                  |
|          Person Detection + PPE Detection                            |
|                    |                                                 |
|                    v                                                 |
|      SimpleCentroidTracker (Multi-Person)                            |
|      Physical Button Press Trigger                                   |
|                    v                                                 |
|  WorkerSessionManager FSM                                            |
|  IDLE -> VERIFYING -> CONFIRMING -> COMPLETED -> COOLDOWN            |
|                    |                                                 |
|      +-------------+--------------------+                           |
|      v             v                    v                           |
|  Face Worker   OCR/QR Worker     Color Classifier                   |
|  (Async Pool)  (Async Pool)      (Real-Time HSV)                    |
|      |             |                    |                           |
|      +-------------+--------------------+                           |
|                    v                                                 |
|      Helmet Crosscheck + Color Occupation Crosscheck                 |
|                    |                                                 |
|      commit_evaluation_session()                                     |
|      +-- EventClient -> /api/attendance                             |
|      +-- EventClient -> /api/events (safety violations)             |
|      +-- PiController -> Green/Red LED + Buzzer + Matrix            |
|                                                                      |
|      push_frame_to_dashboard() -> /api/pi/frame/push (~20 FPS)      |
+----------------------------------------------------------------------+
                                   |
                                   v
+----------------------------------------------------------------------+
|                    Flask Backend (backend/main.py)                   |
|                                                                      |
|  REST API: Auth, Workers, Attendance, Events, Reports, Helmets       |
|  Google OAuth 2.0 | bcrypt Auth | Cloudinary | ReportLab            |
|  In-Process MJPEG Frame Buffer -> Dashboard Live Stream              |
|  Database: Turso (LibSQL / Cloud SQLite)                             |
+----------------------------------------------------------------------+
                                   |
                                   v
+----------------------------------------------------------------------+
|                       Web Dashboard (Frontend)                       |
|                                                                      |
|  HTML + CSS + JavaScript + Tailwind CSS                             |
|  Live Camera Feed | Worker Management | Attendance Log               |
|  Safety Events | Compliance Reports (CSV/PDF) | Helmet Registry      |
|  Pi Configuration | Start/Stop CV Engine                             |
+----------------------------------------------------------------------+
```

---

# Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript, Tailwind CSS |
| Backend | Python 3.11, Flask |
| AI / CV | YOLOv8 (Ultralytics), OpenCV, PyTorch, CUDA, EasyOCR, face_recognition (dlib) |
| Database | Turso (LibSQL / Cloud SQLite) |
| Cloud Storage | Cloudinary |
| IoT / Hardware | Raspberry Pi 4, picamera2, RPi.GPIO, luma.led_matrix |
| Auth | bcrypt, Google OAuth 2.0 |
| Reporting | ReportLab (PDF), csv (CSV) |

---

# Project Structure

```
SiteSentinel/
|
+-- backend/
|   +-- main.py                  Flask app (API + routes + MJPEG frame buffer)
|   +-- database.py              Turso LibSQL query helpers
|   +-- init_db.py               DB schema migration and initialization
|   +-- face__utils.py           Face embedding extraction utilities
|   +-- report_service.py        CSV and PDF compliance report generator
|   +-- reset_admin_password.py  Utility to reset a manager password
|
+-- interface/
|   +-- webcam_detection.py      Main CV engine (YOLO + face + OCR + FSM + HUD)
|   +-- worker_session.py        WorkerSessionManager FSM + PPE scoring + helmet crosscheck
|   +-- helmet_color_classifier.py  Real-time HSV helmet color classifier + occupation mapping
|   +-- pi_controller.py         Non-blocking Raspberry Pi HTTP hardware controller
|   +-- stream_reader.py         MJPEG / local webcam stream reader (PiStream)
|   +-- face_data_manager.py     In-memory face embedding cache + hot-reload
|   +-- event_client.py          Async HTTP client for attendance and safety event logging
|
+-- frontend/
|   +-- pages/
|   |   +-- index.html           Landing page
|   |   +-- login.html           Login / Signup (email + Google OAuth)
|   |   +-- dashboard.html       Main manager dashboard
|   +-- images/                  Frontend static images
|
+-- models/
|   +-- best.pt                  Custom-trained YOLOv8 PPE detection weights
|   +-- last.pt                  Last training checkpoint
|
+-- assets/                      Shared static assets
+-- raspberry_Pi_Code.txt        Full Raspberry Pi server script
+-- requirements_local.txt       Full local dependencies
+-- requirements_render.txt      Render.com deployment dependencies
+-- render.yaml                  Render.com deployment config
+-- vercel.json                  Vercel deployment config
+-- .env                         Environment variables (never commit)
+-- README.md
```

---

# Requirements

Recommended Python Version: 3.11

Python 3.11 is recommended for compatibility with dlib and face_recognition.

---

# Installation

```bash
git clone <repository-url>
cd SiteSentinel
pip install -r requirements_local.txt
```

---

# Environment Variables

Create a `.env` file in the project root:

```
TURSO_DATABASE_URL=
TURSO_AUTH_TOKEN=

CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=

SECRET_KEY=
BACKEND_URL=http://127.0.0.1:5000
```

Never commit `.env` to version control.

---

# Initialize the Database

```bash
python backend/init_db.py
```

---

# Run the Backend

```bash
cd backend
python main.py
```

Backend starts at: http://127.0.0.1:5000

---

# Raspberry Pi Setup

The file `raspberry_Pi_Code.txt` in the root directory contains the full Raspberry Pi server script.

> Do NOT run this on your main computer. It uses hardware-specific libraries (picamera2, RPi.GPIO, luma.led_matrix) only available on a Raspberry Pi.

## Hardware Wiring

| Component | Details |
|---|---|
| Raspberry Pi | Model 4 (recommended) |
| Pi Camera | Connected via CSI ribbon cable |
| Green LED | GPIO pin 17 (BCM) |
| Red LED | GPIO pin 27 (BCM) |
| Buzzer | GPIO pin 18 (BCM) |
| Push Button | Any GPIO with pull-up |
| LED Matrix (MAX7219) | SPI0, CE0 |

## Pi-Side Dependencies

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-picamera2 python3-rpi.gpio
pip3 install luma.led_matrix flask
```

## Deploy to Pi

```bash
scp raspberry_Pi_Code.txt pi@<PI_IP>:/home/pi/Pi_Camera.py
ssh pi@<PI_IP>
python3 Pi_Camera.py
```

---

# Run the AI Gate Monitoring Engine

```bash
python interface/webcam_detection.py --user-id <MANAGER_ID> [--camera-id "Gate 1"] [--imgsz 416] [--headless] [--pi-ip <PI_IP>]
```

| Argument | Description |
|---|---|
| --user-id | Manager user ID (required) |
| --camera-id | Camera label shown in events (default: Pi Camera 01) |
| --imgsz | YOLO inference image size: 320 or 416 (default: 416) |
| --headless | Disable OpenCV display window (for dashboard-only streaming) |
| --pi-ip | Override Pi IP address (otherwise read from database) |

The CV engine can also be started and stopped directly from the dashboard via the Start Detection button.

---

# Security Notes

Never expose or commit:

- Turso Auth Token
- Cloudinary API Secret
- Google Client Secret
- Flask Secret Key
- Raspberry Pi IP (if private)
- .env file

---

# Feature Status

| Feature | Status |
|---|---|
| YOLOv8 PPE Detection | Complete |
| Spatial PPE Anatomical Attribution | Complete |
| 4-Frame Temporal PPE Consensus | Complete |
| Face Recognition (dlib 128-d) | Complete |
| Vectorized In-Memory Face Matching | Complete |
| FaceDataManager Hot-Reload | Complete |
| Helmet QR Code Scanning | Complete |
| Helmet Barcode Scanning | Complete |
| Helmet OCR (Multi-Pass EasyOCR) | Complete |
| Helmet ID Cross-Check | Complete |
| Helmet Color Classification (HSV) | Complete |
| Helmet Color x Occupation Cross-Check | Complete |
| WorkerSessionManager FSM (7 states) | Complete |
| Centroid-Based Person Tracker | Complete |
| Physical Button Gate Trigger | Complete |
| Worker Cooldown (Anti Re-Trigger) | Complete |
| Automatic Check-In / Check-Out | Complete |
| NVIDIA GPU Acceleration | Complete |
| Raspberry Pi Camera Streaming | Complete |
| Green / Red LED Actuation | Complete |
| Buzzer Safety Alerts | Complete |
| 8x8 LED Matrix PPE Score Display | Complete |
| Dashboard MJPEG Live Stream | Complete |
| Flask Backend + REST API | Complete |
| Google OAuth 2.0 | Complete |
| Worker Management (CRUD) | Complete |
| Helmet Registry | Complete |
| Attendance Log | Complete |
| Safety Events Log | Complete |
| CSV Compliance Reports | Complete |
| PDF Compliance Reports (ReportLab) | Complete |
| Turso Cloud Database | Complete |
| Cloudinary Image Storage | Complete |
| Pi IP Configuration from Dashboard | Complete |
| Start / Stop CV Engine from Dashboard | Complete |

---

# Team

- Souvagya Karmakar
- Sushmita Roy
- Anirban Pal
- Ronit Mishra

---

# License

**Proprietary Software - All Rights Reserved**

SiteSentinel is the intellectual property of its authors. Unauthorized copying, modification, distribution, or commercial use is prohibited without explicit permission from the project authors.
