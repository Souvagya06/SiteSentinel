# SiteSentinel

## AI-Powered Construction Site Safety Monitoring & Smart Attendance System

**SiteSentinel** is a production-grade AI, Computer Vision, and IoT-based construction site gate-monitoring platform designed to enforce worker safety compliance and automate workforce attendance in real time.

The system combines a **custom-trained YOLOv8 PPE detection model**, **biometric face recognition**, **helmet OCR / QR / Barcode scanning**, **helmet color role classification**, **Raspberry Pi IoT hardware actuation**, **GPU-accelerated inference**, **Firebase Realtime Cloud telemetry**, **Turso cloud database**, and a **Flask web dashboard** (hosted at [sitesentinel.site](https://sitesentinel.site) and runnable locally) — unified into a resilient, parallel, event-driven computer vision pipeline.

---

## Key Features

### 1. Parallel Event-Driven Computer Vision Engine
Located in `interface/webcam_detection.py`, the core CV engine runs at 20+ FPS with CUDA acceleration:
- **Asynchronous Multiprocessing** — Face recognition, helmet OCR/QR scanning, and helmet color classification execute concurrently in background thread pools (`ThreadPoolExecutor`) without stalling the main YOLO detection loop.
- **7-State Single-Worker Gate State Machine (`WorkerSessionManager`)** — Enforces single-file gate discipline:
  $$\text{IDLE} \longrightarrow \text{VERIFYING} \longrightarrow \text{CONFIRMING} \longrightarrow \text{COMPLETED} \longrightarrow \text{COOLDOWN}$$
  Handles `CROWDED` (multiple workers entering simultaneously) and `LOCKED` states.
- **Auto-Evaluation & Dual Triggering** — Evaluation activates automatically when a single worker enters the gate zone (`ENABLE_AUTO_EVAL`), or manually via the physical Raspberry Pi push button or virtual dashboard button.
- **Centroid-Based Person Tracker (`SimpleCentroidTracker`)** — Maintains consistent identity tracking and prevents track-hopping using centroid distance matching.
- **4-Frame Temporal PPE Consensus** — Smooths PPE observations across a sliding window, requiring 3 consecutive unanimous readings for early decision lock-in.
- **Vectorized In-Memory Face Matching** — 128-dimensional dlib face embeddings are preloaded into NumPy matrices for batch Euclidean distance comparisons without DB latency.
- **Hot-Reloading Face Biometrics (`FaceDataManager`)** — Monitors backend database versions every 8 seconds and refreshes embeddings on-the-fly when workers are added or removed.
- **Dual-Mode Cloud HUD Streaming** — Streams the live annotated video with HUD banners (`MODE: IDLE`, `MODE: VERIFYING`, `MODE: COMPLETED`, `MODE: COOLDOWN`) to both `localhost:5000` and the hosted cloud dashboard (`sitesentinel.site`).

---

### 2. Custom YOLOv8 PPE Detection
Powered by custom-trained weights (`models/best.pt`) optimized for industrial safety gear:

| Class | Description | Gate Impact |
|---|---|---|
| `Person` | Worker detection and gate bounding trigger | Triggers gate evaluation |
| `Hardhat` | Helmet detected on worker | **PASS** |
| `NO-Hardhat` | Helmet missing | **FAIL** (Buzzer sounds) |
| `Safety Vest` | High-visibility vest worn | **PASS** |
| `NO-Safety Vest` | Safety vest missing | **FAIL** (Buzzer sounds) |
| `Safety Cone` | Site safety hazard markers | Tracked |
| `Machinery` / `Vehicle` | Site equipment | Tracked |

#### Spatial PPE Anatomical Attribution
Detections are mapped to anatomical bounding box regions:
- **Head Zone** (Top 40% of person box): Hardhat / NO-Hardhat detections must fall within this boundary.
- **Torso Zone** (20% – 80% of person box): Safety Vest / NO-Safety Vest must reside in this region.
- Prevents false attribution from nearby equipment or bystanders.

#### PPE Scoring Matrix
- **100% (Full Pass):** Hardhat + Safety Vest both present.
- **50% (Partial):** Only one required PPE item detected.
- **0% (Violation):** No PPE detected.

---

### 3. Helmet Identity Verification (QR / Barcode / OCR)
Ensures workers wear their officially assigned safety gear:
1. **QR Code Scanning** — High-speed (<2ms) detection via `cv2.QRCodeDetector`.
2. **Barcode Scanning** — Detection via `cv2.barcode.BarcodeDetector`.
3. **Multi-Pass EasyOCR (CUDA GPU Accelerated)**:
   - **Pass 1:** Direct high-speed inference.
   - **Pass 2:** Grayscale + CLAHE contrast enhancement + 1.8x bicubic upscaling.
   - **Pass 3:** Otsu adaptive binary thresholding.
- Scanned helmet IDs (e.g. `H1`, `H2`) are verified against worker database records. Mismatches abort gate access and log an audit event.

---

### 4. Helmet Color Classification & Role Cross-Check
Enforces industry-standard construction helmet color coding via HSV color space analysis of the helmet's upper dome:

| Helmet Color | Assigned Occupation Role |
|---|---|
| **Yellow** | Labourer / Construction / Heavy-Duty |
| **White + Grey** | Manager / Engineer / Supervisor / Visitor |
| **Red** | Firefighter / Safety Emergency |
| **Brown** | Welder / High Heat Operations |
| **Blue** | Electrician / Technical Specialist |
| **Green** | Safety Officer |
| **Pink** | Female Worker / Specialized Personnel |

Observations are filtered across 6 frames. If a Labourer wears a White helmet or an Engineer wears a Yellow helmet, check-in is aborted and a `HELMET_COLOR_MISMATCH` safety incident is logged.

---

### 5. Biometric Face Recognition & Automated Attendance
- **Worker Registration:** Managers upload photos via the dashboard; images are stored on **Cloudinary**, and 128-dimensional embeddings are generated via `dlib` and saved to **Turso DB**.
- **Real-Time Attendance Logic:**
  - Off-site workers entering gate $\longrightarrow$ **Check-In** (Green LED, Matrix score, gate unlocked).
  - On-site workers exiting gate $\longrightarrow$ **Check-Out** (Red LED, check-out recorded).
- **Anti-Fraud Protections:** 10-second re-trigger cooldown prevents duplicate logs. Unknown faces or safety violations trigger audible buzzer alerts without recording attendance.

---

### 6. Raspberry Pi IoT Edge Integration & Firebase Cloud Relay
The gate hardware runs on Raspberry Pi (Rev 1.3 / Pi 4) with hybrid local + cloud communication:

| Component | Pin / Bus | Operational Function |
|---|---|---|
| **Pi Camera Rev 1.3** | CSI Ribbon | High-framerate MJPEG video capture |
| **Green LED** | BCM 17 | Check-in approval indicator |
| **Red LED** | BCM 27 | Check-out / gate locked indicator |
| **Buzzer** | BCM 18 | Audible safety violation alert |
| **Push Button** | BCM 22 | Physical gate evaluation trigger |
| **MAX7219 8x8 LED Matrix** | SPI0 CE0 | Live scrolling PPE score display (`100%`, `50%`, `0%`) |

#### Cloud & Edge Communication
- **Direct Local API (Port 8080):** High-speed local MJPEG stream (`/stream`), health checks (`/health`), and hardware actuation (`/checkin`, `/checkout`, `/on`, `/off`, `/ppe`, `/status`).
- **Firebase Realtime Cloud Bridge:**
  - **Telemetry Relay:** Pi continuously reports uptime, button state, LED status, and verification state to Firebase.
  - **Remote Actuation:** Hosted cloud dashboards dispatch commands (`buzzer`, `ppe`, `green`, `red`, `verify`) to the Pi through Firebase Cloud with zero router port-forwarding.
- **Adaptive Cloud Pusher:** Non-blocking background worker (`_cloud_pusher_network_worker`) streams camera frames directly to the cloud backend with proxy buffering prevention (`X-Accel-Buffering: no`).

---

### 7. Modern Web Dashboard & Cloud Infrastructure
- **Hosted Domain:** [https://sitesentinel.site](https://sitesentinel.site) (Vercel CDN + Render Backend).
- **Local Dashboard:** `http://127.0.0.1:5000/dashboard`.
- **Live Stream Watchdog:** Automatic reconnect watchdog eliminates frozen streams caused by reverse proxy timeouts.
- **Worker Management:** Add, edit, remove workers; view safety compliance histories.
- **Helmet Registry:** Manage helmet inventories, conditions, and worker assignments.
- **Exportable Compliance Reports:** Download PDF (ReportLab) and CSV audit logs.
- **Manual Hardware Testing:** Test buzzer, LEDs, and matrix displays remotely from the dashboard.
- **Virtual Gate Trigger:** Trigger gate evaluation directly from the UI (`Verify Gate`).

---

## System Architecture

```
+----------------------------------------------------------------------+
|                     Raspberry Pi Edge Gate                          |
|                                                                      |
|  Pi Camera Rev 1.3 -> MJPEG Server (Port 8080)                       |
|  Decoupled Cloud Pusher -> https://sitesentinel-or0h.onrender.com    |
|  GPIO: Green LED (17) | Red LED (27) | Buzzer (18) | Button (22)     |
|  SPI: MAX7219 8x8 Matrix                                            |
|  Firebase Realtime Bridge (pisite_01)                                |
+----------------------------------+-----------------------------------+
                                   | Local MJPEG / Firebase RTDB
                                   v
+----------------------------------------------------------------------+
|             SiteSentinel AI CV Engine (Workstation / GPU)            |
|                                                                      |
|  PiStream (MJPEG / Local Fallback)                                   |
|  YOLOv8 Inference (CUDA) -> Person & PPE Detection                  |
|  SimpleCentroidTracker -> Multi-Person Lock                          |
|  WorkerSessionManager FSM (IDLE -> VERIFYING -> COMPLETED)           |
|      |-- Async Face Worker (128-d Vector Match)                      |
|      |-- Async OCR/QR Worker (EasyOCR + QR + Barcode)                |
|      +-- Real-Time HSV Helmet Color Classification                   |
|  Actuation Dispatch -> PiController (Local HTTP / Firebase)         |
|  HUD Mirror Pusher -> Local & Cloud Frame Buffers                    |
+----------------------------------+-----------------------------------+
                                   | /api/pi/frame/push (~20 FPS)
                                   v
+----------------------------------------------------------------------+
|              Flask Cloud / Local Backend (backend/main.py)           |
|                                                                      |
|  Auth: Email/Password (bcrypt) + Google OAuth 2.0                    |
|  Database: Turso Cloud SQLite (LibSQL Pipeline)                      |
|  Cloud Storage: Cloudinary (Worker Photos)                           |
|  Cloud Bridge: Firebase Realtime Database (Device Commands)          |
|  Streaming: Unbuffered MJPEG (/api/pi/stream) + Latest Snapshot      |
+----------------------------------+-----------------------------------+
                                   | HTTPS / WebSockets
                                   v
+----------------------------------------------------------------------+
|                   Web Management Dashboard (Frontend)               |
|                                                                      |
|  Live HUD Stream with Watchdog & Live FPS Indicator                  |
|  Virtual Gate Trigger (Verify Gate)                                  |
|  Worker Management | Attendance Log | Safety Events Log              |
|  Audit Reports (CSV & PDF) | Manual Hardware GPIO Test Panel         |
+----------------------------------------------------------------------+
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Vanilla HTML5, CSS3 (Design Tokens, Glassmorphism), Modern JavaScript |
| **Backend** | Python 3.11, Flask 3.1, Gunicorn, Werkzeug |
| **AI / Computer Vision** | YOLOv8 (Ultralytics), PyTorch (CUDA), OpenCV, EasyOCR, `face_recognition` (`dlib`) |
| **Database** | Turso (Cloud LibSQL / SQLite) |
| **Cloud Storage** | Cloudinary |
| **IoT / Edge** | Raspberry Pi 4 / Rev 1.3, `picamera2`, `RPi.GPIO`, `luma.led_matrix` |
| **Cloud Bridge** | Firebase Realtime Database (`firebase-admin`) |
| **Authentication** | `bcrypt`, Google OAuth 2.0 |
| **Reporting** | ReportLab (PDF), Python CSV |
| **Hosting & CDN** | Render (Web Service), Vercel (Edge Proxy & DNS) |

---

## Project Structure

```
SiteSentinel/
├── backend/
│   ├── main.py                  # Flask application (REST API, MJPEG buffer, auth)
│   ├── database.py              # Turso LibSQL HTTP pipeline client
│   ├── firebase_service.py      # Firebase Cloud Realtime Database service
│   ├── face__utils.py           # Face embedding extraction utilities
│   ├── report_service.py        # CSV and PDF audit report generator
│   └── init_db.py               # Database table schema migrations
│
├── interface/
│   ├── webcam_detection.py      # Main parallel CV engine with HUD & cloud pusher
│   ├── worker_session.py        # 7-state FSM, PPE consensus, and spatial attribution
│   ├── helmet_color_classifier.py # HSV helmet dome color classification
│   ├── pi_controller.py         # Hardware controller with Firebase fallback poller
│   ├── stream_reader.py         # High-res MJPEG and local webcam stream reader
│   ├── face_data_manager.py     # In-memory vectorized face cache & hot-reloader
│   └── event_client.py          # Asynchronous attendance & safety event logger
│
├── frontend/
│   ├── pages/
│   │   ├── index.html           # Landing page
│   │   ├── login.html           # Auth page (Email + Google OAuth)
│   │   └── dashboard.html       # Management dashboard with live AI HUD
│   └── images/                  # Static assets & icons
│
├── models/
│   └── best.pt                  # Custom-trained YOLOv8 PPE detection weights
│
├── Pi_Camera.py                 # Raspberry Pi server & decoupled cloud pusher
├── pi_firebase_bridge.py        # Pi-side Firebase command & telemetry daemon
├── raspberry_Pi_Code.txt        # Full deployment mirror of Pi camera code
├── firebase-credentials.json    # Firebase service account credentials
├── requirements_local.txt       # Local workstation requirements (GPU CV stack)
├── requirements_render.txt      # Cloud backend requirements (lightweight web stack)
├── render.yaml                  # Render deployment configuration
├── vercel.json                  # Vercel proxy & clean URLs configuration
└── README.md
```

---

## Installation & Setup

### 1. Workstation / Local PC Setup (Runs the AI CV Engine)
Python 3.11 with an NVIDIA CUDA-compatible GPU is recommended.

```bash
# Clone the repository
git clone https://github.com/Souvagya06/SiteSentinel.git
cd SiteSentinel

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements_local.txt
```

### 2. Environment Variables (`.env`)
Create a `.env` file in the project root:

```env
# Turso Database
TURSO_DATABASE_URL=libsql://your-database.turso.io
TURSO_AUTH_TOKEN=your-turso-auth-token

# Cloudinary
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret

# Google OAuth 2.0
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback

# Security & Endpoints
SECRET_KEY=your-random-secret-key
BACKEND_URL=http://127.0.0.1:5000
CLOUD_STREAM_URL=https://sitesentinel-or0h.onrender.com/api/pi/frame/push
ENABLE_CLOUD_PUSH=true
ENABLE_AUTO_EVAL=true

# Firebase Cloud Bridge
FIREBASE_DATABASE_URL=https://your-project-rtdb.firebaseio.com/
FIREBASE_DEVICE_ID=pisite_01
```

### 3. Initialize the Database
```bash
python backend/init_db.py
```

### 4. Run the Local Backend
```bash
cd backend
python main.py
```
Dashboard available at: `http://127.0.0.1:5000`

---

## Raspberry Pi Setup

The file [`Pi_Camera.py`](Pi_Camera.py) (or [`raspberry_Pi_Code.txt`](raspberry_Pi_Code.txt)) runs on the Raspberry Pi at the gate.

### Hardware Pinout

| Pin | BCM GPIO | Component | Function |
|---|---|---|---|
| Pin 11 | GPIO 17 | Green LED | Access Approved |
| Pin 13 | GPIO 27 | Red LED | Access Denied / Cooldown |
| Pin 12 | GPIO 18 | Active Buzzer | Safety Violation Warning |
| Pin 15 | GPIO 22 | Tactile Push Button | Gate Evaluation Trigger |
| Pin 19, 23, 24 | SPI0 (MOSI, SCLK, CE0) | MAX7219 8x8 Matrix | PPE Score Display |
| CSI Port | CSI Ribbon | Camera Module Rev 1.3 | Live Stream Capture |

### Deploy and Run on Pi
```bash
# Copy script and credentials to the Pi
scp Pi_Camera.py pi@<PI_IP>:/home/pi/
scp firebase-credentials.json pi@<PI_IP>:/home/pi/

# SSH into the Pi
ssh pi@<PI_IP>

# Run camera server & cloud pusher
python3 Pi_Camera.py
```

The Pi prints:
```text
=====================================================
  SiteSentinel Pi Server + Firebase Cloud Active
  Stream   -> http://192.168.1.25:8080/stream
  Health   -> http://192.168.1.25:8080/health
  Status   -> http://192.168.1.25:8080/status
  Firebase -> https://sitesentinel-50bfc-default-rtdb...
  Pins     -> Green=17, Red=27, Buzzer=18, Button=22
=====================================================
[Cloud Stream] Fast adaptive pusher active -> https://sitesentinel-or0h.onrender.com/api/pi/frame/push
```

---

## Running the AI Gate Monitoring Engine

Once the Pi and backend are running, launch the AI Computer Vision engine on your local workstation:

```bash
python interface/webcam_detection.py --user-id 1 [--camera-id "Gate 1"] [--imgsz 416] [--headless]
```

| Flag | Description | Default |
|---|---|---|
| `--user-id` | Manager user ID in database | **Required** |
| `--camera-id` | Camera identifier string | `Pi Camera 01` |
| `--imgsz` | YOLO input resolution (320 or 416) | `416` |
| `--headless` | Run without opening local OpenCV window (for cloud/server streaming) | `False` |
| `--pi-ip` | Explicit Pi IP override (otherwise auto-resolved or read from DB) | Auto |
| `--backend-url` | Override target backend endpoint | Read from `.env` |

> **Tip:** You can also start and stop the detection engine directly from the web dashboard by clicking **Start Camera Stream**.

---

## Operational Verification Workflow

```
[Gate Idle]
Camera displays: "MODE: IDLE | Gate Ready | Press Button to Verify"
                      |
Worker enters gate zone OR presses physical/virtual button
                      |
[Verification Mode]
Camera displays: "MODE: VERIFYING" (Amber HUD)
- Bounding box locks onto worker
- Parallel Face Recognition identifies worker name and occupation role
- Parallel OCR/QR scans helmet serial ID
- Real-time HSV analyzes helmet dome color against occupation role
- YOLOv8 calculates spatial PPE consensus score (Hardhat + Vest)
                      |
      +---------------+---------------+
      |                               |
[PPE Violation / Mismatch]       [Compliance Confirmed]
- Gate remains locked            - Gate unlocked
- Buzzer sounds (GPIO 18)        - Green LED ON (GPIO 17)
- Red LED ON (GPIO 27)           - Matrix displays: "PPE: 100%"
- Matrix displays: "PPE: 0%"     - Attendance logged (Check-In / Out)
- Safety Event logged in DB      - HUD displays: "MODE: COMPLETED"
      |                               |
      +---------------+---------------+
                      |
[Cooldown Mode]
Camera displays: "MODE: COOLDOWN" (Blue HUD)
- 10-second re-trigger cooldown prevents duplicate scans
- System resets cleanly to "MODE: IDLE" for the next worker
```

---

## Team

- **Souvagya Karmakar**
- **Sushmita Roy**
- **Anirban Pal**
- **Ronit Mishra**

---

## License

**Proprietary Software — All Rights Reserved.**
SiteSentinel is the intellectual property of its authors. Unauthorized copying, modification, distribution, or commercial use is strictly prohibited.
