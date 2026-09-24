# 🦺 SiteSentinel

## AI-Powered Construction Site Safety Monitoring & Smart Attendance System

**SiteSentinel** is an AI, Computer Vision, and IoT-based construction site safety monitoring platform designed to improve worker safety and automate workforce attendance.

The system combines **YOLOv8-based PPE detection**, **face recognition**, **Raspberry Pi camera streaming**, **GPU acceleration**, a **Turso database** to monitor workers in real time.

A worker can be identified through face recognition, automatically checked in or checked out, and monitored for Personal Protective Equipment (PPE) compliance.

---

# 🚀 Key Features

## 🦺 Real-Time PPE Detection

SiteSentinel uses a custom-trained **YOLOv8 model** to detect safety-related objects in real time.

Currently supported classes include:

* Hardhat
* Mask
* NO-Hardhat
* NO-Mask
* NO-Safety Vest
* Person
* Safety Cone
* Safety Vest
* Machinery
* Vehicle

The system processes the live camera stream and performs real-time object detection.

---

## ⚠️ PPE Compliance Monitoring

The system evaluates PPE compliance based on detected safety equipment.

The current PPE monitoring workflow is:

```text
Camera Stream
      ↓
YOLOv8 Detection
      ↓
Person Detection
      ↓
Hardhat / Safety Vest Analysis
      ↓
PPE Score Calculation
      ↓
Dashboard / Backend Update
```

A PPE score is associated with the worker when a valid worker detection and attendance event occurs.

The system distinguishes between:

* Valid PPE detection
* PPE violation
* No valid person detected

This prevents incorrect PPE scores from being generated when no person is detected.

---

# 👤 Smart Attendance System

SiteSentinel integrates **face recognition with worker management**.

The workflow is:

```text
Camera
   ↓
Face Detection
   ↓
Face Embedding Extraction
   ↓
Compare with Registered Workers
   ↓
Worker Identified
   ↓
Automatic Check-In / Check-Out
```

### Current Features

* Face Recognition-Based Worker Identification
* Automatic Worker Check-In
* Automatic Worker Check-Out
* Worker ID Identification
* Worker Name Identification
* Attendance Status Updates
* Attendance Logging
* Real-Time Worker Tracking
* PPE Score Association with Attendance Events

Example:

```text
Face Recognized
        ↓
Worker: Souvagya Karmakar
Worker ID: 0002
        ↓
Checked IN
        ↓
Status Updated: Active
        ↓
PPE Score Recorded
```

When the same worker is detected again according to the attendance logic:

```text
Worker Identified
        ↓
Checked OUT
        ↓
Status Updated: Off-Site
        ↓
PPE Score Recorded
```

---

# 🎥 Raspberry Pi Camera Streaming

The Raspberry Pi acts as a remote camera and IoT server.

The Pi camera provides a live MJPEG stream that is consumed by the main SiteSentinel AI system.

Architecture:

```text
Raspberry Pi Camera
        │
        ▼
Pi Camera Server
        │
        ├── /stream
        ├── /health
        ├── /checkin
        ├── /checkout
        ├── /on
        ├── /off
        └── /ppe?score=N
        │
        ▼
Main Computer
        │
        ▼
YOLOv8 + Face Recognition
```

The Raspberry Pi server handles:

* Live Camera Streaming
* Health Check
* GPIO Control
* IoT Communication
* PPE Score Display Integration

---

# 🖥️ GPU-Accelerated AI Detection

SiteSentinel supports NVIDIA GPU acceleration using **PyTorch CUDA**.

The current system detects and uses:

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

When CUDA is available:

```text
Using GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

Otherwise:

```text
Using CPU
```

GPU acceleration significantly improves real-time YOLO inference performance.

---

# 🔊 IoT Safety Alerts

SiteSentinel integrates IoT hardware for real-time safety alerts.

Current components include:

* Raspberry Pi 4
* Buzzer
* LED Matrix Display

The system can communicate PPE information to the IoT system.

Example:

```text
AI Detection
      ↓
PPE Score Calculated
      ↓
HTTP Request
      ↓
Raspberry Pi
      ↓
Buzzer / LED Matrix Response
```

---

# 📟 LED Matrix PPE Display

The system can send PPE scores to an LED matrix connected to the Raspberry Pi.

Example:

```text
PPE Score: 100
```

The score is displayed using scrolling text on the LED matrix.

The scrolling speed can be configured in the Raspberry Pi camera server.

---

# 👷 Worker Management

The manager can manage workers through the SiteSentinel dashboard.

Current worker management features include:

* Manager Login
* Manager Signup
* Add New Worker
* Worker ID Management
* Worker Name Management
* Upload Worker Face Image
* Store Worker Information
* Face Embedding Generation
* Worker Status Tracking

Worker information is stored in the database and used for face recognition.

---

# 🧠 Face Recognition Workflow

When a manager adds a worker:

```text
Manager Adds Worker
        ↓
Worker Information Stored
        ↓
Worker Image Uploaded
        ↓
Face Embedding Generated
        ↓
Embedding Stored in Database
        ↓
Worker Ready for Recognition
```

During live monitoring:

```text
Camera Frame
        ↓
Face Detected
        ↓
Face Embedding Generated
        ↓
Compare with Registered Faces
        ↓
Best Match Selected
        ↓
Worker Identified
```

---

# 🗄️ Database

SiteSentinel uses **Turso Database** for cloud-based data storage.

The database manages:

* Manager Accounts
* Worker Information
* Worker IDs
* Worker Face Metadata
* Face Embeddings
* Attendance Information
* Worker Status
* PPE Score Data

### Main Data Components

#### `users`

Stores manager authentication information.

#### `workers`

Stores worker details such as:

* Worker ID
* First Name
* Last Name
* Worker Status
* Face Embedding

#### `attendance_log`

Stores worker attendance activity.

#### Worker Face Data

Stores face-related information required for worker identification.

---

# ☁️ Cloud Storage

Worker images are stored using **Cloudinary**.

Workflow:

```text
Worker Image
      ↓
Cloudinary Upload
      ↓
Image URL Stored
      ↓
Face Embedding Generated
      ↓
Worker Registered
```

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Raspberry Pi 4     │
                    │                      │
                    │   Pi Camera Server   │
                    └──────────┬───────────┘
                               │
                         MJPEG Stream
                               │
                               ▼
┌─────────────────────────────────────────────────────┐
│                  Main AI System                     │
│                                                     │
│              YOLOv8 Object Detection               │
│                         +                           │
│                 Face Recognition                   │
│                                                     │
└───────────────┬───────────────────┬─────────────────┘
                │                   │
                ▼                   ▼
        PPE Detection        Worker Identification
                │                   │
                └─────────┬─────────┘
                          │
                          ▼
                  Attendance System
                          │
                          ▼
                     Turso Database
                          │
                          ▼
                  SiteSentinel Dashboard

                          │
                          ▼

                      IoT System
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
              Buzzer           LED Matrix
```

---

# 🛠️ Tech Stack

## Frontend

* HTML
* CSS
* JavaScript
* Tailwind CSS

## Backend

* Python
* Flask

## AI & Computer Vision

* YOLOv8
* OpenCV
* PyTorch
* CUDA
* Face Recognition
* Dlib

## Database

* Turso
* LibSQL

## Cloud Storage

* Cloudinary

## IoT & Hardware

* Raspberry Pi 4
* Raspberry Pi Camera
* Buzzer
* LED Matrix

---

# 📁 Project Structure

```text
SiteSentinel
│
├── backend
│   ├── main.py
│   ├── database.py
│   ├── init_db.py
│   └── face_utils.py
│
├── frontend
│   ├── assets
│   ├── css
│   ├── js
│   └── pages
│       ├── index.html
│       ├── login.html
│       └── dashboard.html
│
├── interface
│   ├── webcam_detection.py
│   ├── pi_controller.py
│   ├── stream_reader.py
│   └── test_images
│
├── models
│   ├── best.pt
│   └── last.pt
│
├── datasets
│
├── .env
├── .gitignore
├── raspberry_Pi_Code.txt   ← Upload this file to your Raspberry Pi
├── requirements.txt
└── README.md
```

---

# ⚙️ Requirements

## Recommended Python Version

```text
Python 3.11
```

Python 3.11 is recommended because some versions of the following libraries may have compatibility issues with newer Python versions:

* dlib
* face_recognition

---

# 📦 Installation

Clone the repository:

```bash
git clone <repository-url>
cd SiteSentinel
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file.

Example:

```env
TURSO_DATABASE_URL=
TURSO_AUTH_TOKEN=

CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=
```

Never upload your `.env` file to GitHub.

---

# 🗄️ Initialize the Database

Run:

```bash
python backend/init_db.py
```

---

# ▶️ Run the Backend

```bash
cd backend
python main.py
```

The backend will start at:

```text
http://127.0.0.1:5000
```

---

# 📷 Raspberry Pi Setup & Camera Server

The file **`raspberry_Pi_Code.txt`** in the root of this repository contains the complete Raspberry Pi server script. You need to copy this file to your Raspberry Pi and run it there.

> ⚠️ **Do NOT run this script on your main computer.** It uses hardware-specific libraries (`picamera2`, `RPi.GPIO`, `luma.led_matrix`) that are only available on a Raspberry Pi.

---

## 🔌 Hardware Requirements

| Component | Details |
|---|---|
| Raspberry Pi | Model 4 (recommended) |
| Pi Camera Module | Connected via CSI ribbon cable |
| Green LED | GPIO pin **17** (BCM) |
| Red LED | GPIO pin **27** (BCM) |
| Buzzer | GPIO pin **18** (BCM) |
| LED Matrix (MAX7219) | Connected via **SPI0** (CE0) |

---

## 🛠️ Pi-Side Dependencies

Install the following on your Raspberry Pi before running the server:

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install picamera2 (comes pre-installed on Raspberry Pi OS Bullseye+)
# If missing:
sudo apt-get install -y python3-picamera2

# Install RPi.GPIO
sudo apt-get install -y python3-rpi.gpio

# Install luma LED matrix driver
pip3 install luma.led_matrix
```

> **Note:** `picamera2` requires **Raspberry Pi OS Bullseye** or later. Make sure the camera interface is enabled via `sudo raspi-config` → Interface Options → Camera.

---

## 🚀 Deploying the Script

**Step 1 — Copy the file to your Pi**

From your main computer, use `scp` to transfer the file:

```bash
scp raspberry_Pi_Code.txt pi@<PI_IP>:/home/pi/Pi_Camera.py
```

Or manually copy the contents of `raspberry_Pi_Code.txt` into a new file called `Pi_Camera.py` on your Pi.

**Step 2 — SSH into your Raspberry Pi**

```bash
ssh pi@<PI_IP>
```

**Step 3 — Run the server**

```bash
python3 Pi_Camera.py
```

On successful startup you will see:

```text
==================================================
SiteSentinel Pi Server
Stream  → http://<PI_IP>:8080/stream
Health  → http://<PI_IP>:8080/health
GPIO    → /checkin /checkout /on /off /ppe?score=N
==================================================
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/stream` | GET | MJPEG live camera stream |
| `/health` | GET | Server health check |
| `/checkin` | GET | Green LED ON for 3 s (worker check-in) |
| `/checkout` | GET | Red LED ON for 3 s (worker check-out) |
| `/on` | GET | Buzzer ON |
| `/off` | GET | Buzzer OFF |
| `/ppe?score=N` | GET | Scroll PPE score on LED matrix |

Example URLs (replace `<PI_IP>` with your Pi's local IP address):

```text
http://<PI_IP>:8080/stream
http://<PI_IP>:8080/health
http://<PI_IP>:8080/ppe?score=100
```

**Tip:** Find your Pi's IP address by running `hostname -I` on the Pi.

---

## ⚙️ Configuration

The following constants at the top of the script can be adjusted to match your wiring:

```python
GREEN_PIN  = 17   # BCM pin for green LED
RED_PIN    = 27   # BCM pin for red LED
BUZZER_PIN = 18   # BCM pin for buzzer
PORT       = 8080 # HTTP server port
```

The LED matrix is expected on **SPI0, CE0**. Change `spi(port=0, device=0, ...)` if you wired it differently.

---

# 🤖 Run AI Detection and Smart Attendance

On the main computer:

```bash
python interface/webcam_detection.py
```

The system will:

1. Connect to the Raspberry Pi camera stream.
2. Load the YOLOv8 model.
3. Use the NVIDIA GPU if CUDA is available.
4. Load registered worker face embeddings.
5. Detect objects and PPE.
6. Identify registered workers.
7. Perform automatic check-in/check-out.
8. Calculate and record PPE information.
9. Update the backend.
10. Send relevant information to the IoT system.

---

# 🔒 Security Notes

Never expose:

```text
.env
```

Do not commit the following:

* Turso Auth Token
* Cloudinary API Secret
* Database Credentials
* Raspberry Pi IP configuration if private
* API Keys

Ensure `.env` is included in `.gitignore`.

---

# 🚧 Current Development Status

SiteSentinel is currently under active development.

The major integrated components include:

* ✅ Raspberry Pi Camera Streaming
* ✅ YOLOv8 PPE Detection
* ✅ NVIDIA GPU Acceleration
* ✅ Face Recognition
* ✅ Automatic Check-In
* ✅ Automatic Check-Out
* ✅ Worker Identification
* ✅ Turso Database Integration
* ✅ Cloudinary Worker Image Storage
* ✅ Raspberry Pi HTTP API
* ✅ Buzzer Integration
* ✅ LED Matrix PPE Score Display

The PPE detection and scoring logic is currently being refined to ensure accurate association between valid person detection, detected safety equipment, and worker attendance events.

---

# 👥 Team

### Team Members

* **Souvagya Karmakar**
* **Sushmita Roy**
* **Anirban Pal**
* **Ronit Mishra**

---

# 📜 License

**Proprietary Software – All Rights Reserved**

SiteSentinel is the intellectual property of its authors.

Unauthorized copying, modification, distribution, or commercial use is prohibited without permission from the project authors.
