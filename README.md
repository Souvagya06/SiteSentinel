# 🦺 SiteSentinel

AI-Powered Construction Site Safety Monitoring & Smart Attendance System

SiteSentinel is an AI + IoT based construction site monitoring platform that ensures worker safety through real-time PPE detection while simultaneously automating worker attendance using face recognition.

---

## 🚀 Features

### PPE Detection

* Detects Hardhat
* Detects Safety Vest
* Detects Person
* Detects Vehicle
* Detects Machinery
* Detects Safety Cone

### PPE Violation Detection

* No Hardhat Detection
* No Safety Vest Detection
* Real-time Violation Alerts
* ESP32 Buzzer Integration

### Smart Attendance System

* Face Recognition Based Worker Identification
* Automatic Check-In
* Automatic Check-Out
* Attendance Log Generation
* Real-Time Worker Tracking

### Worker Management

* Manager Login & Signup
* Add New Workers
* Upload Worker Images
* Store Worker Records in Database

### Cloud Storage

* Worker Images Stored on Cloudinary

### Database

* User Authentication
* Worker Records
* Attendance Logs
* Face Recognition Metadata

---

# 🏗️ Tech Stack

### Frontend

* HTML
* CSS
* JavaScript
* Tailwind CSS

### Backend

* Python
* Flask

### AI & Computer Vision

* YOLOv8
* OpenCV
* PyTorch
* Face Recognition
* Dlib

### Database

* Turso Database

### Cloud Storage

* Cloudinary

### IoT

* ESP32

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
│   └── test_images
│
├── models
│   └── best.pt
│
├── datasets
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ⚙️ Requirements

### Python Version

This project has been tested on:

```text
Python 3.11.9
```

⚠️ Python 3.13 may cause compatibility issues with:

* dlib
* face_recognition

Python 3.11.9 is strongly recommended.

---

# 🖥️ Windows Setup

Install:

```text
Visual Studio Build Tools 2022
```

Required workload:

```text
Desktop Development with C++
```

This is required for:

* dlib
* face_recognition

---

# 📦 Installation

Clone Repository

```bash
git clone <repository-url>
cd SiteSentinel
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

```env
ESP32_IP=

TURSO_DATABASE_URL=
TURSO_AUTH_TOKEN=

CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=
```

---

# 🗄️ Initialize Database

```bash
python backend/init_db.py
```

---

# ▶️ Run Backend

```bash
python backend/main.py
```

Backend will start on:

```text
http://127.0.0.1:5000
```

---

# 🎥 Run PPE Detection & Attendance System

```bash
python interface/webcam_detection.py
```

---

# 📊 Database Tables

### users

Stores manager accounts.

### workers

Stores worker details and uploaded image information.

### attendance_log

Stores worker attendance records.

### worker_images

Stores worker face recognition metadata.

---

# 🔒 Security Notes

Never upload:

```text
.env
```

Never expose:

* Turso Auth Token
* Cloudinary API Secret
* ESP32 Credentials

---

# 👥 Team

Developed as part of an AI-powered construction site safety and workforce management solution.

---

## License

Proprietary Software – All Rights Reserved.

SiteSentinel is the intellectual property of its authors.
Unauthorized copying, modification, distribution, or commercial use is prohibited.
