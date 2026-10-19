#!/usr/bin/env python3
"""
SiteSentinel — All-in-One Raspberry Pi Camera, GPIO & Firebase Cloud Server
Handles:
- MJPEG video streaming from Pi Camera (/stream)
- Push-and-Hold physical verification button (GPIO 22)
- State tracking (IDLE, ACTIVE, COMPLETED)
- Independent LED indicators (Green: Check-in, Red: Check-out)
- Independent Safety Buzzer (GPIO 18)
- MAX7219 8x8 LED Matrix PPE score display (SPI0)
- Local HTTP REST API (port 8080)
- OUTBOUND Firebase Realtime Database Bridge (Live heartbeat & cloud commands)

Run on Raspberry Pi:
    python3 Pi_Camera.py
"""

import io
import os
import sys
import time
import json
import socket
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

# ── Cloud Bridge Configuration ────────────────────────
FIREBASE_DATABASE_URL = os.getenv(
    "FIREBASE_DATABASE_URL",
    "https://sitesentinel-50bfc-default-rtdb.asia-southeast1.firebasedatabase.app/"
).strip()
DEVICE_ID = os.getenv("FIREBASE_DEVICE_ID", "pisite_01").strip()
CREDENTIALS_FILE = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")

# ── GPIO Initialization ───────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("WARNING: RPi.GPIO not available (Running in simulation mode)")

# ── Pi Camera Initialization ──────────────────────────
try:
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput
    PICAMERA_AVAILABLE = True
except Exception as e:
    PICAMERA_AVAILABLE = False
    print(f"ERROR: picamera2 not available ({e})")

# ── LED Matrix Initialization ─────────────────────────
MATRIX_INIT_ERROR = None
try:
    from luma.led_matrix.device import max7219
    from luma.core.interface.serial import spi, noop
    from luma.core.render import canvas
    from luma.core.legacy import text
    from luma.core.legacy.font import proportional, CP437_FONT
    MATRIX_AVAILABLE = True
    print("LED Matrix library loaded")
except Exception as e:
    MATRIX_AVAILABLE = False
    MATRIX_INIT_ERROR = f"luma.led_matrix library unavailable: {e}"
    print(f"WARNING: luma.led_matrix not available ({e}) — matrix disabled")

# ── GPIO Pin Configuration ────────────────────────────
GREEN_PIN  = 17   # Check-in LED (Physical Pin 11)
RED_PIN    = 27   # Check-out LED (Physical Pin 13)
BUZZER_PIN = 18   # Safety Buzzer (Physical Pin 12)
BUTTON_PIN = 22   # Verification Push Button (Physical Pin 15, GND Pin 14)

# ── Global Hardware & Verification State ──────────────
gpio_lock          = threading.Lock()
button_pressed     = False
verification_state = "IDLE"  # "IDLE", "ACTIVE", "COMPLETED"
press_start_time   = 0.0
last_state_change  = time.time()
server_start_time  = time.time()

green_off_at       = 0.0
red_off_at         = 0.0
matrix_off_at      = 0.0
buzzer_state       = False

if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(GREEN_PIN,  GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(RED_PIN,    GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
    # Button wired between GPIO 22 and GND -> Internal Pull-Up Resistor
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    print(f"GPIO initialized — Green={GREEN_PIN}, Red={RED_PIN}, Buzzer={BUZZER_PIN}, Button={BUTTON_PIN}")

# ── MAX7219 Hardware Device Setup ─────────────────────
matrix_device = None
matrix_lock   = threading.Lock()
matrix_token  = 0
if MATRIX_AVAILABLE:
    try:
        serial        = spi(port=0, device=0, gpio=noop())
        matrix_device = max7219(serial, cascaded=1,
                                 block_orientation=90,
                                 rotate=0)
        matrix_device.contrast(128)  # High contrast: bright and crisp
        matrix_device.clear()
        print("LED Matrix initialized on SPI0 (Contrast 128)")
    except Exception as e:
        MATRIX_INIT_ERROR = f"SPI/MAX7219 init failed: {e}. Check if SPI is enabled in raspi-config"
        print(f"Matrix init failed: {e}")
        matrix_device = None

# ── Direct Actuation Helpers (Thread-Safe) ─────────────
def trigger_checkin():
    global green_off_at, verification_state
    with gpio_lock:
        if GPIO_AVAILABLE:
            GPIO.output(GREEN_PIN, GPIO.HIGH)
            GPIO.output(RED_PIN,   GPIO.LOW)
        green_off_at = time.time() + 3.0
        verification_state = "COMPLETED"
    print("[Actuate] 🟢 Check-in granted -> Green LED ON for 3s")

def trigger_checkout():
    global red_off_at, verification_state
    with gpio_lock:
        if GPIO_AVAILABLE:
            GPIO.output(RED_PIN,   GPIO.HIGH)
            GPIO.output(GREEN_PIN, GPIO.LOW)
        red_off_at = time.time() + 3.0
        verification_state = "COMPLETED"
    print("[Actuate] 🔴 Check-out confirmed -> Red LED ON for 3s")

def trigger_buzzer(duration=1.0):
    global buzzer_state
    with gpio_lock:
        buzzer_state = True
        if GPIO_AVAILABLE:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
    print(f"[Actuate] 🔊 Buzzer ON for {duration}s")
    time.sleep(duration)
    with gpio_lock:
        buzzer_state = False
        if GPIO_AVAILABLE:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
    print("[Actuate] 🔇 Buzzer OFF")

def trigger_matrix_score(score=100, duration=None, show_once=True):
    global matrix_token
    if not matrix_device or not MATRIX_AVAILABLE:
        err = MATRIX_INIT_ERROR or "Matrix hardware not initialized"
        print(f"[Actuate] Matrix not available ({err}) — score={score}")
        return False, err

    try:
        s_int = int(score)
    except Exception:
        s_int = 100

    score_durations = {100: 6.0, 50: 5.0, 0: 4.0}
    dur = duration or score_durations.get(s_int, 5.0)

    try:
        matrix_token += 1
        cur_token = matrix_token

        def scroll_score(s, token, dur):
            with matrix_lock:
                try:
                    matrix_device.contrast(128)  # Crisp & bright
                    # For an 8x8 matrix, scrolling " 100% " is vastly cleaner and immediately readable
                    msg = f" {s}% " if s > 0 else " 0% FAIL "
                    show_until = time.time() + dur
                    max_pos = len(msg) * 8
                    pos = 8
                    while time.time() < show_until and matrix_token == token:
                        with canvas(matrix_device) as draw:
                            text(draw, (pos, 0), msg,
                                 fill="white",
                                 font=proportional(CP437_FONT))
                        pos -= 1
                        if pos < -max_pos:
                            pos = 8
                        time.sleep(0.07)
                    matrix_device.clear()
                except Exception as err:
                    print(f"[Matrix Worker] {err}")

        threading.Thread(
            target=scroll_score,
            args=(s_int, cur_token, dur),
            daemon=True
        ).start()
        print(f"[Actuate] 📊 Matrix scrolling PPE score: {s_int}%")
        return True, f"Displaying {s_int}% on matrix"
    except Exception as e:
        print(f"[Actuate] Matrix draw error: {e}")
        return False, str(e)
        print(f"[Actuate] Matrix draw error: {e}")

def trigger_reset():
    global green_off_at, red_off_at, buzzer_state, verification_state
    with gpio_lock:
        verification_state = "IDLE"
        green_off_at = 0.0
        red_off_at = 0.0
        buzzer_state = False
        if GPIO_AVAILABLE:
            GPIO.output(GREEN_PIN,  GPIO.LOW)
            GPIO.output(RED_PIN,    GPIO.LOW)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        if matrix_device:
            try:
                matrix_device.clear()
            except Exception:
                pass
    print("[Actuate] Hardware & Verification State RESET to IDLE")

# ── Button Debounce & State Monitor Loop ──────────────
def button_monitor_loop():
    global button_pressed, verification_state, press_start_time, last_state_change

    stable_state = 1
    sample_history = [1] * 8
    last_pressed_time = 0.0

    while True:
        if GPIO_AVAILABLE:
            try:
                raw = GPIO.input(BUTTON_PIN)
            except Exception:
                raw = 1
        else:
            raw = 1

        sample_history.pop(0)
        sample_history.append(raw)

        if all(s == 0 for s in sample_history):
            new_state = 0
        elif all(s == 1 for s in sample_history):
            new_state = 1
        else:
            new_state = stable_state

        now = time.time()
        if new_state == 0:
            last_pressed_time = now
            if not button_pressed:
                with gpio_lock:
                    button_pressed = True
                    press_start_time = now
                    last_state_change = now
                    if verification_state != "ACTIVE":
                        verification_state = "ACTIVE"
                        print("[Button] -> PRESSED: Verification state changed to ACTIVE")
        else:
            if button_pressed and (now - last_pressed_time > 0.8):
                with gpio_lock:
                    button_pressed = False
                    press_start_time = 0.0
                    last_state_change = now
                    if verification_state == "ACTIVE":
                        verification_state = "IDLE"
                        print("[Button] -> RELEASED: Verification cancelled, return to IDLE")
                    else:
                        print("[Button] -> RELEASED")

        stable_state = new_state
        time.sleep(0.015)

threading.Thread(target=button_monitor_loop, daemon=True).start()

# ── Non-blocking GPIO Timers & Auto-Reset ─────────────
def gpio_timer_loop():
    global green_off_at, red_off_at, matrix_off_at, verification_state
    while True:
        now = time.time()
        with gpio_lock:
            if green_off_at and now > green_off_at:
                if GPIO_AVAILABLE:
                    GPIO.output(GREEN_PIN, GPIO.LOW)
                green_off_at = 0.0
                if verification_state == "COMPLETED" and not button_pressed:
                    verification_state = "IDLE"

            if red_off_at and now > red_off_at:
                if GPIO_AVAILABLE:
                    GPIO.output(RED_PIN, GPIO.LOW)
                red_off_at = 0.0
                if verification_state == "COMPLETED" and not button_pressed:
                    verification_state = "IDLE"

            if matrix_off_at and now > matrix_off_at:
                if matrix_device:
                    try:
                        matrix_device.clear()
                    except Exception:
                        pass
                matrix_off_at = 0.0
        time.sleep(0.05)

threading.Thread(target=gpio_timer_loop, daemon=True).start()

# ── Camera Streaming Output Setup ─────────────────────
class StreamOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame     = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

output = StreamOutput()
picam2 = None

if PICAMERA_AVAILABLE:
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        picam2.configure(config)
        picam2.start_recording(JpegEncoder(), FileOutput(output))
        print("Pi Camera started (640x480 @ 30fps)")
    except Exception as e:
        print(f"Error starting Pi Camera: {e}")
        picam2 = None

# ── Cloud Stream Frame Pusher ────────────────────────
CLOUD_STREAM_URL = os.getenv("CLOUD_STREAM_URL", "https://sitesentinel.site/api/pi/frame/push").strip()
ENABLE_CLOUD_STREAM = os.getenv("ENABLE_CLOUD_STREAM", "true").lower() in ["true", "1", "yes"]
cloud_stream_active = True
_pushing = False

def _cloud_stream_worker():
    """Continuously pushes the Pi Camera JPEG frames outbound to the hosted dashboard."""
    global cloud_stream_active, _pushing
    if not ENABLE_CLOUD_STREAM or not CLOUD_STREAM_URL:
        return
    import requests
    session = requests.Session()
    print(f"[Cloud Stream] Fast adaptive pusher active: {CLOUD_STREAM_URL}")
    while True:
        if not cloud_stream_active:
            time.sleep(0.5)
            continue
        try:
            with output.condition:
                output.condition.wait(timeout=0.2)
                frame = output.frame
            if frame is not None and not _pushing:
                _pushing = True
                try:
                    session.post(
                        CLOUD_STREAM_URL,
                        data=frame,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=0.35
                    )
                finally:
                    _pushing = False
            time.sleep(0.12)  # ~8 fps: smooth video with zero bufferbloat
        except Exception:
            _pushing = False
            time.sleep(0.5)

if PICAMERA_AVAILABLE:
    t_cloud_stream = threading.Thread(target=_cloud_stream_worker, daemon=True)
    t_cloud_stream.start()

# ── HTTP Handler (Port 8080) ──────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress normal request logs

    def do_GET(self):
        global cloud_stream_active
        parsed = urlparse(self.path)
        path   = parsed.path
        params = parse_qs(parsed.query)

        if path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    if frame is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
            except Exception:
                pass

        elif path in ["/status", "/button"]:
            now = time.time()
            with gpio_lock:
                status_payload = {
                    "button_pressed": button_pressed,
                    "state": verification_state,
                    "press_duration": round(now - press_start_time, 2) if button_pressed else 0.0,
                    "green": bool(green_off_at > now),
                    "red": bool(red_off_at > now),
                    "buzzer": buzzer_state,
                    "uptime": round(now - server_start_time, 1)
                }
            self._json(status_payload)

        elif path == "/checkin":
            trigger_checkin()
            self._ok("Green ON")

        elif path == "/checkout":
            trigger_checkout()
            self._ok("Red ON")

        elif path == "/on":
            with gpio_lock:
                buzzer_state_val = True
                if GPIO_AVAILABLE:
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
            self._ok("Buzzer ON")

        elif path == "/off":
            with gpio_lock:
                buzzer_state_val = False
                if GPIO_AVAILABLE:
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
            self._ok("Buzzer OFF")

        elif path == "/ppe":
            try:
                score = int(params.get("score", ["0"])[0])
            except Exception:
                score = 0
            ok, msg = trigger_matrix_score(score)
            self._json({
                "ok": ok,
                "score": score,
                "message": msg,
                "matrix_available": MATRIX_AVAILABLE,
                "matrix_initialized": matrix_device is not None,
                "matrix_error": MATRIX_INIT_ERROR
            })

        elif path == "/reset":
            trigger_reset()
            self._ok("RESET OK")

        elif path in ["/stream/stop", "/camera/off"]:
            cloud_stream_active = False
            print("[HTTP] Cloud streaming turned OFF")
            self._ok("Stream stopped")

        elif path in ["/stream/start", "/camera/on"]:
            cloud_stream_active = True
            print("[HTTP] Cloud streaming turned ON")
            self._ok("Stream started")

        elif path == "/health":
            self._json({
                "status": "online",
                "service": "SiteSentinel Pi Server",
                "button_pressed": button_pressed,
                "state": verification_state,
                "cloud_stream": cloud_stream_active,
                "matrix_available": MATRIX_AVAILABLE,
                "matrix_initialized": matrix_device is not None,
                "matrix_error": MATRIX_INIT_ERROR
            })

        else:
            self.send_response(404)
            self.end_headers()

    def _ok(self, msg="OK"):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(msg.encode())
        except Exception:
            pass

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(json.dumps(data).encode())
        except Exception:
            pass

class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    def handle_error(self, request, client_address):
        pass

# ── Firebase Realtime Cloud Bridge (Integrated) ───────
def find_credentials_file():
    candidates = [
        Path(CREDENTIALS_FILE),
        Path(__file__).resolve().parent / CREDENTIALS_FILE,
        Path(__file__).resolve().parent / "firebase-credentials.json",
        Path(__file__).resolve().parent / "serviceAccountKey.json",
        Path.home() / "Site" / "firebase-credentials.json",
        Path.home() / "Site" / "serviceAccountKey.json",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None

def start_firebase_bridge():
    """Starts outbound Firebase Realtime Database bridge in background."""
    try:
        import firebase_admin
        from firebase_admin import credentials, db
    except ImportError:
        print("[Firebase] WARNING: firebase-admin not installed. Skipping cloud bridge.")
        return

    cred_path = find_credentials_file()
    if not cred_path:
        print(f"[Firebase] NOTICE: Credentials file not found ({CREDENTIALS_FILE}). Cloud bridge inactive.")
        return

    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            "databaseURL": FIREBASE_DATABASE_URL
        })
        print(f"[Firebase] Connected to Realtime Cloud! Device: {DEVICE_ID}")
    except Exception as e:
        print(f"[Firebase] Init error: {e}")
        return

    device_ref = db.reference(f"devices/{DEVICE_ID}")
    command_ref = db.reference(f"commands/{DEVICE_ID}")

    # 1. Telemetry loop
    def _telemetry_loop():
        while True:
            now = time.time()
            with gpio_lock:
                payload = {
                    "status": "online",
                    "last_seen": now,
                    "device_id": DEVICE_ID,
                    "camera_server": "online",
                    "button_pressed": button_pressed,
                    "verification_state": verification_state,
                    "green": bool(green_off_at > now),
                    "red": bool(red_off_at > now),
                    "buzzer": buzzer_state,
                    "uptime": round(now - server_start_time, 1)
                }
            try:
                device_ref.update(payload)
            except Exception:
                pass
            time.sleep(3.0)

    # 2. Command execution
    last_cmd_id = None
    cmd_lock = threading.Lock()

    def _execute_cloud_command(cmd_data):
        nonlocal last_cmd_id
        global cloud_stream_active
        if not isinstance(cmd_data, dict):
            return
        cmd_id = cmd_data.get("command_id")
        status = cmd_data.get("status")
        action = (cmd_data.get("action") or "").lower().strip()
        params = cmd_data.get("params") or {}

        if not cmd_id or status != "pending":
            return

        with cmd_lock:
            if cmd_id == last_cmd_id:
                return
            last_cmd_id = cmd_id

        print(f"\n[Firebase Command] Received: '{action}' (ID={cmd_id})")
        try:
            command_ref.update({"status": "acknowledged", "ack_at": time.time()})
        except Exception:
            pass

        # Execute hardware actions directly
        try:
            if action in ["checkin", "green"]:
                trigger_checkin()
            elif action in ["checkout", "red"]:
                trigger_checkout()
            elif action == "buzzer":
                threading.Thread(target=trigger_buzzer, args=(1.0,), daemon=True).start()
            elif action == "ppe":
                score_val = int(params.get("score", 100))
                trigger_matrix_score(score_val)
            elif action in ["stream_start", "camera_on"]:
                cloud_stream_active = True
                print("[Firebase Command] Stream started.")
            elif action in ["stream_stop", "camera_off"]:
                cloud_stream_active = False
                print("[Firebase Command] Stream stopped.")
            elif action == "reset":
                trigger_reset()
            elif action == "state":
                with gpio_lock:
                    verification_state = params.get("set", "IDLE")

            command_ref.update({
                "status": "completed",
                "completed_at": time.time(),
                "result": f"Executed {action} successfully"
            })
            print(f"[Firebase Command] Completed: '{action}'")
        except Exception as err:
            command_ref.update({
                "status": "failed",
                "failed_at": time.time(),
                "error": str(err)
            })

    def _command_stream_listener():
        def _on_change(event):
            try:
                if event.event_type in ["put", "patch"] and event.data:
                    data = event.data
                    if isinstance(data, dict) and data.get("status") == "pending":
                        _execute_cloud_command(data)
                    elif isinstance(data, str) and data == "pending":
                        full_cmd = command_ref.get()
                        if isinstance(full_cmd, dict) and full_cmd.get("status") == "pending":
                            _execute_cloud_command(full_cmd)
            except Exception:
                pass

        while True:
            try:
                command_ref.listen(_on_change)
                while True:
                    time.sleep(15)
            except Exception:
                time.sleep(3)

    def _polling_fallback_loop():
        while True:
            try:
                data = command_ref.get()
                if isinstance(data, dict) and data.get("status") == "pending":
                    _execute_cloud_command(data)
            except Exception:
                pass
            time.sleep(1.5)

    threading.Thread(target=_telemetry_loop, daemon=True).start()
    threading.Thread(target=_command_stream_listener, daemon=True).start()
    threading.Thread(target=_polling_fallback_loop, daemon=True).start()

# ── Main Server Start ─────────────────────────────────
if __name__ == "__main__":
    start_firebase_bridge()

    PORT     = 8080
    server   = ThreadedServer(("0.0.0.0", PORT), Handler)
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "hostname -I"

    print(f"\n{'='*55}")
    print(f"  SiteSentinel Pi Server + Firebase Cloud Active")
    print(f"  Stream   -> http://{local_ip}:{PORT}/stream")
    print(f"  Health   -> http://{local_ip}:{PORT}/health")
    print(f"  Status   -> http://{local_ip}:{PORT}/status")
    print(f"  Firebase -> {FIREBASE_DATABASE_URL} (pisite_01)")
    print(f"  Pins     -> Green={GREEN_PIN}, Red={RED_PIN}, Buzzer={BUZZER_PIN}, Button={BUTTON_PIN}")
    print(f"{'='*55}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Pi server...")
    finally:
        if picam2:
            try:
                picam2.stop_recording()
            except Exception:
                pass
        if GPIO_AVAILABLE:
            try:
                GPIO.output(GREEN_PIN,  GPIO.LOW)
                GPIO.output(RED_PIN,    GPIO.LOW)
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                GPIO.cleanup()
            except Exception:
                pass
        if matrix_device:
            try:
                matrix_device.clear()
            except Exception:
                pass
        print("SiteSentinel Pi Server Stopped safely.")
