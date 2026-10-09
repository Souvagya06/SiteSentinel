"""
PiController — sends HTTP commands and queries button/verification state from the Raspberry Pi.
Controls Check-in Green LED, Check-out Red LED, Safety Buzzer, and 8x8 LED Matrix.
All calls are non-blocking with real-time status polling.
"""
import requests
import threading
import time


class PiController:
    def __init__(self, ip: str, port: int = 8080):
        self.raw_ip = str(ip or "").strip()
        self.port = port
        self.is_simulated = self.raw_ip in ["", "0", "1", "local", "webcam", "usb"] or self.raw_ip.isdigit()
        self.base = f"http://{self.raw_ip}:{self.port}" if not self.is_simulated else "simulated"
        
        self._button_pressed = False
        self._verification_state = "IDLE"
        self._last_poll_time = 0.0
        self._lock = threading.Lock()
        self._stop = False
        
        print(f"[PiController] Controller initialized: {'Simulated GPIO' if self.is_simulated else self.base}")

        if not self.is_simulated:
            self._session = requests.Session()
            self._poller_thread = threading.Thread(target=self._status_poller, daemon=True)
            self._poller_thread.start()

    def _status_poller(self):
        """Continuously polls Raspberry Pi /status endpoint at ~15Hz for zero-latency button response."""
        while not self._stop:
            try:
                r = self._session.get(f"{self.base}/status", timeout=0.8)
                if r.status_code == 200:
                    data = r.json()
                    with self._lock:
                        self._button_pressed = bool(data.get("button_pressed", False))
                        self._verification_state = str(data.get("state", "IDLE"))
                        self._last_poll_time = time.time()
            except Exception:
                pass
            time.sleep(0.06)

    def is_button_pressed(self) -> bool:
        """Returns True if physical push button is currently held down."""
        if self.is_simulated:
            return True  # Always active in simulation mode
        with self._lock:
            # If we haven't received a poll update in 2.5s, assume released
            if time.time() - self._last_poll_time > 2.5:
                return False
            return self._button_pressed

    def get_verification_state(self) -> str:
        """Returns current verification state ('IDLE', 'ACTIVE', 'COMPLETED')."""
        if self.is_simulated:
            return "ACTIVE"
        with self._lock:
            return self._verification_state

    def set_state(self, state: str):
        """Notifies Pi of state changes."""
        self._fire("/state", params={"set": state})

    def _get(self, endpoint: str, params: dict | None = None):
        """Internal: non-blocking HTTP GET to Pi."""
        if self.is_simulated:
            print(f"[PiController] [Simulated] Executed command: {endpoint} {params or ''}")
            return

        try:
            r = requests.get(
                f"{self.base}{endpoint}",
                params=params,
                timeout=2.5
            )
            if r.ok:
                print(f"[PiController] Pi hardware response [{endpoint}]: {r.text.strip()}")
            else:
                print(f"[PiController] Pi responded with HTTP {r.status_code} on {endpoint}")
        except Exception as e:
            print(f"[PiController] Pi hardware command {endpoint} failed (Pi unreachable): {e}")

    def _fire(self, endpoint: str, params: dict | None = None):
        """Send command in background thread — never blocks main loop."""
        threading.Thread(
            target=self._get,
            args=(endpoint, params),
            daemon=True
        ).start()

    # ── LED commands ──────────────────────────────────
    def checkin(self):
        """Green LED ON for 3s (Access Granted / On-Site)."""
        print("[PiController] 🟢 CHECK-IN: Turning ON GREEN LED (GPIO 17) for 3s")
        self._fire("/checkin")

    def checkout(self):
        """Red LED ON for 3s (Worker Leaving / Off-Site)."""
        print("[PiController] 🔴 CHECK-OUT: Turning ON RED LED (GPIO 27) for 3s")
        self._fire("/checkout")

    # ── Buzzer commands ───────────────────────────────
    def buzzer_on(self):
        self._fire("/on")

    def buzzer_off(self):
        self._fire("/off")

    # ── LED Matrix ────────────────────────────────────
    def show_score(self, score: int):
        """
        Show PPE score on 8x8 LED matrix for 4 seconds.
        score: 0, 50, or 100
        """
        print(f"[PiController] 📊 MATRIX: Displaying PPE Score {score}%")
        self._fire("/ppe", params={"score": score})

    # ── Health check ──────────────────────────────────
    def ping(self) -> bool:
        if self.is_simulated:
            return True
        try:
            r = requests.get(f"{self.base}/health", timeout=2)
            return r.ok
        except:
            return False

    def close(self):
        self._stop = True
