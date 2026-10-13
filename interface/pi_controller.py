"""
PiController — sends HTTP commands and queries button/verification state from the Raspberry Pi.
Controls Check-in Green LED, Check-out Red LED, Safety Buzzer, and 8x8 LED Matrix.
All calls are non-blocking with real-time status polling.
"""
import sys
import requests
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Default durations per score (seconds)
# 100% has 8 chars ("PPE:100 "), takes ~5.8s for a single scroll pass.
# 50%  has 7 chars ("PPE:50 "),  takes ~4.5s for a single scroll pass.
# 0%   has 6 chars ("PPE:0 "),   takes ~3.5s for a single scroll pass.
DEFAULT_SCORE_DURATIONS = {
    100: 6.0,
    50: 4.5,
    0: 3.5,
}


class PiController:
    def __init__(self, ip: str, port: int = 8080):
        self.raw_ip = str(ip or "").strip()
        self.port = port
        self.is_simulated = self.raw_ip in ["", "0", "1", "local", "webcam", "usb"] or self.raw_ip.isdigit()
        self.base = f"http://{self.raw_ip}:{self.port}" if not self.is_simulated else "simulated"
        
        self._button_pressed = False
        self._verification_state = "IDLE"
        self._last_poll_time = 0.0
        self._last_buzzer_time = 0.0
        self._last_matrix_time = 0.0
        self._last_actuation_time = 0.0
        self._lock = threading.Lock()
        self._stop = False
        
        print(f"[PiController] Controller initialized: {'Simulated GPIO' if self.is_simulated else self.base}")

        if not self.is_simulated:
            self._session = requests.Session()
            self._poller_thread = threading.Thread(target=self._status_poller, daemon=True)
            self._poller_thread.start()

    def _status_poller(self):
        """Continuously polls Raspberry Pi /status endpoint at ~5Hz to keep button state responsive without socket overload."""
        while not self._stop:
            try:
                r = self._session.get(f"{self.base}/status", timeout=1.2)
                if r.status_code == 200:
                    data = r.json()
                    with self._lock:
                        self._button_pressed = bool(data.get("button_pressed", False))
                        self._verification_state = str(data.get("state", "IDLE"))
                        self._last_poll_time = time.time()
            except Exception:
                pass
            time.sleep(0.18)

    def is_button_pressed(self) -> bool:
        """Returns True if physical push button is held down or gate verification is ACTIVE."""
        if self.is_simulated:
            return True  # Always active in simulation mode
        with self._lock:
            # If we haven't received a poll update in 2.5s, assume released
            if time.time() - self._last_poll_time > 2.5:
                return False
            return self._button_pressed or (self._verification_state == "ACTIVE")

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
        now = time.time()
        if now - self._last_actuation_time < 2.0:
            return
        self._last_actuation_time = now
        print("[PiController] 🟢 CHECK-IN: Turning ON GREEN LED (GPIO 17) for 3s")
        self._fire("/checkin")

    def checkout(self):
        """Red LED ON for 3s (Worker Leaving / Off-Site)."""
        now = time.time()
        if now - self._last_actuation_time < 2.0:
            return
        self._last_actuation_time = now
        print("[PiController] 🔴 CHECK-OUT: Turning ON RED LED (GPIO 27) for 3s")
        self._fire("/checkout")

    # ── Buzzer commands ───────────────────────────────
    def buzzer_on(self):
        now = time.time()
        if now - self._last_buzzer_time < 0.6:
            return
        self._last_buzzer_time = now
        self._fire("/on")

    def buzzer_off(self):
        self._fire("/off")

    # ── LED Matrix ────────────────────────────────────
    def show_score(self, score: int, duration: float | None = None, once: bool = True):
        """
        Show PPE score on 8x8 LED matrix.
        score: 0, 50, or 100
        duration: duration in seconds (if None, uses score-specific duration: 100->6.0s, 50->4.5s, 0->3.5s).
        once: if True, scrolls across the matrix exactly once without repeating.
        """
        now = time.time()
        if now - self._last_matrix_time < 2.5:
            return
        self._last_matrix_time = now
        actual_duration = duration if duration is not None else DEFAULT_SCORE_DURATIONS.get(score, 5.0)
        print(f"[PiController] 📊 MATRIX: Displaying PPE Score {score}% for {actual_duration}s (once={once})")
        self._fire("/ppe", params={"score": score, "duration": actual_duration, "once": "true" if once else "false"})

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
