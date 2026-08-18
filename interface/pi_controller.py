"""
PiController — sends HTTP commands to the Raspberry Pi GPIO server.
All calls are fire-and-forget with short timeouts.
"""
import requests
import threading


class PiController:
    def __init__(self, ip: str, port: int = 8080):
        self.base = f"http://{ip}:{port}"
        print(f"PiController → {self.base}")

    def _get(self, endpoint: str, params: dict = None):  #type: ignore
        """Internal: non-blocking HTTP GET to Pi."""
        try:
            requests.get(
                f"{self.base}{endpoint}",
                params=params,
                timeout=2
            )
        except Exception as e:
            print(f"Pi command {endpoint} failed: {e}")

    def _fire(self, endpoint: str, params: dict = None):  #type: ignore
        """Send command in background thread — never blocks main loop."""
        threading.Thread(
            target=self._get,
            args=(endpoint, params),
            daemon=True
        ).start()

    # ── LED commands ──────────────────────────────────
    def checkin(self):
        """Green LED ON for 3s."""
        self._fire("/checkin")

    def checkout(self):
        """Red LED ON for 3s."""
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
        self._fire("/ppe", params={"score": score})

    # ── Health check ──────────────────────────────────
    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.base}/health", timeout=2)
            return r.ok
        except:
            return False
