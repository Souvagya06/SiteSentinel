"""
PiStream — always keeps the LATEST frame from the MJPEG stream.
Runs a background thread that continuously reads and discards old frames,
so the main loop always gets a fresh frame with zero buffer delay.
"""
import cv2
import threading
import time


class PiStream:
    def __init__(self, ip: str, port: int = 8080):
        self.url    = f"http://{ip}:{port}/stream"
        self._frame = None
        self._ok    = False
        self._lock  = threading.Lock()
        self._stop  = False
        self._cap   = None
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        # Wait up to 5s for first frame
        for _ in range(50):
            if self._frame is not None:
                break
            time.sleep(0.1)
        print(f"PiStream connected: {self.url}")

    def _reader(self):
        """Background thread — always drains the stream buffer."""
        while not self._stop:
            try:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap = cap
                while not self._stop:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    with self._lock:
                        self._frame = frame
                        self._ok    = True
                cap.release()
            except Exception as e:
                print(f"PiStream error: {e}")
            if not self._stop:
                print("PiStream reconnecting...")
                time.sleep(1)

    def read(self):
        """Returns (success, latest_frame) — always the newest frame."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def close(self):
        self._stop = True
        if self._cap:
            try: self._cap.release()
            except: pass