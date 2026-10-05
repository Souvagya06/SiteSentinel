"""
SiteSentinel — High Performance Adaptive Video Stream Reader
Supports:
1. Raspberry Pi MJPEG HTTP Stream (Direct fast JPEG chunk parser with auto-reconnect)
2. Local Webcam / USB Camera (Index 0, 1, or 'webcam' / 'local')
3. Seamless fallback to local camera if Raspberry Pi is unreachable, keeping detection alive
"""
import cv2
import threading
import time
import requests
import numpy as np


class PiStream:
    def __init__(self, ip: str, port: int = 8080, fallback_to_local: bool = True):
        self.raw_ip = str(ip or "").strip()
        self.port = port
        self.fallback_to_local = fallback_to_local
        
        self.is_local = self.raw_ip in ["", "0", "1", "local", "webcam", "usb"] or self.raw_ip.isdigit()
        self.cam_index = int(self.raw_ip) if self.raw_ip.isdigit() else 0
        self.url = f"http://{self.raw_ip}:{self.port}/stream" if not self.is_local else f"webcam_{self.cam_index}"
        
        self._frame = None
        self._ok = False
        self._lock = threading.Lock()
        self._stop = False
        self._cap = None
        self.using_local_fallback = False
        
        print(f"[PiStream] Initializing stream source: {'Local Webcam ' + str(self.cam_index) if self.is_local else self.url}")
        
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
        # Wait up to 3 seconds for initial frame
        for _ in range(30):
            if self._frame is not None:
                break
            time.sleep(0.1)

    def _read_mjpeg_http(self):
        """High-speed zero-latency MJPEG byte stream parser via HTTP."""
        session = requests.Session()
        resp = session.get(self.url, stream=True, timeout=3.5)
        if resp.status_code != 200:
            raise ConnectionError(f"HTTP {resp.status_code}")
        
        print(f"[PiStream] Connected to Raspberry Pi MJPEG stream at {self.url}")
        self.using_local_fallback = False
        bytes_buffer = b""
        
        for chunk in resp.iter_content(chunk_size=16384):
            if self._stop:
                break
            if not chunk:
                continue
            bytes_buffer += chunk
            
            # Locate JPEG frame boundaries (Start: 0xFFD8, End: 0xFFD9)
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')
            if a != -1 and b != -1:
                if b > a:
                    jpg = bytes_buffer[a:b+2]
                    bytes_buffer = bytes_buffer[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._lock:
                            self._frame = frame
                            self._ok = True
                else:
                    bytes_buffer = bytes_buffer[a:]

    def _read_cv_stream(self):
        """Fallback OpenCV VideoCapture reader."""
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise ConnectionError(f"OpenCV could not open stream {self.url}")
        
        self._cap = cap
        self.using_local_fallback = False
        while not self._stop:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            with self._lock:
                self._frame = frame
                self._ok = True
        cap.release()

    def _read_local_webcam(self, idx=0):
        """Streams from local PC / USB webcam."""
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"[PiStream] Error: Could not open local webcam index {idx}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap = cap
        print(f"[PiStream] Capturing from local webcam index {idx}")
        
        consecutive_errors = 0
        while not self._stop:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_errors += 1
                time.sleep(0.05)
                if consecutive_errors > 40:
                    break
                continue
            consecutive_errors = 0
            with self._lock:
                self._frame = frame
                self._ok = True
        cap.release()

    def _worker(self):
        """Main background ingestion worker."""
        if self.is_local:
            while not self._stop:
                self._read_local_webcam(self.cam_index)
                if not self._stop:
                    time.sleep(1)
            return

        while not self._stop:
            try:
                # Try direct HTTP MJPEG reader with short timeout
                self._read_mjpeg_http()
            except Exception as e:
                if self._stop:
                    break
                print(f"[PiStream] Warning: Raspberry Pi at {self.url} is offline or unreachable ({e})")
                
                if self.fallback_to_local:
                    print("[PiStream] Fallback: Capturing from local webcam so live stream & detection stay active...")
                    self.using_local_fallback = True
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        self._cap = cap
                        start_fb = time.time()
                        while not self._stop and (time.time() - start_fb < 10.0):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                with self._lock:
                                    self._frame = frame
                                    self._ok = True
                            time.sleep(0.03)
                        cap.release()
                    else:
                        time.sleep(2)
                else:
                    time.sleep(2)

    def read(self):
        """Returns (success, latest_frame_bgr)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def close(self):
        self._stop = True
        if self._cap:
            try:
                self._cap.release()
            except:
                pass