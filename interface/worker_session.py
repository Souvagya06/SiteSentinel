"""
WorkerSessionManager — Parallel Event-Driven Single-Worker Evaluation State Machine,
Spatial PPE Attribution, and Gate Occupancy Lock for SiteSentinel.
"""
import time
from collections import deque
import numpy as np


class SessionState:
    IDLE = "IDLE"
    CROWDED = "CROWDED"
    LOCKED = "LOCKED"
    VERIFYING = "VERIFYING"
    CONFIRMING = "CONFIRMING"
    COMPLETED = "COMPLETED"
    COOLDOWN = "COOLDOWN"


class WorkerSessionManager:
    def __init__(self, required_ppe_frames: int = 4, grace_period_sec: float = 2.0, max_verify_sec: float = 4.5):
        self.state = SessionState.IDLE
        self.active_track_id = None
        self.active_worker = None      # Dict of worker details once matched by face worker
        self.active_bbox = None        # [x1, y1, x2, y2]
        
        self.session_started_at: float = 0.0
        self.last_seen_at: float = 0.0
        self.grace_period_sec = grace_period_sec
        self.max_verify_sec = max_verify_sec
        self.required_ppe_frames = required_ppe_frames
        
        # Adaptive PPE smoothing buffer for the active locked worker (size 4)
        self.ppe_observations = deque(maxlen=required_ppe_frames)
        self.finalized_ppe_score = None  # None, 0, 50, or 100
        self.finalized_action = None     # 'DONE' once dispatched
        self.finalized_helmet_id = ""
        self.ocr_reads = deque(maxlen=5) # Consensus buffer for OCR
        self.finalized_at: float = 0.0

        # UI / HUD Message & Metrics
        self.hud_message = "Gate Ready — Waiting for worker"
        self.crowding_warning = False
        self.elapsed_verify_time = 0.0

    def is_locked(self) -> bool:
        return self.active_track_id is not None and self.state in [
            SessionState.LOCKED,
            SessionState.VERIFYING,
            SessionState.CONFIRMING,
            SessionState.COMPLETED,
            SessionState.COOLDOWN
        ]

    def reset_to_idle(self):
        self.state = SessionState.IDLE
        self.active_track_id = None
        self.active_worker = None
        self.active_bbox = None
        self.session_started_at = 0.0
        self.last_seen_at = 0.0
        self.ppe_observations.clear()
        self.ocr_reads.clear()
        self.finalized_ppe_score = None
        self.finalized_action = None
        self.finalized_helmet_id = ""
        self.finalized_at = 0.0
        self.crowding_warning = False
        self.elapsed_verify_time = 0.0
        self.hud_message = "Gate Ready — Waiting for worker"

    def get_gate_roi(self, frame_w: int, frame_h: int):
        """
        Returns ROI coordinates (x1, y1, x2, y2) representing the gate entry zone.
        Defaults to central 75% of the frame horizontally and full height.
        """
        margin_x = int(frame_w * 0.12)
        return (margin_x, 0, frame_w - margin_x, frame_h)

    def is_in_gate_roi(self, bbox, frame_w: int, frame_h: int) -> bool:
        """Checks if a person's center is inside the gate ROI."""
        gx1, gy1, gx2, gy2 = self.get_gate_roi(frame_w, frame_h)
        px_center = (bbox[0] + bbox[2]) / 2.0
        py_center = (bbox[1] + bbox[3]) / 2.0
        return (gx1 <= px_center <= gx2) and (gy1 <= py_center <= gy2)

    def update_frame(self, person_detections: list, frame_w: int, frame_h: int):
        """
        person_detections: list of dicts: {"track_id": int/str, "bbox": [x1,y1,x2,y2], "conf": float}
        Updates the concurrent state machine based on persons present in the gate zone.
        """
        now = time.time()
        
        # Filter persons in gate zone
        gate_persons = [p for p in person_detections if self.is_in_gate_roi(p["bbox"], frame_w, frame_h)]
        num_in_gate = len(gate_persons)

        # ── State: IDLE ──
        if self.state == SessionState.IDLE:
            if num_in_gate == 0:
                self.hud_message = "Gate Ready — Waiting for worker"
                self.crowding_warning = False
                return

            if num_in_gate > 1:
                self.state = SessionState.CROWDED
                self.hud_message = "One worker at a time! Please step back."
                self.crowding_warning = True
                return

            # Exactly 1 person entered -> Lock onto this person immediately
            person = gate_persons[0]
            self.state = SessionState.VERIFYING
            self.active_track_id = person["track_id"]
            self.active_bbox = person["bbox"]
            self.session_started_at = now
            self.last_seen_at = now
            self.ppe_observations.clear()
            self.ocr_reads.clear()
            self.finalized_ppe_score = None
            self.finalized_action = None
            self.finalized_helmet_id = ""
            self.hud_message = "Worker locked. Verifying Identity, PPE & Helmet in parallel..."
            self.crowding_warning = False
            return

        # ── State: CROWDED ──
        if self.state == SessionState.CROWDED:
            if num_in_gate == 0:
                self.reset_to_idle()
            elif num_in_gate == 1:
                person = gate_persons[0]
                self.state = SessionState.VERIFYING
                self.active_track_id = person["track_id"]
                self.active_bbox = person["bbox"]
                self.session_started_at = now
                self.last_seen_at = now
                self.ppe_observations.clear()
                self.ocr_reads.clear()
                self.finalized_ppe_score = None
                self.finalized_action = None
                self.finalized_helmet_id = ""
                self.hud_message = "Worker locked. Verifying Identity, PPE & Helmet in parallel..."
                self.crowding_warning = False
            else:
                self.hud_message = "One worker at a time! Please clear the gate."
                self.crowding_warning = True
            return

        # ── Active Track Handling ──
        locked_person = next((p for p in gate_persons if p["track_id"] == self.active_track_id), None)
        if locked_person is None:
            locked_person = next((p for p in person_detections if p["track_id"] == self.active_track_id), None)

        if locked_person is not None:
            self.last_seen_at = now
            self.active_bbox = locked_person["bbox"]
            self.crowding_warning = (num_in_gate > 1)
        else:
            if (now - self.last_seen_at) > self.grace_period_sec:
                if self.state in [SessionState.COMPLETED, SessionState.COOLDOWN]:
                    print(f"[WorkerSession] Active worker {self.active_track_id} exited gate.")
                else:
                    print(f"[WorkerSession] Evaluation track {self.active_track_id} timed out / lost.")
                self.reset_to_idle()
                return

        # Track elapsed evaluation time
        if self.session_started_at > 0:
            self.elapsed_verify_time = now - self.session_started_at

        # ── State: VERIFYING (Parallel Checks) ──
        if self.state == SessionState.VERIFYING:
            # Check for fast convergence:
            # 1. Face identified
            # 2. PPE buffer has sufficient stable observations (>= 3 frames with consistent majority)
            has_identity = self.active_worker is not None
            ppe_ready = self._check_ppe_convergence()

            # Dynamic message update
            if self.active_worker is not None:
                id_str = f"ID: {self.active_worker.get('name', 'Worker')}"
            else:
                id_str = "ID: Scanning..."

            ppe_cur = self.get_smoothed_ppe_score()
            ppe_str = f"PPE: {ppe_cur}%" if ppe_cur is not None else "PPE: Evaluating..."
            helm_str = f"Helmet: {self.finalized_helmet_id}" if self.finalized_helmet_id else "Helmet: Scanning..."
            self.hud_message = f"Verifying ({self.elapsed_verify_time:.1f}s) | {id_str} | {ppe_str} | {helm_str}"

            if has_identity and ppe_ready:
                # Both face and PPE verified! Proceed to CONFIRMING
                self.finalized_ppe_score = self.get_smoothed_ppe_score()
                self.state = SessionState.CONFIRMING
                self.finalized_at = now
                self.hud_message = f"Verified in {self.elapsed_verify_time:.2f}s! Validating results..."
                return

            # If max timeout reached without face match, complete with Unknown Worker
            if (now - self.session_started_at) > self.max_verify_sec:
                if not has_identity:
                    print("[WorkerSession] Face match timeout reached without recognition.")
                    self.active_worker = {
                        "worker_id": "UNKNOWN",
                        "name": "Unregistered / Unknown",
                        "helmet_id": self.finalized_helmet_id or "UNKNOWN",
                        "status": "Off-Site"
                    }
                self.finalized_ppe_score = self.get_smoothed_ppe_score() or 0
                self.state = SessionState.CONFIRMING
                self.finalized_at = now
                self.hud_message = f"Verification finished ({self.elapsed_verify_time:.2f}s). Finalizing..."

        # ── State: CONFIRMING ──
        elif self.state == SessionState.CONFIRMING:
            # Immediate transition to COMPLETED (gives 1 tick for validation hooks)
            self.state = SessionState.COMPLETED
            worker_name = self.active_worker.get('name', 'Worker') if self.active_worker is not None else 'Worker'
            self.hud_message = f"Check Complete: {worker_name} | PPE: {self.finalized_ppe_score}%"

        # ── State: COMPLETED ──
        elif self.state == SessionState.COMPLETED:
            if (now - self.finalized_at) > 2.0:
                self.state = SessionState.COOLDOWN
                self.hud_message = "Evaluation complete. Please step forward."

        # ── State: COOLDOWN ──
        elif self.state == SessionState.COOLDOWN:
            if locked_person is None or not self.is_in_gate_roi(locked_person["bbox"], frame_w, frame_h):
                self.reset_to_idle()

    def _check_ppe_convergence(self) -> bool:
        """
        Fast temporal convergence check:
        Returns True if at least 3-4 observations exist and agree with high confidence.
        """
        if len(self.ppe_observations) < 3:
            return False
        
        scores = [obs["score"] for obs in self.ppe_observations]
        # If the last 3 consecutive observations are identical (e.g. all 100 or all 0) -> Converged!
        if len(scores) >= 3 and scores[-1] == scores[-2] == scores[-3]:
            return True

        # If buffer is full (4 frames)
        if len(self.ppe_observations) >= self.required_ppe_frames:
            return True

        return False

    def set_identified_worker(self, worker_dict: dict):
        """Thread-safe callback from Async Face Worker."""
        if self.state in [SessionState.LOCKED, SessionState.VERIFYING]:
            self.active_worker = worker_dict
            if not self.finalized_helmet_id and worker_dict.get("helmet_id"):
                # Use assigned helmet if physical OCR hasn't superseded it yet
                self.finalized_helmet_id = worker_dict["helmet_id"]

    def record_ocr_reading(self, scanned_id: str):
        """Thread-safe callback from Async OCR Worker."""
        if not scanned_id or self.state not in [SessionState.LOCKED, SessionState.VERIFYING]:
            return
        self.ocr_reads.append(scanned_id)
        # Consensus: if 2 identical reads or 1 matching the worker's assigned helmet
        if self.active_worker and scanned_id.upper() == str(self.active_worker.get("helmet_id", "")).upper():
            self.finalized_helmet_id = scanned_id
        elif len(self.ocr_reads) >= 2 and self.ocr_reads[-1] == self.ocr_reads[-2]:
            self.finalized_helmet_id = scanned_id
        elif not self.finalized_helmet_id:
            self.finalized_helmet_id = scanned_id

    def attribute_ppe_detections(self, ppe_detections: list) -> dict:
        """
        Strict Spatial PPE Attribution:
        Given all YOLO PPE detections in the frame, selects only those within the
        locked worker's bounding box and verifies head/torso anatomical zones.
        """
        if self.active_bbox is None:
            return {"hardhat": False, "vest": False, "score": None, "matched_boxes": []}

        px1, py1, px2, py2 = self.active_bbox
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)

        # Anatomical Regions within person bounding box
        head_box = (px1 - pw*0.1, py1 - ph*0.05, px2 + pw*0.1, py1 + ph*0.40)
        torso_box = (px1 - pw*0.1, py1 + ph*0.20, px2 + pw*0.1, py1 + ph*0.80)

        has_hardhat = False
        has_no_hardhat = False
        has_vest = False
        has_no_vest = False
        matched_boxes = []

        for det in ppe_detections:
            dx1, dy1, dx2, dy2 = det["bbox"]
            label = det["label"]
            cx = (dx1 + dx2) / 2.0
            cy = (dy1 + dy2) / 2.0

            if not (px1 - pw*0.1 <= cx <= px2 + pw*0.1 and py1 - ph*0.1 <= cy <= py2 + ph*0.1):
                continue

            if label in ["Hardhat", "NO-Hardhat"]:
                if head_box[0] <= cx <= head_box[2] and head_box[1] <= cy <= head_box[3]:
                    matched_boxes.append(det)
                    if label == "Hardhat":
                        has_hardhat = True
                    elif label == "NO-Hardhat":
                        has_no_hardhat = True

            elif label in ["Safety Vest", "NO-Safety Vest"]:
                if torso_box[0] <= cx <= torso_box[2] and torso_box[1] <= cy <= torso_box[3]:
                    matched_boxes.append(det)
                    if label == "Safety Vest":
                        has_vest = True
                    elif label == "NO-Safety Vest":
                        has_no_vest = True

        hardhat_pass = has_hardhat and not has_no_hardhat
        vest_pass = has_vest and not has_no_vest

        score = 0
        if hardhat_pass:
            score += 50
        if vest_pass:
            score += 50

        observation = {
            "hardhat": hardhat_pass,
            "vest": vest_pass,
            "has_hardhat_det": has_hardhat,
            "has_no_hardhat_det": has_no_hardhat,
            "has_vest_det": has_vest,
            "has_no_vest_det": has_no_vest,
            "score": score,
            "matched_boxes": matched_boxes,
            "timestamp": time.time()
        }

        if self.state == SessionState.VERIFYING:
            self.ppe_observations.append(observation)

        return observation

    def get_smoothed_ppe_score(self) -> int | None:
        """Returns the current smoothed score or None if not evaluated."""
        if self.finalized_ppe_score is not None:
            return self.finalized_ppe_score
        if not self.ppe_observations:
            return None
        scores = [obs["score"] for obs in self.ppe_observations]
        median_score = int(np.median(scores))
        return 50 if 25 <= median_score < 75 else (100 if median_score >= 75 else 0)
