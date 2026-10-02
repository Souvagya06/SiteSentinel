"""
WorkerSessionManager — Single-Worker Evaluation State Machine, Spatial PPE Attribution,
and Gate Occupancy Lock for SiteSentinel.
"""
import time
from collections import deque
import numpy as np


class SessionState:
    IDLE = "IDLE"
    CROWDED = "CROWDED"
    IDENTIFYING = "IDENTIFYING"
    EVALUATING_PPE = "EVALUATING_PPE"
    FINALIZED = "FINALIZED"
    COOLDOWN = "COOLDOWN"


class WorkerSessionManager:
    def __init__(self, required_ppe_frames: int = 8, grace_period_sec: float = 2.0):
        self.state = SessionState.IDLE
        self.active_track_id = None
        self.active_worker = None  # Dict of worker details when matched
        self.active_bbox = None    # [x1, y1, x2, y2]
        
        self.session_started_at: float = 0.0
        self.last_seen_at: float = 0.0
        self.grace_period_sec = grace_period_sec
        self.required_ppe_frames = required_ppe_frames
        
        # PPE smoothing buffer for the active locked worker
        self.ppe_observations = deque(maxlen=required_ppe_frames)
        self.finalized_ppe_score = None  # None, 0, 50, or 100
        self.finalized_action = None     # 'Check-in' or 'Check-out'
        self.finalized_helmet_id = ""
        self.finalized_at: float = 0.0

        # UI / HUD Message
        self.hud_message = "Waiting for worker..."
        self.crowding_warning = False

    def is_locked(self) -> bool:
        return self.active_track_id is not None and self.state in [
            SessionState.IDENTIFYING,
            SessionState.EVALUATING_PPE,
            SessionState.FINALIZED,
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
        self.finalized_ppe_score = None
        self.finalized_action = None
        self.finalized_helmet_id = ""
        self.crowding_warning = False
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
        Updates the state machine based on persons present in the gate zone.
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

            # Exactly 1 person entered
            person = gate_persons[0]
            self.state = SessionState.IDENTIFYING
            self.active_track_id = person["track_id"]
            self.active_bbox = person["bbox"]
            self.session_started_at = now
            self.last_seen_at = now
            self.ppe_observations.clear()
            self.finalized_ppe_score = None
            self.hud_message = "Worker detected. Look at camera for ID..."
            self.crowding_warning = False
            return

        # ── State: CROWDED ──
        if self.state == SessionState.CROWDED:
            if num_in_gate == 0:
                self.reset_to_idle()
            elif num_in_gate == 1:
                # Crowding resolved to 1 person
                person = gate_persons[0]
                self.state = SessionState.IDENTIFYING
                self.active_track_id = person["track_id"]
                self.active_bbox = person["bbox"]
                self.session_started_at = now
                self.last_seen_at = now
                self.ppe_observations.clear()
                self.finalized_ppe_score = None
                self.hud_message = "Worker detected. Look at camera for ID..."
                self.crowding_warning = False
            else:
                self.hud_message = "One worker at a time! Please clear the gate."
                self.crowding_warning = True
            return

        # ── Active Track Handling (IDENTIFYING, EVALUATING_PPE, FINALIZED, COOLDOWN) ──
        # Check if the locked person is still present
        locked_person = next((p for p in gate_persons if p["track_id"] == self.active_track_id), None)
        
        # If not in gate persons, check if they are anywhere in frame to maintain track
        if locked_person is None:
            locked_person = next((p for p in person_detections if p["track_id"] == self.active_track_id), None)

        if locked_person is not None:
            self.last_seen_at = now
            self.active_bbox = locked_person["bbox"]
            # Check for background crowding
            self.crowding_warning = (num_in_gate > 1)
        else:
            # Check grace period
            if (now - self.last_seen_at) > self.grace_period_sec:
                # Worker exited or track lost
                if self.state in [SessionState.FINALIZED, SessionState.COOLDOWN]:
                    print(f"[WorkerSession] Active worker {self.active_track_id} exited gate zone.")
                else:
                    print(f"[WorkerSession] Evaluation track {self.active_track_id} timed out.")
                self.reset_to_idle()
                return

        # ── State transitions for active track ──
        if self.state == SessionState.IDENTIFYING:
            if self.active_worker is not None:
                # Worker identified! Move to PPE evaluation
                self.state = SessionState.EVALUATING_PPE
                self.hud_message = f"Identified: {self.active_worker['name']}. Evaluating PPE..."
            elif (now - self.session_started_at) > 5.0:
                self.hud_message = "Face not recognized. Looking for match..."

        elif self.state == SessionState.EVALUATING_PPE:
            if len(self.ppe_observations) >= self.required_ppe_frames:
                # Finalize PPE score using smoothed majority
                scores = [obs["score"] for obs in self.ppe_observations]
                # Calculate majority / stable score
                final_score = int(np.median(scores)) if scores else 0
                if final_score not in [0, 50, 100]:
                    final_score = 50 if final_score > 25 else 0

                self.finalized_ppe_score = final_score
                self.state = SessionState.FINALIZED
                self.finalized_at = now
                self.hud_message = f"Evaluated PPE: {self.finalized_ppe_score}%"

        elif self.state == SessionState.FINALIZED:
            # Transition to cooldown after brief display
            if (now - self.finalized_at) > 3.0:
                self.state = SessionState.COOLDOWN
                self.hud_message = "Evaluation complete. Step forward through the gate."

        elif self.state == SessionState.COOLDOWN:
            # Hold lock until worker steps out of gate ROI
            if locked_person is None or not self.is_in_gate_roi(locked_person["bbox"], frame_w, frame_h):
                self.reset_to_idle()

    def set_identified_worker(self, worker_dict: dict):
        """Assigns the identified worker to the active session."""
        if self.state in [SessionState.IDENTIFYING, SessionState.EVALUATING_PPE]:
            self.active_worker = worker_dict
            if self.state == SessionState.IDENTIFYING:
                self.state = SessionState.EVALUATING_PPE
                self.hud_message = f"Identified: {worker_dict['name']}. Evaluating PPE..."

    def attribute_ppe_detections(self, ppe_detections: list) -> dict:
        """
        Strict Spatial PPE Attribution:
        Given all YOLO PPE detections in the frame, selects only those within the
        locked worker's bounding box and verifies head/torso anatomical zones.
        
        ppe_detections: list of dicts: {"label": str, "bbox": [x1,y1,x2,y2], "conf": float}
        Returns: {"hardhat": bool, "vest": bool, "score": int, "matched_boxes": list}
        """
        if self.active_bbox is None:
            return {"hardhat": False, "vest": False, "score": None, "matched_boxes": []}

        px1, py1, px2, py2 = self.active_bbox
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)

        # Anatomical Regions within person bounding box
        # Head: top 35%
        head_box = (px1 - pw*0.1, py1 - ph*0.05, px2 + pw*0.1, py1 + ph*0.40)
        # Torso: 20% to 80% from top
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

            # Check if detection center is roughly inside the person box with 10% margin
            if not (px1 - pw*0.1 <= cx <= px2 + pw*0.1 and py1 - ph*0.1 <= cy <= py2 + ph*0.1):
                continue  # Belong to someone else

            # Head PPE (Hardhat / NO-Hardhat)
            if label in ["Hardhat", "NO-Hardhat"]:
                if head_box[0] <= cx <= head_box[2] and head_box[1] <= cy <= head_box[3]:
                    matched_boxes.append(det)
                    if label == "Hardhat":
                        has_hardhat = True
                    elif label == "NO-Hardhat":
                        has_no_hardhat = True

            # Torso PPE (Safety Vest / NO-Safety Vest)
            elif label in ["Safety Vest", "NO-Safety Vest"]:
                if torso_box[0] <= cx <= torso_box[2] and torso_box[1] <= cy <= torso_box[3]:
                    matched_boxes.append(det)
                    if label == "Safety Vest":
                        has_vest = True
                    elif label == "NO-Safety Vest":
                        has_no_vest = True

        # PPE Scoring: 50 for hardhat, 50 for vest
        # If hardhat detected (and not explicitly NO-Hardhat dominant) -> +50
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

        if self.state == SessionState.EVALUATING_PPE:
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
