"""
Verification script for SiteSentinel Upgrades
Tests:
1. WorkerSessionManager state transitions & gate occupancy lock
2. Spatial PPE attribution (Head -> Hardhat, Torso -> Vest)
3. ReportService CSV & PDF report generation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'interface'))

from worker_session import WorkerSessionManager, SessionState
from report_service import generate_csv_report, generate_pdf_report

def test_worker_session():
    print("[Test] Testing WorkerSessionManager...")
    mgr = WorkerSessionManager(required_ppe_frames=3, grace_period_sec=1.5)
    
    # 1. Gate is empty -> IDLE
    mgr.update_frame([], 640, 480)
    assert mgr.state == SessionState.IDLE, f"Expected IDLE, got {mgr.state}"
    
    # 2. Two persons enter at once -> CROWDED
    p1 = {"track_id": 1, "bbox": [200, 100, 300, 400], "conf": 0.9}
    p2 = {"track_id": 2, "bbox": [320, 100, 420, 400], "conf": 0.9}
    mgr.update_frame([p1, p2], 640, 480)
    assert mgr.state == SessionState.CROWDED, f"Expected CROWDED, got {mgr.state}"
    
    # 3. One person steps back -> 1 person remains -> VERIFYING (Parallel lock)
    mgr.update_frame([p1], 640, 480)
    assert mgr.state == SessionState.VERIFYING, f"Expected VERIFYING, got {mgr.state}"
    assert mgr.active_track_id == 1, f"Expected track 1, got {mgr.active_track_id}"
    
    # 4. Face matched asynchronously
    mgr.set_identified_worker({"worker_id": "003", "name": "Souvagya Karmakar", "helmet_id": "H001", "status": "Off-Site"})
    assert mgr.state == SessionState.VERIFYING, f"Expected VERIFYING, got {mgr.state}"
    
    # 5. PPE spatial attribution: Hardhat at top, Vest at middle
    ppe_dets = [
        {"label": "Hardhat", "bbox": [210, 105, 290, 160], "conf": 0.85},
        {"label": "Safety Vest", "bbox": [205, 180, 295, 340], "conf": 0.90}
    ]
    for _ in range(3):
        mgr.attribute_ppe_detections(ppe_dets)
        mgr.update_frame([p1], 640, 480)

    # Since Souvagya is assigned helmet H001 and Hardhat is detected, state remains VERIFYING until helmet is cross-checked
    assert mgr.state == SessionState.VERIFYING, f"Expected VERIFYING while waiting for helmet crosscheck, got {mgr.state}"
    assert not mgr.helmet_crosscheck_passed

    # 6. Simulate Helmet ID OCR/QR read matching assigned H001
    mgr.record_ocr_reading("H001")
    assert mgr.helmet_crosscheck_passed, "Expected helmet crosscheck to pass"
    assert mgr.finalized_helmet_id == "H001"

    # Frame update now proceeds to CONFIRMING / COMPLETED
    mgr.update_frame([p1], 640, 480)
    assert mgr.state in [SessionState.CONFIRMING, SessionState.COMPLETED], f"Expected CONFIRMING or COMPLETED, got {mgr.state}"
    assert mgr.finalized_ppe_score == 100, f"Expected score 100, got {mgr.finalized_ppe_score}"
    assert mgr.get_score_display_sec() == 6.0, f"Expected 6.0s for score 100, got {mgr.get_score_display_sec()}"
    mgr.finalized_ppe_score = 50
    assert mgr.get_score_display_sec() == 4.5, f"Expected 4.5s for score 50, got {mgr.get_score_display_sec()}"
    mgr.finalized_ppe_score = 0
    assert mgr.get_score_display_sec() == 3.5, f"Expected 3.5s for score 0, got {mgr.get_score_display_sec()}"
    print("[PASS] WorkerSessionManager test passed!")

def test_reports():
    print("[Test] Testing ReportService...")
    workers = [{"id": 1, "worker_id": "003", "first_name": "Souvagya", "last_name": "Karmakar", "helmet_id": "H001", "status": "Active", "checkin_time": "09:00 AM", "checkout_time": "", "ppe_score": 100}]
    att = [{"id": 1, "date": "2026-10-03", "timestamp": "2026-10-03 09:00:00", "worker_id": "003", "event": "CHECK-IN", "ppe_score": 100, "helmet_id": "H001"}]
    events = [{"id": 1, "created_at": "2026-10-03 09:00:00", "worker_id": "003", "event_type": "CHECK_IN", "message": "Souvagya checked in", "ppe_score": 100, "helmet_id": "H001", "acknowledged": 1}]
    
    csv_out = generate_csv_report("Site Manager", workers, att, events)
    assert "SITESENTINEL" in csv_out
    assert "Souvagya" in csv_out
    
    pdf_out = generate_pdf_report("Site Manager", workers, att, events)
    assert len(pdf_out) > 500, "PDF output is empty or too short"
    print("[PASS] ReportService test passed!")

from helmet_color_classifier import (
    get_expected_color_for_occupation,
    is_color_matching_occupation,
    classify_helmet_color,
    OCCUPATION_TO_COLOR
)

def test_helmet_color_classification():
    print("[Test] Testing Helmet Color Classifier...")
    # Verify Grey & White combination per user instruction:
    assert get_expected_color_for_occupation("Manager / Supervisor / Visitor") == "White"
    assert get_expected_color_for_occupation("Site Visitor") == "White"
    assert get_expected_color_for_occupation("Foreman") == "White"
    assert get_expected_color_for_occupation("Engineer") == "White"
    assert get_expected_color_for_occupation("Labourer / Construction") == "Yellow"
    assert get_expected_color_for_occupation("Firefighter") == "Red"
    assert get_expected_color_for_occupation("Welder / High Heat") == "Brown"
    assert get_expected_color_for_occupation("Electrician / Technical") == "Blue"
    assert get_expected_color_for_occupation("Safety Officer") == "Green"
    assert get_expected_color_for_occupation("Female Worker / Additional") == "Pink"

    # Match evaluations
    assert is_color_matching_occupation("White", "Manager / Supervisor / Visitor")
    assert is_color_matching_occupation("Grey", "Manager / Supervisor / Visitor") # Grey combined into White
    assert is_color_matching_occupation("Yellow", "Labourer / Construction")
    assert not is_color_matching_occupation("Yellow", "Manager / Supervisor / Visitor")
    assert not is_color_matching_occupation("White", "Labourer / Construction")
    assert is_color_matching_occupation("Blue", "Electrician / Technical")
    assert is_color_matching_occupation("Green", "Safety Officer")
    assert is_color_matching_occupation("Red", "Firefighter")
    assert is_color_matching_occupation("Brown", "Welder / High Heat")
    assert is_color_matching_occupation("Pink", "Female Worker / Additional")
    print("[PASS] Helmet color classifier test passed!")

def test_occupation_crosscheck():
    print("[Test] Testing Occupation & Helmet Color Cross-Check in Session...")
    mgr = WorkerSessionManager(required_ppe_frames=2, grace_period_sec=1.5)
    p1 = {"track_id": 1, "bbox": [200, 100, 300, 400], "conf": 0.9}
    mgr.update_frame([p1], 640, 480)

    # Worker identified as Labourer (expected Yellow helmet)
    mgr.set_identified_worker({
        "worker_id": "007",
        "name": "Ramesh Kumar",
        "helmet_id": "H002",
        "occupation": "Labourer / Construction",
        "status": "Off-Site"
    })
    assert mgr.expected_helmet_color == "Yellow"

    # Simulating PPE detections
    ppe_dets = [
        {"label": "Hardhat", "bbox": [210, 105, 290, 160], "conf": 0.85},
        {"label": "Safety Vest", "bbox": [205, 180, 295, 340], "conf": 0.90}
    ]
    mgr.attribute_ppe_detections(ppe_dets)
    mgr.record_ocr_reading("H002")

    # If camera detects White helmet instead of expected Yellow -> mismatch!
    mgr.record_helmet_color("White")
    assert mgr.detected_helmet_color == "White"
    assert mgr.helmet_color_mismatch is True
    assert mgr.helmet_color_match is False

    # Now if worker wears expected Yellow helmet -> match!
    for _ in range(5):
        mgr.record_helmet_color("Yellow")
    assert mgr.detected_helmet_color == "Yellow"
    assert mgr.helmet_color_match is True
    assert mgr.helmet_color_mismatch is False
    print("[PASS] Occupation & Helmet Color Cross-Check test passed!")

if __name__ == "__main__":
    test_helmet_color_classification()
    test_occupation_crosscheck()
    test_worker_session()
    test_reports()
    print("\nALL UPGRADE VERIFICATIONS PASSED!")

