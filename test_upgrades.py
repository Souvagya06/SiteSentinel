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
    
    # 3. One person steps back -> 1 person remains -> IDENTIFYING
    mgr.update_frame([p1], 640, 480)
    assert mgr.state == SessionState.IDENTIFYING, f"Expected IDENTIFYING, got {mgr.state}"
    assert mgr.active_track_id == 1, f"Expected track 1, got {mgr.active_track_id}"
    
    # 4. Face matched
    mgr.set_identified_worker({"worker_id": "003", "name": "Souvagya Karmakar", "helmet_id": "H001", "status": "Off-Site"})
    assert mgr.state == SessionState.EVALUATING_PPE, f"Expected EVALUATING_PPE, got {mgr.state}"
    
    # 5. PPE spatial attribution: Hardhat at top, Vest at middle
    ppe_dets = [
        {"label": "Hardhat", "bbox": [210, 105, 290, 160], "conf": 0.85},
        {"label": "Safety Vest", "bbox": [205, 180, 295, 340], "conf": 0.90}
    ]
    for _ in range(3):
        mgr.attribute_ppe_detections(ppe_dets)
        mgr.update_frame([p1], 640, 480)
        
    assert mgr.state == SessionState.FINALIZED, f"Expected FINALIZED, got {mgr.state}"
    assert mgr.finalized_ppe_score == 100, f"Expected score 100, got {mgr.finalized_ppe_score}"
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

if __name__ == "__main__":
    test_worker_session()
    test_reports()
    print("\nALL UPGRADE VERIFICATIONS PASSED!")
