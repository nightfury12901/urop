"""
Smart Campus Parking System - One-Click Launcher
Runs the complete full-stack system: Backend API, CV Background Engine, and Web Application.
"""

import sys
import os
import uvicorn

def main():
    print("=" * 75)
    print("🚗 SMART CAMPUS PARKING SYSTEM - PROTOTYPE DEMO SERVER")
    print("=" * 75)
    print("• Customer Portal:    http://localhost:8000")
    print("• Admin Portal:       http://localhost:8000 (Switch tab to 'Admin Portal')")
    print("• Interactive Docs:   http://localhost:8000/docs")
    print("• WebSocket Stream:   ws://localhost:8000/ws/live-parking")
    print("-" * 75)
    print("PROTOTYPE CAMERA FEEDS (50 Slots across 4 Areas):")
    print("• P1: Student Lot A (15 slots - Bikes only) -> Default: Mock / DroidCam")
    print("• P2: Student Lot B (15 slots - Bikes only) -> Default: Mock / DroidCam")
    print("• P3: Faculty Lot   (12 slots - Car or 2 Bikes) -> Default: Mock / DroidCam")
    print("• P4: VIP & Guest   (8 slots  - Car or 2 Bikes) -> Default: Mock / DroidCam")
    print("-" * 75)
    print("CALIBRATION COMMAND FOR YOUR A4 SHEET & DROIDCAM:")
    print("  python calibrate_rois.py --area P1 --source http://<DROIDCAM_IP>:4747/video")
    print("=" * 75)
    print("\nStarting server on http://localhost:8000 ... Press CTRL+C to stop.\n")

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
