"""
Interactive ROI Polygon Calibration Tool for Smart Campus Parking.

Usage:
  python calibrate_rois.py --area P1 --source 0             # Webcam
  python calibrate_rois.py --area P3 --source http://192.168.1.15:4747/video  # DroidCam
  python calibrate_rois.py --area P1 --source mock          # Mock A4 generator
  python calibrate_rois.py --area P1 --source image.png     # Static test photo

Controls:
  - Left Click: Place polygon point (4 points required per slot)
  - 'n' or SPACE: Next slot
  - 'p': Previous slot
  - 'c': Clear points for current slot
  - 's': Save polygons to backend/app/data/campus_config.json
  - 'q' or ESC: Quit calibration
"""

import cv2
import json
import argparse
import os
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "backend", "app", "data", "campus_config.json")

class Calibrator:
    def __init__(self, area_id: str, source: str):
        self.area_id = area_id.upper()
        self.source = source
        self.config = self.load_config()
        self.slots = [s for s in self.config.get("slots", []) if s.get("area_id") == self.area_id]
        
        if not self.slots:
            raise ValueError(f"No slots found for area {self.area_id} in {CONFIG_PATH}")
            
        self.current_slot_idx = 0
        self.current_points = []
        self.saved_polygons = {}
        
        # Load existing polygons
        for s in self.slots:
            if "roi_polygon" in s:
                self.saved_polygons[s["id"]] = [list(pt) for pt in s["roi_polygon"]]
                
        self.cap = None
        self.static_image = None
        self.is_mock = False
        self._init_source()

    def load_config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def save_config(self):
        for s in self.config["slots"]:
            if s["id"] in self.saved_polygons:
                s["roi_polygon"] = self.saved_polygons[s["id"]]
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"[SUCCESS] Saved updated polygons for {len(self.saved_polygons)} slots to {CONFIG_PATH}")

    def _init_source(self):
        if self.source.lower() == "mock":
            from backend.app.services.camera.mock_cam import MockCameraSource
            self.mock_cam = MockCameraSource(area_id=self.area_id)
            for s in self.slots:
                if "roi_polygon" in s:
                    self.mock_cam.register_slot(s["id"], s["roi_polygon"])
            self.is_mock = True
        elif os.path.isfile(self.source):
            self.static_image = cv2.imread(self.source)
            if self.static_image is None:
                raise ValueError(f"Failed to read image {self.source}")
        else:
            dev = int(self.source) if self.source.isdigit() else self.source
            if isinstance(dev, int):
                self.cap = cv2.VideoCapture(dev, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(dev)
            if not self.cap.isOpened():
                print(f"[WARN] Failed to open video source {self.source}. Falling back to blank test card.")

    def get_frame(self):
        if self.is_mock:
            _, f = self.mock_cam.read_frame()
            return f
        if self.static_image is not None:
            return self.static_image.copy()
        if self.cap and self.cap.isOpened():
            ok, f = self.cap.read()
            if ok and f is not None:
                return f
        # Default blank canvas if camera unavailable
        canvas = np.full((480, 640, 3), 235, dtype=np.uint8)
        cv2.putText(canvas, f"NO CAMERA FEED ({self.source})", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        return canvas

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_points) < 4:
                # Store normalized coordinates
                h, w = param["shape"][:2]
                norm_pt = [round(x / w, 3), round(y / h, 3)]
                self.current_points.append(norm_pt)
                print(f"[{self.current_slot['id']}] Point {len(self.current_points)}: ({x}, {y})")

                if len(self.current_points) == 4:
                    self.saved_polygons[self.current_slot["id"]] = list(self.current_points)
                    print(f"[OK] Completed polygon for {self.current_slot['id']}! Press 'n' to advance.")

    @property
    def current_slot(self):
        return self.slots[self.current_slot_idx]

    def run(self):
        window_name = f"ROI Calibrator - {self.area_id} [Press 's' to Save, 'q' to Quit]"
        cv2.namedWindow(window_name)

        while True:
            frame = self.get_frame()
            h, w = frame.shape[:2]
            cv2.setMouseCallback(window_name, self.mouse_callback, {"shape": (h, w)})

            # Draw all existing polygons
            for slot in self.slots:
                sid = slot["id"]
                poly = self.saved_polygons.get(sid)
                if poly and len(poly) >= 4:
                    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], dtype=np.int32)
                    is_active = (sid == self.current_slot["id"])
                    color = (0, 255, 0) if not is_active else (0, 165, 255)
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2 if not is_active else 3)
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    cv2.putText(frame, sid, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)

            # Draw current points being placed
            if self.current_points:
                pts = [[int(p[0] * w), int(p[1] * h)] for p in self.current_points]
                for idx, pt in enumerate(pts):
                    cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
                    cv2.putText(frame, str(idx + 1), (pt[0] + 5, pt[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                if len(pts) > 1:
                    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=1)

            # Overlay status bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 55), (w, h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            status_text = f"Slot: {self.current_slot['id']} ({self.current_slot_idx + 1}/{len(self.slots)}) | Points: {len(self.current_points)}/4"
            cmd_text = "[Click]: Points | [n]: Next | [p]: Prev | [c]: Clear | [s]: Save | [q]: Quit"
            cv2.putText(frame, status_text, (15, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, cmd_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key in (ord('n'), 32):  # 'n' or SPACE
                self.current_slot_idx = (self.current_slot_idx + 1) % len(self.slots)
                self.current_points = []
            elif key == ord('p'):  # Previous
                self.current_slot_idx = (self.current_slot_idx - 1) % len(self.slots)
                self.current_points = []
            elif key == ord('c'):  # Clear
                self.current_points = []
                if self.current_slot["id"] in self.saved_polygons:
                    del self.saved_polygons[self.current_slot["id"]]
                print(f"[RESET] Cleared polygon for {self.current_slot['id']}")
            elif key == ord('s'):  # Save
                self.save_config()

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Parking Slot ROI Polygons")
    parser.add_argument("--area", default="P1", help="Area ID: P1, P2, P3, P4")
    parser.add_argument("--source", default="mock", help="Camera source (0, URL, 'mock', or image path)")
    args = parser.parse_args()

    calib = Calibrator(area_id=args.area, source=args.source)
    calib.run()
