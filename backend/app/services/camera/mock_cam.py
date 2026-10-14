import cv2
import numpy as np
import time
import threading
from typing import Tuple, Optional, Dict, Any, List
from .base import CameraSource

class MockCameraSource(CameraSource):
    """
    Simulated camera stream for offline testing and reproducible demos.
    Generates video frames representing an A4 sheet parking lot.
    Supports dynamic programmatic injection of vehicles into slots.
    """
    def __init__(self, area_id: str, width: int = 640, height: int = 480, fps: int = 20):
        self.area_id = area_id
        self.width = width
        self.height = height
        self.fps = fps
        self.lock = threading.Lock()
        self.is_active = True
        
        # Simulated slot occupancy: slot_id -> "EMPTY" | "CAR" | "BIKE_1" | "BIKE_2"
        self.slot_states: Dict[str, str] = {}
        self.slot_polygons: Dict[str, List[List[float]]] = {}

    def register_slot(self, slot_id: str, polygon: List[List[float]]) -> None:
        """Register slot polygon (normalized coordinates 0.0-1.0)."""
        with self.lock:
            self.slot_polygons[slot_id] = polygon
            if slot_id not in self.slot_states:
                self.slot_states[slot_id] = "EMPTY"

    def set_slot_state(self, slot_id: str, state: str) -> None:
        """Dynamically set slot state: 'EMPTY', 'CAR', 'BIKE_1', 'BIKE_2'."""
        with self.lock:
            self.slot_states[slot_id] = state.upper()

    def _draw_parking_lot(self) -> np.ndarray:
        # A4 Paper Texture / White/Light-Gray Canvas
        frame = np.full((self.height, self.width, 3), 242, dtype=np.uint8)
        
        # Grid texture / asphalt / table
        cv2.rectangle(frame, (20, 20), (self.width - 20, self.height - 20), (225, 225, 225), -1)
        cv2.rectangle(frame, (20, 20), (self.width - 20, self.height - 20), (120, 120, 120), 2)
        
        # Title header on the sheet
        cv2.putText(frame, f"CAM {self.area_id} - CAMPUS LOT", (30, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)

        with self.lock:
            polys = dict(self.slot_polygons)
            states = dict(self.slot_states)

        # Draw each slot
        for slot_id, poly in polys.items():
            pts = np.array([[int(p[0] * self.width), int(p[1] * self.height)] for p in poly], dtype=np.int32)
            
            state = states.get(slot_id, "EMPTY")
            
            # Draw slot boundary
            cv2.polylines(frame, [pts], isClosed=True, color=(160, 160, 160), thickness=2)
            
            # Label
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(frame, slot_id, (cx - 20, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

            # Draw vehicle if occupied
            if state == "CAR":
                # Car body (e.g. Blue or Red toy car rectangle)
                car_pts = (pts * 0.75 + [cx * 0.25, cy * 0.25]).astype(np.int32)
                cv2.fillPoly(frame, [car_pts], color=(180, 50, 40))  # Navy/Crimson car
                cv2.circle(frame, (cx, cy), 5, (250, 250, 250), -1)
            elif state == "BIKE_1":
                # 1 Toy Bike (Left side of slot)
                bx = cx - 12
                cv2.ellipse(frame, (bx, cy), (6, 14), 0, 0, 360, (30, 140, 30), -1)
            elif state == "BIKE_2":
                # 2 Toy Bikes (Both halves occupied)
                bx1, bx2 = cx - 14, cx + 14
                cv2.ellipse(frame, (bx1, cy), (6, 14), 0, 0, 360, (30, 140, 30), -1)
                cv2.ellipse(frame, (bx2, cy), (6, 14), 0, 0, 360, (30, 140, 30), -1)

        # Subtle noise to simulate real sensor camera
        noise = np.random.normal(0, 2, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Timestamp in bottom corner
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"LIVE: {ts}", (self.width - 120, self.height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        return frame

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_active:
            return False, None
        return True, self._draw_parking_lot()

    def is_opened(self) -> bool:
        return self.is_active

    def release(self) -> None:
        self.is_active = False

    def get_info(self) -> Dict[str, Any]:
        return {
            "source": f"mock_camera_{self.area_id}",
            "source_type": "mock",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "is_active": self.is_active
        }
