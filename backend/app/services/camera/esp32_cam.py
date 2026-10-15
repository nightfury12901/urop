"""
esp32_cam.py
------------
Camera source adapter for the ESP32-CAM edge node.

Architecture difference from normal camera sources:
  - Normal cameras: server grabs frames and runs CV locally
  - ESP32 camera:  ESP32 runs CV on-device and exposes JSON occupancy API
                   Server just polls /api/slots to get pre-analysed results

This class implements two interfaces:
  1. CameraSource (read_frame) → proxies the MJPEG stream for admin feed display
  2. get_occupancy()           → polls ESP32's /api/slots for slot states
     VisionWorker detects this method and skips local CV analysis
"""

import time
import threading
import requests
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .base import CameraSource


class ESP32CameraSource(CameraSource):
    """
    Adapter for an ESP32-CAM running the parking_esp32 firmware.

    cam_config format:
      {
        "type": "esp32",
        "source": "http://192.168.1.XX",   # ESP32 base URL (no trailing slash)
        "width": 640,
        "height": 480,
        "poll_interval": 2.0               # seconds between occupancy polls
      }
    """

    def __init__(self, base_url: str, width: int = 640, height: int = 480,
                 poll_interval: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.width    = width
        self.height   = height
        self.poll_interval = poll_interval

        # Cached occupancy result from last ESP32 poll
        self._occupancy: Dict[str, str] = {}      # slot_id -> "FREE" | "OCCUPIED"
        self._last_poll: float = 0.0
        self._lock = threading.Lock()

        # For the MJPEG proxy frame
        self._last_frame: Optional[np.ndarray] = None
        self._opened = True

        print(f"[ESP32CameraSource] Connected to {self.base_url}")

    # ── CameraSource interface ────────────────────────────────────

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Fetches a single JPEG snapshot from the ESP32 and returns it
        as an OpenCV BGR frame (used for the Admin Portal live feed).
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/snapshot",
                timeout=3.0,
                stream=False
            )
            if resp.status_code != 200:
                return self._return_cached()

            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return self._return_cached()

            frame = cv2.resize(frame, (self.width, self.height))
            with self._lock:
                self._last_frame = frame
            return True, frame

        except Exception as e:
            print(f"[ESP32CameraSource] read_frame error: {e}")
            return self._return_cached()

    def _return_cached(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            f = self._last_frame
        return (True, f) if f is not None else (False, None)

    def is_opened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self._opened = False
        print(f"[ESP32CameraSource] Released {self.base_url}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "type":          "esp32",
            "base_url":      self.base_url,
            "width":         self.width,
            "height":        self.height,
            "poll_interval": self.poll_interval,
        }

    # ── ESP32-specific: edge occupancy polling ────────────────────

    def get_occupancy(self) -> Optional[Dict[str, str]]:
        """
        Polls the ESP32's /api/slots endpoint.
        Returns dict {slot_id: "FREE"|"OCCUPIED"} or None on error.
        This is called by VisionWorker instead of frame-based CV
        when the camera source has this method.
        """
        now = time.time()
        if now - self._last_poll < self.poll_interval:
            # Return cached result if within poll interval
            with self._lock:
                return dict(self._occupancy) if self._occupancy else None

        try:
            resp = requests.get(
                f"{self.base_url}/api/slots",
                timeout=4.0
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            slots = data.get("slots", {})
            if not slots:
                return None

            with self._lock:
                self._occupancy = dict(slots)
                self._last_poll = now

            return dict(slots)

        except Exception as e:
            print(f"[ESP32CameraSource] get_occupancy error: {e}")
            return None

    def trigger_calibration(self) -> Dict[str, Any]:
        """
        Sends POST /calibrate to the ESP32 to trigger on-device
        automatic slot grid detection.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/calibrate",
                timeout=15.0
            )
            return resp.json()
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def push_roi_config(self, slots_config: list) -> bool:
        """
        Pushes ROI configuration from campus_config.json to the ESP32
        (normalised coordinates). Returns True on success.
        """
        try:
            payload = {"slots": [
                {
                    "id": s["id"],
                    "x0": s["roi_polygon"][0][0],
                    "y0": s["roi_polygon"][0][1],
                    "x1": s["roi_polygon"][2][0],
                    "y1": s["roi_polygon"][2][1],
                }
                for s in slots_config
                if s.get("roi_polygon") and len(s["roi_polygon"]) >= 3
            ]}
            resp = requests.post(
                f"{self.base_url}/api/slots/config",
                json=payload,
                timeout=5.0
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"[ESP32CameraSource] push_roi_config error: {e}")
            return False
