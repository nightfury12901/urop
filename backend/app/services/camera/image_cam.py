import cv2
import numpy as np
import os
import time
import threading
from typing import Tuple, Optional
from .base import CameraSource

class ImageCameraSource(CameraSource):
    """
    Serves a static or updated image file (e.g., photo of A4 sheet with toy cars)
    as a camera video source.
    Monitors file timestamp and re-reads automatically whenever a new image is saved.
    """
    def __init__(self, image_path: str, width: int = 640, height: int = 480):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        
        self.cached_frame: Optional[np.ndarray] = None
        self.last_mtime: float = 0.0
        self._load_image()

    def _load_image(self) -> None:
        if os.path.exists(self.image_path):
            try:
                mtime = os.path.getmtime(self.image_path)
                if mtime != self.last_mtime or self.cached_frame is None:
                    img = cv2.imread(self.image_path)
                    if img is not None:
                        if (img.shape[1] != self.width) or (img.shape[0] != self.height):
                            img = cv2.resize(img, (self.width, self.height))
                        self.cached_frame = img
                        self.last_mtime = mtime
            except Exception as e:
                print(f"[ImageCameraSource] Error loading {self.image_path}: {e}")
        else:
            # Fallback canvas if image file not yet present
            blank = np.full((self.height, self.width, 3), 235, dtype=np.uint8)
            cv2.rectangle(blank, (20, 20), (self.width - 20, self.height - 20), (210, 210, 210), -1)
            cv2.rectangle(blank, (20, 20), (self.width - 20, self.height - 20), (120, 120, 120), 2)
            cv2.putText(blank, f"Awaiting Photo: {os.path.basename(self.image_path)}", (40, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
            self.cached_frame = blank

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            # Check for file update every second
            self._load_image()
            if self.cached_frame is not None:
                return True, self.cached_frame.copy()
            return False, None

    def is_opened(self) -> bool:
        return self.cached_frame is not None

    def release(self) -> None:
        pass
