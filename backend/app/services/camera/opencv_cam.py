import cv2
import threading
import time
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from .base import CameraSource

class OpenCVCameraSource(CameraSource):
    """
    Production-grade camera capture source for:
    1. Local USB / Laptop Webcams (source as int: 0, 1, 2)
    2. DroidCam HTTP/MJPEG streams (e.g. 'http://192.168.1.15:4747/video')
    3. CCTV RTSP streams (e.g. 'rtsp://user:pass@192.168.1.20:554/stream1')
    
    Includes a background polling thread to eliminate OpenCV buffer lag.
    """
    def __init__(self, source: Union[int, str], width: int = 640, height: int = 480, fps: int = 30):
        # Cast numeric strings to int for local devices
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source
            
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_status: bool = False
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.source_type = self._determine_source_type()
        
        self._initialize_capture()
        self._start_grabber_thread()

    def _determine_source_type(self) -> str:
        if isinstance(self.source, int):
            return "webcam"
        s = str(self.source).lower()
        if s.startswith("rtsp://"):
            return "rtsp"
        if s.startswith("http://") or s.startswith("https://"):
            return "droidcam_mjpeg"
        return "video_file"

    def _initialize_capture(self) -> None:
        try:
            if isinstance(self.source, int):
                # On Windows, DirectShow backend prevents 5s startup delay
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"[OpenCVCameraSource] Error opening source {self.source}: {e}")

    def _start_grabber_thread(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _grab_loop(self) -> None:
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(1.0)
                self._initialize_capture()
                continue
                
            success, frame = self.cap.read()
            if success and frame is not None:
                if (frame.shape[1] != self.target_width) or (frame.shape[0] != self.target_height):
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
                with self.lock:
                    self.latest_frame = frame
                    self.latest_status = True
            else:
                with self.lock:
                    self.latest_status = False
                time.sleep(0.1)
            time.sleep(1.0 / max(self.target_fps, 10))

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_status and self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def is_opened(self) -> bool:
        return self.latest_status and (self.cap is not None) and self.cap.isOpened()

    def release(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_info(self) -> Dict[str, Any]:
        return {
            "source": str(self.source),
            "source_type": self.source_type,
            "width": self.target_width,
            "height": self.target_height,
            "fps": self.target_fps,
            "is_active": self.is_opened()
        }
