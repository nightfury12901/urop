from typing import Dict, Any
from .base import CameraSource
from .opencv_cam import OpenCVCameraSource
from .mock_cam import MockCameraSource
from .image_cam import ImageCameraSource
from .esp32_cam import ESP32CameraSource

def create_camera_source(area_id: str, cam_config: Dict[str, Any]) -> CameraSource:
    """
    Factory function that builds the requested CameraSource.
    cam_config format:
      {
        "type": "mock" | "webcam" | "droidcam" | "rtsp" | "image" | "esp32",
        "source": "0" | "http://..." | "rtsp://..." | "c:/path/to/image.jpg",
        "width": 640,
        "height": 480
      }
    For esp32 type, "source" is the ESP32 base URL (e.g. "http://192.168.1.50").
    """
    src_type = cam_config.get("type", "mock").lower()
    width = cam_config.get("width", 640)
    height = cam_config.get("height", 480)
    source_val = cam_config.get("source", 0)

    if src_type == "esp32":
        poll_interval = cam_config.get("poll_interval", 2.0)
        return ESP32CameraSource(
            base_url=str(source_val),
            width=width,
            height=height,
            poll_interval=poll_interval,
        )

    if src_type in ("image", "photo", "file"):
        return ImageCameraSource(image_path=str(source_val), width=width, height=height)

    if src_type in ("webcam", "droidcam", "rtsp", "opencv"):
        return OpenCVCameraSource(source=source_val, width=width, height=height)
    
    # Default to MockCameraSource
    return MockCameraSource(area_id=area_id, width=width, height=height)
