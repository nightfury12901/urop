from typing import Dict, Any
from .base import CameraSource
from .opencv_cam import OpenCVCameraSource
from .mock_cam import MockCameraSource
from .image_cam import ImageCameraSource

def create_camera_source(area_id: str, cam_config: Dict[str, Any]) -> CameraSource:
    """
    Factory function that builds the requested CameraSource.
    cam_config format:
      {
        "type": "mock" | "webcam" | "droidcam" | "rtsp" | "image",
        "source": "0" | "http://..." | "rtsp://..." | "c:/path/to/image.jpg",
        "width": 640,
        "height": 480
      }
    """
    src_type = cam_config.get("type", "mock").lower()
    width = cam_config.get("width", 640)
    height = cam_config.get("height", 480)
    source_val = cam_config.get("source", 0)

    if src_type in ("image", "photo", "file"):
        return ImageCameraSource(image_path=str(source_val), width=width, height=height)

    if src_type in ("webcam", "droidcam", "rtsp", "opencv"):
        return OpenCVCameraSource(source=source_val, width=width, height=height)
    
    # Default to MockCameraSource
    return MockCameraSource(area_id=area_id, width=width, height=height)
