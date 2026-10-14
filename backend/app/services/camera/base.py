from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np

class CameraSource(ABC):
    """Abstract Base Class for all camera video sources."""

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads and returns the latest available frame.
        Returns:
            (success: bool, frame: np.ndarray | None)
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Checks if the video capture source is currently active."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases underlying device/stream resources."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns metadata about the camera source (type, resolution, fps, status)."""
        pass
