import time
import threading
from typing import Dict, Any, Optional

class OccupancyCache:
    """
    Thread-safe in-memory cache holding real-time slot occupancy.
    Updated every 1-2 seconds by the CV background worker.
    Read in O(1) time (<0.1ms) by incoming user allocation requests.
    """
    _instance: Optional["OccupancyCache"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._data = {}
                cls._instance._data_lock = threading.Lock()
            return cls._instance

    def update_slot(self, slot_id: str, detected_state: str, bikes: int, has_car: bool, confidence: float) -> None:
        with self._data_lock:
            self._data[slot_id] = {
                "slot_id": slot_id,
                "detected_state": detected_state, # "FREE" or "OCCUPIED"
                "bikes": bikes,                   # 0, 1, or 2
                "has_car": has_car,               # True/False
                "confidence": round(confidence, 3),
                "updated_at": time.time()
            }

    def get_slot(self, slot_id: str) -> Optional[Dict[str, Any]]:
        with self._data_lock:
            val = self._data.get(slot_id)
            return dict(val) if val else None

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        with self._data_lock:
            return {k: dict(v) for k, v in self._data.items()}

    def get_area_summary(self, area_id: str) -> Dict[str, Any]:
        with self._data_lock:
            slots_in_area = [v for k, v in self._data.items() if k.startswith(area_id)]
            total = len(slots_in_area)
            free = sum(1 for s in slots_in_area if s["detected_state"] == "FREE")
            occupied = sum(1 for s in slots_in_area if s["detected_state"] == "OCCUPIED")
            return {
                "area_id": area_id,
                "total_slots": total,
                "free_slots": free,
                "occupied_slots": occupied
            }

occupancy_cache = OccupancyCache()
