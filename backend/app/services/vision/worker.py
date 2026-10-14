import time
import threading
import json
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Callable

from .detector import SlotDetector
from .smoothing import TemporalFilter
from .cache import occupancy_cache
from ..camera.base import CameraSource
from ..camera.factory import create_camera_source

class VisionWorker:
    """
    Decoupled background inference worker.
    Continuously loops through camera feeds, performs detection, applies
    temporal majority smoothing, and updates the shared in-memory cache.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.detector = SlotDetector()
        self.smoother = TemporalFilter(window_size=5)
        
        self.cameras: Dict[str, CameraSource] = {}
        self.slots_by_area: Dict[str, List[Dict[str, Any]]] = {}
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.on_slot_update_callback: Optional[Callable[[str, str, int, bool], None]] = None
        
        self._init_cameras_and_slots()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _init_cameras_and_slots(self):
        areas = self.config.get("parking_areas", {})
        slots = self.config.get("slots", [])
        
        for area_id, area_cfg in areas.items():
            cam_src_cfg = area_cfg.get("camera_source", {"type": "mock"})
            cam = create_camera_source(area_id, cam_src_cfg)
            self.cameras[area_id] = cam
            
            # Group slots
            area_slots = [s for s in slots if s.get("area_id") == area_id]
            self.slots_by_area[area_id] = area_slots
            
            # Register slots with Mock cameras if mock is used
            if hasattr(cam, "register_slot"):
                for s in area_slots:
                    cam.register_slot(s["id"], s.get("roi_polygon", []))

    def register_callback(self, cb: Callable[[str, str, int, bool], None]):
        self.on_slot_update_callback = cb

    def update_camera_source(self, area_id: str, cam_config: Dict[str, Any]):
        """Hot-swaps the camera source for an area while worker is running."""
        old_cam = self.cameras.get(area_id)
        if old_cam and hasattr(old_cam, "release"):
            try:
                old_cam.release()
            except Exception:
                pass

        new_cam = create_camera_source(area_id, cam_config)
        self.cameras[area_id] = new_cam

        area_slots = self.slots_by_area.get(area_id, [])
        if hasattr(new_cam, "register_slot"):
            for s in area_slots:
                new_cam.register_slot(s["id"], s.get("roi_polygon", []))

        if "parking_areas" in self.config and area_id in self.config["parking_areas"]:
            self.config["parking_areas"][area_id]["camera_source"] = cam_config

        print(f"[VisionWorker] Camera source for {area_id} updated to: {cam_config}")

    def update_slot_roi(self, slot_id: str, roi_polygon: List[List[float]]):
        """Hot-updates the ROI polygon for a slot in-memory."""
        area_id = slot_id.split("-")[0]
        slots = self.slots_by_area.get(area_id, [])
        for s in slots:
            if s["id"] == slot_id:
                s["roi_polygon"] = roi_polygon
                print(f"[VisionWorker] Hot-updated ROI polygon for slot {slot_id}")
                break

        # Also update in mock camera if mock is active
        cam = self.cameras.get(area_id)
        if cam and hasattr(cam, "register_slot"):
            cam.register_slot(slot_id, roi_polygon)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True, name="CV-Worker")
            self.thread.start()
            print("[VisionWorker] Background detection loop started.")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        for cam in self.cameras.values():
            cam.release()
        print("[VisionWorker] Stopped and cameras released.")

    def _run_loop(self):
        while self.running:
            loop_start = time.time()
            
            for area_id, cam in self.cameras.items():
                slots = self.slots_by_area.get(area_id, [])
                if not slots:
                    continue
                    
                ok, frame = cam.read_frame()
                if not ok or frame is None:
                    continue
                
                # Analyze frame
                raw_results = self.detector.analyze_frame_slots(frame, slots)
                
                for slot_id, res in raw_results.items():
                    # Apply temporal smoothing
                    smooth_state, bikes, has_car, conf = self.smoother.add_observation(
                        slot_id, res["state"], res["bikes"], res["has_car"], res["confidence"]
                    )
                    
                    # Update cache
                    occupancy_cache.update_slot(slot_id, smooth_state, bikes, has_car, conf)
                    
                    # Notify callback if hook is registered
                    if self.on_slot_update_callback:
                        try:
                            self.on_slot_update_callback(slot_id, smooth_state, bikes, has_car)
                        except Exception as e:
                            print(f"[VisionWorker] Callback error: {e}")

            # Sleep to maintain ~1.0-1.5s interval
            elapsed = time.time() - loop_start
            sleep_time = max(0.1, 1.2 - elapsed)
            time.sleep(sleep_time)

    def snapshot_areas(self, area_ids: list) -> bool:
        """
        ON-DEMAND camera snapshot – called synchronously when a user clicks
        "Find & Reserve". Grabs one fresh frame from each requested zone's camera,
        runs the detector, and updates the shared occupancy cache immediately.

        This is separate from the background loop so the allocation request
        always sees the freshest possible occupancy data.
        Returns True if at least one camera was successfully snapshotted.
        """
        any_success = False
        for area_id in area_ids:
            cam = self.cameras.get(area_id)
            slots = self.slots_by_area.get(area_id, [])
            if cam is None or not slots:
                continue
            try:
                ok, frame = cam.read_frame()
                if not ok or frame is None:
                    print(f"[VisionWorker] Snapshot: camera {area_id} returned no frame.")
                    continue

                raw_results = self.detector.analyze_frame_slots(frame, slots)
                for slot_id, res in raw_results.items():
                    smooth_state, bikes, has_car, conf = self.smoother.add_observation(
                        slot_id, res["state"], res["bikes"], res["has_car"], res["confidence"]
                    )
                    occupancy_cache.update_slot(slot_id, smooth_state, bikes, has_car, conf)
                    if self.on_slot_update_callback:
                        try:
                            self.on_slot_update_callback(slot_id, smooth_state, bikes, has_car)
                        except Exception:
                            pass
                any_success = True
                print(f"[VisionWorker] On-demand snapshot complete for zone {area_id} "
                      f"({len(raw_results)} slots analysed)")
            except Exception as e:
                print(f"[VisionWorker] Snapshot error for zone {area_id}: {e}")
        return any_success

    def get_annotated_frame(self, area_id: str) -> Optional[np.ndarray]:
        """
        Draws live slot polygon overlays, labels, and detection status
        on the frame for live Admin streaming or debugging.
        """
        cam = self.cameras.get(area_id)
        if not cam:
            return None
            
        ok, frame = cam.read_frame()
        if not ok or frame is None:
            return None
            
        h, w = frame.shape[:2]
        slots = self.slots_by_area.get(area_id, [])
        cache_data = occupancy_cache.get_all()

        for s in slots:
            slot_id = s["id"]
            poly = s.get("roi_polygon")
            if not poly or len(poly) < 4:
                continue
                
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in poly], dtype=np.int32)
            cached = cache_data.get(slot_id, {})
            state = cached.get("detected_state", "FREE")
            bikes = cached.get("bikes", 0)
            has_car = cached.get("has_car", False)
            
            # Color coding: Green for FREE, Red for OCCUPIED, Amber for partial
            if state == "FREE":
                color = (0, 200, 0)
                label = f"{slot_id} [FREE]"
            elif has_car:
                color = (0, 0, 220)
                label = f"{slot_id} [CAR]"
            elif bikes > 0:
                color = (0, 140, 255) if bikes == 1 else (0, 0, 220)
                label = f"{slot_id} [{bikes} BIKE{'S' if bikes > 1 else ''}]"
            else:
                color = (0, 0, 220)
                label = f"{slot_id} [OCC]"

            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(frame, label, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, label, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return frame
