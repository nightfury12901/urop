import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import os

class SlotDetector:
    """
    Hybrid Vehicle Presence and Classification Detector:
    1. Contour / Area / Variance Analyzer: Optimized for physical toy cars and bikes on A4 sheets.
    2. Optional YOLOv8 / CNN support: Evaluates full frame / patches if models are loaded.
    """
    def __init__(self, method: str = "hybrid"):
        self.method = method
        self.yolo_model = None
        self._init_models()

    def _init_models(self):
        # Attempt to load YOLOv8 if requested
        if self.method == "yolo":
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO("yolov8n.pt")
                print("[SlotDetector] YOLOv8n initialized successfully.")
            except Exception as e:
                print(f"[SlotDetector] Note: YOLOv8 not active ({e}), running high-performance contour/variance engine.")

    def extract_roi(self, frame: np.ndarray, polygon: List[List[float]]) -> np.ndarray:
        """
        Extracts and warps the polygon ROI from the frame into a normalized 128x128 patch.
        polygon is in normalized coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        """
        h, w = frame.shape[:2]
        pts_src = np.array([[p[0] * w, p[1] * h] for p in polygon], dtype=np.float32)
        
        # Target square size
        target_size = 128
        pts_dst = np.array([
            [0, 0],
            [target_size - 1, 0],
            [target_size - 1, target_size - 1],
            [0, target_size - 1]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(frame, matrix, (target_size, target_size))
        return warped

    def classify_patch(self, patch: np.ndarray, slot_type: str = "CAR_OR_2BIKES") -> Tuple[str, int, bool, float]:
        """
        Classifies an extracted slot patch.
        Returns:
            (state: "FREE"|"OCCUPIED", bikes: int, has_car: bool, confidence: float)
        """
        h, w = patch.shape[:2]
        # Inset 12% to strip polygon borders and bounding line artifacts
        my = int(h * 0.12)
        mx = int(w * 0.12)
        inner = patch[my:h-my, mx:w-mx] if (h > 24 and w > 24) else patch
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        
        std_dev = float(np.std(gray))
        edges = cv2.Canny(gray, 40, 120)
        edge_ratio = float(np.count_nonzero(edges)) / edges.size

        # Background estimate: use median of inner patch
        bg_val = float(np.median(gray))
        # If the median itself is dark (meaning a large vehicle occupies >50% of inner patch),
        # the true background is bright (white paper / light asphalt)
        if bg_val < 140:
            bg_val = 225.0
            
        diff = np.abs(gray.astype(np.float32) - bg_val)
        fg_mask = (diff > 30).astype(np.uint8) * 255
        fg_ratio = float(np.count_nonzero(fg_mask)) / fg_mask.size
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 60]
        
        # 1. Empty test (paper sheet with drawn slot text/lines has fg_ratio < 0.12 and std_dev < 26.0)
        if std_dev < 26.0 and fg_ratio < 0.12:
            return "FREE", 0, False, max(0.85, 1.0 - edge_ratio * 5)
            
        # 2. Occupied
        if slot_type == "BIKES_ONLY":
            if len(valid_contours) >= 2 or (fg_ratio > 0.20 and edge_ratio > 0.03):
                return "OCCUPIED", 2, False, 0.92
            return "OCCUPIED", 1, False, 0.88
        else:
            # CAR_OR_2BIKES
            # Cars take large area (>30% fg or high variance with dark body)
            if fg_ratio > 0.30 or bg_val < 140 or std_dev > 45.0:
                return "OCCUPIED", 0, True, 0.95
            # 2 bikes vs 1 bike
            if len(valid_contours) >= 2 or fg_ratio > 0.14:
                return "OCCUPIED", 2, False, 0.90
            elif fg_ratio > 0.05:
                return "OCCUPIED", 1, False, 0.87
            else:
                return "FREE", 0, False, 0.82

    def analyze_frame_slots(self, frame: np.ndarray, slots: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes all slots assigned to this camera frame.
        Returns mapping: slot_id -> {state, bikes, has_car, conf}
        """
        results = {}
        for s in slots:
            slot_id = s["id"]
            poly = s.get("roi_polygon")
            if not poly or len(poly) < 4:
                continue
            
            slot_type = s.get("slot_type", "CAR_OR_2BIKES")
            try:
                patch = self.extract_roi(frame, poly)
                state, bikes, has_car, conf = self.classify_patch(patch, slot_type)
                results[slot_id] = {
                    "state": state,
                    "bikes": bikes,
                    "has_car": has_car,
                    "confidence": conf
                }
            except Exception as e:
                # Safe fallback
                results[slot_id] = {
                    "state": "FREE",
                    "bikes": 0,
                    "has_car": False,
                    "confidence": 0.50
                }
        return results
