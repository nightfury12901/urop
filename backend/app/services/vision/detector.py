import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import os


class SlotDetector:
    """
    Hybrid Vehicle Presence and Classification Detector.

    Calibrated for the Smart Campus Parking demo setup:
    - Camera: ESP32-CAM OV2640 (QVGA/VGA, typically dark/underexposed)
    - Surface: white A4 paper with dark pen/marker drawn slot grid
    - Vehicles: physical toy cars and bikes placed on paper slots

    Detection pipeline per slot patch:
    1. Extract & inset 18% to strip drawn grid-line artifacts at borders
    2. Compute RAW std_dev on gray patch — primary FREE/OCCUPIED discriminator:
         Empty slot (uniform dark or bright background): std_dev < 25
         Physical vehicle (non-uniform, dark mass): std_dev >= 28+
    3. Apply CLAHE normalisation → compute fg_ratio to distinguish
         1 bike vs 2 bikes vs car as secondary check
    """

    def __init__(self, method: str = "hybrid"):
        self.method = method
        self.yolo_model = None
        self._init_models()

    def _init_models(self):
        if self.method == "yolo":
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO("yolov8n.pt")
                print("[SlotDetector] YOLOv8n initialized successfully.")
            except Exception as e:
                print(f"[SlotDetector] Note: YOLOv8 not active ({e}), using contour/variance engine.")

    def extract_roi(self, frame: np.ndarray, polygon: List[List[float]]) -> np.ndarray:
        """
        Extracts and warps the polygon ROI from the frame into a 128x128 patch.
        polygon: normalized coordinates [[x,y], [x,y], [x,y], [x,y]]
        """
        h, w = frame.shape[:2]
        pts_src = np.array([[p[0] * w, p[1] * h] for p in polygon], dtype=np.float32)
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
        Returns: (state: "FREE"|"OCCUPIED", bikes: int, has_car: bool, confidence: float)
        """
        h, w = patch.shape[:2]

        # ── Step 1: strip 18% border to avoid grid-line artifacts ──────────────
        my = int(h * 0.18)
        mx = int(w * 0.18)
        inner = patch[my:h - my, mx:w - mx] if (h > 24 and w > 24) else patch
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)

        # ── Step 2: PRIMARY discriminator — raw standard deviation ─────────────
        # Measured on live ESP32 stream:
        #   Empty paper slot:      std_dev = 2–24   (uniform paper background)
        #   Slot with toy bike:    std_dev ≥ 30+    (distinct 3D dark mass)
        #   Slot with toy car:     std_dev ≥ 40+    (large dark/colored object)
        std_dev = float(np.std(gray))

        # Fast path: low variance = definitively empty
        if std_dev < 24.0:
            edges_quick = cv2.Canny(gray, 40, 120)
            edge_ratio_q = float(np.count_nonzero(edges_quick)) / max(1, edges_quick.size)
            return "FREE", 0, False, max(0.90, 1.0 - edge_ratio_q * 2)

        # ── Step 3: Raw gray foreground extraction (avoids CLAHE noise amplification) ──
        bg_val = float(np.median(gray))
        diff = np.abs(gray.astype(np.float32) - bg_val)
        fg_mask = (diff > 35).astype(np.uint8) * 255
        fg_ratio = float(np.count_nonzero(fg_mask)) / max(1, fg_mask.size)

        edges = cv2.Canny(gray, 40, 120)
        edge_ratio = float(np.count_nonzero(edges)) / max(1, edges.size)

        # Contours on foreground mask — must be substantial to be an actual vehicle
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_vehicle_area = 80.0
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_vehicle_area]

        # ── Step 4: Robust FREE checks ─────────────────────────────────────────
        # If foreground ratio is small and no substantial object contour exists, it is empty paper
        if fg_ratio < 0.13 and len(valid_contours) == 0:
            return "FREE", 0, False, max(0.88, 1.0 - edge_ratio * 2)

        if std_dev < 30.0 and len(valid_contours) == 0:
            return "FREE", 0, False, max(0.85, 1.0 - edge_ratio * 2)

        # ── Step 5: OCCUPIED classification ────────────────────────────────────
        if slot_type == "BIKES_ONLY":
            if len(valid_contours) >= 2 or (fg_ratio > 0.32 and edge_ratio > 0.04):
                return "OCCUPIED", 2, False, 0.92
            elif fg_ratio > 0.14 and len(valid_contours) >= 1:
                return "OCCUPIED", 1, False, 0.88
            else:
                return "FREE", 0, False, 0.90
        else:
            # CAR_OR_2BIKES: large fg or high std = car
            if fg_ratio > 0.38 or std_dev > 50.0:
                return "OCCUPIED", 0, True, 0.95
            elif len(valid_contours) >= 2 or fg_ratio > 0.28:
                return "OCCUPIED", 2, False, 0.90
            elif fg_ratio > 0.14 and len(valid_contours) >= 1:
                return "OCCUPIED", 1, False, 0.86
            else:
                return "FREE", 0, False, 0.90

    def analyze_frame_slots(self, frame: np.ndarray, slots: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes all slots assigned to this camera frame.
        Returns mapping: slot_id -> {state, bikes, has_car, confidence}
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
                results[slot_id] = {
                    "state": "FREE",
                    "bikes": 0,
                    "has_car": False,
                    "confidence": 0.50
                }
        return results
