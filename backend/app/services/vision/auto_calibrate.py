"""
auto_calibrate.py
-----------------
Intelligent, structure-aware automatic parking slot ROI detection from camera feed.

Supports:
  - Multi-bay layouts with central driving aisles (e.g. 2 bays, top and bottom rows)
  - Dynamic discovery of bays and dividers from drawn pen/pencil lines or real markings
  - Extrapolation for partially drawn grids
  - Seamless hot-reloading into backend vision worker
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
from typing import Optional, List, Tuple, Dict, Any
import json
import os


class AutoCalibrationResult:
    def __init__(self, success: bool, polygons=None, bays_found=None, slot_count=0, message=""):
        self.success = success
        self.polygons = polygons or []
        self.slot_count = slot_count or len(self.polygons)
        self.bays_found = bays_found or []
        self.h_lines = []
        self.v_lines = []
        self.message = message

    def to_dict(self):
        return {
            "success": self.success,
            "polygons": self.polygons,
            "slot_count": self.slot_count,
            "bays_count": len(self.bays_found),
            "message": self.message,
        }


def auto_detect_slots(
    frame: np.ndarray,
    expected_rows: int = 2,
    expected_cols: int = 10,
    min_line_length_ratio: float = 0.08,
    hough_threshold: int = 35,
    structure: str = "auto",
    target_slots: Optional[int] = None
) -> AutoCalibrationResult:
    """
    Main entry point for automatic parking slot grid detection.

    Args:
        frame: BGR frame from camera
        expected_rows: number of parking bays/rows (default 2)
        expected_cols: number of slots per bay/row (default 10)
        structure: "auto" (detect bays dynamically), "dual_bay" (top/bottom with aisle), "uniform" (strict grid)
        target_slots: total number of slots to generate (defaults to expected_rows * expected_cols)
    """
    if frame is None or frame.size == 0:
        return AutoCalibrationResult(False, message="Empty frame received")

    h, w = frame.shape[:2]
    total_slots = target_slots or (expected_rows * expected_cols)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive threshold to isolate lines (works for dark drawn lines on paper or bright lines on asphalt)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=10
    )

    # Morphological kernels to extract horizontal and vertical lines
    v_len = max(18, int(h * 0.04))
    h_len = max(22, int(w * 0.04))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))

    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    bays: List[Tuple[int, int]] = []

    if structure == "uniform":
        # Strict uniform grid mode without aisles
        row_h = h / expected_rows
        for r in range(expected_rows):
            bays.append((int(r * row_h), int((r + 1) * row_h)))
    else:
        # ── 1. Dynamic Bay Detection along Y axis via vertical line projection ──
        y_proj = np.sum(v_lines > 0, axis=1)
        y_smooth = np.convolve(y_proj, np.ones(9) / 9.0, mode='same')
        y_thresh = np.max(y_smooth) * 0.22 if np.max(y_smooth) > 0 else 10.0

        in_bay = False
        start_y = 0
        for y in range(h):
            if y_smooth[y] > y_thresh and not in_bay:
                in_bay = True
                start_y = y
            elif y_smooth[y] <= y_thresh and in_bay:
                in_bay = False
                if y - start_y > 35:  # bay height must be at least 35px
                    bays.append((start_y, y))
        if in_bay and (h - start_y > 35):
            bays.append((start_y, h))

        # Fallback if no distinct bays detected: default to expected_rows bays
        if not bays:
            if expected_rows == 2:
                # Top bay (16% to 46%) and bottom bay (54% to 86%), aisle in middle
                bays = [(int(h * 0.16), int(h * 0.46)), (int(h * 0.54), int(h * 0.86))]
            else:
                bays = [(int(h * 0.12), int(h * 0.88))]

    # Distribute total slots evenly across detected bays
    num_bays = len(bays)
    slots_per_bay = []
    base_count = total_slots // num_bays
    remainder = total_slots % num_bays
    for b in range(num_bays):
        slots_per_bay.append(base_count + (1 if b < remainder else 0))

    all_slot_polygons = []

    for bay_idx, (by0, by1) in enumerate(bays):
        bay_v = v_lines[by0:by1, :]
        x_proj = np.sum(bay_v > 0, axis=0)

        # Detect vertical divider peaks along X
        peaks, _ = find_peaks(x_proj, distance=max(22, int(w * 0.04)), prominence=8)
        # Filter out extreme margin (x < 18) or extreme right margin (x > w - 10)
        peaks = [int(p) for p in peaks if 18 <= p <= w - 10]

        actual_slot_spans = []
        for i in range(len(peaks) - 1):
            x0, x1 = peaks[i], peaks[i + 1]
            slot_w = x1 - x0
            # Only genuine drawn parking slots: width between 28px and 68px
            # Large gaps (e.g. blank empty paper areas) are strictly skipped
            if 28 <= slot_w <= 68:
                actual_slot_spans.append((x0, x1))

        if structure == "uniform" or not actual_slot_spans:
            # Fallback for explicit uniform grid mode
            left_x = peaks[0] if len(peaks) >= 2 else int(w * 0.08)
            right_x = peaks[-1] if len(peaks) >= 2 else int(w * 0.95)
            n_slots = slots_per_bay[bay_idx]
            slot_xs = np.linspace(left_x, right_x, n_slots + 1)
            slot_spans = [(slot_xs[i], slot_xs[i + 1]) for i in range(n_slots)]
        else:
            slot_spans = actual_slot_spans

        # Inset 3% from boundaries so the polygon does not overlap drawn divider lines
        slot_h = by1 - by0
        inset_y = max(2, int(slot_h * 0.03))

        for (sx0, sx1) in slot_spans:
            inset_x = max(2, int((sx1 - sx0) * 0.04))

            x0 = max(0.0, (sx0 + inset_x) / w)
            x1 = min(1.0, (sx1 - inset_x) / w)
            y0 = max(0.0, (by0 + inset_y) / h)
            y1 = min(1.0, (by1 - inset_y) / h)

            poly = [
                [round(float(x0), 4), round(float(y0), 4)],
                [round(float(x1), 4), round(float(y0), 4)],
                [round(float(x1), 4), round(float(y1), 4)],
                [round(float(x0), 4), round(float(y1), 4)],
            ]
            all_slot_polygons.append(poly)

    msg = f"Auto-detected {len(all_slot_polygons)} slots across {len(bays)} parking bay(s)."
    return AutoCalibrationResult(
        True,
        polygons=all_slot_polygons,
        bays_found=bays,
        slot_count=len(all_slot_polygons),
        message=msg
    )


def apply_auto_calibration_to_config(config_path: str, area_id: str, polygons: List[List[List[float]]]) -> Tuple[bool, str]:
    """
    Writes detected slot polygons back into campus_config.json for the given area.
    Slot order: row-major (left to right, top to bottom) matching slot numbering.
    """
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)

        area_slots = sorted(
            [s for s in cfg["slots"] if s["area_id"] == area_id],
            key=lambda s: s["slot_number"]
        )

        n_to_update = min(len(area_slots), len(polygons))
        for i in range(n_to_update):
            area_slots[i]["roi_polygon"] = polygons[i]

        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        return True, f"Updated {n_to_update} slots in {area_id}"

    except Exception as e:
        return False, f"Failed to save auto-calibration: {e}"
