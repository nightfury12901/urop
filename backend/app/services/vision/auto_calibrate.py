"""
auto_calibrate.py
-----------------
Automatic parking slot ROI detection from camera feed.

Pipeline:
  1. Grab a frame from the camera
  2. Detect white/bright parking line markings via adaptive threshold + Hough lines
  3. Cluster the lines into a regular grid of horizontal & vertical separators
  4. Intersect the grid to produce (rows x cols) rectangular slot polygons
  5. Return normalised [[x,y], ...] quads for each slot, ready to write into campus_config.json

Falls back gracefully: if fewer lines are detected than needed, returns None
so the caller knows to prompt the user for manual calibration.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import json
import os


def _cluster_lines(positions, gap):
    if not positions:
        return []
    positions = sorted(positions)
    clusters = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] < gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [float(np.mean(c)) for c in clusters]


def _detect_grid_lines(frame, min_line_length_ratio=0.30, hough_threshold=60):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=-10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    min_h_len = int(w * min_line_length_ratio)
    min_v_len = int(h * min_line_length_ratio)

    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min(min_h_len, min_v_len),
        maxLineGap=20
    )

    h_positions = []
    v_positions = []

    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = np.hypot(dx, dy)
            if dx == 0 and dy == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 15 and length >= min_h_len:
                h_positions.append((y1 + y2) / 2.0)
            elif angle > 75 and length >= min_v_len:
                v_positions.append((x1 + x2) / 2.0)

    h_clustered = _cluster_lines(h_positions, gap=h * 0.05)
    v_clustered = _cluster_lines(v_positions, gap=w * 0.05)
    return h_clustered, v_clustered


def _regularise(positions, target_count):
    if len(positions) >= target_count:
        indices = np.round(np.linspace(0, len(positions) - 1, target_count)).astype(int)
        return [positions[i] for i in indices]
    else:
        return list(np.linspace(positions[0], positions[-1], target_count))


def _lines_to_slot_polygons(h_lines, v_lines, frame_h, frame_w, expected_rows, expected_cols):
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    h = _regularise(h_lines, expected_rows + 1)
    v = _regularise(v_lines, expected_cols + 1)

    polygons = []
    for row in range(expected_rows):
        for col in range(expected_cols):
            y0 = max(0.0, h[row] / frame_h)
            y1 = min(1.0, h[row + 1] / frame_h)
            x0 = max(0.0, v[col] / frame_w)
            x1 = min(1.0, v[col + 1] / frame_w)
            polygons.append([
                [round(x0, 4), round(y0, 4)],
                [round(x1, 4), round(y0, 4)],
                [round(x1, 4), round(y1, 4)],
                [round(x0, 4), round(y1, 4)],
            ])
    return polygons


class AutoCalibrationResult:
    def __init__(self, success, polygons=None, h_lines=None, v_lines=None, frame_shape=None, message=""):
        self.success = success
        self.polygons = polygons or []
        self.h_lines = h_lines or []
        self.v_lines = v_lines or []
        self.frame_shape = frame_shape
        self.message = message

    def to_dict(self):
        return {
            "success": self.success,
            "polygons": self.polygons,
            "h_lines": self.h_lines,
            "v_lines": self.v_lines,
            "frame_shape": list(self.frame_shape) if self.frame_shape else None,
            "message": self.message,
            "slot_count": len(self.polygons),
        }


def auto_detect_slots(frame, expected_rows, expected_cols, min_line_length_ratio=0.25, hough_threshold=50):
    """
    Main entry point. Detects parking slot grid lines from the camera frame and
    returns normalised ROI polygons for each slot.

    Args:
        frame: BGR OpenCV frame
        expected_rows: number of slot rows (e.g. 4 for P1 4x5 grid)
        expected_cols: number of slot columns (e.g. 5 for P1 4x5 grid)
    Returns:
        AutoCalibrationResult
    """
    if frame is None or frame.size == 0:
        return AutoCalibrationResult(False, message="Empty frame received")

    frame_h, frame_w = frame.shape[:2]

    h_lines, v_lines = _detect_grid_lines(
        frame,
        min_line_length_ratio=min_line_length_ratio,
        hough_threshold=hough_threshold,
    )

    if len(h_lines) < 2:
        return AutoCalibrationResult(
            False, h_lines=h_lines, v_lines=v_lines, frame_shape=(frame_h, frame_w),
            message=f"Only {len(h_lines)} horizontal lines detected (need at least 2). "
                    "Markings may not be visible - use manual calibration."
        )

    if len(v_lines) < 2:
        return AutoCalibrationResult(
            False, h_lines=h_lines, v_lines=v_lines, frame_shape=(frame_h, frame_w),
            message=f"Only {len(v_lines)} vertical lines detected (need at least 2). "
                    "Markings may not be visible - use manual calibration."
        )

    polygons = _lines_to_slot_polygons(h_lines, v_lines, frame_h, frame_w, expected_rows, expected_cols)

    if not polygons:
        return AutoCalibrationResult(
            False, h_lines=h_lines, v_lines=v_lines, frame_shape=(frame_h, frame_w),
            message="Could not form slot grid from detected lines."
        )

    return AutoCalibrationResult(
        True,
        polygons=polygons,
        h_lines=h_lines,
        v_lines=v_lines,
        frame_shape=(frame_h, frame_w),
        message=f"Auto-detected {len(polygons)} slots ({expected_rows}x{expected_cols} grid) "
                f"from {len(h_lines)} H-lines and {len(v_lines)} V-lines."
    )


def apply_auto_calibration_to_config(config_path, area_id, polygons):
    """
    Writes detected slot polygons back into campus_config.json for the given area.
    Slot order: row-major (left to right, top to bottom) matching slot numbering.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        area_slots = sorted(
            [s for s in config["slots"] if s["area_id"] == area_id],
            key=lambda s: s["slot_number"]
        )

        if len(polygons) != len(area_slots):
            return False, (
                f"Polygon count mismatch: detected {len(polygons)} polygons "
                f"but area {area_id} has {len(area_slots)} slots."
            )

        for slot, poly in zip(area_slots, polygons):
            slot["roi_polygon"] = poly

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return True, f"Saved {len(polygons)} ROI polygons for area {area_id}"
    except Exception as e:
        return False, f"Failed to update config: {e}"
