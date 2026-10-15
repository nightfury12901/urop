/**
 * slot_detector.h
 * Edge-side parking slot occupancy detection for ESP32 + OV2640.
 *
 * Algorithm (mirrors the Python SlotDetector.classify_patch logic):
 *   1. Extract a rectangular patch from the QVGA grayscale frame
 *      corresponding to a slot ROI (normalized coordinates stored in SPIFFS)
 *   2. Inset border by BORDER_INSET to exclude the painted line pixels
 *   3. Compute mean, std-deviation, and foreground pixel ratio of the patch
 *   4. Classify as FREE or OCCUPIED based on thresholds in config.h
 */
#pragma once
#include <Arduino.h>
#include "config.h"

// ── Slot ROI (stored in SPIFFS as JSON) ──────────────────────────
struct SlotROI {
  char  id[12];        // e.g. "P1-01"
  float x0, y0;        // top-left  (normalised 0..1)
  float x1, y1;        // bottom-right (normalised 0..1)
  bool  occupied;      // last detection result
};

// ── Detection result ─────────────────────────────────────────────
struct SlotResult {
  bool  occupied;
  float std_dev;
  float fg_ratio;
  float mean;
};

// ── API ───────────────────────────────────────────────────────────

/**
 * Load slot ROI configuration from SPIFFS (/slot_config.json).
 * Returns true on success. Populates rois[] and sets slotCount.
 */
bool loadSlotConfig(SlotROI* rois, int* slotCount, int maxSlots);

/**
 * Save slot ROI configuration to SPIFFS (/slot_config.json).
 */
bool saveSlotConfig(const SlotROI* rois, int slotCount);

/**
 * Generate default evenly-spaced grid ROIs for SLOT_ROWS x SLOT_COLS.
 * Used on first boot before any calibration.
 */
void generateDefaultROIs(SlotROI* rois, int* slotCount);

/**
 * Analyse a single slot patch from a QVGA grayscale frame buffer.
 * frameBuf: pointer to raw 8-bit grayscale pixels (width x height)
 * frameW, frameH: frame dimensions
 * roi: the slot region (normalised coords)
 * Returns SlotResult with occupancy decision.
 */
SlotResult analyseSlot(const uint8_t* frameBuf, int frameW, int frameH,
                       const SlotROI& roi);

/**
 * Run analyseSlot on every slot in rois[] and update roi.occupied.
 */
void analyseAllSlots(const uint8_t* frameBuf, int frameW, int frameH,
                     SlotROI* rois, int slotCount);
