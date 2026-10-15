/**
 * auto_calibrate.h
 * Automatic parking slot grid detection on the ESP32.
 *
 * Algorithm (pure C, no OpenCV — designed for QVGA grayscale frames):
 *   1. Compute per-row and per-column brightness profiles
 *   2. Differentiate profiles to find sharp brightness transitions (= lines)
 *   3. Threshold the derivative to get line candidate positions
 *   4. Cluster nearby candidates → grid row/column positions
 *   5. Use the (SLOT_ROWS+1) horizontal and (SLOT_COLS+1) vertical positions
 *      to generate the slot ROI grid
 *   6. Save to SPIFFS via saveSlotConfig()
 */
#pragma once
#include <Arduino.h>
#include "slot_detector.h"
#include "config.h"

/**
 * Run automatic slot detection on a grayscale QVGA frame.
 * Writes detected ROIs into rois[] and updates slotCount.
 * Returns true if enough lines were found to form a valid grid.
 *
 * frameBuf: raw 8-bit grayscale pixel buffer (frameW x frameH)
 */
bool autoDetectSlots(const uint8_t* frameBuf, int frameW, int frameH,
                     SlotROI* rois, int* slotCount);
