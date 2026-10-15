/**
 * auto_calibrate.cpp
 * Lightweight grid-line detection for ESP32 QVGA grayscale frames.
 * No OpenCV — pure C array operations to fit in 520KB SRAM.
 */
#include "auto_calibrate.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ── Internal helpers ──────────────────────────────────────────────

/**
 * Compute a 1D brightness profile by averaging pixels along one axis.
 * axis=0 → horizontal profile (one value per row, averaged across columns)
 * axis=1 → vertical profile   (one value per col, averaged across rows)
 */
static void computeProfile(const uint8_t* buf, int W, int H,
                            int axis, float* profile, int* profileLen) {
  if (axis == 0) {
    // Row profile: average brightness of each row
    *profileLen = H;
    for (int y = 0; y < H; y++) {
      float sum = 0;
      for (int x = 0; x < W; x++) sum += buf[y * W + x];
      profile[y] = sum / W;
    }
  } else {
    // Column profile: average brightness of each column
    *profileLen = W;
    for (int x = 0; x < W; x++) {
      float sum = 0;
      for (int y = 0; y < H; y++) sum += buf[y * W + x];
      profile[x] = sum / H;
    }
  }
}

/**
 * Find local peaks in the absolute first-derivative of a profile.
 * A peak at position i means there is a sharp brightness transition
 * at pixel i (= a line crossing at that row/column).
 *
 * Returns the number of peaks found (sorted ascending).
 * peaks[]: output array, caller must ensure size >= profileLen
 */
static int findPeaks(const float* profile, int len,
                     float edgeThresh, int* peaks) {
  int count = 0;
  for (int i = 1; i < len - 1; i++) {
    float d = fabsf(profile[i] - profile[i - 1]);
    if (d >= edgeThresh) {
      // Local maximum of derivative
      if (count == 0 || i - peaks[count - 1] > 3) {
        peaks[count++] = i;
      } else if (d > fabsf(profile[peaks[count - 1]] - profile[peaks[count - 1] - 1])) {
        peaks[count - 1] = i; // Replace with stronger peak nearby
      }
    }
  }
  return count;
}

/**
 * Cluster peak positions that are within `gap` pixels of each other.
 * Replaces each cluster with its mean position.
 * Returns number of clusters.
 */
static int clusterPeaks(int* peaks, int count, int gap, int* centres) {
  if (count == 0) return 0;

  // peaks[] is already sorted ascending from findPeaks
  int nClusters = 0;
  int clusterStart = 0;

  for (int i = 1; i <= count; i++) {
    bool flush = (i == count) || (peaks[i] - peaks[i - 1] > gap);
    if (flush) {
      // Mean of cluster [clusterStart, i-1]
      float sum = 0;
      for (int j = clusterStart; j < i; j++) sum += peaks[j];
      centres[nClusters++] = (int)(sum / (i - clusterStart));
      clusterStart = i;
    }
  }
  return nClusters;
}

/**
 * Select exactly `targetCount` evenly-spaced positions from `centres`.
 * If fewer centres than needed: interpolate between min and max.
 * If more centres than needed: pick evenly spaced subset.
 */
static void regularise(int* centres, int count, int targetCount,
                        int dimSize, int* out) {
  if (count >= targetCount) {
    // Pick targetCount evenly spaced
    for (int i = 0; i < targetCount; i++) {
      int idx = (int)round((float)i * (count - 1) / (targetCount - 1));
      out[i] = centres[idx];
    }
  } else if (count >= 2) {
    // Interpolate between first and last detected lines
    for (int i = 0; i < targetCount; i++) {
      out[i] = centres[0] + (int)((float)i * (centres[count - 1] - centres[0]) /
                                   (targetCount - 1));
    }
  } else {
    // Fallback: evenly divide frame with 5% margin
    int margin = (int)(dimSize * 0.05f);
    for (int i = 0; i < targetCount; i++) {
      out[i] = margin + (int)((float)i * (dimSize - 2 * margin) / (targetCount - 1));
    }
  }
}

// ── Public API ────────────────────────────────────────────────────
bool autoDetectSlots(const uint8_t* frameBuf, int frameW, int frameH,
                     SlotROI* rois, int* slotCount) {
  // Allocate profiles on heap (QVGA: max 320 floats per profile)
  float* rowProfile = (float*)malloc(frameH * sizeof(float));
  float* colProfile = (float*)malloc(frameW * sizeof(float));
  int*   rowPeaks   = (int*)malloc(frameH * sizeof(int));
  int*   colPeaks   = (int*)malloc(frameW * sizeof(int));
  int*   rowCentres = (int*)malloc(frameH * sizeof(int));
  int*   colCentres = (int*)malloc(frameW * sizeof(int));
  int*   hLines     = (int*)malloc((SLOT_ROWS + 1) * sizeof(int));
  int*   vLines     = (int*)malloc((SLOT_COLS + 1) * sizeof(int));

  if (!rowProfile || !colProfile || !rowPeaks || !colPeaks ||
      !rowCentres || !colCentres || !hLines || !vLines) {
    Serial.println("[AutoCal] Heap allocation failed");
    free(rowProfile); free(colProfile);
    free(rowPeaks);   free(colPeaks);
    free(rowCentres); free(colCentres);
    free(hLines);     free(vLines);
    return false;
  }

  // ── Compute brightness profiles ─────────────────────────────────
  int rowLen = 0, colLen = 0;
  computeProfile(frameBuf, frameW, frameH, 0, rowProfile, &rowLen);
  computeProfile(frameBuf, frameW, frameH, 1, colProfile, &colLen);

  // ── Detect edge peaks ───────────────────────────────────────────
  float edgeThresh = AUTOCAL_EDGE_THRESHOLD;
  int   nRowPeaks  = findPeaks(rowProfile, rowLen, edgeThresh, rowPeaks);
  int   nColPeaks  = findPeaks(colProfile, colLen, edgeThresh, colPeaks);

  Serial.printf("[AutoCal] Row peaks: %d, Col peaks: %d\n", nRowPeaks, nColPeaks);

  // ── Cluster peaks ────────────────────────────────────────────────
  int gapH = (int)(frameH * 0.04f);  // 4% of frame height
  int gapW = (int)(frameW * 0.04f);  // 4% of frame width
  int nRowCentres = clusterPeaks(rowPeaks, nRowPeaks, gapH, rowCentres);
  int nColCentres = clusterPeaks(colPeaks, nColPeaks, gapW, colCentres);

  Serial.printf("[AutoCal] Row clusters: %d (need %d), Col clusters: %d (need %d)\n",
                nRowCentres, SLOT_ROWS + 1, nColCentres, SLOT_COLS + 1);

  // ── Regularise to expected grid ──────────────────────────────────
  regularise(rowCentres, nRowCentres, SLOT_ROWS + 1, frameH, hLines);
  regularise(colCentres, nColCentres, SLOT_COLS + 1, frameW, vLines);

  // ── Generate slot ROIs ───────────────────────────────────────────
  int idx = 0;
  for (int row = 0; row < SLOT_ROWS; row++) {
    for (int col = 0; col < SLOT_COLS; col++) {
      SlotROI& r = rois[idx];
      snprintf(r.id, sizeof(r.id), "%s-%02d", PARKING_ZONE_ID, idx + 1);
      r.x0 = (float)vLines[col]     / frameW;
      r.y0 = (float)hLines[row]     / frameH;
      r.x1 = (float)vLines[col + 1] / frameW;
      r.y1 = (float)hLines[row + 1] / frameH;
      // Clamp
      r.x0 = max(0.0f, min(1.0f, r.x0));
      r.y0 = max(0.0f, min(1.0f, r.y0));
      r.x1 = max(0.0f, min(1.0f, r.x1));
      r.y1 = max(0.0f, min(1.0f, r.y1));
      r.occupied = false;
      idx++;
    }
  }
  *slotCount = idx;

  bool success = (nRowCentres >= 2 && nColCentres >= 2);
  Serial.printf("[AutoCal] %s — %d ROIs generated\n",
                success ? "SUCCESS" : "PARTIAL (used interpolation)", idx);

  free(rowProfile); free(colProfile);
  free(rowPeaks);   free(colPeaks);
  free(rowCentres); free(colCentres);
  free(hLines);     free(vLines);

  return success;
}
