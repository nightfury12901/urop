/**
 * auto_calibrate.cpp
 * Automatic grid-line and slot detection for ESP32.
 * Supports dual-row parking lot layouts (top row + bottom row with central aisle).
 */
#include "auto_calibrate.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ── Internal 1D profile helpers ───────────────────────────────────

static void computeProfile(const uint8_t* buf, int W, int H,
                            int axis, float* profile, int* profileLen) {
  if (axis == 0) {
    *profileLen = H;
    for (int y = 0; y < H; y++) {
      float sum = 0;
      for (int x = 0; x < W; x++) sum += buf[y * W + x];
      profile[y] = sum / W;
    }
  } else {
    *profileLen = W;
    for (int x = 0; x < W; x++) {
      float sum = 0;
      for (int y = 0; y < H; y++) sum += buf[y * W + x];
      profile[x] = sum / H;
    }
  }
}

static int findPeaks(const float* profile, int len,
                     float edgeThresh, int* peaks) {
  int count = 0;
  for (int i = 1; i < len - 1; i++) {
    float d = fabsf(profile[i] - profile[i - 1]);
    if (d >= edgeThresh) {
      if (count == 0 || i - peaks[count - 1] > 3) {
        peaks[count++] = i;
      } else if (d > fabsf(profile[peaks[count - 1]] - profile[peaks[count - 1] - 1])) {
        peaks[count - 1] = i;
      }
    }
  }
  return count;
}

static int clusterPeaks(int* peaks, int count, int gap, int* centres) {
  if (count == 0) return 0;
  int nClusters = 0;
  int clusterStart = 0;
  for (int i = 1; i <= count; i++) {
    bool flush = (i == count) || (peaks[i] - peaks[i - 1] > gap);
    if (flush) {
      float sum = 0;
      for (int j = clusterStart; j < i; j++) sum += peaks[j];
      centres[nClusters++] = (int)(sum / (i - clusterStart));
      clusterStart = i;
    }
  }
  return nClusters;
}

// ── Public API ────────────────────────────────────────────────────
bool autoDetectSlots(const uint8_t* frameBuf, int frameW, int frameH,
                     SlotROI* rois, int* slotCount) {
  // Allocate profiles
  float* rowProfile = (float*)malloc(frameH * sizeof(float));
  float* colProfile = (float*)malloc(frameW * sizeof(float));
  int*   rowPeaks   = (int*)malloc(frameH * sizeof(int));
  int*   colPeaks   = (int*)malloc(frameW * sizeof(int));
  int*   rowCentres = (int*)malloc(frameH * sizeof(int));
  int*   colCentres = (int*)malloc(frameW * sizeof(int));

  if (!rowProfile || !colProfile || !rowPeaks || !colPeaks ||
      !rowCentres || !colCentres) {
    Serial.println("[AutoCal] Heap allocation failed");
    free(rowProfile); free(colProfile);
    free(rowPeaks);   free(colPeaks);
    free(rowCentres); free(colCentres);
    generateDefaultROIs(rois, slotCount);
    return false;
  }

  // Compute overall row and column gradients
  int rowLen = 0, colLen = 0;
  computeProfile(frameBuf, frameW, frameH, 0, rowProfile, &rowLen);
  computeProfile(frameBuf, frameW, frameH, 1, colProfile, &colLen);

  int nRowPeaks = findPeaks(rowProfile, rowLen, AUTOCAL_EDGE_THRESHOLD, rowPeaks);
  int nColPeaks = findPeaks(colProfile, colLen, AUTOCAL_EDGE_THRESHOLD, colPeaks);

  int gapH = (int)(frameH * 0.04f);
  int gapW = (int)(frameW * 0.04f);
  int nRowCentres = clusterPeaks(rowPeaks, nRowPeaks, gapH, rowCentres);
  int nColCentres = clusterPeaks(colPeaks, nColPeaks, gapW, colCentres);

  Serial.printf("[AutoCal] Found %d row lines, %d col lines\n", nRowCentres, nColCentres);

  // If we found at least 2 horizontal row markers and 2 vertical markers,
  // adapt the detected boundaries:
  float topY0 = 0.12f, topY1 = 0.38f;
  float botY0 = 0.62f, botY1 = 0.88f;

  if (nRowCentres >= 2) {
    // Top boundary line
    float yA = (float)rowCentres[0] / frameH;
    float yB = (float)rowCentres[nRowCentres - 1] / frameH;
    if (yA < yB && yB - yA > 0.40f) {
      topY0 = max(0.06f, yA);
      topY1 = topY0 + 0.25f;
      botY1 = min(0.94f, yB);
      botY0 = botY1 - 0.25f;
    }
  }

  // Generate the 20 slots (8 top + 12 bottom)
  int idx = 0;

  // ── Top Row: 8 slots on right half ──
  const float topX0 = 0.48f;
  const float topX1 = 0.92f;
  const int   topCount = 8;
  const float topW = (topX1 - topX0) / topCount;

  for (int i = 0; i < topCount; i++) {
    SlotROI& r = rois[idx];
    snprintf(r.id, sizeof(r.id), "%s-%02d", PARKING_ZONE_ID, idx + 1);
    r.x0 = topX0 + i * topW;
    r.y0 = topY0;
    r.x1 = r.x0 + topW;
    r.y1 = topY1;
    r.occupied = false;
    idx++;
  }

  // ── Bottom Row: 12 slots along bottom ──
  const float botX0 = 0.10f;
  const float botX1 = 0.92f;
  const int   botCount = 12;
  const float botW = (botX1 - botX0) / botCount;

  for (int i = 0; i < botCount; i++) {
    SlotROI& r = rois[idx];
    snprintf(r.id, sizeof(r.id), "%s-%02d", PARKING_ZONE_ID, idx + 1);
    r.x0 = botX0 + i * botW;
    r.y0 = botY0;
    r.x1 = r.x0 + botW;
    r.y1 = botY1;
    r.occupied = false;
    idx++;
  }

  *slotCount = idx;
  Serial.printf("[AutoCal] Configured %d slot ROIs\n", idx);

  free(rowProfile); free(colProfile);
  free(rowPeaks);   free(colPeaks);
  free(rowCentres); free(colCentres);

  return true;
}
