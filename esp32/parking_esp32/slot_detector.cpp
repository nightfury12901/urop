/**
 * slot_detector.cpp
 * Implementation of edge-side parking slot occupancy detection.
 */
#include "slot_detector.h"
#include <SPIFFS.h>
#include <ArduinoJson.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

// ── Config file path ──────────────────────────────────────────────
static const char* CONFIG_FILE = "/slot_config.json";

// ── Default ROI generation ────────────────────────────────────────
void generateDefaultROIs(SlotROI* rois, int* slotCount) {
  // Build a uniform SLOT_ROWS x SLOT_COLS grid
  // Leave 5% margin on each edge
  const float marginX = 0.05f;
  const float marginY = 0.05f;
  const float slotW = (1.0f - 2 * marginX) / SLOT_COLS;
  const float slotH = (1.0f - 2 * marginY) / SLOT_ROWS;

  int idx = 0;
  for (int row = 0; row < SLOT_ROWS; row++) {
    for (int col = 0; col < SLOT_COLS; col++) {
      SlotROI& r = rois[idx];
      snprintf(r.id, sizeof(r.id), "%s-%02d", PARKING_ZONE_ID, idx + 1);
      r.x0 = marginX + col * slotW;
      r.y0 = marginY + row * slotH;
      r.x1 = r.x0 + slotW;
      r.y1 = r.y0 + slotH;
      r.occupied = false;
      idx++;
    }
  }
  *slotCount = idx;
  Serial.printf("[SlotDetector] Generated %d default ROIs (%dx%d grid)\n",
                idx, SLOT_ROWS, SLOT_COLS);
}

// ── SPIFFS persistence ────────────────────────────────────────────
bool loadSlotConfig(SlotROI* rois, int* slotCount, int maxSlots) {
  if (!SPIFFS.exists(CONFIG_FILE)) {
    Serial.println("[SlotDetector] No config file — using defaults");
    generateDefaultROIs(rois, slotCount);
    saveSlotConfig(rois, *slotCount);
    return true;
  }

  File f = SPIFFS.open(CONFIG_FILE, "r");
  if (!f) {
    Serial.println("[SlotDetector] Failed to open config file");
    generateDefaultROIs(rois, slotCount);
    return false;
  }

  // Use a dynamic JSON document sized for up to 50 slots
  DynamicJsonDocument doc(8192);
  DeserializationError err = deserializeJson(doc, f);
  f.close();

  if (err) {
    Serial.printf("[SlotDetector] JSON parse error: %s\n", err.c_str());
    generateDefaultROIs(rois, slotCount);
    return false;
  }

  JsonArray slots = doc["slots"].as<JsonArray>();
  int count = 0;
  for (JsonObject s : slots) {
    if (count >= maxSlots) break;
    strlcpy(rois[count].id, s["id"] | "UNKN", sizeof(rois[count].id));
    rois[count].x0 = s["x0"] | 0.0f;
    rois[count].y0 = s["y0"] | 0.0f;
    rois[count].x1 = s["x1"] | 1.0f;
    rois[count].y1 = s["y1"] | 1.0f;
    rois[count].occupied = false;
    count++;
  }
  *slotCount = count;
  Serial.printf("[SlotDetector] Loaded %d slots from SPIFFS\n", count);
  return count > 0;
}

bool saveSlotConfig(const SlotROI* rois, int slotCount) {
  DynamicJsonDocument doc(8192);
  JsonArray arr = doc.createNestedArray("slots");
  for (int i = 0; i < slotCount; i++) {
    JsonObject s = arr.createNestedObject();
    s["id"] = rois[i].id;
    s["x0"] = rois[i].x0;
    s["y0"] = rois[i].y0;
    s["x1"] = rois[i].x1;
    s["y1"] = rois[i].y1;
  }
  doc["zone"]  = PARKING_ZONE_ID;
  doc["rows"]  = SLOT_ROWS;
  doc["cols"]  = SLOT_COLS;

  File f = SPIFFS.open(CONFIG_FILE, "w");
  if (!f) {
    Serial.println("[SlotDetector] Failed to write config file");
    return false;
  }
  serializeJson(doc, f);
  f.close();
  Serial.printf("[SlotDetector] Saved %d slot ROIs to SPIFFS\n", slotCount);
  return true;
}

// ── Core occupancy analysis ───────────────────────────────────────
SlotResult analyseSlot(const uint8_t* frameBuf, int frameW, int frameH,
                       const SlotROI& roi) {
  SlotResult result = {false, 0.0f, 0.0f, 0.0f};

  // Convert normalised ROI to pixel coords
  int x0 = (int)(roi.x0 * frameW);
  int y0 = (int)(roi.y0 * frameH);
  int x1 = (int)(roi.x1 * frameW);
  int y1 = (int)(roi.y1 * frameH);

  // Clamp
  x0 = max(0, min(x0, frameW - 1));
  x1 = max(0, min(x1, frameW - 1));
  y0 = max(0, min(y0, frameH - 1));
  y1 = max(0, min(y1, frameH - 1));

  int w = x1 - x0;
  int h = y1 - y0;
  if (w < 4 || h < 4) return result;

  // Inset border to skip line pixels
  int inX = max(1, (int)(w * BORDER_INSET));
  int inY = max(1, (int)(h * BORDER_INSET));
  int ix0 = x0 + inX, ix1 = x1 - inX;
  int iy0 = y0 + inY, iy1 = y1 - inY;

  if (ix1 <= ix0 || iy1 <= iy0) return result;

  // ── Pass 1: compute mean ──────────────────────────────────────
  float sum = 0.0f;
  int   n   = 0;
  for (int y = iy0; y < iy1; y++) {
    for (int x = ix0; x < ix1; x++) {
      sum += frameBuf[y * frameW + x];
      n++;
    }
  }
  if (n == 0) return result;
  float mean = sum / n;

  // Background estimate: if mean is very dark a vehicle fills >50% of patch;
  // true background (asphalt/paper) is light, so flip to 225
  float bgVal = (mean < 140.0f) ? 225.0f : mean;

  // ── Pass 2: std-dev and foreground ratio ──────────────────────
  float varSum   = 0.0f;
  int   fgCount  = 0;
  for (int y = iy0; y < iy1; y++) {
    for (int x = ix0; x < ix1; x++) {
      float pix  = (float)frameBuf[y * frameW + x];
      float diff = pix - mean;
      varSum    += diff * diff;
      if (fabsf(pix - bgVal) > FG_DIFF_THRESHOLD) fgCount++;
    }
  }
  float stdDev  = sqrtf(varSum / n);
  float fgRatio = (float)fgCount / n;

  result.mean     = mean;
  result.std_dev  = stdDev;
  result.fg_ratio = fgRatio;

  // ── Classification (mirrors Python classify_patch) ────────────
  // FREE: low texture variance + low foreground coverage
  if (stdDev < FREE_STD_THRESHOLD && fgRatio < FREE_FG_RATIO) {
    result.occupied = false;
  } else {
    result.occupied = true;
  }

  return result;
}

void analyseAllSlots(const uint8_t* frameBuf, int frameW, int frameH,
                     SlotROI* rois, int slotCount) {
  for (int i = 0; i < slotCount; i++) {
    SlotResult r = analyseSlot(frameBuf, frameW, frameH, rois[i]);
    rois[i].occupied = r.occupied;
  }
}
