/**
 * config.h
 * User-editable WiFi credentials and system settings.
 * Edit this file before flashing.
 */
#pragma once

// ── WiFi Settings ────────────────────────────────────────────────
// Set to 1 to connect to your existing WiFi (recommended for integration)
// Set to 0 to create a standalone Access Point (useful for field testing)
#define WIFI_MODE_STATION  1       // 1 = STA (connect), 0 = AP (hotspot)

// Station mode credentials (your college/lab/home WiFi)
#define WIFI_STA_SSID      "YourWiFiSSID"
#define WIFI_STA_PASSWORD  "YourWiFiPassword"

// Access Point mode credentials (used when WIFI_MODE_STATION = 0)
#define WIFI_AP_SSID       "ESP32-Parking-P1"
#define WIFI_AP_PASSWORD   "parking123"

// ── Parking Zone ─────────────────────────────────────────────────
#define PARKING_ZONE_ID    "P1"
#define PARKING_ZONE_NAME  "Student Parking (Bikes Only)"
#define TOTAL_SLOTS        20

// ── Detection Thresholds ─────────────────────────────────────────
// A slot patch is FREE if its grayscale std-deviation < this value
// (empty paper under indoor light is 4-12 std-dev; toy cars/bikes are 25-60+)
#define FREE_STD_THRESHOLD   18.0f

// A slot patch is FREE if foreground pixel ratio < this value
#define FREE_FG_RATIO        0.12f

// Pixel is foreground if |pixel - local_mean| > this value
#define FG_DIFF_THRESHOLD    20

// Border inset (fraction of patch) to ignore slot boundary pencil lines
#define BORDER_INSET         0.10f

// ── Auto-Calibration Settings ────────────────────────────────────
#define AUTOCAL_MIN_LINE_RATIO   0.15f
#define AUTOCAL_EDGE_THRESHOLD   8

// ── Camera Settings ──────────────────────────────────────────────
#define DETECTION_FRAMESIZE  FRAMESIZE_QVGA
#define STREAM_FRAMESIZE     FRAMESIZE_VGA

// ── HTTP Server ───────────────────────────────────────────────────
#define HTTP_PORT            80

// ── Status LED (AI-Thinker built-in LED on GPIO 33, active LOW) ──
#define STATUS_LED_PIN       33
#define LED_ON               LOW
#define LED_OFF              HIGH
