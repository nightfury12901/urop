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

// Station mode credentials (your college/lab WiFi)
#define WIFI_STA_SSID      "YourWiFiSSID"
#define WIFI_STA_PASSWORD  "YourWiFiPassword"

// Access Point mode credentials (used when WIFI_MODE_STATION = 0)
#define WIFI_AP_SSID       "ESP32-Parking-P1"
#define WIFI_AP_PASSWORD   "parking123"

// ── Parking Zone ─────────────────────────────────────────────────
// Zone identifier reported in API responses
#define PARKING_ZONE_ID    "P1"
#define PARKING_ZONE_NAME  "Student Parking (Bikes Only)"

// Grid layout — must match campus_config.json for this zone
#define SLOT_ROWS          4       // Rows of parking slots
#define SLOT_COLS          5       // Columns of parking slots
#define TOTAL_SLOTS        (SLOT_ROWS * SLOT_COLS)  // 20 for P1

// ── Detection Thresholds ─────────────────────────────────────────
// A slot patch is FREE if its grayscale std-deviation < this value
// (empty asphalt/paper is uniform; vehicles create high variance)
#define FREE_STD_THRESHOLD   28.0f

// A slot patch is FREE if its foreground pixel ratio < this value
// (vehicles cover a large percentage of the slot area)
#define FREE_FG_RATIO        0.10f

// Foreground pixel: |pixel - background| > this many gray levels
#define FG_DIFF_THRESHOLD    30

// Border inset (fraction of patch) to ignore slot boundary lines
#define BORDER_INSET         0.12f

// ── Auto-Calibration Settings ────────────────────────────────────
// Minimum line length (as fraction of frame dimension) to count as grid line
#define AUTOCAL_MIN_LINE_RATIO   0.28f

// Gradient threshold for line detection (edge strength 0-255)
#define AUTOCAL_EDGE_THRESHOLD   20

// ── Camera Settings ──────────────────────────────────────────────
// Resolution for occupancy DETECTION (smaller = faster inference)
// Options: FRAMESIZE_QQVGA (160x120), FRAMESIZE_QVGA (320x240), FRAMESIZE_CIF (400x296)
#define DETECTION_FRAMESIZE  FRAMESIZE_QVGA

// Resolution for MJPEG live stream endpoint
#define STREAM_FRAMESIZE     FRAMESIZE_VGA

// ── HTTP Server ───────────────────────────────────────────────────
#define HTTP_PORT            80

// ── Status LED (AI-Thinker built-in LED on GPIO 33, active LOW) ──
#define STATUS_LED_PIN       33
#define LED_ON               LOW
#define LED_OFF              HIGH
