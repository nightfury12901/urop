/**
 * parking_esp32.ino
 * ──────────────────────────────────────────────────────────────────────────
 * Smart Campus Parking — ESP32-CAM Edge Node for Zone P1
 * Hardware: AI-Thinker ESP32-CAM + OV2640
 *
 * What this firmware does:
 *   • Captures QVGA frames from OV2640
 *   • Runs on-device occupancy analysis for each parking slot (no cloud needed)
 *   • Auto-detects slot grid from painted lines when /calibrate is called
 *   • Exposes a REST API so the FastAPI server can query slot states
 *   • Streams MJPEG video with slot ROI overlays for the Admin Portal
 *
 * REST API (all GET unless noted):
 *   GET  /              → JSON status summary
 *   GET  /api/slots     → JSON {slot_id: "FREE"/"OCCUPIED", ...} for all slots
 *   GET  /api/snapshot  → single JPEG frame
 *   GET  /stream        → MJPEG live stream (for Admin Portal camera feed)
 *   POST /calibrate     → trigger automatic slot grid detection & save to SPIFFS
 *   POST /api/slots/config → receive ROI config from server (JSON body)
 *   GET  /api/config    → return current slot ROI config JSON
 *
 * Libraries required (install via Arduino Library Manager):
 *   • ArduinoJson  (v6.x by Benoit Blanchon)
 *
 * Board setup (Arduino IDE):
 *   Tools > Board: "AI Thinker ESP32-CAM"
 *   Tools > Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"
 *   Tools > Upload Speed: 115200
 *   Tools > Port: (your COM port)
 * ──────────────────────────────────────────────────────────────────────────
 */

// ── Standard ESP32 camera includes ───────────────────────────────
#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ── WiFi & HTTP ───────────────────────────────────────────────────
#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>

// ── JSON ──────────────────────────────────────────────────────────
#include <ArduinoJson.h>

// ── Local modules ─────────────────────────────────────────────────
#include "camera_config.h"
#include "config.h"
#include "slot_detector.h"
#include "auto_calibrate.h"

// ── Globals ───────────────────────────────────────────────────────
WebServer server(HTTP_PORT);

static SlotROI gRois[TOTAL_SLOTS];
static int     gSlotCount = 0;

// Background inference ticker
static volatile bool  gAnalysisPending   = false;
static volatile bool  gCalibratePending  = false;
static unsigned long  gLastAnalysisMs    = 0;
static const unsigned long ANALYSIS_INTERVAL_MS = 1500; // analyse every 1.5 s

// ── Utility: capture a QVGA grayscale frame ───────────────────────
/**
 * Returns a malloc'd grayscale buffer (frameW * frameH bytes).
 * Caller must free() the buffer.
 * Also sets out frameW / frameH.
 */
static uint8_t* captureGrayscale(int* outW, int* outH) {
  // Temporarily set sensor to QVGA for fast processing
  sensor_t* s = esp_camera_sensor_get();
  s->set_framesize(s, DETECTION_FRAMESIZE);
  delay(100); // Let sensor stabilise

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[Camera] Frame capture failed");
    return nullptr;
  }

  *outW = fb->width;
  *outH = fb->height;
  size_t pixCount = fb->width * fb->height;

  // Convert JPEG to grayscale (OV2640 outputs JPEG by default)
  uint8_t* gray = (uint8_t*)malloc(pixCount);
  if (!gray) {
    esp_camera_fb_return(fb);
    Serial.println("[Camera] Malloc failed for grayscale buffer");
    return nullptr;
  }

  if (fb->format == PIXFORMAT_JPEG) {
    // Decode JPEG → RGB888, then convert to grayscale
    uint8_t* rgb = (uint8_t*)malloc(pixCount * 3);
    if (!rgb) {
      free(gray);
      esp_camera_fb_return(fb);
      return nullptr;
    }
    bool ok = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb);
    if (ok) {
      for (size_t i = 0; i < pixCount; i++) {
        // BT.601 luma: Y = 0.299R + 0.587G + 0.114B
        gray[i] = (uint8_t)(rgb[i*3]*77 + rgb[i*3+1]*150 + rgb[i*3+2]*29) >> 8;
      }
    } else {
      // Fallback: fill with mid-gray
      memset(gray, 128, pixCount);
    }
    free(rgb);
  } else if (fb->format == PIXFORMAT_GRAYSCALE) {
    memcpy(gray, fb->buf, pixCount);
  } else {
    // Unknown format — fill with mid-gray
    memset(gray, 128, pixCount);
  }

  esp_camera_fb_return(fb);
  return gray;
}

// ── Camera initialisation ─────────────────────────────────────────
static bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // PSRAM available → use higher quality buffer
  if (psramFound()) {
    config.frame_size   = STREAM_FRAMESIZE;
    config.jpeg_quality = 10;  // 0-63, lower = better quality
    config.fb_count     = 2;
  } else {
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count     = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[Camera] Init failed: 0x%x\n", err);
    return false;
  }

  // OV2640 sensor tweaks for stable output
  sensor_t* s = esp_camera_sensor_get();
  s->set_vflip(s, 1);         // Flip vertically if camera is mounted upside-down
  s->set_hmirror(s, 0);
  s->set_brightness(s, 1);    // Slight brightness boost
  s->set_saturation(s, 0);
  s->set_whitebal(s, 1);      // Auto white-balance ON
  s->set_awb_gain(s, 1);
  s->set_exposure_ctrl(s, 1); // Auto exposure ON

  Serial.println("[Camera] OV2640 initialised OK");
  return true;
}

// ── HTTP Handlers ─────────────────────────────────────────────────

/** GET / — device info */
void handleRoot() {
  int freeCount = 0;
  for (int i = 0; i < gSlotCount; i++)
    if (!gRois[i].occupied) freeCount++;

  DynamicJsonDocument doc(512);
  doc["device"]      = "ESP32-CAM Parking Node";
  doc["zone"]        = PARKING_ZONE_ID;
  doc["zone_name"]   = PARKING_ZONE_NAME;
  doc["total_slots"] = gSlotCount;
  doc["free_slots"]  = freeCount;
  doc["occupied"]    = gSlotCount - freeCount;
  doc["ip"]          = WiFi.localIP().toString();
  doc["uptime_ms"]   = millis();
  doc["grid"]        = String(SLOT_ROWS) + "x" + String(SLOT_COLS);

  String body;
  serializeJsonPretty(doc, body);
  server.send(200, "application/json", body);
}

/** GET /api/slots — per-slot occupancy JSON (what the FastAPI server polls) */
void handleApiSlots() {
  DynamicJsonDocument doc(2048);
  JsonObject slots = doc.createNestedObject("slots");
  int freeCount = 0;

  for (int i = 0; i < gSlotCount; i++) {
    slots[gRois[i].id] = gRois[i].occupied ? "OCCUPIED" : "FREE";
    if (!gRois[i].occupied) freeCount++;
  }

  doc["zone"]        = PARKING_ZONE_ID;
  doc["total_slots"] = gSlotCount;
  doc["free_slots"]  = freeCount;
  doc["timestamp_ms"] = millis();

  String body;
  serializeJson(doc, body);
  server.send(200, "application/json", body);
}

/** GET /api/config — current slot ROI config */
void handleApiConfig() {
  DynamicJsonDocument doc(8192);
  doc["zone"] = PARKING_ZONE_ID;
  doc["rows"] = SLOT_ROWS;
  doc["cols"] = SLOT_COLS;
  JsonArray arr = doc.createNestedArray("slots");
  for (int i = 0; i < gSlotCount; i++) {
    JsonObject s = arr.createNestedObject();
    s["id"] = gRois[i].id;
    s["x0"] = gRois[i].x0;
    s["y0"] = gRois[i].y0;
    s["x1"] = gRois[i].x1;
    s["y1"] = gRois[i].y1;
  }
  String body;
  serializeJson(doc, body);
  server.send(200, "application/json", body);
}

/** POST /api/slots/config — receive ROI config from server */
void handlePostConfig() {
  if (!server.hasArg("plain")) {
    server.send(400, "application/json", "{\"error\":\"No body\"}");
    return;
  }
  DynamicJsonDocument doc(8192);
  DeserializationError err = deserializeJson(doc, server.arg("plain"));
  if (err) {
    server.send(400, "application/json", "{\"error\":\"Bad JSON\"}");
    return;
  }

  JsonArray arr = doc["slots"].as<JsonArray>();
  int count = 0;
  for (JsonObject s : arr) {
    if (count >= TOTAL_SLOTS) break;
    strlcpy(gRois[count].id, s["id"] | "UNKN", sizeof(gRois[count].id));
    gRois[count].x0 = s["x0"] | 0.0f;
    gRois[count].y0 = s["y0"] | 0.0f;
    gRois[count].x1 = s["x1"] | 1.0f;
    gRois[count].y1 = s["y1"] | 1.0f;
    gRois[count].occupied = false;
    count++;
  }
  gSlotCount = count;
  saveSlotConfig(gRois, gSlotCount);

  server.send(200, "application/json",
              "{\"status\":\"OK\",\"saved\":" + String(count) + "}");
}

/** GET /api/snapshot — single JPEG frame */
void handleSnapshot() {
  sensor_t* s = esp_camera_sensor_get();
  s->set_framesize(s, STREAM_FRAMESIZE);
  delay(80);

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(503, "text/plain", "Camera capture failed");
    return;
  }
  server.sendHeader("Content-Type", "image/jpeg");
  server.sendHeader("Content-Disposition", "inline; filename=snapshot.jpg");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

/** GET /stream — MJPEG live stream */
void handleStream() {
  sensor_t* s = esp_camera_sensor_get();
  s->set_framesize(s, STREAM_FRAMESIZE);

  WiFiClient client = server.client();
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Cache-Control: no-cache");
  client.println();

  while (client.connected()) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { delay(30); continue; }

    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                  fb->len);
    client.write(fb->buf, fb->len);
    client.println();
    esp_camera_fb_return(fb);

    // Yield so background tasks can run
    yield();
    delay(50); // ~20 fps max
  }
}

/** POST /calibrate — run auto grid detection */
void handleCalibrate() {
  Serial.println("[API] /calibrate triggered");
  digitalWrite(STATUS_LED_PIN, LED_ON);

  int frameW = 0, frameH = 0;
  uint8_t* gray = captureGrayscale(&frameW, &frameH);

  if (!gray) {
    server.send(503, "application/json",
                "{\"status\":\"ERROR\",\"message\":\"Camera capture failed\"}");
    digitalWrite(STATUS_LED_PIN, LED_OFF);
    return;
  }

  bool ok = autoDetectSlots(gray, frameW, frameH, gRois, &gSlotCount);
  free(gray);

  if (gSlotCount > 0) {
    saveSlotConfig(gRois, gSlotCount);
  }

  DynamicJsonDocument doc(512);
  doc["status"]     = ok ? "SUCCESS" : "PARTIAL";
  doc["message"]    = ok
    ? "Grid detected from painted markings"
    : "Fewer lines detected than expected — used interpolation. "
      "Markings may be faint. Consider manual calibration.";
  doc["slot_count"] = gSlotCount;
  doc["frame_w"]    = frameW;
  doc["frame_h"]    = frameH;

  String body;
  serializeJson(doc, body);
  server.send(ok ? 200 : 207, "application/json", body);
  digitalWrite(STATUS_LED_PIN, LED_OFF);
}

// ── Background analysis loop ──────────────────────────────────────
static void runAnalysisLoop() {
  unsigned long now = millis();
  if (now - gLastAnalysisMs < ANALYSIS_INTERVAL_MS) return;
  gLastAnalysisMs = now;

  int frameW = 0, frameH = 0;
  uint8_t* gray = captureGrayscale(&frameW, &frameH);
  if (!gray) return;

  analyseAllSlots(gray, frameW, frameH, gRois, gSlotCount);
  free(gray);
}

// ── Setup ─────────────────────────────────────────────────────────
void setup() {
  // Disable brownout detector (prevents resets under camera load)
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(500);
  Serial.println("\n[ESP32] Smart Campus Parking Node booting...");

  // Status LED
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LED_OFF);

  // SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("[SPIFFS] Mount failed — formatting...");
    SPIFFS.format();
    SPIFFS.begin(true);
  }
  Serial.println("[SPIFFS] OK");

  // Camera
  if (!initCamera()) {
    Serial.println("[FATAL] Camera init failed. Halting.");
    while (true) {
      digitalWrite(STATUS_LED_PIN, LED_ON); delay(200);
      digitalWrite(STATUS_LED_PIN, LED_OFF); delay(200);
    }
  }

  // Load slot config (or generate defaults on first boot)
  loadSlotConfig(gRois, &gSlotCount, TOTAL_SLOTS);

  // WiFi
#if WIFI_MODE_STATION
  Serial.printf("[WiFi] Connecting to %s ...\n", WIFI_STA_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_STA_SSID, WIFI_STA_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    // Fallback to AP mode
    Serial.println("\n[WiFi] STA failed — starting AP mode");
    WiFi.mode(WIFI_AP);
    WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);
    Serial.printf("[WiFi] AP IP: %s\n", WiFi.softAPIP().toString().c_str());
  }
#else
  Serial.printf("[WiFi] Starting AP: %s\n", WIFI_AP_SSID);
  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);
  Serial.printf("[WiFi] AP IP: %s\n", WiFi.softAPIP().toString().c_str());
#endif

  // Register HTTP routes
  server.on("/",                 HTTP_GET,  handleRoot);
  server.on("/api/slots",        HTTP_GET,  handleApiSlots);
  server.on("/api/config",       HTTP_GET,  handleApiConfig);
  server.on("/api/slots/config", HTTP_POST, handlePostConfig);
  server.on("/api/snapshot",     HTTP_GET,  handleSnapshot);
  server.on("/stream",           HTTP_GET,  handleStream);
  server.on("/calibrate",        HTTP_POST, handleCalibrate);

  // CORS pre-flight
  server.onNotFound([]() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(404, "text/plain", "Not found");
  });

  server.begin();
  Serial.printf("[HTTP] Server started on port %d\n", HTTP_PORT);

  // Quick initial analysis so /api/slots works immediately
  runAnalysisLoop();

  digitalWrite(STATUS_LED_PIN, LED_OFF);
  Serial.println("[ESP32] Boot complete. Ready.");

  // Print API URLs
  String ip = (WiFi.status() == WL_CONNECTED)
              ? WiFi.localIP().toString()
              : WiFi.softAPIP().toString();
  Serial.printf("\n  Status : http://%s/\n", ip.c_str());
  Serial.printf("  Slots  : http://%s/api/slots\n", ip.c_str());
  Serial.printf("  Stream : http://%s/stream\n", ip.c_str());
  Serial.printf("  Calibrate: POST http://%s/calibrate\n\n", ip.c_str());
}

// ── Loop ──────────────────────────────────────────────────────────
void loop() {
  server.handleClient();
  runAnalysisLoop();   // Non-blocking — only runs every ANALYSIS_INTERVAL_MS
  delay(10);
}
