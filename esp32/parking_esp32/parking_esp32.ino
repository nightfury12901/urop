/**
 * parking_esp32.ino
 * ---------------------------------------------------------------------------
 * Smart Campus Parking - ESP32-CAM Edge Node for Zone P1
 * Hardware: AI-Thinker ESP32-CAM + OV2640
 *
 * What this firmware does:
 *   - Captures QVGA frames from the OV2640
 *   - Runs on-device occupancy analysis for each parking slot (no cloud needed)
 *   - Auto-detects the slot grid from painted/drawn lines when /calibrate is called
 *   - Exposes a REST API so a server can query slot states
 *   - Streams MJPEG video and provides an in-browser visual HUD overlay at /view
 *
 * REST API:
 *   GET       /                  -> JSON status summary
 *   GET       /view              -> Live in-browser visual overlay (Green/Red boxes)
 *   GET       /api/slots         -> JSON {slot_id: "FREE"/"OCCUPIED", ...}
 *   GET       /api/config        -> current slot ROI config JSON
 *   POST      /api/slots/config  -> receive ROI config (JSON body)
 *   GET       /api/snapshot      -> single JPEG frame (VGA)
 *   GET       /stream            -> MJPEG live stream (VGA)
 *   GET/POST  /calibrate         -> automatic slot grid detection, saved to SPIFFS
 *
 * Board setup (Arduino IDE):
 *   Tools > Board:            "AI Thinker ESP32-CAM"
 *   Tools > PSRAM:            "Enabled"
 *   Tools > Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"
 *   Tools > Upload Speed:     115200
 * ---------------------------------------------------------------------------
 */

#include "esp_camera.h"
#include "img_converters.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>

#include "camera_config.h"
#include "config.h"
#include "slot_detector.h"
#include "auto_calibrate.h"

WebServer server(HTTP_PORT);

static SlotROI gRois[TOTAL_SLOTS];
static int     gSlotCount = 0;
static bool    gCameraOk  = false;

static unsigned long       gLastAnalysisMs     = 0;
static const unsigned long ANALYSIS_INTERVAL_MS = 1500;

static const size_t MAX_DECODE_PIXELS = 640UL * 480UL;

// ── Memory and Frame Helpers ──────────────────────────────────────────────
static void* bigAlloc(size_t n) {
  if (psramFound()) {
    void* p = ps_malloc(n);
    if (p) return p;
  }
  return malloc(n);
}

static void flushFrames(int n) {
  for (int i = 0; i < n; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
}

static void ensureFrameSize(framesize_t fs) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;
  if ((int)s->status.framesize == (int)fs) return;
  s->set_framesize(s, fs);
  delay(150);
  flushFrames(3);
}

static bool jpegDims(const uint8_t* b, size_t len, int* w, int* h) {
  if (len < 4 || b[0] != 0xFF || b[1] != 0xD8) return false;
  size_t i = 2;
  while (i + 9 < len) {
    if (b[i] != 0xFF) { i++; continue; }
    uint8_t m = b[i + 1];
    if (m == 0xFF) { i++; continue; }
    if (m == 0xD8 || m == 0x01 || (m >= 0xD0 && m <= 0xD7)) { i += 2; continue; }
    if (m == 0xC0 || m == 0xC2) {
      *h = (b[i + 5] << 8) | b[i + 6];
      *w = (b[i + 7] << 8) | b[i + 8];
      return (*w > 0 && *h > 0);
    }
    i += 2 + ((b[i + 2] << 8) | b[i + 3]);
  }
  return false;
}

static uint8_t* captureGrayscale(int* outW, int* outH) {
  if (!gCameraOk) return nullptr;

  ensureFrameSize(DETECTION_FRAMESIZE);

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[Camera] Frame capture failed");
    return nullptr;
  }

  int w = fb->width, h = fb->height;
  if (fb->format == PIXFORMAT_JPEG) {
    int jw = 0, jh = 0;
    if (!jpegDims(fb->buf, fb->len, &jw, &jh)) {
      Serial.println("[Camera] Bad JPEG header, skipping frame");
      esp_camera_fb_return(fb);
      return nullptr;
    }
    w = jw;
    h = jh;
  }

  size_t pixCount = (size_t)w * (size_t)h;
  if (pixCount == 0 || pixCount > MAX_DECODE_PIXELS) {
    Serial.printf("[Camera] Unexpected frame size %dx%d, skipping\n", w, h);
    esp_camera_fb_return(fb);
    return nullptr;
  }

  *outW = w;
  *outH = h;

  uint8_t* gray = (uint8_t*)malloc(pixCount);
  if (!gray) {
    Serial.println("[Camera] Malloc failed for grayscale buffer");
    esp_camera_fb_return(fb);
    return nullptr;
  }

  if (fb->format == PIXFORMAT_JPEG) {
    uint8_t* rgb = (uint8_t*)bigAlloc(pixCount * 3);
    if (!rgb) {
      Serial.println("[Camera] Malloc failed for RGB buffer");
      free(gray);
      esp_camera_fb_return(fb);
      return nullptr;
    }
    bool ok = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb);
    if (ok) {
      for (size_t i = 0; i < pixCount; i++) {
        gray[i] = (uint8_t)((rgb[i * 3] * 77 + rgb[i * 3 + 1] * 150 + rgb[i * 3 + 2] * 29) >> 8);
      }
    } else {
      Serial.println("[Camera] JPEG decode failed");
      free(rgb);
      free(gray);
      esp_camera_fb_return(fb);
      return nullptr;
    }
    free(rgb);
  } else if (fb->format == PIXFORMAT_GRAYSCALE) {
    if (fb->len < pixCount) {
      free(gray);
      esp_camera_fb_return(fb);
      return nullptr;
    }
    memcpy(gray, fb->buf, pixCount);
  } else {
    free(gray);
    esp_camera_fb_return(fb);
    return nullptr;
  }

  esp_camera_fb_return(fb);
  return gray;
}

// ── Camera Initialization ─────────────────────────────────────────────────
static bool initCamera() {
  camera_config_t config = {};

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
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
#else
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
#endif
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size   = STREAM_FRAMESIZE;
    config.jpeg_quality = 10;
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

  sensor_t* s = esp_camera_sensor_get();
  s->set_vflip(s, 1);
  s->set_hmirror(s, 0);
  s->set_brightness(s, 1);
  s->set_saturation(s, 0);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_exposure_ctrl(s, 1);

  Serial.println("[Camera] OV2640 initialised OK");
  return true;
}

// ── HTTP Handlers ─────────────────────────────────────────────────────────

/** GET /view - Live visual video stream with real-time green/red bounding boxes */
void handleView() {
  String html = R"rawliteral(<!DOCTYPE html>
<html>
<head>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Zone P1 Camera HUD</title>
<style>
  body { background: #0f172a; color: #fff; font-family: system-ui, sans-serif; text-align: center; margin: 0; padding: 15px; }
  h2 { margin: 5px 0 12px; font-size: 20px; }
  .badge { display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: bold; margin: 0 4px; font-size: 14px; }
  .free-badge { background: #10b981; }
  .occ-badge { background: #ef4444; }
  .wrapper { position: relative; display: inline-block; max-width: 640px; width: 100%; border-radius: 10px; overflow: hidden; box-shadow: 0 8px 30px rgba(0,0,0,0.6); }
  img#stream { width: 100%; display: block; }
  canvas#overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .btn { background: #3b82f6; color: #fff; border: none; padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: bold; cursor: pointer; margin-top: 15px; }
  .btn:hover { background: #2563eb; }
</style>
</head>
<body>
  <h2>Zone P1 - Smart Campus Parking (ESP32-CAM)</h2>
  <div style='margin-bottom:12px;'>
    <span class='badge free-badge' id='freeBadge'>FREE: --</span>
    <span class='badge occ-badge' id='occBadge'>OCCUPIED: --</span>
  </div>
  <div class='wrapper'>
    <img id='stream' src='/stream'>
    <canvas id='overlay'></canvas>
  </div>
  <div>
    <button class='btn' onclick='calibrate()'>⚡ Run Auto-Calibration</button>
  </div>
  <script>
    const stream = document.getElementById('stream');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    let config = null;

    async function loadConfig() {
      try {
        const res = await fetch('/api/config');
        config = await res.json();
      } catch(e) {}
    }

    async function update() {
      try {
        const res = await fetch('/api/slots');
        const data = await res.json();
        document.getElementById('freeBadge').innerText = 'FREE: ' + data.free_slots;
        document.getElementById('occBadge').innerText = 'OCCUPIED: ' + (data.total_slots - data.free_slots);

        canvas.width = stream.clientWidth || 640;
        canvas.height = stream.clientHeight || 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (config && config.slots) {
          ctx.lineWidth = 2;
          ctx.font = 'bold 11px sans-serif';
          config.slots.forEach(s => {
            const isOcc = data.slots && data.slots[s.id] === 'OCCUPIED';
            const x = s.x0 * canvas.width;
            const y = s.y0 * canvas.height;
            const w = (s.x1 - s.x0) * canvas.width;
            const h = (s.y1 - s.y0) * canvas.height;

            ctx.strokeStyle = isOcc ? '#ef4444' : '#10b981';
            ctx.fillStyle = isOcc ? 'rgba(239,68,68,0.22)' : 'rgba(16,185,129,0.22)';
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = isOcc ? '#ef4444' : '#10b981';
            ctx.fillRect(x, y - 14, 38, 14);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(s.id.replace('P1-', ''), x + 3, y - 3);
          });
        }
      } catch(e) {}
    }

    async function calibrate() {
      if(!confirm('Trigger auto-calibration on ESP32?')) return;
      await fetch('/calibrate', { method: 'POST' });
      await loadConfig();
    }

    loadConfig();
    setInterval(update, 1000);
  </script>
</body>
</html>)rawliteral";
  server.send(200, "text/html", html);
}

/** GET / - device info */
void handleRoot() {
  int freeCount = 0;
  for (int i = 0; i < gSlotCount; i++)
    if (!gRois[i].occupied) freeCount++;

  DynamicJsonDocument doc(768);
  doc["device"]      = "ESP32-CAM Parking Node";
  doc["zone"]        = PARKING_ZONE_ID;
  doc["zone_name"]   = PARKING_ZONE_NAME;
  doc["total_slots"] = gSlotCount;
  doc["free_slots"]  = freeCount;
  doc["occupied"]    = gSlotCount - freeCount;
  doc["camera_ok"]   = gCameraOk;
  doc["ip"]          = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP().toString()
                                                       : WiFi.softAPIP().toString();
  doc["uptime_ms"]   = millis();
  doc["free_heap"]   = ESP.getFreeHeap();
  doc["free_psram"]  = ESP.getFreePsram();
  doc["view_hud"]    = "http://" + doc["ip"].as<String>() + "/view";

  String body;
  serializeJsonPretty(doc, body);
  server.send(200, "application/json", body);
}

/** GET /api/slots - per-slot occupancy JSON */
void handleApiSlots() {
  DynamicJsonDocument doc(2048);
  JsonObject slots = doc.createNestedObject("slots");
  int freeCount = 0;

  for (int i = 0; i < gSlotCount; i++) {
    slots[gRois[i].id] = gRois[i].occupied ? "OCCUPIED" : "FREE";
    if (!gRois[i].occupied) freeCount++;
  }

  doc["zone"]         = PARKING_ZONE_ID;
  doc["total_slots"]  = gSlotCount;
  doc["free_slots"]   = freeCount;
  doc["timestamp_ms"] = millis();

  String body;
  serializeJson(doc, body);
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", body);
}

/** GET /api/config - current slot ROI config */
void handleApiConfig() {
  DynamicJsonDocument doc(8192);
  doc["zone"]        = PARKING_ZONE_ID;
  doc["total_slots"] = gSlotCount;
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
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", body);
}

/** POST /api/slots/config - receive ROI config from server */
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

/** GET /api/snapshot - single JPEG frame */
void handleSnapshot() {
  if (!gCameraOk) {
    server.send(503, "text/plain", "Camera not available");
    return;
  }
  ensureFrameSize(STREAM_FRAMESIZE);

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(503, "text/plain", "Camera capture failed");
    return;
  }
  server.sendHeader("Content-Disposition", "inline; filename=snapshot.jpg");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.client().write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

/** GET /stream - MJPEG live stream */
void handleStream() {
  if (!gCameraOk) {
    server.send(503, "text/plain", "Camera not available");
    return;
  }
  ensureFrameSize(STREAM_FRAMESIZE);

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
                  (unsigned)fb->len);
    size_t written = client.write(fb->buf, fb->len);
    client.println();
    esp_camera_fb_return(fb);

    if (written == 0) break;
    yield();
    delay(50);
  }
}

/** POST or GET /calibrate - run auto grid detection */
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
    ? "Dual-row grid detected from markings"
    : "Faint lines - used calibrated dual-row parking layout";
  doc["slot_count"] = gSlotCount;
  doc["frame_w"]    = frameW;
  doc["frame_h"]    = frameH;

  String body;
  serializeJson(doc, body);
  server.send(200, "application/json", body);
  digitalWrite(STATUS_LED_PIN, LED_OFF);
}

// ── Background Analysis ───────────────────────────────────────────────────
static void runAnalysisLoop() {
  if (!gCameraOk || gSlotCount <= 0) return;

  unsigned long now = millis();
  if (now - gLastAnalysisMs < ANALYSIS_INTERVAL_MS) return;
  gLastAnalysisMs = now;

  int frameW = 0, frameH = 0;
  uint8_t* gray = captureGrayscale(&frameW, &frameH);
  if (!gray) return;

  analyseAllSlots(gray, frameW, frameH, gRois, gSlotCount);
  free(gray);

  static int lastFree = -1;
  int freeCount = 0;
  for (int i = 0; i < gSlotCount; i++)
    if (!gRois[i].occupied) freeCount++;
  if (freeCount != lastFree) {
    Serial.printf("[Analysis] %d free / %d occupied (frame %dx%d)\n",
                  freeCount, gSlotCount - freeCount, frameW, frameH);
    lastFree = freeCount;
  }
}

// ── Setup ─────────────────────────────────────────────────────────────────
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(500);
  Serial.println("\n[ESP32] Smart Campus Parking Node booting...");
  Serial.printf("[Mem] PSRAM: %s, free heap: %u, free PSRAM: %u\n",
                psramFound() ? "found" : "NOT FOUND (enable Tools > PSRAM)",
                (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LED_OFF);

  if (!SPIFFS.begin(true)) {
    Serial.println("[SPIFFS] Mount failed - formatting...");
    SPIFFS.format();
    SPIFFS.begin(true);
  }
  Serial.println("[SPIFFS] OK");

  gCameraOk = initCamera();
  if (!gCameraOk) {
    Serial.println("[FATAL] Camera init failed. Check ribbon cable and power.");
    while (true) {
      digitalWrite(STATUS_LED_PIN, LED_ON);  delay(200);
      digitalWrite(STATUS_LED_PIN, LED_OFF); delay(200);
    }
  }

  loadSlotConfig(gRois, &gSlotCount, TOTAL_SLOTS);

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
    Serial.println("\n[WiFi] STA failed - starting AP mode");
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
  WiFi.setSleep(false);

  server.on("/",                 HTTP_GET,  handleRoot);
  server.on("/view",             HTTP_GET,  handleView);
  server.on("/api/slots",        HTTP_GET,  handleApiSlots);
  server.on("/api/config",       HTTP_GET,  handleApiConfig);
  server.on("/api/slots/config", HTTP_POST, handlePostConfig);
  server.on("/api/snapshot",     HTTP_GET,  handleSnapshot);
  server.on("/stream",           HTTP_GET,  handleStream);
  server.on("/calibrate",        HTTP_ANY,  handleCalibrate);

  server.onNotFound([]() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(404, "text/plain", "Not found");
  });

  server.begin();
  Serial.printf("[HTTP] Server started on port %d\n", HTTP_PORT);

  runAnalysisLoop();

  digitalWrite(STATUS_LED_PIN, LED_OFF);
  Serial.println("[ESP32] Boot complete. Ready.");

  String ip = (WiFi.status() == WL_CONNECTED)
              ? WiFi.localIP().toString()
              : WiFi.softAPIP().toString();
  Serial.printf("\n  Status    : http://%s/\n", ip.c_str());
  Serial.printf("  Visual HUD: http://%s/view\n", ip.c_str());
  Serial.printf("  Slots     : http://%s/api/slots\n", ip.c_str());
  Serial.printf("  Snapshot  : http://%s/api/snapshot\n", ip.c_str());
  Serial.printf("  Stream    : http://%s/stream\n", ip.c_str());
  Serial.printf("  Calibrate : http://%s/calibrate\n\n", ip.c_str());
}

// ── Loop ──────────────────────────────────────────────────────────────────
void loop() {
  server.handleClient();
  runAnalysisLoop();
  delay(10);
}
