# Smart Campus Parking System (UROP)

An intelligent, full-stack campus parking allocation and computer vision system. It automatically assigns the optimal parking spot based on vehicle type and user permit, cross-references physical availability via live camera feeds, and provides real Google Maps turn-by-turn routing from the college entrance gate directly to the assigned slot.

---

## 🚀 Key Features

- **Zone-Specific Role Allocation**:
  - **Zone P1 (20 slots)**: Exclusively for Students (**2-wheelers only**). Supports dual-bike occupancy per slot.
  - **Zone P3 (15 slots)**: Faculty, Staff, and VIP parking (**4-wheelers and 2-wheelers**).
- **Live Camera Verification**:
  - Automatically queries the responsible camera (**P1 camera** for students, **P3 camera** for faculty) when a user requests a spot.
  - Background CV detection with temporal smoothing to prevent false alerts.
  - Resilient fallback to database state if cameras are offline or unconfigured.
- **Real GPS Turn-by-Turn Routing**:
  - Exact GPS coordinates for College Gate (`12.823115, 80.041077`), P1 (`12.823339, 80.042857`), and P3 (`12.823572, 80.046243`).
  - Dijkstra shortest path engine with Haversine distance calculations.
  - One-tap **"Navigate via Google Maps"** button that opens direct navigation in Google Maps.
- **Portals**:
  - **Customer Portal**: Fast 1-tap spot reservation, hold countdown timer (5 min), and GPS directions.
  - **Admin Portal**: Live MJPEG video stream with ROI overlays, 35-slot visual matrix, manual override/release controls, misparking anomaly alerts, and analytics.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, SQLite (WAL mode) / PostgreSQL, SQLAlchemy, OpenCV, NumPy, WebSockets.
- **Frontend**: Vanilla HTML5, CSS3 (Modern Glassmorphic Dark UI), JavaScript (ES6+), WebSocket live sync.
- **Computer Vision**: Patch-level occupancy classification, HSV foreground ratio & standard deviation variance analysis, multi-frame temporal voting.
- **Hardware Compatibility**: DroidCam (Android/iOS via WiFi), ESP32-CAM, USB Webcams, RTSP CCTV streams.

---

## 📦 Project Structure

```
├── backend/
│   ├── app/
│   │   ├── data/                 # campus_config.json, SQLite DB, camera snapshots
│   │   ├── models/               # SQLAlchemy database models & init logic
│   │   ├── routers/              # API endpoints (auth, parking, admin, websocket)
│   │   ├── schemas/              # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── camera/           # OpenCV, DroidCam, RTSP, and Mock camera drivers
│   │   │   ├── vision/           # Detector, temporal smoother, cache, background worker
│   │   │   ├── allocation.py     # Atomic slot allocation engine
│   │   │   ├── routing.py        # Dijkstra shortest path router
│   │   │   └── reservation_manager.py # TTL watcher, auto-release, anomalies
│   │   └── main.py               # FastAPI entry point & lifespan management
│   ├── generate_config.py        # Layout generator (GPS, ROIs, graph edges)
│   └── tests/                    # Concurrency and TTL unit tests
├── frontend/
│   ├── index.html                # Single-page Customer & Admin portal
│   ├── styles.css                # Responsive glassmorphism dark theme
│   └── app.js                    # Reactive state, WebSocket hub, reservation manager
├── Dockerfile                    # Production container specification
├── docker-compose.yml            # Container orchestration config
└── requirements.txt              # Python dependencies
```

---

## 💻 Local Setup & Execution

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nightfury12901/urop.git
   cd urop
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Generate configuration & seed slots**:
   ```bash
   python backend/generate_config.py
   ```

4. **Launch the application**:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Open in browser**:
   Navigate to [http://localhost:8000](http://localhost:8000).

---

## 🌐 Production Deployment Guide

### Option 1: Docker (Recommended)

1. **Build and start the container**:
   ```bash
   docker-compose up -d --build
   ```
2. The system will start on port `8000` with automatic restart and health checks.

---

### Option 2: Cloud VPS (Ubuntu / Debian on AWS, DigitalOcean, Hetzner, Linode)

1. **Connect to your server via SSH**:
   ```bash
   ssh root@your-server-ip
   ```

2. **Install system packages & Python**:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git libgl1 libglib2.0-0 nginx
   ```

3. **Clone and setup**:
   ```bash
   git clone https://github.com/nightfury12901/urop.git /var/www/smartpark
   cd /var/www/smartpark
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python backend/generate_config.py
   ```

4. **Run as a systemd background service**:
   Create `/etc/systemd/system/smartpark.service`:
   ```ini
   [Unit]
   Description=Smart Campus Parking System
   After=network.target

   [Service]
   User=root
   WorkingDirectory=/var/www/smartpark
   ExecStart=/var/www/smartpark/venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 2
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable smartpark
   sudo systemctl start smartpark
   ```

5. **Configure Nginx as Reverse Proxy**:
   Edit `/etc/nginx/sites-available/default`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com; # or your public server IP

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }
   }
   ```
   Restart Nginx:
   ```bash
   sudo nginx -t && sudo systemctl restart nginx
   ```

---

### 📷 Connecting Cameras (DroidCam / ESP-CAM) in Production

- **On Local Campus WiFi**:
  - Install DroidCam on your phone or power on your ESP32-CAM.
  - In the Admin Portal, paste the stream URL:
    - **DroidCam**: `http://192.168.1.X:4747/video`
    - **ESP32-CAM**: `http://192.168.1.X/stream`
- **If Backend is in the Cloud & Cameras are on Local WiFi**:
  - Expose your camera stream securely to the cloud using **Cloudflare Tunnel**, **Tailscale**, or **Ngrok**:
    ```bash
    ngrok http 4747
    ```
    Then paste the forwarding URL (e.g. `https://xxxx.ngrok-free.app/video`) into the Admin Portal camera source input.
