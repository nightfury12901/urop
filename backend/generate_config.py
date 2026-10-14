import json
import os
import math

# ─────────────────────────────────────────────
# REAL GPS COORDINATES
# ─────────────────────────────────────────────
GATE_LAT, GATE_LNG = 12.823115, 80.041077
P1_LAT,   P1_LNG   = 12.823339, 80.042857   # Student bikes – 20 slots
P3_LAT,   P3_LNG   = 12.823572, 80.046243   # Teacher cars  – 15 slots

GOOGLE_MAPS_ORIGIN = f"{GATE_LAT},{GATE_LNG}"

# ─────────────────────────────────────────────
# Camera sources — swap "mock" → "droidcam" when phones are ready
# DroidCam example: {"type": "droidcam", "source": "http://192.168.1.X:4747/video", "width": 640, "height": 480}
# ESPCam example:   {"type": "rtsp",     "source": "http://192.168.1.X/stream", "width": 640, "height": 480}
# ─────────────────────────────────────────────
P1_CAMERA = {"type": "mock", "source": "0", "width": 640, "height": 480}
P3_CAMERA = {"type": "mock", "source": "1", "width": 640, "height": 480}


def gps_distance_meters(lat1, lng1, lat2, lng2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def google_maps_link(dest_lat, dest_lng):
    return (
        f"https://www.google.com/maps/dir/?api=1"
        f"&origin={GOOGLE_MAPS_ORIGIN}"
        f"&destination={dest_lat},{dest_lng}"
        f"&travelmode=driving"
    )


def slot_gps(area_lat, area_lng, slot_index, cols=5, spread_lat=0.00005, spread_lng=0.00006):
    col = slot_index % cols
    row = slot_index // cols
    lat = area_lat + (row - 0.5) * spread_lat
    lng = area_lng + (col - cols / 2.0) * spread_lng
    return round(lat, 7), round(lng, 7)


# ─────────────────────────────────────────────
# CONFIG — P1 (students, 20 bike slots) + P3 (teachers, 15 car slots)
# ─────────────────────────────────────────────
config = {
    "campus_info": {
        "name": "Smart Campus Parking System",
        "gate_id": "GATE",
        "gate_name": "Main College Gate",
        "gate_gps_lat": GATE_LAT,
        "gate_gps_lng": GATE_LNG,
        "google_maps_origin": GOOGLE_MAPS_ORIGIN
    },
    "parking_areas": {
        "P1": {
            "id": "P1",
            "name": "Student Parking (Bikes Only)",
            "allowed_roles": ["STUDENT"],
            "allowed_vehicles": ["TWO_WHEELER"],
            "slot_count": 20,
            "entry_node": "ENTRY_P1",
            "gps_lat": P1_LAT,
            "gps_lng": P1_LNG,
            "camera_source": P1_CAMERA
        },
        "P3": {
            "id": "P3",
            "name": "Faculty & Staff Parking",
            "allowed_roles": ["TEACHER", "VIP"],
            "allowed_vehicles": ["FOUR_WHEELER", "TWO_WHEELER"],
            "slot_count": 15,
            "entry_node": "ENTRY_P3",
            "gps_lat": P3_LAT,
            "gps_lng": P3_LNG,
            "camera_source": P3_CAMERA
        }
    },
    "slots": [],
    "graph": {
        "nodes": {
            "GATE":     {"label": "College Gate",  "gps_lat": GATE_LAT, "gps_lng": GATE_LNG, "x": 80,  "y": 300},
            "JUNC_1":   {"label": "Main Junction", "gps_lat": GATE_LAT, "gps_lng": P1_LNG,   "x": 300, "y": 300},
            "ENTRY_P1": {"label": "P1 Entrance",   "gps_lat": P1_LAT,   "gps_lng": P1_LNG,   "x": 300, "y": 150},
            "ENTRY_P3": {"label": "P3 Entrance",   "gps_lat": P3_LAT,   "gps_lng": P3_LNG,   "x": 650, "y": 150}
        },
        "edges": [
            {"from": "GATE",     "to": "JUNC_1",   "weight": gps_distance_meters(GATE_LAT, GATE_LNG, GATE_LAT, P1_LNG)},
            {"from": "JUNC_1",   "to": "GATE",     "weight": gps_distance_meters(GATE_LAT, GATE_LNG, GATE_LAT, P1_LNG)},
            {"from": "JUNC_1",   "to": "ENTRY_P1", "weight": gps_distance_meters(GATE_LAT, P1_LNG, P1_LAT, P1_LNG)},
            {"from": "ENTRY_P1", "to": "JUNC_1",   "weight": gps_distance_meters(GATE_LAT, P1_LNG, P1_LAT, P1_LNG)},
            {"from": "JUNC_1",   "to": "ENTRY_P3", "weight": gps_distance_meters(GATE_LAT, P1_LNG, P3_LAT, P3_LNG)},
            {"from": "ENTRY_P3", "to": "JUNC_1",   "weight": gps_distance_meters(GATE_LAT, P1_LNG, P3_LAT, P3_LNG)}
        ]
    }
}

# ─────────────────────────────────────────────
# SLOT GENERATION
# ─────────────────────────────────────────────
AREA_META = [
    # (id, slot_type, count, lat, lng, entry_node, cols, map_origin, map_stride)
    ("P1", "BIKES_ONLY",    20, P1_LAT, P1_LNG, "ENTRY_P1", 5, (160, 80),  (30, 32)),
    ("P3", "CAR_OR_2BIKES", 15, P3_LAT, P3_LNG, "ENTRY_P3", 5, (490, 80),  (38, 32)),
]

ROI_PARAMS = {
    "P1": (0.06, 0.12, 0.17, 0.14, 0.22),
    "P3": (0.08, 0.12, 0.19, 0.17, 0.22),
}

for (area_id, slot_type, count, area_lat, area_lng, entry_node, cols, map_origin, map_stride) in AREA_META:
    rx0, ry0, stride, rw, rh = ROI_PARAMS[area_id]
    map_ox, map_oy = map_origin
    map_sx, map_sy = map_stride

    for i in range(1, count + 1):
        col = (i - 1) % cols
        row = (i - 1) // cols
        slot_id = f"{area_id}-{i:02d}"
        s_lat, s_lng = slot_gps(area_lat, area_lng, i - 1, cols)
        dist_m = gps_distance_meters(GATE_LAT, GATE_LNG, s_lat, s_lng)

        rx = rx0 + col * stride
        ry = ry0 + row * stride
        poly = [
            [round(rx,      3), round(ry,      3)],
            [round(rx + rw, 3), round(ry,      3)],
            [round(rx + rw, 3), round(ry + rh, 3)],
            [round(rx,      3), round(ry + rh, 3)]
        ]

        map_x = map_ox + col * map_sx
        map_y = map_oy + row * map_sy

        config["slots"].append({
            "id": slot_id,
            "area_id": area_id,
            "slot_number": i,
            "slot_type": slot_type,
            "bike_capacity": 2,
            "distance_from_gate": dist_m,
            "gps_lat": s_lat,
            "gps_lng": s_lng,
            "google_maps_url": google_maps_link(s_lat, s_lng),
            "map_coord": {"x": map_x, "y": map_y},
            "roi_polygon": poly
        })

        config["graph"]["nodes"][slot_id] = {
            "x": map_x, "y": map_y, "label": slot_id,
            "gps_lat": s_lat, "gps_lng": s_lng
        }
        w = gps_distance_meters(area_lat, area_lng, s_lat, s_lng) + 5
        config["graph"]["edges"].append({"from": entry_node, "to": slot_id, "weight": w})
        config["graph"]["edges"].append({"from": slot_id, "to": entry_node, "weight": w})

# ─────────────────────────────────────────────
# WRITE OUTPUT
# ─────────────────────────────────────────────
os.makedirs("backend/app/data", exist_ok=True)
out_path = "backend/app/data/campus_config.json"
with open(out_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"[OK] campus_config.json — {len(config['slots'])} slots, {len(config['graph']['nodes'])} graph nodes")
print(f"     P1 (Students, 20 bike slots):  {P1_LAT}, {P1_LNG} | {gps_distance_meters(GATE_LAT,GATE_LNG,P1_LAT,P1_LNG)}m from gate")
print(f"     P3 (Teachers, 15 car slots):   {P3_LAT}, {P3_LNG} | {gps_distance_meters(GATE_LAT,GATE_LNG,P3_LAT,P3_LNG)}m from gate")
