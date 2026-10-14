from fastapi import APIRouter, HTTPException, Depends, Response, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import json
import os
import cv2
import time
import shutil

from ..models.database import SessionLocal, Slot, Reservation, User, OccupancyLog
from ..services.vision.cache import occupancy_cache
from ..services.reservation_manager import reservation_manager

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "campus_config.json")
CAMERA_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "camera_images")
os.makedirs(CAMERA_IMAGES_DIR, exist_ok=True)

router = APIRouter(prefix="/api/v1/admin", tags=["Admin Portal & Diagnostics"])

# Reference to VisionWorker set from main.py
vision_worker_ref = None

def set_vision_worker(worker):
    global vision_worker_ref
    vision_worker_ref = worker

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class OverrideRequest(BaseModel):
    new_state: str # "FREE", "OCCUPIED", "MAINTENANCE"
    reason: Optional[str] = "Admin manual override"

class SimulateSlotRequest(BaseModel):
    slot_id: str
    state: str # "CAR", "BIKE_1", "BIKE_2", "EMPTY"

class SaveROIRequest(BaseModel):
    slot_id: str
    roi_polygon: List[List[float]]

class CameraConfigRequest(BaseModel):
    area_id: str
    type: str # "mock" | "droidcam" | "webcam" | "rtsp" | "image"
    source: str # "0" | "http://..." | "c:/path/to/img.jpg"
    width: Optional[int] = 640
    height: Optional[int] = 480

@router.get("/slots")
def get_all_slots_matrix(db: Session = Depends(get_db)):
    """Full 50-slot status matrix with live CV detections and coordinates."""
    slots = db.query(Slot).order_by(Slot.area_id, Slot.slot_number).all()
    cached_data = occupancy_cache.get_all()
    
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    slot_coords = {s["id"]: s.get("map_coord", {"x": 0, "y": 0}) for s in cfg.get("slots", [])}

    output = []
    for s in slots:
        cv_state = cached_data.get(s.id, {})
        output.append({
            "id": s.id,
            "area_id": s.area_id,
            "slot_number": s.slot_number,
            "slot_type": s.slot_type,
            "db_state": s.state,
            "bike_capacity": s.bike_capacity,
            "current_bikes": s.current_bikes,
            "has_car": s.has_car,
            "distance_from_gate": s.distance_from_gate,
            "map_coord": slot_coords.get(s.id, {"x": 0, "y": 0}),
            "cv_detected_state": cv_state.get("detected_state", "FREE"),
            "cv_confidence": cv_state.get("confidence", 0.95),
            "updated_at": s.updated_at.isoformat() if s.updated_at else None
        })
    return output

@router.post("/slot/{slot_id}/override")
def override_slot_state(slot_id: str, req: OverrideRequest, db: Session = Depends(get_db)):
    """Manually forces a slot state for maintenance or VIP reservation."""
    slot = db.query(Slot).filter(Slot.id == slot_id).first()
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
        
    st = req.new_state.upper()
    if st not in ("FREE", "OCCUPIED", "MAINTENANCE", "RESERVED"):
        raise HTTPException(status_code=400, detail="Invalid state")

    slot.state = st
    if st == "FREE":
        slot.has_car = False
        slot.current_bikes = 0
    elif st == "OCCUPIED":
        if slot.slot_type == "CAR_OR_2BIKES":
            slot.has_car = True
        else:
            slot.current_bikes = 2

    db.commit()
    return {"status": "SUCCESS", "slot_id": slot_id, "new_state": slot.state}

@router.post("/slot/{slot_id}/release")
def force_release_slot(slot_id: str, db: Session = Depends(get_db)):
    """Admin force releases any active reservation on this slot."""
    active_res = db.query(Reservation).filter(
        Reservation.slot_id == slot_id,
        Reservation.status == "ACTIVE"
    ).all()
    for r in active_res:
        r.status = "CANCELLED"
        
    slot = db.query(Slot).filter(Slot.id == slot_id).first()
    if slot:
        slot.state = "FREE"
        slot.has_car = False
        slot.current_bikes = 0

    db.commit()
    return {"status": "SUCCESS", "slot_id": slot_id, "released_reservations": len(active_res)}

@router.get("/anomalies")
def get_parking_anomalies():
    """Returns detected wrong-slot misparking alerts."""
    return reservation_manager.get_anomalies()

@router.post("/simulate/slot")
def simulate_slot_presence(req: SimulateSlotRequest):
    """
    Live Demo simulator: injects a toy CAR, BIKE_1, BIKE_2, or EMPTY into the mock camera
    for any slot. Automatically triggers the CV pipeline!
    """
    if not vision_worker_ref:
        raise HTTPException(status_code=500, detail="Vision worker not running")
        
    area_id = req.slot_id.split("-")[0]
    cam = vision_worker_ref.cameras.get(area_id)
    if not cam or not hasattr(cam, "set_slot_state"):
        raise HTTPException(status_code=400, detail=f"Camera for area {area_id} is not in Mock/Simulation mode.")

    cam.set_slot_state(req.slot_id, req.state)
    return {"status": "SUCCESS", "slot_id": req.slot_id, "simulated_state": req.state}

@router.post("/roi/save")
def save_slot_roi(req: SaveROIRequest):
    """Updates ROI polygon in campus_config.json."""
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    updated = False
    for s in cfg.get("slots", []):
        if s["id"] == req.slot_id:
            s["roi_polygon"] = req.roi_polygon
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail="Slot not found in configuration")

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

    return {"status": "SUCCESS", "message": f"ROI polygon saved for {req.slot_id}"}

@router.get("/analytics")
def get_analytics(db: Session = Depends(get_db)):
    """Computes analytics: peak distribution, no-show rate, reservation breakdown."""
    total_res = db.query(Reservation).count()
    completed = db.query(Reservation).filter(Reservation.status == "COMPLETED").count()
    expired = db.query(Reservation).filter(Reservation.status == "EXPIRED").count()
    cancelled = db.query(Reservation).filter(Reservation.status == "CANCELLED").count()
    active = db.query(Reservation).filter(Reservation.status == "ACTIVE").count()

    slots = db.query(Slot).all()
    free_slots = sum(1 for s in slots if s.state == "FREE")
    occupied_slots = sum(1 for s in slots if s.state == "OCCUPIED")
    reserved_slots = sum(1 for s in slots if s.state == "RESERVED")
    maintenance_slots = sum(1 for s in slots if s.state == "MAINTENANCE")

    no_show_rate = round((expired / max(1, total_res)) * 100, 1)
    utilization = round(((occupied_slots + reserved_slots) / max(1, len(slots))) * 100, 1)

    return {
        "total_slots": len(slots),
        "occupancy_rate_pct": utilization,
        "no_show_rate_pct": no_show_rate,
        "slots_breakdown": {
            "free": free_slots,
            "occupied": occupied_slots,
            "reserved": reserved_slots,
            "maintenance": maintenance_slots
        },
        "reservations_summary": {
            "total": total_res,
            "completed_arrivals": completed,
            "no_shows": expired,
            "voluntary_cancelled": cancelled,
            "currently_active": active
        }
    }

@router.get("/cameras/{area_id}/feed")
def get_camera_stream(area_id: str):
    """MJPEG live stream proxy with real-time ROI polygon overlays."""
    if not vision_worker_ref:
        raise HTTPException(status_code=500, detail="Vision worker not running")

    def frame_generator():
        while True:
            frame = vision_worker_ref.get_annotated_frame(area_id.upper())
            if frame is None:
                # Blank placeholder
                frame = cv2.imread(CONFIG_PATH) # None
                time.sleep(0.5)
                continue
                
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                time.sleep(0.1)
                continue
                
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.1)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/cameras/config")
def get_all_camera_configs():
    """Returns the current camera source configurations for all areas (P1-P4)."""
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    
    areas = cfg.get("parking_areas", {})
    configs = {}
    for aid, ainfo in areas.items():
        configs[aid] = {
            "area_id": aid,
            "name": ainfo.get("name", aid),
            "camera_source": ainfo.get("camera_source", {"type": "mock", "source": "0"}),
            "image_path": os.path.join(CAMERA_IMAGES_DIR, f"{aid}.jpg"),
            "image_exists": os.path.exists(os.path.join(CAMERA_IMAGES_DIR, f"{aid}.jpg"))
        }
    return {
        "cameras": configs,
        "images_dir": os.path.abspath(CAMERA_IMAGES_DIR)
    }

@router.post("/cameras/config")
def update_camera_config(req: CameraConfigRequest):
    """
    Updates the camera source for an area (mock, droidcam, webcam, rtsp, or image)
    and hot-swaps it into the live vision worker without restarting the server.
    """
    area_id = req.area_id.upper()
    cam_cfg = {
        "type": req.type.lower(),
        "source": req.source,
        "width": req.width or 640,
        "height": req.height or 480
    }

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    if "parking_areas" not in cfg or area_id not in cfg["parking_areas"]:
        raise HTTPException(status_code=404, detail=f"Parking area {area_id} not found.")

    cfg["parking_areas"][area_id]["camera_source"] = cam_cfg

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

    if vision_worker_ref:
        vision_worker_ref.update_camera_source(area_id, cam_cfg)

    return {
        "status": "SUCCESS",
        "message": f"Camera for {area_id} configured as {cam_cfg['type']}",
        "config": cam_cfg
    }

@router.post("/cameras/{area_id}/upload-image")
async def upload_camera_image(area_id: str, file: UploadFile = File(...)):
    """
    Uploads a photo of the A4 sheet model for an area (P1, P2, P3, P4).
    Automatically saves it to backend/app/data/camera_images/{area_id}.jpg,
    sets the area's camera source to 'image', and hot-swaps it into the vision worker!
    """
    aid = area_id.upper()
    if aid not in ("P1", "P3"):
        raise HTTPException(status_code=400, detail=f"Invalid area_id '{area_id}'. Must be P1 (Students) or P3 (Faculty).")

    save_path = os.path.join(CAMERA_IMAGES_DIR, f"{aid}.jpg")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cam_cfg = {
        "type": "image",
        "source": save_path,
        "width": 640,
        "height": 480
    }

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    if "parking_areas" in cfg and aid in cfg["parking_areas"]:
        cfg["parking_areas"][aid]["camera_source"] = cam_cfg
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)

    if vision_worker_ref:
        vision_worker_ref.update_camera_source(aid, cam_cfg)

    return {
        "status": "SUCCESS",
        "message": f"Photo for {aid} uploaded successfully and set as live camera feed.",
        "file_path": save_path,
        "camera_config": cam_cfg
    }
