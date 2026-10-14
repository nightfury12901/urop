from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from ..models.database import SessionLocal, Slot, Reservation, User
from ..schemas.parking import (
    AreaSummaryResponse, AllocationRequest, ReservationResponse, RouteResponse, SlotSummary
)
from ..services.allocation import AllocationService
from ..services.reservation_manager import reservation_manager
from ..services.routing import CampusRouter
from ..services.vision.cache import occupancy_cache

import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "campus_config.json")

router = APIRouter(prefix="/api/v1/parking", tags=["Parking Operations"])
alloc_service = AllocationService(CONFIG_PATH)
campus_router = CampusRouter(CONFIG_PATH)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/lots", response_model=List[AreaSummaryResponse])
def get_parking_lots(db: Session = Depends(get_db)):
    """Summary of all 4 parking areas with real-time occupancy counts."""
    import json
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
        
    areas_info = cfg.get("parking_areas", {})
    results = []
    
    for area_id, info in areas_info.items():
        slots = db.query(Slot).filter(Slot.area_id == area_id).all()
        total = len(slots)
        free = sum(1 for s in slots if s.state == "FREE")
        occupied = sum(1 for s in slots if s.state == "OCCUPIED")
        reserved = sum(1 for s in slots if s.state == "RESERVED")
        
        results.append(AreaSummaryResponse(
            id=area_id,
            name=info.get("name", area_id),
            allowed_roles=info.get("allowed_roles", []),
            allowed_vehicles=info.get("allowed_vehicles", []),
            total_slots=total,
            free_slots=free,
            occupied_slots=occupied,
            reserved_slots=reserved
        ))
    return results

@router.post("/allocate")
def allocate_parking_slot(
    req: AllocationRequest,
    x_idempotency_key: Optional[str] = Header(None)
):
    """
    1-Tap Atomic slot assignment.
    Finds nearest suitable slot to Gate based on role, vehicle type, and live occupancy.
    """
    idemp_key = x_idempotency_key or req.idempotency_key
    try:
        res = alloc_service.allocate_slot(user_id=req.user_id, idempotency_key=idemp_key)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Allocation failure: {str(e)}")

@router.get("/reservation/active/{user_id}")
def get_active_reservation(user_id: str, db: Session = Depends(get_db)):
    """Fetches user's current active reservation, countdown timer, and route."""
    res = db.query(Reservation).filter(
        Reservation.user_id == user_id,
        Reservation.status == "ACTIVE",
        Reservation.expires_at > datetime.utcnow()
    ).first()
    if not res:
        return {"has_active": False}

    user = db.query(User).filter(User.id == user_id).first()
    res_dict = alloc_service._build_reservation_dict(db, res, user)
    return {"has_active": True, "reservation": res_dict}

@router.post("/reservation/{reservation_id}/cancel")
def cancel_reservation(reservation_id: str, user_id: str):
    """Cancels an active reservation and frees the slot."""
    try:
        return reservation_manager.cancel_reservation(reservation_id, user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/reservation/{reservation_id}/extend")
def extend_reservation(reservation_id: str, user_id: str):
    """Extends the hold timer by +3 minutes (1 time max)."""
    try:
        return reservation_manager.extend_reservation(reservation_id, user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/route/{slot_id}", response_model=RouteResponse)
def get_route_to_slot(slot_id: str):
    """Computes Dijkstra shortest path and waypoints from Gate to slot."""
    try:
        route = campus_router.get_route(slot_id)
        area_id = slot_id.split("-")[0]
        return RouteResponse(
            slot_id=slot_id,
            area_id=area_id,
            total_distance_meters=route["total_distance"],
            path_nodes=route["path_nodes"],
            waypoints=route["waypoints"],
            turn_by_turn=route["turn_by_turn"]
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
