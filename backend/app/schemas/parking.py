from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class UserRegisterRequest(BaseModel):
    name: str = Field(..., example="Rahul Sharma")
    role: str = Field(..., example="STUDENT") # STUDENT, TEACHER, VIP
    vehicle_number: str = Field(..., example="KA-05-EV-1234")
    vehicle_type: str = Field(..., example="TWO_WHEELER") # TWO_WHEELER, FOUR_WHEELER

class UserResponse(BaseModel):
    id: str
    name: str
    role: str
    vehicle_number: str
    vehicle_type: str
    created_at: datetime

class Coordinate(BaseModel):
    x: float
    y: float
    lat: Optional[float] = None
    lng: Optional[float] = None

class RouteResponse(BaseModel):
    slot_id: str
    area_id: str
    total_distance_meters: float
    path_nodes: List[str]
    waypoints: List[Coordinate]
    turn_by_turn: List[str]

class AllocationRequest(BaseModel):
    user_id: str
    idempotency_key: Optional[str] = None

class ReservationResponse(BaseModel):
    reservation_id: str
    user_id: str
    user_name: str
    user_role: str
    slot_id: str
    area_id: str
    vehicle_number: str
    vehicle_type: str
    status: str
    reserved_at: datetime
    expires_at: datetime
    seconds_remaining: int
    extensions_used: int
    google_maps_url: Optional[str] = None
    camera_verified: Optional[bool] = False
    route: RouteResponse

class SlotSummary(BaseModel):
    id: str
    area_id: str
    slot_number: int
    slot_type: str
    state: str
    current_bikes: int
    bike_capacity: int
    has_car: bool
    distance_from_gate: float
    detected_state: Optional[str] = "FREE"

class AreaSummaryResponse(BaseModel):
    id: str
    name: str
    allowed_roles: List[str]
    allowed_vehicles: List[str]
    total_slots: int
    free_slots: int
    occupied_slots: int
    reserved_slots: int
