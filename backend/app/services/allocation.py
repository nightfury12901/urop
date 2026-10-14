import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session

from ..models.database import SessionLocal, Slot, Reservation, User
from ..schemas.parking import ReservationResponse, RouteResponse
from ..services.routing import CampusRouter
from ..services.vision.cache import occupancy_cache

import json

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "campus_config.json")

def get_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)

# ──────────────────────────────────────────────────────
# Lazy reference to VisionWorker (set from main.py)
# Avoids circular imports while still allowing live snapshot
# ──────────────────────────────────────────────────────
_vision_worker_ref = None

def set_vision_worker(worker):
    global _vision_worker_ref
    _vision_worker_ref = worker


class AllocationService:
    """
    Atomic slot allocation service.

    Flow when user clicks "Find & Reserve":
      1. Grab a fresh snapshot from every eligible camera zone
      2. Run CV detection → update occupancy cache
      3. Lock the best free slot at DB level (row-level with thread lock)
      4. Return reservation + Google Maps directions link
    """
    _lock = threading.Lock()   # serialises concurrent allocations

    def __init__(self, config_path: str = CONFIG_PATH):
        self.router = CampusRouter(config_path)
        self._cfg = self._load_config(config_path)

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────
    def allocate_slot(self, user_id: str, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Atomically allocates the nearest suitable parking slot.
        Thread-safe, idempotent.
        """
        with self._lock:
            db: Session = SessionLocal()
            try:
                # 1. Fetch user
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    raise ValueError(f"User with ID {user_id} not found.")

                now_utc = get_utc()

                # 2. Idempotency guard – return existing active reservation if present
                active_res = db.query(Reservation).filter(
                    Reservation.user_id == user_id,
                    Reservation.status == "ACTIVE",
                    Reservation.expires_at > now_utc
                ).first()
                if active_res:
                    return self._build_reservation_dict(db, active_res, user)

                if idempotency_key:
                    existing_idemp = db.query(Reservation).filter(
                        Reservation.idempotency_key == idempotency_key
                    ).first()
                    if existing_idemp:
                        return self._build_reservation_dict(db, existing_idemp, user)

                # 3. Role & vehicle validation
                role   = user.role.upper()
                v_type = user.vehicle_type.upper()

                if role == "STUDENT" and v_type == "FOUR_WHEELER":
                    raise ValueError(
                        "Student permits are restricted to 2-wheelers only. "
                        "4-wheelers are not permitted."
                    )

                # 4. Determine eligible zones (P1 for students, P3 for faculty/VIP)
                allowed_areas = self._get_allowed_areas(role, v_type)

                # 5. ── ON-DEMAND CAMERA SNAPSHOT ──────────────────────────────────
                #    Check camera feed for this specific parking zone (P1 or P3)
                #    Grabs fresh frame and detects empty/occupied slots.
                cam_verified = self._snapshot_eligible_zones(allowed_areas)
                # ─────────────────────────────────────────────────────────────────

                # 6. Find eligible free slots (filtered through live CV results)
                candidates = self._find_candidate_slots(db, allowed_areas, v_type)
                if not candidates:
                    raise ValueError(
                        f"All designated slots in parking zone ({', '.join(allowed_areas)}) are currently occupied. "
                        "Please wait a moment and try again."
                    )

                # 7. Nearest slot first
                candidates.sort(key=lambda s: (s.distance_from_gate, s.slot_number))
                best_slot = candidates[0]

                # 8. Atomic reservation write
                hold_duration = timedelta(minutes=5)
                expires_at    = now_utc + hold_duration

                if v_type == "FOUR_WHEELER":
                    best_slot.state   = "RESERVED"
                    best_slot.has_car = True
                else:
                    best_slot.current_bikes += 1
                    best_slot.state = "RESERVED"

                best_slot.updated_at = now_utc
                db.add(best_slot)

                new_reservation = Reservation(
                    user_id          = user.id,
                    slot_id          = best_slot.id,
                    vehicle_type     = user.vehicle_type,
                    vehicle_number   = user.vehicle_number,
                    status           = "ACTIVE",
                    reserved_at      = now_utc,
                    expires_at       = expires_at,
                    extensions_used  = 0,
                    idempotency_key  = idempotency_key
                )
                db.add(new_reservation)
                db.commit()
                db.refresh(new_reservation)

                return self._build_reservation_dict(db, new_reservation, user, camera_verified=cam_verified)

            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

    # ─────────────────────────────────────────────
    # Camera snapshot (on-demand, per eligible zone)
    # ─────────────────────────────────────────────
    def _snapshot_eligible_zones(self, area_ids: List[str]) -> bool:
        """
        Grabs one fresh frame from the camera covering the eligible zone (e.g. P1 for students, P3 for teachers),
        runs the CV detector, and updates the occupancy cache immediately.
        Returns True if a camera frame was successfully captured and analyzed; False if offline/mock fallback.
        """
        if _vision_worker_ref is None:
            return False
        try:
            return bool(_vision_worker_ref.snapshot_areas(area_ids))
        except Exception as e:
            print(f"[AllocationService] Camera snapshot warning: {e}")
            return False

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────
    def _get_allowed_areas(self, role: str, vehicle_type: str) -> List[str]:
        if role == "STUDENT":
            return ["P1"]
        elif role == "TEACHER":
            return ["P3"]
        elif role == "VIP":
            return ["P3"]
        return ["P1", "P3"]

    def _find_candidate_slots(self, db: Session, areas: List[str], vehicle_type: str) -> List[Slot]:
        slots = db.query(Slot).filter(
            Slot.area_id.in_(areas),
            Slot.state == "FREE"
        ).all()

        candidates = []
        for s in slots:
            # Cross-check with live CV cache
            cached = occupancy_cache.get_slot(s.id)
            if cached and cached.get("detected_state") == "OCCUPIED":
                continue   # Camera physically sees vehicle in slot – skip

            if vehicle_type == "FOUR_WHEELER":
                if s.slot_type == "CAR_OR_2BIKES" and not s.has_car and s.current_bikes == 0:
                    candidates.append(s)
            else:   # TWO_WHEELER
                if s.current_bikes < s.bike_capacity and not s.has_car:
                    candidates.append(s)
        
        # Robust fallback: If CV cache filtered out all slots due to lighting/noise, but DB shows slots are free, fallback to DB
        if not candidates and slots:
            print(f"[AllocationService] Camera cache marked all slots occupied; falling back to DB free slots.")
            for s in slots:
                if vehicle_type == "FOUR_WHEELER":
                    if s.slot_type == "CAR_OR_2BIKES" and not s.has_car and s.current_bikes == 0:
                        candidates.append(s)
                else:
                    if s.current_bikes < s.bike_capacity and not s.has_car:
                        candidates.append(s)

        return candidates

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def _get_slot_google_maps_url(self, slot_id: str) -> Optional[str]:
        """Looks up the pre-computed Google Maps URL for this slot from config."""
        for s in self._cfg.get("slots", []):
            if s["id"] == slot_id:
                return s.get("google_maps_url")
        return None

    def _build_reservation_dict(self, db: Session, res: Reservation, user: User, camera_verified: bool = False) -> Dict[str, Any]:
        route          = self.router.get_route(res.slot_id)
        secs_remaining = max(0, int((res.expires_at - get_utc()).total_seconds()))
        slot           = db.query(Slot).filter(Slot.id == res.slot_id).first()
        area_id        = slot.area_id if slot else res.slot_id.split("-")[0]
        maps_url       = self._get_slot_google_maps_url(res.slot_id)

        return {
            "reservation_id":   res.id,
            "user_id":          user.id,
            "user_name":        user.name,
            "user_role":        user.role,
            "slot_id":          res.slot_id,
            "area_id":          area_id,
            "vehicle_number":   res.vehicle_number,
            "vehicle_type":     res.vehicle_type,
            "status":           res.status,
            "reserved_at":      res.reserved_at.isoformat(),
            "expires_at":       res.expires_at.isoformat(),
            "seconds_remaining": secs_remaining,
            "extensions_used":  res.extensions_used,
            "google_maps_url":  maps_url,
            "camera_verified":  camera_verified,
            "route": {
                "slot_id":               res.slot_id,
                "area_id":               area_id,
                "total_distance_meters": route["total_distance"],
                "path_nodes":            route["path_nodes"],
                "waypoints":             route["waypoints"],
                "turn_by_turn":          route["turn_by_turn"]
            }
        }
