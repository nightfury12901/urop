import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from ..models.database import SessionLocal, Slot, Reservation, User, OccupancyLog
from ..services.vision.cache import occupancy_cache

def get_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class ReservationManager:
    """
    Manages the full lifecycle of reservations:
    - 5-minute hold timer countdown
    - Voluntary cancellation
    - 1-time hold extension (+3 min)
    - Background TTL expiry and auto-release on no-show
    - CV physical arrival trigger (RESERVED -> OCCUPIED)
    - Wrong-slot misparking anomaly detection
    """
    def __init__(self, check_interval_seconds: float = 1.0):
        self.check_interval = check_interval_seconds
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.anomalies: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._expiry_loop, daemon=True, name="Reservation-TTL-Watcher")
            self.thread.start()
            print("[ReservationManager] TTL Watcher background service started.")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def cancel_reservation(self, reservation_id: str, user_id: str) -> Dict[str, Any]:
        """User voluntarily cancels their active reservation."""
        db: Session = SessionLocal()
        try:
            res = db.query(Reservation).filter(
                Reservation.id == reservation_id,
                Reservation.user_id == user_id,
                Reservation.status == "ACTIVE"
            ).first()
            if not res:
                raise ValueError("Active reservation not found or unauthorized.")

            res.status = "CANCELLED"
            slot = db.query(Slot).filter(Slot.id == res.slot_id).first()
            if slot:
                if res.vehicle_type == "FOUR_WHEELER":
                    slot.has_car = False
                    slot.state = "FREE"
                else: # TWO_WHEELER
                    slot.current_bikes = max(0, slot.current_bikes - 1)
                    if slot.current_bikes == 0 and not slot.has_car:
                        slot.state = "FREE"

            db.commit()
            return {"status": "SUCCESS", "message": f"Reservation {reservation_id} cancelled. Slot {res.slot_id} is released."}
        finally:
            db.close()

    def extend_reservation(self, reservation_id: str, user_id: str) -> Dict[str, Any]:
        """Extends the 5-minute hold timer by 3 minutes (allowed exactly once)."""
        db: Session = SessionLocal()
        try:
            res = db.query(Reservation).filter(
                Reservation.id == reservation_id,
                Reservation.user_id == user_id,
                Reservation.status == "ACTIVE"
            ).first()
            if not res:
                raise ValueError("Active reservation not found or unauthorized.")

            if res.extensions_used >= 1:
                raise ValueError("Reservation hold timer has already been extended the maximum permitted (1 time).")

            res.extensions_used += 1
            res.expires_at = res.expires_at + timedelta(minutes=3)
            db.commit()
            
            secs_remaining = int((res.expires_at - get_utc()).total_seconds())
            return {
                "status": "SUCCESS",
                "message": "Hold timer extended by +3 minutes.",
                "new_expires_at": res.expires_at.isoformat(),
                "seconds_remaining": max(0, secs_remaining)
            }
        finally:
            db.close()

    def handle_cv_detection_event(self, slot_id: str, detected_state: str, bikes: int, has_car: bool):
        """
        Called when the camera vision loop updates a slot:
        - If slot was RESERVED and camera sees vehicle -> transitions to OCCUPIED!
        - Checks for wrong-slot parking anomaly.
        """
        db: Session = SessionLocal()
        try:
            slot = db.query(Slot).filter(Slot.id == slot_id).first()
            if not slot:
                return

            now_utc = get_utc()
            # Case 1: Slot is RESERVED and vehicle has physically arrived
            if slot.state == "RESERVED" and detected_state == "OCCUPIED":
                active_res = db.query(Reservation).filter(
                    Reservation.slot_id == slot_id,
                    Reservation.status == "ACTIVE"
                ).first()
                if active_res:
                    active_res.status = "COMPLETED"
                
                slot.state = "OCCUPIED"
                slot.updated_at = now_utc
                db.commit()
                print(f"[ReservationManager] Vehicle arrived! Slot {slot_id} transitioned from RESERVED -> OCCUPIED.")

            # Case 2: Slot was OCCUPIED and vehicle departed
            elif slot.state == "OCCUPIED" and detected_state == "FREE":
                slot.state = "FREE"
                slot.has_car = False
                slot.current_bikes = 0
                slot.updated_at = now_utc
                db.commit()
                print(f"[ReservationManager] Vehicle departed! Slot {slot_id} transitioned from OCCUPIED -> FREE.")

            # Case 3: Misparking Anomaly Detection
            # If an unreserved slot suddenly became occupied in the same area where another slot is waiting on a reservation
            elif slot.state == "FREE" and detected_state == "OCCUPIED":
                # Check if there is an active reservation in this area waiting for arrival
                waiting_res = db.query(Reservation).join(Slot).filter(
                    Slot.area_id == slot.area_id,
                    Reservation.status == "ACTIVE"
                ).first()
                if waiting_res and waiting_res.slot_id != slot_id:
                    with self.lock:
                        if not any(a["expected_slot"] == waiting_res.slot_id and a["occupied_slot"] == slot_id and not a.get("resolved") for a in self.anomalies):
                            anomaly = {
                                "type": "MISPARK_SUSPECTED",
                                "slot_id": slot_id,
                                "expected_slot": waiting_res.slot_id,
                                "occupied_slot": slot_id,
                                "anomaly_reason": f"Driver assigned {waiting_res.slot_id} but parked in {slot_id}",
                                "area_id": slot.area_id,
                                "user_id": waiting_res.user_id,
                                "timestamp": now_utc.isoformat(),
                                "resolved": False
                            }
                            self.anomalies.append(anomaly)
                            print(f"[ALERT: MISPARK] User assigned {waiting_res.slot_id} may have parked in {slot_id}!")
        finally:
            db.close()

    def _expiry_loop(self):
        """Continuously checks for expired reservation hold timers and auto-releases slots."""
        while self.running:
            try:
                db: Session = SessionLocal()
                now = get_utc()
                expired_reservations = db.query(Reservation).filter(
                    Reservation.status == "ACTIVE",
                    Reservation.expires_at < now
                ).all()

                for res in expired_reservations:
                    res.status = "EXPIRED"
                    slot = db.query(Slot).filter(Slot.id == res.slot_id).first()
                    if slot:
                        if res.vehicle_type == "FOUR_WHEELER":
                            slot.has_car = False
                            slot.state = "FREE"
                        else:
                            slot.current_bikes = max(0, slot.current_bikes - 1)
                            if slot.current_bikes == 0 and not slot.has_car:
                                slot.state = "FREE"
                        slot.updated_at = now
                    print(f"[ReservationManager] Hold timeout for reservation {res.id}. Slot {res.slot_id} auto-released!")

                if expired_reservations:
                    db.commit()
                db.close()
            except Exception as e:
                print(f"[ReservationManager] Error in expiry loop: {e}")

            time.sleep(self.check_interval)

    def get_anomalies(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.anomalies)

reservation_manager = ReservationManager()
