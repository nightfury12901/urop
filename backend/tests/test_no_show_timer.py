"""
Automated Validation Suite - Edge Case A:
Hold Timers, Extension, Suppression of CV False-Free, and Auto-Release on No-Show.
"""

import time
import os
import unittest
from datetime import datetime, timedelta
from backend.app.models.database import init_db, SessionLocal, User, Slot, Reservation
from backend.app.services.allocation import AllocationService
from backend.app.services.reservation_manager import ReservationManager
from backend.app.services.vision.cache import occupancy_cache

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "data", "campus_config.json")

class TestNoShowReservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db(CONFIG_PATH)
        cls.alloc_svc = AllocationService(CONFIG_PATH)
        cls.res_mgr = ReservationManager(check_interval_seconds=0.2)

    def setUp(self):
        self.db = SessionLocal()
        # Create test faculty user
        self.user = User(
            name="Dr. Alan Turing",
            role="TEACHER",
            vehicle_number="KA-01-AI-1950",
            vehicle_type="FOUR_WHEELER"
        )
        self.db.add(self.user)
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_reservation_hold_and_no_show_expiry(self):
        print("\n=== RUNNING TEST 1: NO-SHOW RESERVATION LIFECYCLE ===")
        # 1. Allocate a slot
        res = self.alloc_svc.allocate_slot(self.user.id)
        res_id = res["reservation_id"]
        slot_id = res["slot_id"]
        print(f"[Step 1] Slot {slot_id} assigned to user. Hold timer: {res['seconds_remaining']}s")

        # Verify DB slot state is RESERVED
        slot = self.db.query(Slot).filter(Slot.id == slot_id).first()
        self.assertEqual(slot.state, "RESERVED")
        self.assertTrue(slot.has_car)

        # 2. Test CV False-Free Suppression:
        # Camera sees an empty space (which is true, user hasn't arrived yet).
        # Camera detection must NOT flip RESERVED to FREE!
        occupancy_cache.update_slot(slot_id, "FREE", 0, False, 0.99)
        self.res_mgr.handle_cv_detection_event(slot_id, "FREE", 0, False)
        
        self.db.refresh(slot)
        self.assertEqual(slot.state, "RESERVED", "CRITICAL ERROR: CV loop incorrectly flipped RESERVED slot to FREE!")
        print("[Step 2] CV empty observation correctly suppressed. Slot remains RESERVED.")

        # 3. Test Extension: Extend hold once (+3 min)
        ext = self.res_mgr.extend_reservation(res_id, self.user.id)
        self.assertEqual(ext["status"], "SUCCESS")
        print(f"[Step 3] Hold timer extended once (+3 min). New time remaining: {ext['seconds_remaining']}s")

        # Try extending a second time (must fail with policy error)
        with self.assertRaises(ValueError):
            self.res_mgr.extend_reservation(res_id, self.user.id)
        print("[Step 4] Second extension blocked as expected (Max 1 extension allowed).")

        # 4. Simulate Timer Expiry (User never arrived -> No-Show)
        # Fast-forward expires_at to 10 seconds in the past
        from datetime import timezone
        past_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=10)
        self.db.query(Reservation).filter(Reservation.id == res_id).update({"expires_at": past_time})
        self.db.commit()

        # Trigger expiry watcher
        self.res_mgr.start()
        time.sleep(0.8)
        self.res_mgr.stop()

        # Verify Slot is released to FREE and Reservation marked EXPIRED
        self.db.refresh(slot)
        res_record = self.db.query(Reservation).filter(Reservation.id == res_id).first()
        
        print(f"[Step 5] After hold timeout: Slot state is '{slot.state}', Reservation status is '{res_record.status}'.")
        self.assertEqual(slot.state, "FREE", "Slot was not auto-released after countdown timeout!")
        self.assertEqual(res_record.status, "EXPIRED", "Reservation was not marked EXPIRED after timeout!")
        self.assertFalse(slot.has_car)
        print("=== TEST 1 PASSED: Edge Case A Verified Successfully! ===\n")

if __name__ == "__main__":
    unittest.main()
