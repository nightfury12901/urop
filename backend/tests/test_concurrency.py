"""
Automated Validation Suite - Edge Case B:
Simultaneous Arrivals: 10 concurrent requests, Atomic Row Locking, Idempotency, and Zero Overbooking.
"""

import os
import uuid
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.app.models.database import init_db, SessionLocal, User, Slot, Reservation
from backend.app.services.allocation import AllocationService

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "data", "campus_config.json")

class TestSimultaneousAllocations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db(CONFIG_PATH)
        cls.alloc_svc = AllocationService(CONFIG_PATH)

    def setUp(self):
        db = SessionLocal()
        # Reset all slots to FREE and delete old test reservations for a clean test
        db.query(Reservation).delete()
        db.query(Slot).update({"state": "FREE", "has_car": False, "current_bikes": 0})
        db.commit()

        # Create 10 distinct student users arriving simultaneously
        self.user_data = []
        for i in range(1, 11):
            u = User(
                name=f"Student {i:02d}",
                role="STUDENT",
                vehicle_number=f"DL-01-BK-{1000 + i}",
                vehicle_type="TWO_WHEELER"
            )
            db.add(u)
            db.flush()
            self.user_data.append({"id": u.id, "name": u.name, "vehicle_number": u.vehicle_number})
        db.commit()
        db.close()

    def tearDown(self):
        pass

    def test_10_simultaneous_requests(self):
        print("\n=== RUNNING TEST 2: 10 SIMULTANEOUS ALLOCATION REQUESTS ===")
        print(f"Submitting 10 allocation requests concurrently across 10 threads...")

        run_id = uuid.uuid4().hex[:8]
        allocated_slots = []
        reservation_ids = []
        errors = []

        def worker_task(user_id, index):
            idemp_key = f"key_{run_id}_{index}"
            return self.alloc_svc.allocate_slot(user_id=user_id, idempotency_key=idemp_key)

        # Launch 10 requests at the EXACT same instant
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(worker_task, u["id"], idx): u 
                for idx, u in enumerate(self.user_data)
            }
            for future in as_completed(futures):
                user = futures[future]
                try:
                    res = future.result()
                    allocated_slots.append(res["slot_id"])
                    reservation_ids.append(res["reservation_id"])
                    print(f"  [OK] User {user['name']} ({user['vehicle_number']}) -> Assigned Slot {res['slot_id']} ({res['route']['total_distance_meters']}m)")
                except Exception as e:
                    errors.append((user['name'], str(e)))

        print(f"\nResults: {len(allocated_slots)}/10 allocated successfully. Errors: {len(errors)}")
        self.assertEqual(len(errors), 0, f"Encountered unexpected allocation errors: {errors}")
        self.assertEqual(len(allocated_slots), 10, "Not all 10 requests were fulfilled!")

        # Verify: All 10 assigned slot IDs are valid and no slot was double-booked beyond capacity!
        # Remember: A 2-wheeler slot can hold up to 2 bikes.
        slot_counts = {}
        for s in allocated_slots:
            slot_counts[s] = slot_counts.get(s, 0) + 1

        for slot_id, count in slot_counts.items():
            self.assertLessEqual(count, 2, f"Slot {slot_id} was double-booked with {count} bikes beyond max capacity 2!")

        # Verify in DB using fresh session:
        verify_db = SessionLocal()
        slots_in_db = verify_db.query(Slot).filter(Slot.id.in_(list(slot_counts.keys()))).all()
        for s in slots_in_db:
            self.assertEqual(s.current_bikes, slot_counts[s.id], f"DB current_bikes mismatch for slot {s.id}")
            self.assertEqual(s.state, "RESERVED", f"Slot {s.id} is not in RESERVED state")
        verify_db.close()

        # Test Idempotency: Repeating request for user 0 with same key must return SAME reservation
        print("\nTesting Idempotency: Repeating request for User 1 with same key...")
        repeat_res = self.alloc_svc.allocate_slot(user_id=self.user_data[0]["id"], idempotency_key=f"key_{run_id}_0")
        self.assertEqual(repeat_res["reservation_id"], reservation_ids[0])
        print(f"  [Idempotency OK] Returned identical reservation {repeat_res['reservation_id']} without duplicate booking.")

        print("=== TEST 2 PASSED: Edge Case B Concurrency Verified Successfully! ===\n")

if __name__ == "__main__":
    unittest.main()
