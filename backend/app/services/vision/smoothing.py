from collections import deque, Counter
from typing import Dict, Any, Tuple
import threading

class TemporalFilter:
    """
    Temporal sliding window smoother (N frames majority vote).
    Suppresses transient sensor flickers, shadow fluctuations, and lighting noise.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # slot_id -> deque of (state, bikes, has_car)
        self.buffers: Dict[str, deque] = {}
        self.lock = threading.Lock()

    def add_observation(self, slot_id: str, state: str, bikes: int, has_car: bool, conf: float) -> Tuple[str, int, bool, float]:
        with self.lock:
            if slot_id not in self.buffers:
                self.buffers[slot_id] = deque(maxlen=self.window_size)
            
            self.buffers[slot_id].append((state, bikes, has_car, conf))
            
            # Majority vote on state
            states = [obs[0] for obs in self.buffers[slot_id]]
            state_counts = Counter(states)
            majority_state = state_counts.most_common(1)[0][0]
            
            # Find the most recent observation matching majority state to get bikes/car flags
            matching = [obs for obs in self.buffers[slot_id] if obs[0] == majority_state]
            avg_conf = sum(obs[3] for obs in matching) / max(len(matching), 1)
            latest_match = matching[-1]
            
            return majority_state, latest_match[1], latest_match[2], avg_conf

    def reset_slot(self, slot_id: str) -> None:
        with self.lock:
            if slot_id in self.buffers:
                self.buffers[slot_id].clear()
