import json
import heapq
from typing import Dict, List, Tuple, Any, Optional

class CampusRouter:
    """
    Dijkstra shortest-path routing engine on the college campus layout.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.nodes = self.config.get("graph", {}).get("nodes", {})
        self.edges = self.config.get("graph", {}).get("edges", [])
        self.gate_id = self.config.get("campus_info", {}).get("gate_id", "GATE")
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {}
        self._build_adjacency()

    def _build_adjacency(self):
        for n in self.nodes:
            self.adjacency[n] = []
        for e in self.edges:
            u, v, w = e["from"], e["to"], float(e["weight"])
            if u in self.adjacency and v in self.nodes:
                self.adjacency[u].append((v, w))

    def get_route(self, target_slot_id: str, start_node: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs Dijkstra from start_node (default: College Gate) to target_slot_id.
        Returns:
            {
                "slot_id": target_slot_id,
                "total_distance": float,
                "path_nodes": List[str],
                "waypoints": List[{"x": float, "y": float}],
                "turn_by_turn": List[str]
            }
        """
        start = start_node or self.gate_id
        if target_slot_id not in self.nodes:
            raise ValueError(f"Target node {target_slot_id} not found in campus graph.")

        # Dijkstra priority queue: (distance, current_node, path)
        pq = [(0.0, start, [start])]
        visited = set()
        min_dist = {n: float("inf") for n in self.nodes}
        min_dist[start] = 0.0

        best_path = []
        best_dist = float("inf")

        while pq:
            dist, curr, path = heapq.heappop(pq)
            if curr in visited:
                continue
            visited.add(curr)

            if curr == target_slot_id:
                best_dist = dist
                best_path = path
                break

            for neighbor, weight in self.adjacency.get(curr, []):
                new_dist = dist + weight
                if new_dist < min_dist.get(neighbor, float("inf")):
                    min_dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor, path + [neighbor]))

        # Build coordinate waypoints
        waypoints = []
        for node_id in best_path:
            node_info = self.nodes.get(node_id, {})
            waypoints.append({
                "x": node_info.get("x", 0.0),
                "y": node_info.get("y", 0.0),
                "lat": node_info.get("gps_lat"),
                "lng": node_info.get("gps_lng")
            })

        # Build turn-by-turn guidance
        directions = self._generate_directions(best_path, target_slot_id)

        return {
            "slot_id": target_slot_id,
            "total_distance": round(best_dist, 1),
            "path_nodes": best_path,
            "waypoints": waypoints,
            "turn_by_turn": directions
        }

    def _generate_directions(self, path: List[str], target_slot_id: str) -> List[str]:
        steps = []
        steps.append("1. Enter campus through the Main College Gate (12.8231°N, 80.0410°E).")
        
        area_code = target_slot_id.split("-")[0]
        if area_code == "P1":
            area_name = "Student Parking Lot (Bikes Only)"
        elif area_code == "P3":
            area_name = "Faculty & Staff Parking Lot"
        else:
            area_name = f"Parking Area {area_code}"

        for node in path:
            if node == "JUNC_1":
                steps.append("2. Proceed along Campus Main Road towards the parking zones.")
            elif node == "ENTRY_P1":
                steps.append(f"3. Turn into {area_name} entrance.")
            elif node == "ENTRY_P3":
                steps.append(f"3. Continue straight and turn into {area_name} entrance.")

        steps.append(f"-> Arrive and park directly inside assigned Spot {target_slot_id}.")
        return steps
