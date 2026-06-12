"""
Grid navigator — BFS pathfinding on the FIRA competition grid.

The grid is a 5×3 Cartesian space with named QR-code waypoints:

    (0,2) -- (1,2) -- (2,2) -- (3,2) -- (4,2)
      |        |        |        |        |
    (0,1) -- (1,1) -- (2,1) -- (3,1) -- (4,1)
      |        |        |        |        |
    (0,0) -- (1,0) -- (2,0) -- (3,0) -- (4,0)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cardinal direction vectors
# ---------------------------------------------------------------------------
Dir = Tuple[int, int]

DIR_N: Dir = (0, 1)
DIR_E: Dir = (1, 0)
DIR_S: Dir = (0, -1)
DIR_W: Dir = (-1, 0)

# QR text → (x, y) coordinates on the 5×3 grid
GRID_MAP: Dict[str, Tuple[int, int]] = {
    "S,W,N,W,1": (0, 2),
    "ToNorth": (2, 2),
    "ToEast": (4, 2),
    "ToWest": (0, 0),
    "N,S,W,S,3": (2, 0),
    "BUILDING": (3, 0),
    "ToSouth": (4, 0),
}


class GridNavigator:
    """Manages grid position, heading, and BFS pathfinding."""

    def __init__(self, start_heading: str = "NORTH") -> None:
        self.current_node: Optional[str] = None
        self.target_node: str = "S,W,N,W,1"
        self.path: List[Tuple[int, int]] = []

        heading_map: Dict[str, Dir] = {
            "NORTH": DIR_N,
            "EAST": DIR_E,
            "SOUTH": DIR_S,
            "WEST": DIR_W,
        }
        self.current_heading: Dir = heading_map.get(start_heading, DIR_N)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def set_target(self, target_qr: str) -> bool:
        """Set a new target waypoint by QR code text."""
        if target_qr in GRID_MAP:
            self.target_node = target_qr
            logger.info("Target set to %r.", self.target_node)
            return True
        return False

    def calculate_path(self) -> List[Tuple[int, int]]:
        """BFS from current node to target node. Returns list of grid coords."""
        if not self.current_node:
            return []

        start = GRID_MAP.get(self.current_node)
        end = GRID_MAP.get(self.target_node)

        if start is None or end is None:
            return []
        if start == end:
            return []

        queue: deque = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            x, y = path[-1]

            if (x, y) == end:
                return path

            for dx, dy in (DIR_N, DIR_E, DIR_S, DIR_W):
                nx, ny = x + dx, y + dy
                if 0 <= nx <= 4 and 0 <= ny <= 2:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        new_path = list(path)
                        new_path.append((nx, ny))
                        queue.append(new_path)

        return []  # No path found

    def get_turn_angle(self, next_pos: Tuple[int, int]) -> Tuple[int, Dir]:
        """Compute required turn angle and new facing direction.

        Returns
        -------
        tuple[int, Dir]
            (angle_degrees, new_heading)
            Angle: 0, 90, -90, or 180. Positive = right (CW), negative = left.
        """
        if self.current_node not in GRID_MAP:
            return 0, self.current_heading

        curr_pos = GRID_MAP[self.current_node]
        req_dir: Dir = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])

        if req_dir == self.current_heading:
            return 0, req_dir

        cross = (
            self.current_heading[0] * req_dir[1] - self.current_heading[1] * req_dir[0]
        )
        dot = (
            self.current_heading[0] * req_dir[0] + self.current_heading[1] * req_dir[1]
        )

        if dot == -1:
            return 180, req_dir  # U-turn
        if cross > 0:
            return -90, req_dir  # Turn left (CCW)
        return 90, req_dir  # Turn right (CW)
