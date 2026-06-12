"""IndoorKingphoenix — FIRA 2026 autonomous indoor drone.

Grid-based navigation with BFS pathfinding, dual-camera vision
(line + gate detection), T265 odometry, building traversal,
QR-centric precision landing, and a Flask web GCS.
"""

from king_phoenix.config import ConfigManager
from king_phoenix.drone import DroneController
from king_phoenix.front_vision import FrontCamera
from king_phoenix.gcs import create_app
from king_phoenix.mission import MissionController, MissionState
from king_phoenix.navigator import GRID_MAP, GridNavigator
from king_phoenix.t265 import T265Handler
from king_phoenix.vision import BottomVision

__version__ = "2.0.0"
__all__ = [
    "ConfigManager",
    "DroneController",
    "BottomVision",
    "FrontCamera",
    "GridNavigator",
    "GRID_MAP",
    "MissionController",
    "MissionState",
    "T265Handler",
    "create_app",
]
