"""
IndoorKingphoenix — Main entry point.

Starts all subsystems (drone, bottom cam, front cam, T265, navigator,
mission controller, Flask GCS) and runs until shutdown.

Usage::

    python main.py
"""

from __future__ import annotations

import logging
import signal
import sys
import threading

from king_phoenix.config import ConfigManager
from king_phoenix.drone import DroneController
from king_phoenix.front_vision import FrontCamera
from king_phoenix.gcs import create_app
from king_phoenix.mission import MissionController
from king_phoenix.navigator import GRID_MAP, GridNavigator
from king_phoenix.t265 import T265Handler
from king_phoenix.vision import BottomVision


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("pymavlink").setLevel(logging.WARNING)


def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    # --- Bootstrap ---
    cfg = ConfigManager()
    drone = DroneController(cfg)
    nav = GridNavigator(cfg.get("system", "start_heading", "NORTH"))
    bottom = BottomVision(cfg, GRID_MAP)
    front = FrontCamera(cfg)
    t265 = T265Handler(cfg, drone)
    mission = MissionController(cfg, drone, bottom, front, t265, nav)

    # --- Cameras ---
    bottom.start()
    front.start()

    # --- Background mission thread ---
    mission_thread = threading.Thread(target=mission.run, daemon=True)
    mission_thread.start()

    # --- Flask GCS ---
    app = create_app(bottom, front, drone, mission)

    def _shutdown(signum, frame) -> None:  # noqa: ARG001
        logger.info("Signal %d — shutting down.", signum)
        mission.stop_mission()
        bottom.stop()
        front.stop()
        t265.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    port = cfg.get("system", "http_port", 5000)
    logger.info("GCS: http://0.0.0.0:%d", port)

    try:
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except Exception:
        logger.exception("Flask error")
    finally:
        _shutdown(0, None)


if __name__ == "__main__":
    main()
