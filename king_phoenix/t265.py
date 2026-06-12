"""
Intel RealSense T265 visual-inertial odometry bridge.

Streams pose data from the T265 tracking camera to the Pixhawk via
the MAVLink ``VISION_POSITION_ESTIMATE`` message for non-GPS navigation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from king_phoenix.config import ConfigManager
    from king_phoenix.drone import DroneController

# ---------------------------------------------------------------------------
# Graceful import of optional dependencies
# ---------------------------------------------------------------------------
try:
    import pyrealsense2 as rs

    _RS_AVAILABLE = True
except ImportError:
    rs = None  # type: ignore[assignment]
    _RS_AVAILABLE = False

try:
    import transformations as tf

    _TF_AVAILABLE = True
except ImportError:
    tf = None  # type: ignore[assignment]
    _TF_AVAILABLE = False


# Coordinate transformation: T265 camera frame → NED aero frame
_H_AEROREF_T265REF = np.array(
    [
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

_H_T265BODY_AEROBODY = np.linalg.inv(_H_AEROREF_T265REF)


class T265Handler:
    """Manages the T265 tracking camera pipeline.

    Runs a background thread that reads pose frames and publishes them
    as MAVLink ``VISION_POSITION_ESTIMATE`` messages.

    Parameters
    ----------
    config_manager : ConfigManager
        Provides the ``t265`` configuration section.
    drone : DroneController
        The MAVLink controller used to send vision messages.
    """

    def __init__(self, config_manager: ConfigManager, drone: DroneController) -> None:
        self._cfg = config_manager
        self._drone = drone
        self._pipe: "rs.pipeline | None" = None  # type: ignore[name-defined]
        self._running: bool = False
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Launch the T265 pipeline and background thread.

        Returns ``True`` on success, ``False`` if dependencies are missing
        or the camera cannot be initialised.
        """
        if not _RS_AVAILABLE or not _TF_AVAILABLE:
            logger.error("T265 requires 'pyrealsense2' and 'transformations' packages.")
            return False

        try:
            self._pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.pose)
            self._pipe.start(cfg)
            self._running = True

            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            logger.info("T265 service started.")
            return True
        except Exception as exc:
            logger.error("Failed to start T265: %s", exc)
            return False

    def stop(self) -> None:
        """Stop the T265 pipeline and join the background thread."""
        self._running = False
        if self._pipe is not None:
            try:
                self._pipe.stop()
                logger.info("T265 pipeline stopped.")
            except Exception as exc:
                logger.warning("Error stopping T265 pipeline: %s", exc)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        """Background thread: read pose and publish to MAVLink."""
        scale = self._cfg.get("t265", "scale_factor", 1.0)

        while self._running:
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=1000)  # type: ignore[union-attr]
                pose = frames.get_pose_frame()
                if not pose:
                    continue

                data = pose.get_pose_data()

                # Build homogeneous transform from T265 data
                H_t265 = tf.quaternion_matrix(
                    [  # type: ignore[union-attr]
                        data.rotation.w,
                        data.rotation.x,
                        data.rotation.y,
                        data.rotation.z,
                    ]
                )
                H_t265[0, 3] = data.translation.x * scale
                H_t265[1, 3] = data.translation.y * scale
                H_t265[2, 3] = data.translation.z * scale

                # Convert to NED frame
                H_final = _H_AEROREF_T265REF @ H_t265 @ _H_T265BODY_AEROBODY

                # Extract Euler angles
                rpy = tf.euler_from_matrix(H_final, "sxyz")  # type: ignore[union-attr]

                # Publish via MAVLink
                now_us = int(round(time.time() * 1_000_000))
                if self._drone.conn is not None:
                    self._drone.conn.mav.vision_position_estimate_send(
                        now_us,
                        H_final[0, 3],
                        H_final[1, 3],
                        H_final[2, 3],
                        rpy[0],
                        rpy[1],
                        rpy[2],
                        [0.0] * 21,  # covariance (unused)
                        0,  # reset counter
                    )

            except RuntimeError:
                # Pipeline was stopped from another thread
                break
            except Exception as exc:
                logger.warning("T265 loop error: %s", exc)
