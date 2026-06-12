"""
Front-camera vision — gate detection, victim scanning, wall-QR centring.

Runs in its own thread.  Produces:
- ``gate_altitude_request`` — ``None``, ``gate_hold_alt``, or ``gate_avoid_alt``
- ``wall_qr_center`` — (x, y) of largest QR for building hover
- ``found_victims`` — list of victim QR codes seen
- ``display_frame`` for the MJPEG stream
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pyzbar.pyzbar import decode as decode_qr

from king_phoenix.vision import _make_placeholder

logger = logging.getLogger(__name__)


class FrontCamera:
    """Processes the forward-facing camera for gates and victim QR codes."""

    def __init__(self, config_manager) -> None:
        self._cfg = config_manager
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()

        # --- Outputs ---
        self.gate_altitude_request: Optional[float] = None
        self.wall_qr_center: Optional[Tuple[float, float]] = None
        self.found_victims: List[str] = []
        self.display_frame: Optional[np.ndarray] = None

        # --- Gate state ---
        self._last_red_seen: float = 0.0
        self._gate_pass_delay: float = 15.0

        # --- Enable / disable gate logic from mission ---
        self.gate_logic_enabled: bool = True

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> bool:
        """Open the front camera and launch the processing thread."""
        idx = self._cfg.get("system", "front_cam_index", 6)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)  # fallback
        if not cap.isOpened():
            logger.error("Front camera unavailable.")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self._cap = cap
        self._running = True

        # Warm-up
        for _ in range(30):
            cap.read()
            time.sleep(0.01)

        threading.Thread(target=self._loop, daemon=True).start()
        logger.info("Front camera thread started on index %d.", idx)
        return True

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()

    @property
    def is_active(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    def get_jpeg(self) -> bytes:
        if self.display_frame is not None:
            return cv2.imencode(".jpg", self.display_frame)[1].tobytes()
        return _make_placeholder("FRONT CAMERA\nOFFLINE", (320, 240))

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        gates_cfg = self._cfg.data["vision"]["gates"]

        while self._running:
            if self._cap is None:
                time.sleep(0.5)
                continue
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            # ---- A. Victim & centring QR ----
            self.wall_qr_center = None
            largest_area = 0

            qr_objs = decode_qr(frame)
            for obj in qr_objs:
                qr_data = obj.data.decode("utf-8")
                cv2.rectangle(
                    frame,
                    (obj.rect.left, obj.rect.top),
                    (obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height),
                    (0, 255, 0),
                    2,
                )
                if "VICTIM" in qr_data and qr_data not in self.found_victims:
                    self.found_victims.append(qr_data)
                    logger.info("VICTIM FOUND: %s", qr_data)

                area = obj.rect.width * obj.rect.height
                if area > largest_area:
                    largest_area = area
                    cx = obj.rect.left + obj.rect.width / 2
                    cy = obj.rect.top + obj.rect.height / 2
                    self.wall_qr_center = (cx, cy)

            if self.wall_qr_center:
                cv2.circle(
                    frame,
                    (int(self.wall_qr_center[0]), int(self.wall_qr_center[1])),
                    5,
                    (255, 0, 0),
                    -1,
                )

            # ---- B. Gate detection ----
            if self.gate_logic_enabled:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                mask_r = cv2.inRange(
                    hsv,
                    np.array(gates_cfg["red_lower1"]),
                    np.array(gates_cfg["red_upper1"]),
                ) + cv2.inRange(
                    hsv,
                    np.array(gates_cfg["red_lower2"]),
                    np.array(gates_cfg["red_upper2"]),
                )
                mask_y = cv2.inRange(
                    hsv,
                    np.array(gates_cfg["yellow_lower"]),
                    np.array(gates_cfg["yellow_upper"]),
                )

                cnt_red, _ = cv2.findContours(
                    mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                cnt_yel, _ = cv2.findContours(
                    mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )

                min_gate = gates_cfg["min_gate_area"]
                found_red = any(cv2.contourArea(c) > min_gate for c in cnt_red)
                found_yel = any(cv2.contourArea(c) > min_gate for c in cnt_yel)

                if found_red:
                    self.gate_altitude_request = gates_cfg["gate_avoid_alt"]
                    self._last_red_seen = time.time()
                    cv2.putText(
                        frame,
                        "RED GATE",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                elif found_yel:
                    self.gate_altitude_request = gates_cfg["gate_hold_alt"]
                    self._last_red_seen = 0
                    cv2.putText(
                        frame,
                        "YELLOW GATE",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                else:
                    if (time.time() - self._last_red_seen) < self._gate_pass_delay:
                        self.gate_altitude_request = gates_cfg["gate_avoid_alt"]
                    else:
                        self.gate_altitude_request = None

            with self._lock:
                self.display_frame = frame.copy()
            time.sleep(0.05)
