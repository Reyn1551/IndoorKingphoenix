"""
Bottom-camera vision — line detection with EMA smoothing and QR decoding.

Runs in its own thread.  Produces:
- ``line_detected``, ``error_x`` (smoothed lateral offset)
- ``line_angle_error`` (smoothed yaw error from fitLine)
- ``qr_data`` buffer, ``current_qr_center``, ``target_qr_center``
- ``display_frame`` for the MJPEG stream
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from pyzbar.pyzbar import decode as decode_qr

logger = logging.getLogger(__name__)


def _make_placeholder(text: str, size: tuple = (320, 240)) -> bytes:
    """Generate a dark placeholder JPEG with centred text."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = (18, 23, 31)  # dark surface colour
    # Border
    cv2.rectangle(img, (0, 0), (size[0] - 1, size[1] - 1), (30, 42, 58), 2)
    lines = text.split("\n")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, 0.55, 2)
        x = (size[0] - tw) // 2
        y = size[1] // 2 - (len(lines) - 1) * 12 + i * 24
        cv2.putText(img, line, (x, y), font, 0.55, (86, 101, 122), 2, cv2.LINE_AA)
    # Diagonal cross-hatch
    for i in range(0, size[0] + size[1], 20):
        cv2.line(img, (i, 0), (i - size[1], size[1]), (22, 28, 38), 1)
    return cv2.imencode(".jpg", img)[1].tobytes()


class BottomVision:
    """Processes the downward camera for line-following and QR waypoints."""

    def __init__(self, config_manager, grid_map: dict) -> None:
        self._cfg = config_manager
        self._grid_map = grid_map
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()

        # --- Smoothed outputs ---
        self.line_detected: bool = False
        self.error_x: float = 0.0
        self.line_angle_error: float = 0.0

        # --- QR state ---
        self.detected_qr_buffer: Optional[str] = None
        self.current_qr_center: Optional[Tuple[float, float]] = None
        self.target_qr_center: Optional[Tuple[float, float]] = None
        self.target_qr_angle: float = 0.0
        self.last_line_time: float = 0.0

        # --- Velocity commands (written by mission, read by send loop) ---
        self._current_vx: float = 0.0
        self._current_vy: float = 0.0

        # --- Display frame (for Flask) ---
        self.display_frame: Optional[np.ndarray] = None

        # --- Internal state ---
        self.last_known_direction: int = 0
        self._smooth_error_x: float = 0.0
        self._smooth_angle: float = 0.0
        self._alpha: float = 0.35  # EMA smoothing factor

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> bool:
        """Open the camera and launch the processing thread."""
        idx = self._cfg.get("system", "bottom_cam_index", 4)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            logger.error("Bottom camera %d unavailable.", idx)
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self._cap = cap
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info("Bottom camera thread started on index %d.", idx)
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
        return _make_placeholder("BOTTOM CAMERA\nOFFLINE", (320, 240))

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        while self._running:
            if self._cap is None:
                time.sleep(0.5)
                continue

            success, frame = self._cap.read()
            if not success:
                time.sleep(0.5)
                continue

            h, w = frame.shape[:2]
            vis = self._cfg.data["vision"]
            roi_ratio: float = vis["roi_height_ratio"]
            thr_val: int = vis["threshold_val"]
            is_black: bool = vis["is_black_line"]
            min_area: float = vis["min_area"]
            min_ar: float = vis["min_aspect_ratio"]
            cx_offset: int = vis.get("camera_offset_x", 0)
            ctr_weight: float = vis.get("center_priority_weight", 10.0)

            cx_screen = w // 2 + cx_offset
            cut_y = int(h * roi_ratio)
            roi = frame[cut_y:h, 0:w]

            # Pre-processing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            t_mode = cv2.THRESH_BINARY_INV if is_black else cv2.THRESH_BINARY
            _, thresh = cv2.threshold(blur, thr_val, 255, t_mode)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            best_cnt = None
            best_score = -1e9

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                _, _, bw, bh = cv2.boundingRect(cnt)
                if bh > 0 and (bh / bw) < min_ar:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx_cnt = int(M["m10"] / M["m00"])
                dist = abs(cx_cnt - cx_screen)
                score = area - (dist * ctr_weight)
                if score > best_score:
                    best_score = score
                    best_cnt = cnt

            # --- Update line state ---
            temp_detected = False
            if best_cnt is not None:
                temp_detected = True
                self.last_line_time = time.time()
                M = cv2.moments(best_cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    real_cy = cut_y + int(M["m01"] / M["m00"])
                    cv2.drawContours(roi, [best_cnt], -1, (0, 255, 255), 2)
                    cv2.line(frame, (cx_screen, real_cy), (cx, real_cy), (0, 0, 255), 2)

                    # --- fitLine for yaw angle ---
                    try:
                        [vx, vy, _, _] = cv2.fitLine(
                            best_cnt, cv2.DIST_L2, 0, 0.01, 0.01
                        )
                        angle_rad = math.atan2(vy, vx)
                        angle_deg = math.degrees(angle_rad)
                        if angle_deg < 0:
                            angle_deg += 180
                        raw_angle_error = angle_deg - 90
                        self._smooth_angle = (
                            self._alpha * raw_angle_error
                            + (1.0 - self._alpha) * self._smooth_angle
                        )
                        self.line_angle_error = self._smooth_angle
                    except Exception:
                        self.line_angle_error = 0.0

                    # --- EMA-smoothed lateral error ---
                    raw_err_x = float(cx - cx_screen)
                    self._smooth_error_x = (
                        self._alpha * raw_err_x
                        + (1.0 - self._alpha) * self._smooth_error_x
                    )
                    self.error_x = self._smooth_error_x

                    if abs(self.error_x) > 10:
                        self.last_known_direction = 1 if self.error_x > 0 else -1

            else:
                self.line_angle_error = 0.0

            self.line_detected = temp_detected

            # ---- QR detection ----
            self.detected_qr_buffer = None
            self.current_qr_center = None
            self.target_qr_center = None
            self.target_qr_angle = 0.0

            qr_objects = decode_qr(frame)
            for obj in qr_objects:
                qr_text = obj.data.decode("utf-8")
                pts = np.array([obj.polygon], np.int32)
                cv2.polylines(frame, [pts], True, (255, 0, 255), 2)

                if qr_text in self._grid_map:
                    self.detected_qr_buffer = qr_text

                # Current-node QR → centre for hovering
                if qr_text == self._get_current_node():
                    r = obj.rect
                    cx_q = r.left + r.width / 2
                    cy_q = r.top + r.height / 2
                    self.current_qr_center = (cx_q, cy_q)
                    cv2.circle(frame, (int(cx_q), int(cy_q)), 5, (255, 165, 0), -1)

                # Target-node QR → centre + orientation for landing
                if qr_text == self._get_target_node():
                    r = obj.rect
                    cx_q = r.left + r.width / 2
                    cy_q = r.top + r.height / 2
                    self.target_qr_center = (cx_q, cy_q)

                    poly = obj.polygon
                    sorted_y = sorted(poly, key=lambda p: p.y)
                    if len(sorted_y) >= 2:
                        t1, t2 = sorted_y[0], sorted_y[1]
                        tl, tr = (t1, t2) if t1.x < t2.x else (t2, t1)
                        dx, dy = tr.x - tl.x, tr.y - tl.y
                        if dx != 0:
                            self.target_qr_angle = math.degrees(math.atan2(dy, dx))
                        cv2.circle(frame, (int(cx_q), int(cy_q)), 5, (0, 0, 255), -1)
                        cv2.line(
                            frame,
                            (tl.x, tl.y),
                            (tr.x, tr.y),
                            (0, 255, 0),
                            2,
                        )

            # --- Debug overlays ---
            cv2.line(frame, (cx_screen, 0), (cx_screen, h), (0, 255, 0), 2)
            if self.line_detected:
                vis_len = 100
                vis_rad = math.radians(self.line_angle_error)
                tip_x = int(cx_screen + vis_len * math.sin(vis_rad))
                tip_y = int(h / 2 - vis_len * math.cos(vis_rad))
                cv2.line(frame, (cx_screen, h // 2), (tip_x, tip_y), (255, 0, 0), 2)

            with self._lock:
                self.display_frame = frame
            time.sleep(0.01)

    # ------------------------------------------------------------------ #
    #  Hooks provided by mission controller
    # ------------------------------------------------------------------ #

    def _get_current_node(self) -> str:
        """Override via :meth:`set_node_callbacks`."""
        return ""

    def _get_target_node(self) -> str:
        """Override via :meth:`set_node_callbacks`."""
        return ""

    def set_node_callbacks(self, current_fn, target_fn) -> None:
        """Wire the navigator node lookups into the vision thread."""
        self._get_current_node = current_fn  # type: ignore[method-assign]
        self._get_target_node = target_fn  # type: ignore[method-assign]
