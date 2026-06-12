"""
MAVLink drone controller — telemetry, commands, and PID line-following.

Communicates with a Pixhawk / ArduPilot flight stack via MAVLink over UDP.
Exposes thread-safe telemetry state and PID-computed velocity commands.
"""

from __future__ import annotations

import logging
import math
import threading
import time

from pymavlink import mavutil

logger = logging.getLogger(__name__)


class DroneController:
    """High-level interface to a MAVLink-connected multi-rotor.

    Parameters
    ----------
    config_manager : ConfigManager
        Shared config object that provides ``flight`` and ``control`` sections.
    """

    def __init__(self, config_manager) -> None:
        self._cfg = config_manager
        self.conn = None

        # --- Telemetry (read by other subsystems) ---
        self.armed: bool = False
        self.flight_mode: str = "UNKNOWN"
        self.altitude: float = 0.0  # metres, relative
        self.battery: float = 0.0  # volts
        self.heading: float = 0.0  # degrees

        # --- Yaw-rate state (set by perform_rotation, read by mission) ---
        self._commanded_yaw_rate: float = 0.0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Connection & telemetry
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        """Open the MAVLink connection and wait for a heartbeat."""
        conn_str = self._cfg.get("system", "mavlink_connect", "udpin:0.0.0.0:14550")
        logger.info("Connecting to %s …", conn_str)
        try:
            self.conn = mavutil.mavlink_connection(
                conn_str,
                autoreconnect=True,
                source_system=1,
                source_component=191,
            )
            self.conn.wait_heartbeat()
            logger.info("MAVLink heartbeat received.")
            return True
        except Exception as exc:
            logger.error("MAVLink connection failed: %s", exc)
            return False

    def update_telemetry(self) -> None:
        """Drain the incoming MAVLink buffer and update telemetry fields.

        Call this once per control-loop iteration (non-blocking).
        """
        if self.conn is None:
            return

        while True:
            msg = self.conn.recv_match(
                type=["HEARTBEAT", "SYS_STATUS", "GLOBAL_POSITION_INT", "ATTITUDE"],
                blocking=False,
            )
            if msg is None:
                break

            msg_type = msg.get_type()
            with self._lock:
                if msg_type == "HEARTBEAT":
                    self.armed = (
                        msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                    ) != 0
                    self.flight_mode = self.conn.flightmode  # type: ignore[union-attr]
                elif msg_type == "SYS_STATUS":
                    self.battery = msg.voltage_battery / 1000.0  # mV → V
                elif msg_type == "GLOBAL_POSITION_INT":
                    self.altitude = msg.relative_alt / 1000.0  # mm → m
                elif msg_type == "ATTITUDE":
                    self.heading = math.degrees(msg.yaw)

    # ------------------------------------------------------------------ #
    #  MAVLink commands
    # ------------------------------------------------------------------ #

    def set_mode(self, mode: str) -> None:
        """Set the flight mode (e.g. ``"GUIDED"``, ``"LAND"``)."""
        if self.conn is None:
            logger.warning("set_mode(%r) ignored — not connected.", mode)
            return
        try:
            mode_id = self.conn.mode_mapping()[mode]
        except KeyError:
            logger.error("Unknown flight mode: %r", mode)
            return

        self.conn.mav.set_mode_send(
            self.conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        logger.debug("Flight mode set to %s.", mode)

    def arm(self) -> None:
        """Send the arm command."""
        if self.conn is None:
            return
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        logger.info("Arm command sent.")

    def land(self) -> None:
        """Switch to LAND mode."""
        self.set_mode("LAND")
        logger.info("Land command sent.")

    def takeoff(self) -> None:
        """Initiate takeoff to the configured altitude."""
        if self.conn is None:
            return
        alt = self._cfg.get("flight", "takeoff_alt", 1.0)
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            alt,
        )
        logger.info("Takeoff sent — target %.1f m.", alt)

    # ------------------------------------------------------------------ #
    #  Advanced flight commands
    # ------------------------------------------------------------------ #

    def send_velocity_ned(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        yaw_rate: float = 0.0,
        *,
        hold_heading: bool = True,
    ) -> None:
        """Send velocity setpoint with optional yaw-rate control.

        Parameters
        ----------
        hold_heading : bool
            If ``True``, the flight controller maintains current heading.
            If ``False``, ``yaw_rate`` is used to command rotation.
        """
        if self.conn is None:
            return

        if hold_heading:
            # Bit 10 = 1 → ignore yaw rate (hold heading)
            mask = 0b0000111111000111
            yaw_rate = 0.0
        else:
            # Bit 10 = 0 → enable yaw rate control
            mask = 0b011111000111

        self.conn.mav.set_position_target_local_ned_send(
            0,
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            mask,
            0,
            0,
            0,
            vx,
            vy,
            vz,
            0,
            0,
            0,
            0,
            yaw_rate,
        )

    # ------------------------------------------------------------------ #
    #  Yaw-rate state (public property for mission)
    # ------------------------------------------------------------------ #

    @property
    def commanded_yaw_rate(self) -> float:
        return self._commanded_yaw_rate

    def perform_rotation(self, angle: float, direction: int) -> float:
        """Execute a blocking rotation.

        Parameters
        ----------
        angle : float
            Degrees to rotate.
        direction : int
            ``1`` for clockwise, ``-1`` for counter-clockwise.

        Returns
        -------
        float
            Settling delay in seconds (typically 1.5).
        """
        yaw_speed_deg = self._cfg.get("flight", "yaw_speed", 30)
        yaw_rate_rad = math.radians(yaw_speed_deg) * (1 if direction == 1 else -1)
        duration = abs(angle) / yaw_speed_deg

        logger.debug(
            "Rotating %d deg at %d deg/s (%.1f s).", angle, yaw_speed_deg, duration
        )

        # Start rotation
        self._commanded_yaw_rate = yaw_rate_rad
        self.send_velocity_ned(yaw_rate=yaw_rate_rad, hold_heading=False)
        time.sleep(duration)
        # Stop
        self._commanded_yaw_rate = 0.0
        self.send_velocity_ned(yaw_rate=0.0, hold_heading=True)

        return 1.5  # settling delay

    def compute_altitude_vz(self, target_alt: float, current_alt: float) -> float:
        """Compute vertical velocity (NED) to reach target altitude.

        Returns a value suitable for ``vz`` in ``send_velocity_ned``
        (negative = up in NED).
        """
        error = target_alt - current_alt
        vz = max(min(error * 1.5, 0.5), -0.5)
        return -vz  # NED: negative Z = up
