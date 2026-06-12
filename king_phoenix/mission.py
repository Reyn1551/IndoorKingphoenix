"""
Mission state machine — full FIRA 2026 autonomous flight controller.

Coordinates drone, bottom vision, front camera, T265, and grid navigator
through a multi-state finite-state machine:

    INIT → WAIT_USER → TAKEOFF → INITIAL_SCAN → CALCULATING
      → TURNING → PUSH_OUT → FOLLOW_LINE → AT_NODE
      → [BUILDING_ASCEND → BUILDING_SCAN_HOVER → BUILDING_CREATE_GAP
         → BUILDING_ALIGN_CENTER → BUILDING_TURN → BUILDING_DESCEND
         → BUILDING_RECOVER]
      → ALIGN_NORTH → CENTERING → LANDING
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from king_phoenix.config import ConfigManager
    from king_phoenix.drone import DroneController
    from king_phoenix.front_vision import FrontCamera
    from king_phoenix.navigator import GridNavigator
    from king_phoenix.t265 import T265Handler
    from king_phoenix.vision import BottomVision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  State enum
# ---------------------------------------------------------------------------


class MissionState(IntEnum):
    INIT = 0
    WAIT_USER = 1
    TAKEOFF = 2
    INITIAL_SCAN = 5
    FOLLOW_LINE = 10
    AT_NODE = 20
    CALCULATING = 30
    TURNING = 40
    PUSH_OUT = 50
    ALIGN_NORTH = 60
    CENTERING = 80
    LANDING = 99
    SCANNING = 100
    BUILDING_INIT = 200
    BUILDING_ASCEND = 201
    BUILDING_SCAN_HOVER = 202
    BUILDING_FIND_EDGE = 203
    BUILDING_CREATE_GAP = 204
    BUILDING_ALIGN_CENTER = 205
    BUILDING_TURN = 206
    BUILDING_DESCEND = 207
    BUILDING_RECOVER = 208


# ---------------------------------------------------------------------------
#  Shared telemetry dataclass (thread-safe reads)
# ---------------------------------------------------------------------------


class Telemetry:
    """Thread-safe bag of telemetry values for the GCS."""

    def __init__(self) -> None:
        self.mode: str = "INIT"
        self.armed: bool = False
        self.alt: float = 0.0
        self.msg: str = "Booting..."
        self.curr: str = "?"
        self.target: str = "NONE"
        self._lock = threading.Lock()

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "armed": self.armed,
                "alt": self.alt,
                "msg": self.msg,
                "curr": self.curr,
                "target": self.target,
            }


# ---------------------------------------------------------------------------
#  Mission controller
# ---------------------------------------------------------------------------


class MissionController:
    """Orchestrates the full autonomous mission."""

    def __init__(
        self,
        config: ConfigManager,
        drone: DroneController,
        bottom: BottomVision,
        front: FrontCamera,
        t265: T265Handler,
        navigator: GridNavigator,
    ) -> None:
        self._cfg = config
        self._drone = drone
        self._bot = bottom
        self._front = front
        self._t265 = t265
        self._nav = navigator
        self._telemetry = Telemetry()

        # State
        self._state: MissionState = MissionState.INIT
        self._running = False
        self._lock = threading.Lock()
        self._start_cmd = False

        # Timers & helpers
        self._state_timer: float = 0.0
        self._building_side_count: int = 0
        self._visited_nodes: Set[str] = set()

        # Wire vision callbacks
        self._bot.set_node_callbacks(
            lambda: self._nav.current_node or "",
            lambda: self._nav.target_node,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> MissionState:
        with self._lock:
            return self._state

    @property
    def telemetry(self) -> Telemetry:
        return self._telemetry

    def start_mission(self) -> None:
        self._start_cmd = True

    def stop_mission(self) -> None:
        with self._lock:
            self._state = MissionState.LANDING

    def set_target(self, qr_text: str) -> bool:
        return self._nav.set_target(qr_text)

    # ------------------------------------------------------------------ #
    #  Main loop (called from background thread)
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Blocking main loop — connect, spin up subsystems, then tick."""
        self._drone.connect()
        self._t265.start()
        self._running = True

        # Wait for T265 confidence
        self._progress("INITIALISING T265...")
        while self._running:
            # Non-blocking check — gate.py used a global `data` for this;
            # our T265 handler publishes directly, so proceed when ready.
            self._update_telemetry()
            time.sleep(0.5)
            if self._t265._running:  # type: ignore[union-attr]
                break

        self._progress("READY: SET TARGET")
        self._transition(MissionState.WAIT_USER)

        while self._running:
            self._update_telemetry()
            self._tick()
            time.sleep(0.1)  # 10 Hz mission loop

        self._shutdown()

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _progress(self, msg: str) -> None:
        print(msg, file=sys.stdout, flush=True)
        self._telemetry.update(msg=msg)
        logger.info(msg)

    def _transition(self, new_state: MissionState) -> None:
        old = self._state
        self._state = new_state
        self._telemetry.update(mode=new_state.name)
        logger.debug("State: %s → %s", old.name, new_state.name)

    def _update_telemetry(self) -> None:
        self._drone.update_telemetry()
        alt = self._drone.altitude
        self._telemetry.update(
            armed=self._drone.armed,
            alt=alt,
            curr=self._nav.current_node or "?",
            target=self._nav.target_node,
        )

    def _tick(self) -> None:
        """Execute one step of the state machine."""
        state = self._state

        # ---- INIT ----
        if state == MissionState.INIT:
            pass  # handled in run()

        # ---- WAIT_USER ----
        elif state == MissionState.WAIT_USER:
            if self._start_cmd:
                self._progress("STARTING...")
                time.sleep(1)
                self._transition(MissionState.TAKEOFF)

        # ---- TAKEOFF ----
        elif state == MissionState.TAKEOFF:
            self._drone.set_mode("GUIDED")
            self._drone.arm()
            timeout = time.time()
            while not self._drone.armed and (time.time() - timeout) < 10:
                time.sleep(0.5)
                self._update_telemetry()
            self._drone.takeoff()
            target_alt = self._cfg.get("flight", "takeoff_alt", 0.5)
            while self._drone.altitude < (target_alt - 0.2):
                time.sleep(0.5)
                self._update_telemetry()
            self._progress("HOVER: SCANNING FOR START QR...")
            self._bot.detected_qr_buffer = None
            self._transition(MissionState.INITIAL_SCAN)

        # ---- INITIAL_SCAN ----
        elif state == MissionState.INITIAL_SCAN:
            qr = self._bot.detected_qr_buffer
            if qr and qr in self._nav._grid_map:
                self._nav.current_node = qr
                self._nav.current_heading = (0, 1)  # DIR_N
                self._progress(f"LOCATED: {qr} (RESET N)")
                time.sleep(1.0)
                self._transition(MissionState.CALCULATING)

        # ---- CALCULATING ----
        elif state == MissionState.CALCULATING:
            self._reset_pid()
            self._bot.detected_qr_buffer = None

            if self._nav.current_node == self._nav.target_node:
                self._progress("TARGET REACHED. ALIGNING NORTH...")
                self._transition(MissionState.ALIGN_NORTH)
                return

            path = self._nav.calculate_path()
            if len(path) < 2:
                self._progress("NO PATH / ALREADY THERE")
                self._transition(MissionState.LANDING)
                return

            next_coord = path[1]
            angle, new_dir = self._nav.get_turn_angle(next_coord)

            if angle != 0:
                self._progress(f"TURN NEEDED: {angle} deg")
                self._transition(MissionState.TURNING)
                turn_dir = 1 if angle > 0 else -1
                dur = self._drone.perform_rotation(abs(angle), turn_dir)
                time.sleep(dur)
                self._nav.current_heading = new_dir
                post_delay = self._cfg.get("flight", "post_turn_delay", 2.0)
                time.sleep(post_delay)
                self._progress("PUSH OUT (BLIND)")
                self._transition(MissionState.PUSH_OUT)
                self._state_timer = time.time()
            else:
                self._progress("FORWARD (NO TURN)")
                self._transition(MissionState.PUSH_OUT)
                self._state_timer = time.time()

        # ---- PUSH_OUT ----
        elif state == MissionState.PUSH_OUT:
            blind_time = self._cfg.get("flight", "blind_fwd_time", 2.5)
            if time.time() - self._state_timer > blind_time:
                self._progress("FOLLOW LINE TO NEXT")
                self._transition(MissionState.FOLLOW_LINE)

        # ---- FOLLOW_LINE ----
        elif state == MissionState.FOLLOW_LINE:
            qr = self._bot.detected_qr_buffer
            if qr and qr != self._nav.current_node:
                self._progress(f"ARRIVED AT {qr}")
                self._nav.current_node = qr
                self._transition(MissionState.AT_NODE)
                self._state_timer = time.time()

        # ---- AT_NODE ----
        elif state == MissionState.AT_NODE:
            if self._bot.current_qr_center is not None:
                cx, cy = self._bot.current_qr_center
                err_x = cx - 160
                err_y = 120 - cy
                # Light centring P-gain
                self._bot._current_vx = max(min(err_y * 0.00075, 0.15), -0.15)
                self._bot._current_vy = max(min(err_x * 0.00075, 0.15), -0.15)
                dist = math.sqrt(err_x**2 + err_y**2)
                centred = dist < 5
            else:
                self._bot._current_vx = 0.0
                self._bot._current_vy = 0.0
                centred = False

            if centred or (time.time() - self._state_timer > 8.0):
                self._bot._current_vx = 0.0
                self._bot._current_vy = 0.0
                self._progress("NODE CENTERED. PROCEEDING...")

                if self._nav.current_node == "BUILDING":
                    if "BUILDING" not in self._visited_nodes:
                        self._progress("BUILDING FOUND. ORIENTING SOUTH...")
                        time.sleep(0.5)
                        curr = self._nav.current_heading
                        from king_phoenix.navigator import DIR_E, DIR_N, DIR_S, DIR_W

                        angle = 0
                        d = 1
                        if curr == DIR_N:
                            angle, d = 180, 1
                        elif curr == DIR_E:
                            angle, d = 90, 1
                        elif curr == DIR_W:
                            angle, d = 90, -1
                        if angle:
                            dur = self._drone.perform_rotation(angle, d)
                            time.sleep(dur)
                        self._nav.current_heading = DIR_S
                        self._progress(
                            "FACING BUILDING. STEPPING BACK TO CLEAR WALL..."
                        )
                        self._drone.send_velocity_ned(vx=-0.15)
                        time.sleep(1.0)
                        self._drone.send_velocity_ned()
                        self._progress("CLIMBING TO TARGET ALTITUDE...")
                        self._building_side_count = 0
                        self._state_timer = time.time()
                        self._transition(MissionState.BUILDING_ASCEND)
                    else:
                        self._progress("BUILDING ALREADY VISITED. SKIPPING...")
                        time.sleep(1.0)
                        self._transition(MissionState.CALCULATING)
                else:
                    time.sleep(0.5)
                    self._transition(MissionState.CALCULATING)

        # ---- BUILDING_ASCEND ----
        elif state == MissionState.BUILDING_ASCEND:
            self._bot._current_vx = 0.0
            self._bot._current_vy = 0.0
            # gate_altitude_request will be set by send loop
            if self._drone.altitude >= 1.4:
                self._progress(f"SIDE {self._building_side_count + 1}/4: SCANNING...")
                self._state_timer = time.time()
                self._transition(MissionState.BUILDING_SCAN_HOVER)

        # ---- BUILDING_SCAN_HOVER ----
        elif state == MissionState.BUILDING_SCAN_HOVER:
            elapsed = time.time() - self._state_timer
            if elapsed <= 3.0:
                self._bot._current_vx = -0.15
            else:
                self._bot._current_vx = 0.0
            self._bot._current_vy = 0.0

            wc = self._front.wall_qr_center
            if wc is not None:
                self._bot._current_vx = 0.0
                err_x = wc[0] - 160
                self._bot._current_vy = max(min(err_x * 0.002, 0.1), -0.1)
                if abs(err_x) < 20:
                    self._progress("TOP QR LOCKED. STARTING TRANSITION...")
                    self._bot._current_vy = 0.0
                    self._state_timer = time.time()
                    self._transition(MissionState.BUILDING_CREATE_GAP)
            elif elapsed > 6.0:
                self._progress("NO QR SEEN AT TOP. BLIND TRANSITION.")
                self._bot._current_vx = 0.0
                self._state_timer = time.time()
                self._transition(MissionState.BUILDING_CREATE_GAP)

        # ---- BUILDING_CREATE_GAP ----
        elif state == MissionState.BUILDING_CREATE_GAP:
            self._bot._current_vx = 0.0
            self._bot._current_vy = 0.15
            if time.time() - self._state_timer > 5:
                self._bot._current_vy = 0.0
                self._progress("CORNER CLEARED. MOVING FORWARD...")
                self._state_timer = time.time()
                self._transition(MissionState.BUILDING_ALIGN_CENTER)

        # ---- BUILDING_ALIGN_CENTER ----
        elif state == MissionState.BUILDING_ALIGN_CENTER:
            self._bot._current_vx = 0.15
            self._bot._current_vy = 0.0
            if time.time() - self._state_timer > 7.25:
                self._bot._current_vx = 0.0
                self._progress("ALIGNED. TURNING...")
                self._state_timer = time.time()
                self._transition(MissionState.BUILDING_TURN)

        # ---- BUILDING_TURN ----
        elif state == MissionState.BUILDING_TURN:
            dur = self._drone.perform_rotation(90, -1)
            time.sleep(dur)
            self._building_side_count += 1
            if self._building_side_count < 4:
                self._progress(f"SIDE {self._building_side_count + 1} READY.")
                self._state_timer = time.time()
                self._transition(MissionState.BUILDING_SCAN_HOVER)
            else:
                self._progress("ALL SIDES DONE. DESCENDING.")
                self._transition(MissionState.BUILDING_DESCEND)

        # ---- BUILDING_DESCEND ----
        elif state == MissionState.BUILDING_DESCEND:
            self._nav.target_node = "BUILDING"
            alt = self._drone.altitude
            vx_c = 0.0
            vy_c = 0.0

            tc = self._bot.target_qr_center
            if tc is not None:
                cx, cy = tc
                err_x = cx - 160
                err_y = 120 - cy
                pos_kp = 0.003 + (max(0, 1.0 - alt) * 0.005)
                vy_c = max(min(err_x * pos_kp, 0.25), -0.25)
                vx_c = max(min(err_y * pos_kp, 0.25), -0.25)
                dist_err = math.sqrt(err_x**2 + err_y**2)
                if (
                    dist_err < 60
                    and self._front.gate_altitude_request is not None
                    and self._front.gate_altitude_request > 0.4
                ):  # type: ignore[operator]
                    self._front.gate_altitude_request = (
                        self._front.gate_altitude_request - 0.015
                    )  # type: ignore[operator]
            else:
                vx_c = 0.0
                vy_c = 0.0

            self._bot._current_vx = vx_c
            self._bot._current_vy = vy_c

            if alt < 0.45:
                self._progress("REACHED NAV ALT. RECOVERING...")
                self._transition(MissionState.BUILDING_RECOVER)
                self._state_timer = time.time()

        # ---- BUILDING_RECOVER ----
        elif state == MissionState.BUILDING_RECOVER:
            self._bot._current_vx = 0.0
            self._bot._current_vy = 0.0
            self._progress("ON TARGET. TURNING 180 TO NORTH...")
            dur = self._drone.perform_rotation(180, 1)
            time.sleep(1.0)
            self._nav.current_heading = (0, 1)  # DIR_N
            self._visited_nodes.add("BUILDING")
            self._nav.current_node = "BUILDING"
            self._transition(MissionState.CALCULATING)

        # ---- ALIGN_NORTH ----
        elif state == MissionState.ALIGN_NORTH:
            from king_phoenix.navigator import DIR_E, DIR_N, DIR_S, DIR_W

            curr = self._nav.current_heading
            angle = 0
            d = 1
            if curr == DIR_N:
                self._progress("ALREADY FACING NORTH")
            elif curr == DIR_E:
                self._progress("FACING EAST → TURNING LEFT")
                angle, d = 90, -1
            elif curr == DIR_S:
                self._progress("FACING SOUTH → TURNING 180")
                angle, d = 180, 1
            elif curr == DIR_W:
                self._progress("FACING WEST → TURNING RIGHT")
                angle, d = 90, 1
            if angle:
                dur = self._drone.perform_rotation(angle, d)
                time.sleep(dur)
                self._nav.current_heading = DIR_N
                time.sleep(1.0)
            self._progress("ALIGN COMPLETE. CENTERING ON QR...")
            self._state_timer = time.time()
            self._transition(MissionState.CENTERING)

        # ---- CENTERING ----
        elif state == MissionState.CENTERING:
            tc = self._bot.target_qr_center
            if tc is not None:
                cx, cy = tc
                dist_err = math.sqrt((cx - 160) ** 2 + (cy - 120) ** 2)
                ang_err = abs(self._bot.target_qr_angle)
                alt = self._drone.altitude
                is_aligned = dist_err < 25 and ang_err < 8.0
                is_low = alt < 0.20
                if is_aligned and is_low:
                    self._progress(f"TOUCHDOWN: Alt={alt:.2f}m. CUT MOTORS!")
                    self._transition(MissionState.LANDING)
                if time.time() - self._state_timer > 20.0:
                    self._progress("CENTERING TIMEOUT. FORCING LAND.")
                    self._transition(MissionState.LANDING)
            else:
                if time.time() - self._state_timer > 4.0:
                    self._progress("LOST QR. FORCING LAND.")
                    self._transition(MissionState.LANDING)

        # ---- LANDING ----
        elif state == MissionState.LANDING:
            self._drone.land()
            time.sleep(1)
            if not self._drone.armed:
                self._start_cmd = False
                self._transition(MissionState.WAIT_USER)
                self._progress("LANDED")

        # Send velocity command for the current state
        self._send_velocity_cmd()

    # ------------------------------------------------------------------ #
    #  Velocity command dispatch
    # ------------------------------------------------------------------ #

    def _send_velocity_cmd(self) -> None:
        """Build the MAVLink velocity message based on current state."""
        state = self._state
        vx = getattr(self._bot, "_current_vx", 0.0)
        vy = getattr(self._bot, "_current_vy", 0.0)

        # Altitude target
        target_alt = self._cfg.get("flight", "takeoff_alt", 0.5)
        gate_req = self._front.gate_altitude_request
        if gate_req is not None:
            target_alt = gate_req
        vz = self._drone.compute_altitude_vz(target_alt, self._drone.altitude)

        # Building states: override vz
        if state >= 200:
            req = gate_req if gate_req is not None else 0.5
            vz = -max(min((req - self._drone.altitude) * 1.5, 0.5), -0.5)
            self._drone.send_velocity_ned(
                vx=vx,
                vy=vy,
                vz=vz,
                yaw_rate=self._drone._commanded_yaw_rate
                if abs(getattr(self._drone, "_commanded_yaw_rate", 0)) > 0.01
                else 0.0,
                hold_heading=abs(getattr(self._drone, "_commanded_yaw_rate", 0))
                <= 0.01,
            )
            return

        # Brake / hold / rotate states
        if state in (
            MissionState.CALCULATING,
            MissionState.TURNING,
            MissionState.INITIAL_SCAN,
            MissionState.ALIGN_NORTH,
        ):
            yaw_active = state in (MissionState.TURNING, MissionState.ALIGN_NORTH)
            self._drone.send_velocity_ned(vz=vz, hold_heading=not yaw_active)
            return

        # Node centring
        if state == MissionState.AT_NODE:
            self._drone.send_velocity_ned(vx=vx, vy=vy, vz=vz)
            return

        # Push out
        if state == MissionState.PUSH_OUT:
            self._drone.send_velocity_ned(
                vx=self._cfg.get("flight", "forward_speed", 0.1), vz=vz
            )
            return

        # Follow line
        if state == MissionState.FOLLOW_LINE:
            if self._bot.line_detected:
                yaw_kp = self._cfg.get("control", "yaw_kp", 0.015)
                deadband = 8.0
                ang = self._bot.line_angle_error
                if abs(ang) > deadband:
                    active_err = abs(ang) - deadband
                    yaw_dir = 1 if ang > 0 else -1
                    yaw_rate = active_err * yaw_kp * yaw_dir
                else:
                    yaw_rate = 0.0
                yaw_rate = max(min(yaw_rate, 0.5), -0.5)
                self._drone.send_velocity_ned(
                    vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate, hold_heading=False
                )
            else:
                search_vx = self._cfg.get("control", "search_fwd_vel", 0.05)
                search_vy = (
                    -self._cfg.get("control", "search_vel", 0.08)
                    if self._bot.last_known_direction < 0
                    else self._cfg.get("control", "search_vel", 0.08)
                )
                self._drone.send_velocity_ned(vx=search_vx, vy=search_vy, vz=vz)
            return

        # Centring (precision landing)
        if state == MissionState.CENTERING:
            tc = self._bot.target_qr_center
            if tc is not None:
                cx, cy = tc
                err_x = cx - 160
                err_y = 120 - cy
                alt = self._drone.altitude
                pos_kp = 0.0025 + (max(0, 0.5 - alt) * 0.005)
                yaw_kp = 0.03
                prec_vy = max(min(err_x * pos_kp, 0.2), -0.2)
                prec_vx = max(min(err_y * pos_kp, 0.2), -0.2)
                prec_yaw = max(min(self._bot.target_qr_angle * yaw_kp, 0.4), -0.4)
                dist = math.sqrt(err_x**2 + err_y**2)
                if dist < 40 and abs(self._bot.target_qr_angle) < 10:
                    vz = 0.15
                elif dist > 80:
                    pass  # use global vz
                else:
                    vz = 0.0
                self._drone.send_velocity_ned(
                    vx=prec_vx, vy=prec_vy, vz=vz, yaw_rate=prec_yaw, hold_heading=False
                )
            else:
                self._drone.send_velocity_ned(vz=vz)
            return

        # Wait user / idle
        if state in (MissionState.WAIT_USER, MissionState.INIT):
            self._drone.send_velocity_ned()
            return

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _reset_pid(self) -> None:
        self._bot._current_vx = 0.0  # type: ignore[attr-defined]
        self._bot._current_vy = 0.0  # type: ignore[attr-defined]
        self._bot.prev_error = 0.0
        self._bot.integral_error = 0.0

    def _shutdown(self) -> None:
        self._running = False
        self._bot.stop()
        self._front.stop()
        self._t265.stop()
        if self._drone.armed:
            self._drone.land()
            time.sleep(1)
        logger.info("Mission shutdown.")
