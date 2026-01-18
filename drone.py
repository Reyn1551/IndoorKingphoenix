import time
import math
import threading
from pymavlink import mavutil

class DroneController:
    def __init__(self, config_manager):
        self.cfg_mgr = config_manager
        self.conn = None
        
        # Telemetry State (Thread-Safe)
        self.armed = False
        self.flight_mode = "UNKNOWN"
        self.altitude = 0.0
        self.battery = 0.0
        self.heading = 0.0
        
        # PID Controller State
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.last_valid_direction = 0 # -1 Left, 1 Right
        
        # Thread locking
        self.lock = threading.Lock()

    def connect(self):
        """Establishes Mavlink connection."""
        conn_str = self.cfg_mgr.data["system"]["mavlink_connect"]
        print(f"[DRONE] Connecting to {conn_str}...")
        self.conn = mavutil.mavlink_connection(conn_str, autoreconnect=True, source_system=1, source_component=191)
        self.conn.wait_heartbeat()
        print("[DRONE] Heartbeat Received!")

    def update_telemetry(self):
        """
        Reads incoming Mavlink messages.
        Call this frequently (e.g., inside the main loop).
        """
        if not self.conn: return

        # Read all available messages (non-blocking)
        while True:
            msg = self.conn.recv_match(type=['HEARTBEAT', 'SYS_STATUS', 'GLOBAL_POSITION_INT', 'ATTITUDE'], blocking=False)
            if not msg: break
            
            with self.lock:
                if msg.get_type() == 'HEARTBEAT':
                    self.armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                    # Decode custom mode (ArduPilot specific)
                    if self.conn.flightmode == "CUSTOM":
                        self.flight_mode = "GUIDED" # Simplified for now
                    else:
                        self.flight_mode = self.conn.flightmode
                        
                elif msg.get_type() == 'SYS_STATUS':
                    self.battery = msg.voltage_battery / 1000.0
                    
                elif msg.get_type() == 'GLOBAL_POSITION_INT':
                    self.altitude = msg.relative_alt / 1000.0
                    
                elif msg.get_type() == 'ATTITUDE':
                    self.heading = math.degrees(msg.yaw)

    # ================= COMMANDS =================

    def set_mode(self, mode):
        if mode not in self.conn.mode_mapping():
            print(f"[DRONE] Unknown mode: {mode}")
            return
        mode_id = self.conn.mode_mapping()[mode]
        self.conn.mav.set_mode_send(
            self.conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

    def arm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

    def land(self):
        self.set_mode("LAND")

    def takeoff(self):
        target_alt = self.cfg_mgr.data["flight"]["takeoff_alt"]
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, target_alt
        )

    def send_velocity(self, vx, vy, vz, yaw_rate=0):
        """
        Sends velocity setpoints (Body Frame).
        vx: Forward, vy: Right, vz: Down (Keep 0 for alt hold)
        """
        # Mask: Ignore Position (bit 0-2), Use Velocity (bit 3-5), Use Yaw Rate (bit 9)
        type_mask = 0b0000101111000111
        
        self.conn.mav.set_position_target_local_ned_send(
            0, # Time boot ms
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0, 0, 0,        # Position (Ignored)
            vx, vy, vz,     # Velocity
            0, 0, 0,        # Accel
            0, yaw_rate     # Yaw, Yaw Rate
        )

    # ================= LOGIC =================

    def compute_pid(self, error, dt=0.1):
        """
        Calculates the lateral velocity (Vy) to correct the error.
        dt: Time delta since last calculation.
        """
        conf = self.cfg_mgr.data["control"]
        
        # 1. Update Direction Memory (for recovery if line is lost)
        if abs(error) > 20:
            self.last_valid_direction = 1 if error > 0 else -1

        # 2. PID Math
        # Proportional
        p_term = error * conf["pid_kp"]
        
        # Integral (with anti-windup clamping)
        self.integral_error += error * dt
        self.integral_error = max(min(self.integral_error, 500), -500)
        i_term = self.integral_error * conf["pid_ki"]
        
        # Derivative
        derivative = (error - self.prev_error) / dt
        d_term = derivative * conf["pid_kd"]
        
        # Output
        output = p_term + i_term + d_term
        self.prev_error = error
        
        # 3. Constrain to Max Velocity
        max_v = self.cfg_mgr.data["flight"]["max_lat_vel"]
        output = max(min(output, max_v), -max_v)
        
        return output