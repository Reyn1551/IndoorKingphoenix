#!/usr/bin/env python3

###########################################################
##                    KING PHOENIX FIRA 2026             ##
###########################################################

import sys, os, time, threading, math as m, argparse
import numpy as np
import cv2
import json
import copy
from flask import Flask, Response, render_template_string, jsonify, request
from collections import deque

# Mengaktifkan MAVLink 2.0
os.environ["MAVLINK20"] = "1"
import pyrealsense2 as rs
import transformations as tf
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
from pyzbar.pyzbar import decode

import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ==========================================
# 0. GRID MAPPING
# ==========================================
# (0,1) -- (1,1) -- (2,1)
#   |        |        |
# (0,0) -- (1,0) -- (2,0)

GRID_MAP = {
    "S,W,N,W,1": (0, 2), "ToNorth": (2, 2), "ToEast": (4, 2),
    "ToWest": (0, 0), "N,S,W,S,3": (2, 0), "BUILDING": (3,0), "ToSouth": (4, 0)
}

# DIRECTION VECTORS
DIR_N = (0, 1); DIR_E = (1, 0); DIR_S = (0, -1); DIR_W = (-1, 0)
DIR_NAMES = {DIR_N: "NORTH", DIR_E: "EAST", DIR_S: "SOUTH", DIR_W: "WEST"}

class GridNavigator:
    def __init__(self):
        self.current_node = None     # Will be set by Auto-Scan
        self.current_heading = DIR_N # Start facing North default
        self.target_node = "S,W,N,W,1"
        self.path = []

    def set_target(self, target_qr):
        if target_qr in GRID_MAP:
            self.target_node = target_qr
            print(f"NAV: Target Set to {self.target_node}")
            return True
        return False

    def calculate_path(self):
        if not self.current_node: return [] 
        start_pos = GRID_MAP.get(self.current_node)
        end_pos = GRID_MAP.get(self.target_node)
        
        if not start_pos or not end_pos: return []
        if start_pos == end_pos: return []

        queue = deque([[start_pos]])
        visited = {start_pos}

        while queue:
            path = queue.popleft()
            x, y = path[-1]

            if (x, y) == end_pos:
                return path 

            for dx, dy in [DIR_N, DIR_E, DIR_S, DIR_W]:
                next_x, next_y = x + dx, y + dy
                if 0 <= next_x <= 4 and 0 <= next_y <= 2:
                    if (next_x, next_y) not in visited:
                        visited.add((next_x, next_y))
                        new_path = list(path)
                        new_path.append((next_x, next_y))
                        queue.append(new_path)
        return []

    def get_turn_angle(self, next_pos):
        curr_pos = GRID_MAP[self.current_node]
        req_dir = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])

        print(f"DEBUG: CurrHead={self.current_heading} | ReqDir={req_dir}")
        
        if req_dir == self.current_heading: return 0, req_dir 
        
        cross = self.current_heading[0] * req_dir[1] - self.current_heading[1] * req_dir[0]
        dot = self.current_heading[0] * req_dir[0] + self.current_heading[1] * req_dir[1]
        
        if dot == -1: return 180, req_dir # U-Turn
        if cross > 0: return -90, req_dir # Turn Left (CCW)
        else: return 90, req_dir # Turn Right (CW)

navigator = GridNavigator()

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEFAULT_CONFIG = {
    "flight": {
        "takeoff_alt": 0.5,         
        "forward_speed": 0.1,       
        "max_lat_vel": 0.3,         
        "yaw_speed": 30,            
        "pre_turn_delay": 2.0,      
        "post_turn_delay": 2.0,     
        "blind_fwd_time": 2.5,  
        "alt_source": "T265"        
    },
    "control": {
        "pid_kp": 0.008,    
        "pid_ki": 0.0001,   
        "pid_kd": 0.003,    
        "search_vel": 0.08,         
        "search_fwd_vel": 0.05,
        "yaw_kp": 0.03
    },
    "vision": {
        "is_black_line": True,       
        "threshold_val": 80,        
        "roi_height_ratio": 0.30,    
        "min_area": 1000,            
        "min_aspect_ratio": 0.8,     
        "lost_timeout": 3.0,        
        "camera_offset_x": 0,
        "gates": {
            "red_lower1": [0, 148, 102],   "red_upper1": [179, 255, 255],
            "red_lower2": [170, 120, 70], "red_upper2": [180, 255, 255],
            "yellow_lower": [20, 100, 100], "yellow_upper": [30, 255, 255],
            "min_gate_area": 10000,
            "gate_hold_alt": 0.5,   # Height to fly THROUGH (Yellow)
            "gate_avoid_alt": 1.3   # Height to fly OVER (Red)
        }     
    },
    "t265": {
        "scale_factor": 1.0,        
        "confidence_threshold": 3,   
        "ignore_quality": False      
    },
    "system": {
        "http_port": 5000,
        "mavlink_connect": "udpin:0.0.0.0:14550",
        "start_heading": "NORTH"
    }
}

CONFIG_FILE = "config.json"
CONFIG = {}

def load_config():
    global CONFIG
    CONFIG = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_conf = json.load(f)
                CONFIG.update(saved_conf)
                print(f"LOADED CONFIG")
        except: pass
    
    h_str = CONFIG["system"].get("start_heading", "NORTH")
    if h_str == "NORTH": navigator.current_heading = DIR_N
    elif h_str == "SOUTH": navigator.current_heading = DIR_S
    elif h_str == "WEST": navigator.current_heading = DIR_W
    else: navigator.current_heading = DIR_E

def save_config_to_file():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        return True
    except: return False

load_config()

# ==========================================
# 2. GLOBAL STATE MACHINE
# ==========================================
STATE_INIT = 0
STATE_WAIT_USER = 1
STATE_TAKEOFF = 2
STATE_INITIAL_SCAN = 5
STATE_LANDING = 99
STATE_SCANNING = 100

STATE_FOLLOW_LINE = 10
STATE_AT_NODE = 20
STATE_CALCULATING = 30
STATE_TURNING = 40
STATE_PUSH_OUT = 50
STATE_ALIGN_NORTH = 60
STATE_CENTERING = 80

# NEW BUILDING STATES ---
STATE_BUILDING_INIT = 200
STATE_BUILDING_ASCEND = 201
STATE_BUILDING_SCAN_HOVER = 202
STATE_BUILDING_FIND_EDGE = 203    
STATE_BUILDING_CREATE_GAP = 204   
STATE_BUILDING_ALIGN_CENTER = 205 
STATE_BUILDING_TURN = 206         
STATE_BUILDING_DESCEND = 207
STATE_BUILDING_RECOVER = 208
building_side_count = 0
found_victims = []
visited_nodes = set()



mission_state = STATE_INIT
mission_start_command = False
line_detected = False
last_line_time = time.time()
fcu_altitude = 0.0
line_angle_error = 0.0
gate_altitude_request = None
last_red_seen_time = 0.0
GATE_PASS_DELAY = 15.0  # Seconds to stay high after gate disappears
target_qr_center = None # format: (x, y)
target_qr_angle = 0.0
commanded_yaw_rate = 0.0
wall_qr_center = None # (x, y) or None

prev_error = 0.0; integral_error = 0.0; last_known_direction = 0 
state_timer = 0.0
web_data = {"mode": "INIT", "armed": False, "alt": 0.0, "msg": "Booting...", "target": "NONE", "curr": "?"}

# TWO GLOBAL FRAMES NOW
global_frame = None; frame_lock = threading.Lock() # Bottom Camera
global_front_frame = None; front_frame_lock = threading.Lock() # Front Camera

pipe, data, prev_data, H_aeroRef_aeroBody = None, None, None, None
reset_counter = 1; current_time_us = 0
mavlink_thread_should_exit = False
mav_lock = threading.Lock()
active_cam_index = None
app = Flask(__name__)

def progress(s): 
    print(s, file=sys.stdout); sys.stdout.flush()
    web_data["msg"] = s

# ==========================================
# 3. HARDWARE & MAVLINK
# ==========================================

wall_qr_center = None 
def front_camera_thread():
    global gate_altitude_request, last_red_seen_time, global_front_frame, found_victims, wall_qr_center
    
    FRONT_CAM_INDEX = 4 
    cap_front = cv2.VideoCapture(FRONT_CAM_INDEX) 
    if not cap_front.isOpened(): cap_front = cv2.VideoCapture(1)
    cap_front.set(3, 320); cap_front.set(4, 240)
    
    print("FRONT CAM: Warming up...")
    for i in range(30): cap_front.read(); time.sleep(0.01)
    print("FRONT CAM THREAD STARTED")
    
    while True:
        ret, frame = cap_front.read()
        if not ret: time.sleep(0.5); continue

        # --- A. VICTIM & CENTERING ---
        wall_qr_center = None 
        largest_area = 0
        
        if mission_state >= 200:
            qr_objs = decode(frame)
            for obj in qr_objs:
                qr_data = obj.data.decode("utf-8")
                if "VICTIM" in qr_data:
                    cv2.rectangle(frame, (obj.rect.left, obj.rect.top), (obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height), (0, 255, 0), 2)
                    if qr_data not in found_victims:
                        found_victims.append(qr_data)
                        print(f"!!! FOUND VICTIM: {qr_data} !!!")
                
                area = obj.rect.width * obj.rect.height
                if area > largest_area:
                    largest_area = area
                    cx = obj.rect.left + obj.rect.width / 2
                    cy = obj.rect.top + obj.rect.height / 2
                    wall_qr_center = (cx, cy)
            
            if wall_qr_center:
                cv2.circle(frame, (int(wall_qr_center[0]), int(wall_qr_center[1])), 5, (255,0,0), -1)

        # --- B. GATE DETECTION ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_r = cv2.inRange(hsv, np.array(CONFIG["vision"]["gates"]["red_lower1"]), np.array(CONFIG["vision"]["gates"]["red_upper1"])) + \
                 cv2.inRange(hsv, np.array(CONFIG["vision"]["gates"]["red_lower2"]), np.array(CONFIG["vision"]["gates"]["red_upper2"]))
        mask_y = cv2.inRange(hsv, np.array(CONFIG["vision"]["gates"]["yellow_lower"]), np.array(CONFIG["vision"]["gates"]["yellow_upper"]))
        
        cnt_red, _ = cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_yel, _ = cv2.findContours(mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        found_red = any(cv2.contourArea(c) > CONFIG["vision"]["gates"]["min_gate_area"] for c in cnt_red)
        found_yel = any(cv2.contourArea(c) > CONFIG["vision"]["gates"]["min_gate_area"] for c in cnt_yel)

        # [CRITICAL FIX] Only update altitude if NOT in Building Mission
        if mission_state < 200:
            if found_red:
                gate_altitude_request = CONFIG["vision"]["gates"]["gate_avoid_alt"]
                last_red_seen_time = time.time()
                cv2.putText(frame, "RED GATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            elif found_yel:
                gate_altitude_request = CONFIG["vision"]["gates"]["gate_hold_alt"]
                last_red_seen_time = 0
                cv2.putText(frame, "YELLOW GATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            else:
                if (time.time() - last_red_seen_time) < GATE_PASS_DELAY: 
                    gate_altitude_request = CONFIG["vision"]["gates"]["gate_avoid_alt"]
                else:
                    gate_altitude_request = None

        with front_frame_lock: global_front_frame = frame.copy()
        time.sleep(0.05)

def mavlink_loop(conn, callbacks):
    interesting = list(callbacks.keys())
    while not mavlink_thread_should_exit:
        conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0)
        msg = conn.recv_match(type=interesting, timeout=1, blocking=True)
        if msg: callbacks[msg.get_type()](msg)

def heartbeat_cb(msg):
    mode_map = conn.mode_mapping()
    for name, id in mode_map.items():
        if msg.custom_mode == id: web_data["mode"] = name; break
    web_data["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

def statustext_cb(msg): web_data["msg"] = msg.text.upper()
def sys_status_cb(msg): pass
def global_pos_cb(msg): global fcu_altitude; fcu_altitude = msg.relative_alt / 1000.0

def send_vision_msg():
    global current_time_us, H_aeroRef_aeroBody, reset_counter, data, fcu_altitude
    with mav_lock:
        if H_aeroRef_aeroBody is not None and data is not None:
            rpy = np.array(tf.euler_from_matrix(H_aeroRef_aeroBody, 'sxyz'))
            cov = 0.01 * pow(10, 3 - int(data.tracker_confidence))
            conn.mav.vision_position_estimate_send(current_time_us, 
                H_aeroRef_aeroBody[0][3], H_aeroRef_aeroBody[1][3], H_aeroRef_aeroBody[2][3],
                rpy[0], rpy[1], rpy[2], [cov]*21, reset_counter)
            
            if CONFIG["flight"].get("alt_source", "T265") == "FCU": web_data["alt"] = fcu_altitude
            else: web_data["alt"] = -H_aeroRef_aeroBody[2][3]

def set_mode(mode):
    conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()[mode])

def perform_rotation(angle, direction):
    global commanded_yaw_rate
    
    # Direction: 1 = CW (Positive), -1 = CCW (Negative)
    # BUT in NED Frame: Positive Yaw is Clockwise. 
    # Let's stick to standard math: Right (CW) is positive speed.
    
    mav_dir = 1 if direction == 1 else -1 
    yaw_spd_deg = CONFIG["flight"].get("yaw_speed", 30)
    
    # Convert to Radians per Second for SET_POSITION_TARGET
    yaw_rate_rad = m.radians(yaw_spd_deg) * mav_dir
    
    progress(f"CMD: YAW {angle} deg (Rate: {yaw_spd_deg}/s)")
    
    # Calculate duration
    duration = abs(angle) / yaw_spd_deg
    
    # 1. Start Turning
    commanded_yaw_rate = yaw_rate_rad
    
    # 2. Wait for the turn to complete
    time.sleep(duration)
    
    # 3. Stop Turning
    commanded_yaw_rate = 0.0
    
    # Return delay for settling
    return 1.5

# ==========================================
# 4. VISION LOGIC (BOTTOM CAMERA)
# ==========================================
current_vx = 0.0; current_vy = 0.0
detected_qr_buffer = None

def vision_thread_func():
    global current_vx, current_vy, line_detected, last_line_time, prev_error, integral_error, last_known_direction, global_frame, detected_qr_buffer, line_angle_error, target_qr_center, target_qr_angle 
    BOTTOM_CAM_INDEX = 6 
    cap = cv2.VideoCapture(BOTTOM_CAM_INDEX)
    cap.set(3, 320); cap.set(4, 240)
    
    print("VISION THREAD STARTED")

    while True:
        success, frame = cap.read()
        if not success: time.sleep(0.5); cap.release(); cap = cv2.VideoCapture(BOTTOM_CAM_INDEX); continue
        
        h, w, _ = frame.shape
        cut_y = int(h * CONFIG["vision"]["roi_height_ratio"])
        roi = frame[cut_y:h, 0:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        t_type = cv2.THRESH_BINARY_INV if CONFIG["vision"]["is_black_line"] else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blur, CONFIG["vision"]["threshold_val"], 255, t_type)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None; max_score = -999999
        offset_val = int(CONFIG["vision"].get("camera_offset_x", 0)); cx_scr = int(w/2) + offset_val 

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < CONFIG["vision"]["min_area"]: continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if (float(bh)/bw) < CONFIG["vision"]["min_aspect_ratio"]: continue
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx_cnt = int(M['m10'] / M['m00']); dist = abs(cx_cnt - cx_scr)
            score = area - (dist * 10)
            if score > max_score: max_score = score; best_cnt = cnt
        
        temp_line_detected = False
        if best_cnt is not None:
            temp_line_detected = True; last_line_time = time.time()
            M = cv2.moments(best_cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00']); real_cy = cut_y + int(M['m01'] / M['m00'])
                cv2.drawContours(roi, [best_cnt], -1, (0,255,255), 2)
                cv2.line(frame, (cx_scr, real_cy), (cx, real_cy), (0,0,255), 2)
                
                [vx, vy, x, y] = cv2.fitLine(best_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                try:
                    angle_rad = m.atan2(vy, vx)
                    angle_deg = m.degrees(angle_rad)
                    if angle_deg < 0: angle_deg += 180
                    line_angle_error = angle_deg - 90 
                except: line_angle_error = 0.0

                error_x = cx - cx_scr
                if abs(error_x) > 10: last_known_direction = 1 if error_x > 0 else -1
                
            if mission_state == STATE_FOLLOW_LINE:
                kp = CONFIG["control"]["pid_kp"]; ki = CONFIG["control"]["pid_ki"]; kd = CONFIG["control"]["pid_kd"]
                integral_error += error_x; integral_error = max(min(integral_error, 500), -500) 
                derivative = error_x - prev_error
                raw_vy = (error_x * kp) + (integral_error * ki) + (derivative * kd)
                prev_error = error_x; max_lat = CONFIG["flight"]["max_lat_vel"]
                current_vy = max(min(raw_vy, max_lat), -max_lat)
                current_vx = CONFIG["flight"]["forward_speed"]
        else:
            temp_line_detected = False
            # [CHANGE IS HERE] -> Wrap zeroing in this IF
            if mission_state == STATE_FOLLOW_LINE:
                current_vx = 0.0; current_vy = 0.0
                integral_error = 0.0; prev_error = 0.0
                
            line_angle_error = 0.0
                
        line_detected = temp_line_detected

        # QR Logic
        qr_objects = decode(frame)
        target_qr_center = None; target_qr_angle = 0.0 

        for obj in qr_objects:
            qr_text = obj.data.decode("utf-8")
            pts = np.array([obj.polygon], np.int32)
            cv2.polylines(frame, [pts], True, (255,0,255), 2)
            
            if qr_text in GRID_MAP: detected_qr_buffer = qr_text
            
            if qr_text == navigator.target_node:
                r = obj.rect
                cx = r.left + (r.width / 2); cy = r.top + (r.height / 2)
                target_qr_center = (cx, cy)
                
                poly_pts = obj.polygon
                sorted_by_y = sorted(poly_pts, key=lambda p: p.y)
                if len(sorted_by_y) >= 2: # Safety check
                    top_1 = sorted_by_y[0]; top_2 = sorted_by_y[1]
                    if top_1.x < top_2.x: tl, tr = top_1, top_2
                    else: tl, tr = top_2, top_1
                    
                    delta_x = tr.x - tl.x; delta_y = tr.y - tl.y
                    if delta_x != 0: target_qr_angle = m.degrees(m.atan2(delta_y, delta_x))
                    
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                    cv2.line(frame, (tl.x, tl.y), (tr.x, tr.y), (0, 255, 0), 2) 

        cv2.line(frame, (cx_scr, 0), (cx_scr, h), (0, 255, 0), 2) 
        if line_detected:
            vis_len = 100; vis_rad = m.radians(line_angle_error)
            tip_x = int(cx_scr + vis_len * m.sin(vis_rad))
            tip_y = int((h/2) - vis_len * m.cos(vis_rad))
            cv2.line(frame, (cx_scr, int(h/2)), (tip_x, tip_y), (255, 0, 0), 2)

        with frame_lock: global_frame = frame
        time.sleep(0.01)

# ==========================================
# 5. CONTROL & MISSION LOGIC
# ==========================================
def reset_pid():
    global prev_error, integral_error, current_vx, current_vy
    prev_error = 0.0; integral_error = 0.0; current_vx = 0.0; current_vy = 0.0

def send_vel_cmd():
    global mission_state, current_vx, current_vy, last_known_direction, state_timer
    global fcu_altitude, gate_altitude_request
    
    # --- 1. CALCULATE ALTITUDE CORRECTION FIRST ---
    # We do this every loop so EVERY state holds altitude
    req_alt = CONFIG["flight"]["takeoff_alt"] 
    if gate_altitude_request is not None:
        req_alt = gate_altitude_request
        
    alt_error = req_alt - web_data["alt"] 
    # P-Controller for Altitude
    vz = max(min(alt_error * 1.5, 0.5), -0.5)
    mav_vz = -vz # NED convention (Negative is Up)

# BUILDING ALTITUDE OVERRIDE
# BUILDING ALTITUDE OVERRIDE (FIXED FOR YAW)
    if mission_state >= 200:
        req_alt = gate_altitude_request if gate_altitude_request else 0.5
        vz = -max(min((req_alt - web_data["alt"]) * 1.5, 0.5), -0.5)
        
        # [FIX START] Check if we are commanding a turn
        if abs(commanded_yaw_rate) > 0.01:
             mask = 0b011111000111 # Enable Yaw Rate Control (Bit 10 = 0)
             y_rate = commanded_yaw_rate
        else:
             mask = 0b0000111111000111 # Hold Heading (Bit 10 = 1)
             y_rate = 0.0
        # [FIX END]

        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, 
            mavutil.mavlink.MAV_FRAME_BODY_NED, mask, 0,0,0, current_vx, current_vy, vz, 0,0,0, 0, y_rate)
        return

# --- 2. BRAKE & HOLD & ROTATE STATES ---
    # (Includes TURNING, SCANNING, ALIGNING)
    if mission_state in [STATE_AT_NODE, STATE_CALCULATING, STATE_TURNING, STATE_INITIAL_SCAN, STATE_ALIGN_NORTH]:
        
        # FIX: Check if we have an active yaw command OR if we are in a turning state
        is_turning = (mission_state in [STATE_TURNING, STATE_ALIGN_NORTH]) or (abs(commanded_yaw_rate) > 0.01)
        
        if is_turning:
             # Bit 10 is 0 (Enable Yaw Rate Control)
             mask = 0b011111000111 
             y_rate = commanded_yaw_rate
        else:
             # Bit 10 is 1 (Ignore Yaw Rate -> Hold Heading)
             mask = 0b0000111111000111
             y_rate = 0.0

        conn.mav.set_position_target_local_ned_send(
            0, conn.target_system, conn.target_component, 
            mavutil.mavlink.MAV_FRAME_BODY_NED, 
            mask, 
            0,0,0, 
            0, 0, mav_vz, 
            0,0,0, 
            0, y_rate) 
        return

    # --- 3. PUSH OUT ---
    if mission_state == STATE_PUSH_OUT:
        conn.mav.set_position_target_local_ned_send(
            0, conn.target_system, conn.target_component, 
            mavutil.mavlink.MAV_FRAME_BODY_NED, 
            0b0000111111000111, 
            0,0,0, 
            CONFIG["flight"]["forward_speed"], 0, mav_vz, 
            0,0,0, 0,0)
        return

    # --- 4. FOLLOW LINE ---
    if mission_state == STATE_FOLLOW_LINE:
        if line_detected:
            yaw_kp = CONFIG["control"].get("yaw_kp", 0.025)
            if abs(line_angle_error) > 10.0:
                yaw_rate = line_angle_error * yaw_kp
            else:
                yaw_rate = 0.0 
            
            yaw_rate = max(min(yaw_rate, 0.5), -0.5)
            mask = 0b011111000111 
            print(f"CMD: Vx={current_vx:.2f} Vy={current_vy:.2f} YawRate={yaw_rate:.2f}")
            conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, 
                mavutil.mavlink.MAV_FRAME_BODY_NED, 
                mask, 
                0,0,0, 
                current_vx, current_vy, mav_vz, 
                0,0,0, 
                0, yaw_rate)
        else:
            # Simple Searching
            search_vx = CONFIG["control"]["search_fwd_vel"]
            search_vy = CONFIG["control"]["search_vel"] if last_known_direction < 0 else -CONFIG["control"]["search_vel"]
            conn.mav.set_position_target_local_ned_send(
                0, conn.target_system, conn.target_component, 
                mavutil.mavlink.MAV_FRAME_BODY_NED, 
                0b0000111111000111, 
                0,0,0, 
                search_vx, search_vy, mav_vz,
                0,0,0, 
                0,0)

    # --- 5. CENTERING (PRECISION LANDING) ---
    elif mission_state == STATE_CENTERING:
        if target_qr_center is not None:
            cam_w, cam_h = 320, 240
            center_x, center_y = target_qr_center
            
            err_x_px = center_x - (cam_w / 2) 
            err_y_px = (cam_h / 2) - center_y 
            err_yaw_deg = target_qr_angle 

            # Calculate total distance error (pixels)
            dist_error = m.sqrt(err_x_px**2 + err_y_px**2)
            
            # --- PID GAINS ---
            # As we get lower, we need more aggressive correction because 
            # 1 pixel of error means less distance in meters.
            current_alt = web_data["alt"]
            
            # Adaptive P-Gain: Increase gain as altitude decreases
            # Base 0.0025, max 0.005
            pos_kp = 0.0025 + (max(0, 0.5 - current_alt) * 0.005)
            yaw_kp = 0.03 
            
            # Calculate Horizontal Velocities
            prec_vy = max(min(err_x_px * pos_kp, 0.2), -0.2)
            prec_vx = max(min(err_y_px * pos_kp, 0.2), -0.2)
            prec_yaw = max(min(err_yaw_deg * yaw_kp, 0.4), -0.4) 
            
            # --- DESCENT LOGIC (The "Cone") ---
            # default mav_vz calculated at top of function tries to HOLD 0.5m.
            # We override it here based on alignment.
            
            target_descend_vel = 0.0
            
            if dist_error < 40 and abs(err_yaw_deg) < 10:
                # If we are roughly centered, descend slowly
                # Positive Z velocity = DOWN in NED
                target_descend_vel = 0.15 # Descend at 15 cm/s
            elif dist_error > 80:
                # If we drift too far, STOP descending (or climb back if too low)
                # Use the global mav_vz which pulls us back to takeoff_alt
                target_descend_vel = mav_vz 
            else:
                # In between: Hold current altitude (Velocity 0)
                target_descend_vel = 0.0

            conn.mav.set_position_target_local_ned_send(
                0, conn.target_system, conn.target_component, 
                mavutil.mavlink.MAV_FRAME_BODY_NED, 
                0b011111000111, 
                0,0,0, 
                prec_vx, prec_vy, target_descend_vel, # Send dynamic Z vel
                0,0,0, 
                0, prec_yaw) 
                
        else:
            # LOST QR CODE (Missed Approach)
            # If we lose the QR code, the global 'mav_vz' (calculated at top)
            # will automatically try to climb back to 'takeoff_alt' (0.5m).
            # This is a safety feature!
            conn.mav.set_position_target_local_ned_send(
                0, conn.target_system, conn.target_component, 
                mavutil.mavlink.MAV_FRAME_BODY_NED, 
                0b0000111111000111, 
                0,0,0, 
                0, 0, mav_vz, # Climb back to safety height
                0,0,0, 0,0)

    # --- 6. HOLD (WAIT USER) ---
    elif mission_state in [STATE_WAIT_USER]:
        # During Wait User (Disarmed), we send 0 velocity to keep it satisfied
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0, 0, 0, 0,0,0, 0,0)
def mission_logic_thread():
    global mission_state, detected_qr_buffer, mission_start_command, state_timer, current_vx, current_vy, building_side_count, wall_qr_center, gate_altitude_request, visited_nodes
    
    while True:
        time.sleep(0.1)
        
        if mission_state == STATE_INIT:
            conf_ok = (data and data.tracker_confidence >= CONFIG["t265"]["confidence_threshold"])
            if conf_ok or CONFIG["t265"]["ignore_quality"]: 
                progress("READY: SET TARGET"); mission_state = STATE_WAIT_USER

        elif mission_state == STATE_WAIT_USER:
            web_data["curr"] = navigator.current_node if navigator.current_node else "?"
            web_data["target"] = navigator.target_node
            if mission_start_command: 
                progress("STARTING..."); time.sleep(1); mission_state = STATE_TAKEOFF

        elif mission_state == STATE_TAKEOFF:
            set_mode("GUIDED")
            conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            while not web_data["armed"]: time.sleep(0.5)
            conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, CONFIG["flight"]["takeoff_alt"])
            while web_data["alt"] < (CONFIG["flight"]["takeoff_alt"] - 0.2): time.sleep(0.5)
            progress("HOVER: SCANNING FOR START QR...")
            detected_qr_buffer = None
            mission_state = STATE_INITIAL_SCAN

        elif mission_state == STATE_INITIAL_SCAN:
            if detected_qr_buffer and detected_qr_buffer in GRID_MAP:
                navigator.current_node = detected_qr_buffer
                navigator.current_heading = DIR_N 
                progress(f"LOCATED: {navigator.current_node} (RESET N)")
                time.sleep(1.0)
                mission_state = STATE_CALCULATING

        elif mission_state == STATE_CALCULATING:
            reset_pid()
            detected_qr_buffer = None 

            if navigator.current_node == navigator.target_node:
                progress("TARGET REACHED. ALIGNING NORTH..."); 
                mission_state = STATE_ALIGN_NORTH; 
                continue

            path_coords = navigator.calculate_path()
            if len(path_coords) < 2:
                progress("NO PATH / ALREADY THERE"); mission_state = STATE_LANDING; continue
            
            next_coord = path_coords[1]
            angle, new_dir = navigator.get_turn_angle(next_coord)
            
            if angle != 0:
                progress(f"TURN NEEDED: {angle} deg")
                mission_state = STATE_TURNING
                turn_dir = 1 if angle > 0 else -1
                dur = perform_rotation(abs(angle), turn_dir)
                time.sleep(dur)
                navigator.current_heading = new_dir 
                time.sleep(CONFIG["flight"]["post_turn_delay"])
                progress("PUSH OUT (BLIND)")
                mission_state = STATE_PUSH_OUT
                state_timer = time.time()
            else:
                progress("FORWARD (NO TURN)")
                mission_state = STATE_PUSH_OUT
                state_timer = time.time()

        elif mission_state == STATE_AT_NODE:
            if navigator.current_node == "BUILDING":
                # [FIXED LOGIC] Only scan if NOT visited
                if "BUILDING" not in visited_nodes:
                    progress("BUILDING FOUND. ORIENTING SOUTH...")
                    time.sleep(0.5)
                    
                    curr = navigator.current_heading
                    turn_angle = 0; turn_dir = 1
                    if curr == DIR_N: turn_angle = 180; turn_dir = 1
                    elif curr == DIR_E: turn_angle = 90; turn_dir = 1 
                    elif curr == DIR_W: turn_angle = 90; turn_dir = -1
                    
                    if turn_angle > 0:
                        dur = perform_rotation(turn_angle, turn_dir)
                        time.sleep(dur)
                    navigator.current_heading = DIR_S
                    
                    progress("FACING BUILDING. CLIMBING NOW...")
                    building_side_count = 0
                    state_timer = time.time()
                    mission_state = STATE_BUILDING_ASCEND
                else:
                    progress("BUILDING ALREADY VISITED. SKIPPING...")
                    time.sleep(1.0)
                    mission_state = STATE_CALCULATING

        elif mission_state == STATE_BUILDING_ASCEND:
            # [CRITICAL FIX] Force stop moving before climbing
            current_vx = 0.0
            current_vy = 0.0
            
            gate_altitude_request = 1.4
            
            if web_data["alt"] >= 1.3:
                progress(f"SIDE {building_side_count+1}/4: SCANNING...")
                state_timer = time.time()
                mission_state = STATE_BUILDING_SCAN_HOVER

        elif mission_state == STATE_BUILDING_SCAN_HOVER:
            current_vx = 0.0; current_vy = 0.0
            
            # Try to center on Wall QR using P-Controller
            if wall_qr_center is not None:
                err_x = wall_qr_center[0] - 160 # 320/2
                # If QR is to the Right (err > 0), strafe Right (+vy)
                current_vy = max(min(err_x * 0.002, 0.1), -0.1)
                
                if abs(err_x) < 20: # Locked
                    progress("QR LOCKED. STARTING TRANSITION...")
                    current_vy = 0.0
                    state_timer = time.time()
                    mission_state = STATE_BUILDING_CREATE_GAP
            
            # If we waited 4 seconds and found nothing, move anyway
            if time.time() - state_timer > 4.0:
                progress("NO QR LOCK. BLIND TRANSITION.")
                state_timer = time.time()
                mission_state = STATE_BUILDING_CREATE_GAP

        elif mission_state == STATE_BUILDING_CREATE_GAP:
            # 1. STRAFE RIGHT to clear corner (Timer)
            current_vx = 0.0; current_vy = 0.15
            if time.time() - state_timer > 5.5: # Strafe 7s (~1.0m)
                current_vy = 0.0
                progress("CORNER CLEARED. MOVING FORWARD...")
                state_timer = time.time()
                mission_state = STATE_BUILDING_ALIGN_CENTER

        elif mission_state == STATE_BUILDING_ALIGN_CENTER:
            # 2. MOVE FORWARD to next side (Timer)
            current_vx = 0.15; current_vy = 0.0
            if time.time() - state_timer > 7.25: # Forward 8s (~1.2m)
                current_vx = 0.0
                progress("ALIGNED. TURNING...")
                state_timer = time.time()
                mission_state = STATE_BUILDING_TURN

        elif mission_state == STATE_BUILDING_TURN:
            # 3. TURN LEFT
            dur = perform_rotation(90, -1)
            time.sleep(dur)
            building_side_count += 1
            
            if building_side_count < 4:
                progress(f"SIDE {building_side_count+1} READY.")
                state_timer = time.time()
                mission_state = STATE_BUILDING_SCAN_HOVER
            else:
                progress("ALL SIDES DONE. DESCENDING.")
                mission_state = STATE_BUILDING_DESCEND

        elif mission_state == STATE_BUILDING_DESCEND:
            navigator.target_node = "BUILDING"
            current_alt = web_data["alt"]
            
            # P-Controller (Bottom Camera)
            vx_corr = 0.0; vy_corr = 0.0
            
            # Use target_qr_center from global scope
            if target_qr_center is not None:
                cx, cy = target_qr_center
                err_x = cx - 160 
                err_y = 120 - cy
                
                vy_corr = max(min(err_x * 0.0025, 0.15), -0.15) 
                vx_corr = max(min(err_y * 0.0025, 0.15), -0.15) 
                
                # Two-Stage Descent
                if current_alt > 0.7:
                     gate_altitude_request = 0.7 # Stage 1: Drop to 0.6
                else:
                     # Stage 2: 0.6 -> 0.4
                     if abs(err_x) < 40 and abs(err_y) < 40:
                         gate_altitude_request = max(current_alt - 0.05, 0.4) 
                     else:
                         gate_altitude_request = current_alt # Pause to center
            else:
                gate_altitude_request = 0.7 # Blind drop to 0.6
                vx_corr = 0.0; vy_corr = 0.0

            current_vx = vx_corr
            current_vy = vy_corr
            
            # Exit Condition
            if current_alt < 0.45:
                 progress("REACHED NAV ALT. RECOVERING...")
                 mission_state = STATE_BUILDING_RECOVER
                 state_timer = time.time()

        elif mission_state == STATE_BUILDING_RECOVER:
            current_vx = -0.15 
            if time.time() - state_timer > 2.0:
                current_vx = 0.0
                dur = perform_rotation(180, 1) # Turn back to North
                time.sleep(1.0)
                navigator.current_heading = DIR_N
                visited_nodes.add("BUILDING")
                navigator.current_node = "BUILDING"
                mission_state = STATE_CALCULATING

        elif mission_state == STATE_ALIGN_NORTH:
            current = navigator.current_heading
            req_angle = 0; turn_dir = 1 

            if current == DIR_N: progress("ALREADY FACING NORTH"); req_angle = 0
            elif current == DIR_E: progress("FACING EAST -> TURNING LEFT"); req_angle = 90; turn_dir = -1
            elif current == DIR_S: progress("FACING SOUTH -> TURNING 180"); req_angle = 180; turn_dir = 1
            elif current == DIR_W: progress("FACING WEST -> TURNING RIGHT"); req_angle = 90; turn_dir = 1
            
            if req_angle > 0:
                dur = perform_rotation(req_angle, turn_dir)
                time.sleep(dur)
                navigator.current_heading = DIR_N 
                time.sleep(1.0) 

            progress("ALIGN COMPLETE. CENTERING ON QR...")
            state_timer = time.time()
            mission_state = STATE_CENTERING

        elif mission_state == STATE_CENTERING:
            if target_qr_center is not None:
                cam_w, cam_h = 320, 240
                cx, cy = target_qr_center
                
                # Check Position Error
                dist_err = m.sqrt((cx - cam_w/2)**2 + (cy - cam_h/2)**2)
                ang_err = abs(target_qr_angle)
                current_alt = web_data["alt"]
                
                # 
                # Logic: ALIGNED + LOW ALTITUDE = SUCCESS
                
                # 1. Check if perfectly aligned
                is_aligned = (dist_err < 25 and ang_err < 8.0)
                
                # 2. Check if close to ground (e.g., 15cm)
                is_low = (current_alt < 0.20)
                
                if is_aligned and is_low:
                    progress(f"TOUCHDOWN: Alt={current_alt:.2f}m. CUT MOTORS!")
                    mission_state = STATE_LANDING
                
                # Timeout logic: If it takes too long (>20s), just land to prevent battery drain
                if time.time() - state_timer > 20.0:
                    progress("CENTERING TIMEOUT. FORCING LAND.")
                    mission_state = STATE_LANDING
            else:
                # If lost for > 4 seconds, abort
                if time.time() - state_timer > 4.0:
                     progress("LOST QR. FORCING LAND.")
                     mission_state = STATE_LANDING

        elif mission_state == STATE_PUSH_OUT:
            if time.time() - state_timer > CONFIG["flight"]["blind_fwd_time"]:
                progress("FOLLOW LINE TO NEXT")
                mission_state = STATE_FOLLOW_LINE

        elif mission_state == STATE_FOLLOW_LINE:
            if detected_qr_buffer and detected_qr_buffer in GRID_MAP:
                if detected_qr_buffer != navigator.current_node:
                    progress(f"ARRIVED AT {detected_qr_buffer}")
                    navigator.current_node = detected_qr_buffer
                    mission_state = STATE_AT_NODE
                    state_timer = time.time()

        elif mission_state == STATE_AT_NODE:
            if time.time() - state_timer > 2.0:
                mission_state = STATE_CALCULATING

        elif mission_state == STATE_LANDING:
            set_mode("LAND"); time.sleep(1)
            if not web_data["armed"]: mission_start_command = False; mission_state = STATE_WAIT_USER; progress("LANDED")

# ==========================================
# 6. FLASK & API
# ==========================================
def get_display_frame():
    while True:
        with frame_lock:
            if global_frame is None:
                img = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(img, "NO BOTTOM CAM", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                encoded = cv2.imencode('.jpg', img)[1].tobytes()
            else:
                encoded = cv2.imencode('.jpg', global_frame)[1].tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')
        time.sleep(0.1)

# NEW: Generator for Front Camera
def get_front_display_frame():
    while True:
        with front_frame_lock:
            if global_front_frame is None:
                img = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(img, "NO FRONT CAM", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                encoded = cv2.imencode('.jpg', img)[1].tobytes()
            else:
                encoded = cv2.imencode('.jpg', global_front_frame)[1].tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')
        time.sleep(0.1)

@app.route('/status')
def status(): return jsonify(web_data)

@app.route('/set_target', methods=['POST'])
def set_target_api():
    t = request.json.get("target")
    if navigator.set_target(t): return jsonify({"status": "ok", "msg": f"Target: {t}"})
    return jsonify({"status": "error", "msg": "Invalid QR Code"})

@app.route('/start_mission')
def start_cmd():
    global mission_start_command
    mission_start_command = True; return jsonify({"status": "ok"})

@app.route('/stop_mission')
def stop_cmd(): global mission_state; mission_state = STATE_LANDING; return jsonify({"status": "ok"})

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html><html><head><title>KING PHOENIX V41</title>
    <style>body{background:#111;color:#0f0;font-family:monospace;padding:20px}
    .btn{padding:10px 20px;background:#333;color:#fff;border:1px solid #0f0;cursor:pointer;margin:5px}
    .grid-btn{width:80px;height:60px;margin:5px}
    .cam-container{display:flex; flex-wrap:wrap;}
    .cam-box{margin-right:10px; border:2px solid #0f0;}
    </style></head><body>
    <h1>KING PHOENIX V41: OBSTACLE EDITION</h1>
    
    <div class="cam-container">
        <div class="cam-box">
            <div style="background:#000;color:#fff;padding:2px">BOTTOM CAM (LINE/QR)</div>
            <img src="/video_feed" width="320">
        </div>
        <div class="cam-box">
            <div style="background:#000;color:#fff;padding:2px">FRONT CAM (GATES)</div>
            <img src="/video_feed_front" width="320">
        </div>
    </div>

    <h3>STATUS: <span id="st">INIT</span> | MSG: <span id="msg">-</span></h3>
    <h3>CURRENT: <span id="cur">-</span> | TARGET: <span id="tgt">-</span></h3>
    <h3>ALT: <span id="alt">0.0</span>m</h3>
    
    <div>
        <button class="btn" onclick="fetch('/start_mission')">START</button>
        <button class="btn" style="border-color:red" onclick="fetch('/stop_mission')">LAND</button>
    </div>
    <br>
    <div>
        <b>SELECT TARGET:</b><br>
        <button class="btn grid-btn" onclick="st('S,W,N,W,1')">Kiri Atas</button>
        <button class="btn grid-btn" onclick="st('ToNorth')">Tengah Atas</button>
        <button class="btn grid-btn" onclick="st('ToEast')">Kanan Atas</button><br>
        <button class="btn grid-btn" onclick="st('ToWest')">Kiri Bawah</button>
        <button class="btn grid-btn" onclick="st('N,S,W,S,3')">Tengah Bawah</button>
        <button class="btn grid-btn" onclick="st('ToSouth')">Kanan Bawah</button>
    </div>

    <script>
    function st(t){
        fetch('/set_target', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({target:t})})
    }
    setInterval(()=>{
        fetch('/status').then(r=>r.json()).then(d=>{
            document.getElementById('st').innerText=d.mode;
            document.getElementById('msg').innerText=d.msg;
            document.getElementById('cur').innerText=d.curr;
            document.getElementById('tgt').innerText=d.target;
            document.getElementById('alt').innerText=d.alt.toFixed(2);
        });
    }, 500);
    </script>
    </body></html>
    ''')

@app.route('/video_feed')
def video_feed(): return Response(get_display_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ROUTE FOR FRONT CAMERA
@app.route('/video_feed_front')
def video_feed_front(): return Response(get_front_display_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask(): app.run(host='0.0.0.0', port=CONFIG["system"]["http_port"], threaded=True)

# ==========================================
# 7. MAIN STARTUP
# ==========================================
parser = argparse.ArgumentParser()
default_conn = CONFIG["system"]["mavlink_connect"]
parser.add_argument('--connect', default=default_conn)
args = parser.parse_args()

progress("BOOTING KING PHOENIX FIRA 2026...")
conn = mavutil.mavlink_connection(args.connect, autoreconnect=True, source_system=1, source_component=191)
callbacks = {'HEARTBEAT': heartbeat_cb, 'STATUSTEXT': statustext_cb, 'GLOBAL_POSITION_INT': global_pos_cb}
threading.Thread(target=mavlink_loop, args=(conn, callbacks)).start()

try: pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.pose); pipe.start(cfg)
except: progress("T265 FAIL"); sys.exit(1)

threading.Thread(target=run_flask, daemon=True).start()
sched = BackgroundScheduler()
sched.add_job(send_vision_msg, 'interval', seconds=1/30.0)
sched.add_job(send_vel_cmd, 'interval', seconds=1/10.0)
sched.start()

threading.Thread(target=vision_thread_func, daemon=True).start()
threading.Thread(target=front_camera_thread, daemon=True).start()
threading.Thread(target=mission_logic_thread, daemon=True).start()


progress("SYSTEM READY. CALIBRATE T265...")
H_aeroRef_T265Ref = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
H_T265body_aeroBody = np.linalg.inv(H_aeroRef_T265Ref)

try:
    while True:
        frames = pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        if pose:
            with mav_lock:
                current_time_us = int(round(time.time() * 1000000))
                data = pose.get_pose_data()
                H_265 = tf.quaternion_matrix([data.rotation.w, data.rotation.x, data.rotation.y, data.rotation.z])
                H_265[0][3] = data.translation.x * CONFIG["t265"]["scale_factor"]
                H_265[1][3] = data.translation.y * CONFIG["t265"]["scale_factor"]
                H_265[2][3] = data.translation.z * CONFIG["t265"]["scale_factor"]
                H_aeroRef_aeroBody = H_aeroRef_T265Ref.dot(H_265.dot(H_T265body_aeroBody))
                if prev_data and m.sqrt((data.translation.x-prev_data.translation.x)**2) > 0.1: reset_counter+=1
                prev_data = data
except KeyboardInterrupt: pipe.stop(); mavlink_thread_should_exit=True