#!/usr/bin/env python3

###########################################################
##   KING PHOENIX V41: AUTO-LOCATOR GRID NAV             ##
##   (Features: Auto-Detect Start Node after Takeoff)    ##
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

# ==========================================
# 0. GRID MAPPING
# ==========================================
# (0,1) -- (1,1) -- (2,1)
#   |        |        |
# (0,0) -- (1,0) -- (2,0)

GRID_MAP = {
    "S,W,N,W,1": (0, 1), "ToNorth": (1, 1), "ToEast": (2, 1),
    "ToWest": (0, 0), "Landing": (1, 0), "ToSouth": (2, 0)
}

# DIRECTION VECTORS
DIR_N = (0, 1); DIR_E = (1, 0); DIR_S = (0, -1); DIR_W = (-1, 0)
DIR_NAMES = {DIR_N: "NORTH", DIR_E: "EAST", DIR_S: "SOUTH", DIR_W: "WEST"}

class GridNavigator:
    def __init__(self):
        self.current_node = None     # Will be set by Auto-Scan
        self.current_heading = DIR_N # Start facing North default (Configurable)
        self.target_node = "S,W,N,W,1"
        self.path = []

    def set_target(self, target_qr):
        if target_qr in GRID_MAP:
            self.target_node = target_qr
            print(f"NAV: Target Set to {self.target_node}")
            return True
        return False

    def calculate_path(self):
        if not self.current_node: return [] # Unknown start
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
                if 0 <= next_x <= 2 and 0 <= next_y <= 1:
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
        "takeoff_alt": 1.0,         
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
        "search_fwd_vel": 0.05      
    },
    "vision": {
        "is_black_line": True,       
        "threshold_val": 80,         
        "roi_height_ratio": 0.30,    
        "min_area": 1000,            
        "min_aspect_ratio": 0.8,     
        "lost_timeout": 3.0,         
        "camera_offset_x": 0         
    },
    "t265": {
        "scale_factor": 1.0,         
        "confidence_threshold": 3,   
        "ignore_quality": False      
    },
    "system": {
        "http_port": 5000,
        "mavlink_connect": "udpin:0.0.0.0:14550",
        # start_node REMOVED (Auto-detected now)
        "start_heading": "NORTH" # NORTH, EAST, SOUTH, WEST
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
    
    # Initialize Heading Only (Position is now detected at runtime)
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
STATE_INITIAL_SCAN = 5   # <--- NEW STATE: Hover and look for start QR
STATE_LANDING = 99
STATE_SCANNING = 100

STATE_FOLLOW_LINE = 10
STATE_AT_NODE = 20
STATE_CALCULATING = 30
STATE_TURNING = 40
STATE_PUSH_OUT = 50

mission_state = STATE_INIT
mission_start_command = False
line_detected = False
last_line_time = time.time()
fcu_altitude = 0.0

prev_error = 0.0; integral_error = 0.0; last_known_direction = 0 
state_timer = 0.0
web_data = {"mode": "INIT", "armed": False, "alt": 0.0, "msg": "Booting...", "target": "NONE", "curr": "?"}

global_frame = None; frame_lock = threading.Lock()
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
def find_working_camera():
    global active_cam_index
    if active_cam_index is not None:
        cap = cv2.VideoCapture(active_cam_index)
        if cap.isOpened(): return cap
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                active_cam_index = index
                return cap
            else: cap.release()
    return None

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
    mav_dir = 1 if direction == 1 else -1 
    yaw_spd = CONFIG["flight"].get("yaw_speed", 30)
    progress(f"CMD: YAW {angle} deg")
    for i in range(3):
        conn.mav.command_long_send(conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 
            abs(angle), yaw_spd, mav_dir, 1, 0, 0, 0)
        time.sleep(0.1)
    return (abs(angle) / yaw_spd) + 1.5

# ==========================================
# 4. VISION LOGIC
# ==========================================
current_vx = 0.0; current_vy = 0.0
detected_qr_buffer = None

def vision_thread_func():
    global current_vx, current_vy, line_detected, last_line_time, prev_error, integral_error, last_known_direction, global_frame, detected_qr_buffer
    
    cap = find_working_camera()
    while not cap: time.sleep(1); cap = find_working_camera()
    cap.set(3, 320); cap.set(4, 240)
    print("VISION THREAD STARTED")

    while True:
        success, frame = cap.read()
        if not success: 
            cap.release(); time.sleep(0.5); cap = find_working_camera(); continue
        
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
                error_x = cx - cx_scr
                if abs(error_x) > 10: last_known_direction = 1 if error_x > 0 else -1
                kp = CONFIG["control"]["pid_kp"]; ki = CONFIG["control"]["pid_ki"]; kd = CONFIG["control"]["pid_kd"]
                integral_error += error_x; integral_error = max(min(integral_error, 500), -500) 
                derivative = error_x - prev_error
                raw_vy = (error_x * kp) + (integral_error * ki) + (derivative * kd)
                prev_error = error_x; max_lat = CONFIG["flight"]["max_lat_vel"]
                current_vy = max(min(raw_vy, max_lat), -max_lat)
                current_vx = CONFIG["flight"]["forward_speed"]
        else:
            temp_line_detected = False; integral_error = 0.0; prev_error = 0.0; current_vx = 0.0; current_vy = 0.0
        
        line_detected = temp_line_detected

        # --- QR DETECTION ---
        qr_objects = decode(frame)
        for obj in qr_objects:
            qr_text = obj.data.decode("utf-8")
            if qr_text in GRID_MAP:
                detected_qr_buffer = qr_text # Send to Mission Thread
                pts = np.array([obj.polygon], np.int32)
                cv2.polylines(frame, [pts], True, (255,0,255), 2)
                cv2.putText(frame, qr_text, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

        # UI
        curr_str = navigator.current_node if navigator.current_node else "SCANNING..."
        right_panel = np.zeros((h, 150, 3), dtype=np.uint8)
        cv2.putText(right_panel, f"QR: {curr_str}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        cv2.putText(right_panel, f"TGT: {navigator.target_node}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        combined = cv2.hconcat([frame, right_panel])
        
        with frame_lock: global_frame = combined
        time.sleep(0.01)

# ==========================================
# 5. CONTROL & MISSION LOGIC
# ==========================================
def reset_pid():
    global prev_error, integral_error, current_vx, current_vy
    prev_error = 0.0; integral_error = 0.0; current_vx = 0.0; current_vy = 0.0

def send_vel_cmd():
    global mission_state, current_vx, current_vy, last_known_direction, state_timer
    
    if mission_state == STATE_TURNING:
        return
    
    # BRAKE & HOLD
    if mission_state in [STATE_AT_NODE, STATE_CALCULATING, STATE_TURNING, STATE_INITIAL_SCAN]:
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0, 0, 0, 0,0,0, 0,0)
        return

    # PUSH OUT (BLIND FORWARD)
    if mission_state == STATE_PUSH_OUT:
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, CONFIG["flight"]["forward_speed"], 0, 0, 0,0,0, 0,0)
        return

    # FOLLOW LINE
    if mission_state == STATE_FOLLOW_LINE:
        if line_detected:
            conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, current_vx, current_vy, 0, 0,0,0, 0,0)
        else:
            # Simple Searching
            search_vx = CONFIG["control"]["search_fwd_vel"]
            search_vy = CONFIG["control"]["search_vel"] if last_known_direction < 0 else -CONFIG["control"]["search_vel"]
            conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, search_vx, search_vy, 0, 0,0,0, 0,0)

    # HOLD
    elif mission_state in [STATE_WAIT_USER]:
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0, 0, 0, 0,0,0, 0,0)

def mission_logic_thread():
    global mission_state, detected_qr_buffer, mission_start_command, state_timer
    
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
            
            # --- START SCANNING INSTEAD OF CALCULATING ---
            progress("HOVER: SCANNING FOR START QR..."); 
            detected_qr_buffer = None
            mission_state = STATE_INITIAL_SCAN

        elif mission_state == STATE_INITIAL_SCAN:
            if detected_qr_buffer and detected_qr_buffer in GRID_MAP:
                navigator.current_node = detected_qr_buffer
                progress(f"LOCATED: {navigator.current_node}")
                time.sleep(1.0) # Hover briefly to stabilize
                mission_state = STATE_CALCULATING
            else:
                # If taking too long, just wait.
                pass

        elif mission_state == STATE_CALCULATING:
            reset_pid()
            detected_qr_buffer = None # Clear buffer

            if navigator.current_node == navigator.target_node:
                progress("TARGET REACHED. LANDING.")
                mission_state = STATE_LANDING
                continue

            path_coords = navigator.calculate_path()
            
            if len(path_coords) < 2:
                progress("NO PATH / ALREADY THERE")
                mission_state = STATE_LANDING
                continue
            
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
            else:
                progress("FORWARD (NO TURN)")
            
            progress("PUSH OUT")
            state_timer = time.time()
            mission_state = STATE_PUSH_OUT
            
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
                encoded = cv2.imencode('.jpg', img)[1].tobytes()
            else:
                encoded = cv2.imencode('.jpg', global_frame)[1].tobytes()
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
    </style></head><body>
    <h1>KING PHOENIX V41: AUTO-START NAV</h1>
    <img src="/video_feed" style="border:2px solid #0f0;width:640px"><br>
    <h3>STATUS: <span id="st">INIT</span> | MSG: <span id="msg">-</span></h3>
    <h3>CURRENT: <span id="cur">-</span> | TARGET: <span id="tgt">-</span></h3>
    
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
        <button class="btn grid-btn" onclick="st('Landing')">Tengah Bawah</button>
        <button class="btn grid-btn" onclick="st('ToSouth')">Kanan Bawah</button>
    </div>

    <script>
    function st(t){
        fetch('/set_target', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({target:t})})
    }
    setInterval(()=>{
        fetch('/status').then(r=>r.json()).then(d=>{
            document.getElementById('st').innerText=d.state;
            document.getElementById('msg').innerText=d.msg;
            document.getElementById('cur').innerText=d.curr;
            document.getElementById('tgt').innerText=d.target;
        });
    }, 500);
    </script>
    </body></html>
    ''')

@app.route('/video_feed')
def video_feed(): return Response(get_display_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
def run_flask(): app.run(host='0.0.0.0', port=CONFIG["system"]["http_port"], threaded=True)

# ==========================================
# 7. MAIN STARTUP
# ==========================================
parser = argparse.ArgumentParser()
default_conn = CONFIG["system"]["mavlink_connect"]
parser.add_argument('--connect', default=default_conn)
args = parser.parse_args()

progress("BOOTING KING PHOENIX V41 (AUTO-LOCATOR)...")
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