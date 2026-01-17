#!/usr/bin/env python3

#####################################################
##    KING PHOENIX V12: EMERGENCY STOP & RESTART   ##
#####################################################

import sys, os, time, threading, math as m, argparse
import numpy as np
import cv2
import json
from flask import Flask, Response, render_template_string, jsonify, request

# Set MAVLink protocol to 2.
os.environ["MAVLINK20"] = "1"
import pyrealsense2 as rs
import transformations as tf
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
from pyzbar.pyzbar import decode

# ==========================================
# 1. CONFIGURATION MANAGER
# ==========================================
DEFAULT_CONFIG = {
    "flight": {
        "takeoff_alt": 1.0,         
        "forward_speed": 0.1,       
        "max_lat_vel": 0.3,         
        "rotation_angle": 180,      
        "rotation_time": 6.0,       
        "land_after_mission": True, 
        "alt_source": "T265"        
    },
    "control": {
        "pid_kp": 0.008,   
        "pid_ki": 0.0001,  
        "pid_kd": 0.003,   
        "search_vel": 0.08 
    },
    "vision": {
        "is_black_line": True,      
        "threshold_val": 80,        
        "roi_height_ratio": 0.30,   
        "min_area": 1000,           
        "min_aspect_ratio": 0.8,    
        "lost_timeout": 3.0,        
        "camera_offset_x": 0,
        "center_priority_weight": 10.0,
        "qr_confirm_count": 1
    },
    "t265": {
        "scale_factor": 1.0,        
        "confidence_threshold": 3,  
        "ignore_quality": False     
    },
    "system": {
        "http_port": 5000,
        "mavlink_connect": "udpin:0.0.0.0:14550", 
        "qr_keyword": "E,N,N,W,0"      
    }
}

PARAM_DESCRIPTIONS = {
    "pid_kp": "POWER: Besar koreksi. Naikkan jika drone malas kembali ke tengah.",
    "pid_ki": "ANTI-DRIFT: Akumulasi error. Naikkan (0.0001) jika drone suka hanyut pelan.",
    "pid_kd": "STABILIZER: Mencegah overshoot/goyang.",
    "search_vel": "Kecepatan geser otomatis saat garis hilang.",
    "takeoff_alt": "Target ketinggian (m).",
    "forward_speed": "Kecepatan maju (m/s).",
    "max_lat_vel": "Batas maksimum kecepatan geser/roll.",
    "roi_height_ratio": "Area pandang kamera (0.3 = 70% bawah).",
    "camera_offset_x": "Geser target tengah manual (Pixel).",
    "center_priority_weight": "Prioritas garis tengah vs noise pinggir."
}

CONFIG_FILE = "config.json"
CONFIG = {}

def load_config():
    global CONFIG
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_conf = json.load(f)
                CONFIG = DEFAULT_CONFIG.copy()
                for section in saved_conf:
                    if section in CONFIG:
                        CONFIG[section].update(saved_conf[section])
            print(f"LOADED CONFIG FROM {CONFIG_FILE}")
        except:
            CONFIG = DEFAULT_CONFIG.copy()
    else:
        CONFIG = DEFAULT_CONFIG.copy()

def save_config_to_file():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        return True
    except Exception as e:
        print(f"SAVE ERROR: {e}")
        return False

load_config()

# ==========================================
# 2. GLOBAL STATE
# ==========================================
STATE_INIT = 0; STATE_WAIT_USER = 1; STATE_TAKEOFF = 2; STATE_OUTBOUND = 3
STATE_SCANNING = 4; STATE_ROTATING = 5; STATE_INBOUND = 6; STATE_LANDING = 7

mission_state = STATE_INIT
mission_start_command = False
line_detected = False
last_line_time = time.time()
qr_data = ""
fcu_altitude = 0.0

# PID State
prev_error = 0.0
integral_error = 0.0
last_known_direction = 0 

web_data = {"mode": "INIT", "armed": False, "alt": 0.0, "bat": 0.0, "msg": "Booting...", "t265": "Wait...", "state": "INIT"}

pipe, data, prev_data, H_aeroRef_aeroBody = None, None, None, None
reset_counter = 1; current_time_us = 0
mavlink_thread_should_exit = False
lock = threading.Lock()
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
def sys_status_cb(msg): web_data["bat"] = msg.voltage_battery / 1000.0
def global_pos_cb(msg): global fcu_altitude; fcu_altitude = msg.relative_alt / 1000.0

def send_vision_msg():
    global current_time_us, H_aeroRef_aeroBody, reset_counter, data, fcu_altitude
    with lock:
        if H_aeroRef_aeroBody is not None and data is not None:
            rpy = np.array(tf.euler_from_matrix(H_aeroRef_aeroBody, 'sxyz'))
            cov = 0.01 * pow(10, 3 - int(data.tracker_confidence))
            conn.mav.vision_position_estimate_send(current_time_us, 
                H_aeroRef_aeroBody[0][3], H_aeroRef_aeroBody[1][3], H_aeroRef_aeroBody[2][3],
                rpy[0], rpy[1], rpy[2], [cov]*21, reset_counter)
            
            conf_str = ["FAILED", "LOW", "MED", "HIGH"]
            web_data["t265"] = f"{conf_str[data.tracker_confidence]} (X:{H_aeroRef_aeroBody[0][3]:.1f})"
            
            if CONFIG["flight"].get("alt_source", "T265") == "FCU":
                web_data["alt"] = fcu_altitude
            else:
                web_data["alt"] = -H_aeroRef_aeroBody[2][3]

def set_mode(mode):
    conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()[mode])

def perform_rotation():
    conn.mav.command_long_send(conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, CONFIG["flight"]["rotation_angle"], 0, 1, 1, 0, 0, 0)

# ==========================================
# 4. MISSION LOGIC
# ==========================================
current_vx = 0.0; current_vy = 0.0

def send_vel_cmd():
    global mission_state, current_vx, current_vy, last_known_direction
    
    if mission_state in [STATE_OUTBOUND, STATE_INBOUND] and line_detected:
        vx = current_vx 
        vy = current_vy 
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, vx, vy, 0, 0,0,0, 0,0)
    
    elif mission_state in [STATE_OUTBOUND, STATE_INBOUND] and not line_detected:
        # RECOVERY
        search_spd = CONFIG["control"].get("search_vel", 0.08)
        search_vy = search_spd if last_known_direction < 0 else -search_spd 
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0, search_vy, 0, 0,0,0, 0,0)

    elif mission_state in [STATE_SCANNING, STATE_WAIT_USER]:
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0, 0, 0, 0,0,0, 0,0)

def mission_logic_thread():
    global mission_state, qr_data, last_line_time, mission_start_command
    
    state_desc = [
        "MENUNGGU T265", "MENUNGGU START", "TAKEOFF", 
        "TRACKING (PERGI)", "SCAN QR (END)", "PUTAR BALIK", 
        "TRACKING (PULANG)", "LANDING"
    ]
    
    while True:
        time.sleep(0.1)
        web_data["state"] = state_desc[mission_state]

        if mission_state == STATE_INIT:
            quality_ok = (data and data.tracker_confidence >= CONFIG["t265"]["confidence_threshold"])
            ignore_quality = CONFIG["t265"].get("ignore_quality", False)
            if quality_ok or (data and ignore_quality):
                progress("T265 SIAP. MENUNGGU PILOT...")
                mission_state = STATE_WAIT_USER

        elif mission_state == STATE_WAIT_USER:
            if mission_start_command:
                progress("MISI DIMULAI...")
                time.sleep(2); mission_state = STATE_TAKEOFF

        elif mission_state == STATE_TAKEOFF:
            set_mode("GUIDED")
            conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            while not web_data["armed"]: time.sleep(0.5)
            conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, CONFIG["flight"]["takeoff_alt"])
            while web_data["alt"] < (CONFIG["flight"]["takeoff_alt"] - 0.2): time.sleep(0.5)
            progress("TAKEOFF OK. TRACKING..."); mission_state = STATE_OUTBOUND

        elif mission_state == STATE_OUTBOUND:
            if not line_detected and (time.time() - last_line_time > CONFIG["vision"]["lost_timeout"]):
                progress("GARIS HABIS. SCAN QR..."); mission_state = STATE_SCANNING; qr_data = ""

        elif mission_state == STATE_SCANNING:
            if CONFIG["system"]["qr_keyword"] in qr_data:
                progress(f"QR '{qr_data}' FOUND. U-TURN..."); mission_state = STATE_ROTATING
                perform_rotation()
                time.sleep(CONFIG["flight"]["rotation_time"])
                last_line_time = time.time(); progress("PULANG..."); mission_state = STATE_INBOUND

        elif mission_state == STATE_INBOUND:
            if not line_detected and (time.time() - last_line_time > CONFIG["vision"]["lost_timeout"]):
                progress("SAMPAI RUMAH. LANDING..."); mission_state = STATE_LANDING

        elif mission_state == STATE_LANDING:
            # FORCE LAND (Entah dari normal atau emergency)
            set_mode("LAND")
            time.sleep(1)
            
            # Jika sudah Disarm, Reset Logic
            if not web_data["armed"]:
                mission_start_command = False
                mission_state = STATE_WAIT_USER
                progress("LANDING SELESAI. SIAP LAGI.")

# ==========================================
# 5. VISION & PID CONTROLLER
# ==========================================
def gen_frames():
    global current_vx, current_vy, line_detected, last_line_time, qr_data
    global prev_error, integral_error, last_known_direction
    
    cap = find_working_camera()
    if not cap:
        while True: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((240,640,3),np.uint8))[1].tobytes() + b'\r\n'); time.sleep(1)
    cap.set(3, 320); cap.set(4, 240)

    while True:
        success, frame = cap.read()
        if not success: cap.release(); cap = find_working_camera(); continue
        h, w, _ = frame.shape
        
        cut_y = int(h * CONFIG["vision"]["roi_height_ratio"])
        roi = frame[cut_y:h, 0:w]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        t_type = cv2.THRESH_BINARY_INV if CONFIG["vision"]["is_black_line"] else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blur, CONFIG["vision"]["threshold_val"], 255, t_type)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None; max_score = -999999
        offset_val = int(CONFIG["vision"].get("camera_offset_x", 0))
        cx_scr = int(w/2) + offset_val 
        penalty_weight = CONFIG["vision"].get("center_priority_weight", 10.0)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < CONFIG["vision"]["min_area"]: continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if (float(bh)/bw) < CONFIG["vision"]["min_aspect_ratio"]: continue

            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx_cnt = int(M['m10'] / M['m00'])
            dist_from_center = abs(cx_cnt - cx_scr)
            score = area - (dist_from_center * penalty_weight)
            
            if score > max_score: max_score = score; best_cnt = cnt
        
        line_detected = False
        current_vx = 0.0; current_vy = 0.0
        error_x = 0
        
        if best_cnt is not None:
            line_detected = True; last_line_time = time.time()
            M = cv2.moments(best_cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                real_cy = cut_y + int(M['m01'] / M['m00'])
                
                # Visual Debug
                cv2.line(frame, (cx_scr, 0), (cx_scr, h), (255,0,0), 2)
                cv2.line(frame, (cx_scr, real_cy), (cx, real_cy), (0,0,255), 3) 
                cv2.drawContours(roi, [best_cnt], -1, (0,255,255), 2)
                
                # PID
                error_x = cx_scr - cx 
                if abs(error_x) > 10: last_known_direction = 1 if error_x > 0 else -1
                
                kp = CONFIG["control"].get("pid_kp", 0.005)
                ki = CONFIG["control"].get("pid_ki", 0.000)
                kd = CONFIG["control"].get("pid_kd", 0.002)
                
                integral_error += error_x
                integral_error = max(min(integral_error, 500), -500) 
                derivative = error_x - prev_error
                raw_vy = (error_x * kp) + (integral_error * ki) + (derivative * kd)
                
                prev_error = error_x 
                max_lat = CONFIG["flight"]["max_lat_vel"]
                current_vy = max(min(raw_vy, max_lat), -max_lat)
                current_vx = CONFIG["flight"]["forward_speed"]
        else:
            integral_error = 0.0
            prev_error = 0.0

        qr_objects = decode(frame)
        for obj in qr_objects:
            qr_text = obj.data.decode("utf-8"); qr_data = qr_text
            pts = np.array([obj.polygon], np.int32)
            cv2.polylines(frame, [pts], True, (255,0,255), 2)
            cv2.putText(frame, qr_text, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        right_panel = np.zeros_like(frame)
        right_panel[cut_y:h, 0:w] = thresh_color
        cv2.putText(right_panel, "AI VIEW", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.line(right_panel, (cx_scr, 0), (cx_scr, h), (255, 0, 0), 2) 

        combined = cv2.hconcat([frame, right_panel])
        
        cv2.rectangle(combined, (0, 0), (640, 60), (0,0,0), -1) 
        cv2.putText(combined, f"MISI: {web_data['state']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(combined, f"ALT : {web_data['alt']:.2f}m", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        col_pid = (0, 255, 0) if abs(error_x) < 20 else (0, 0, 255)
        cv2.putText(combined, f"ERR : {error_x} px", (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_pid, 2)
        cv2.putText(combined, f"OUT : {current_vy:.3f} m/s", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        
        action_txt = "CENTER"
        if current_vy > 0.02: action_txt = "<< GESER KIRI"
        elif current_vy < -0.02: action_txt = "GESER KANAN >>"
        elif not line_detected: action_txt = "MENCARI..."
        cv2.putText(combined, action_txt, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', combined)[1].tobytes() + b'\r\n')

# ==========================================
# 6. WEB SERVER
# ==========================================
@app.route('/status')
def status(): return jsonify(web_data)

@app.route('/start_mission')
def start_cmd():
    global mission_start_command
    t265_ready = web_data["t265"].startswith("HIGH") or CONFIG["t265"].get("ignore_quality", False)
    if t265_ready:
        mission_start_command = True; return jsonify({"status": "ok"})
    return jsonify({"status": "error", "msg": "T265 Not Ready"})

@app.route('/stop_mission')
def stop_cmd():
    global mission_state, mission_start_command
    # Force state to LANDING
    mission_state = STATE_LANDING
    mission_start_command = False
    return jsonify({"status": "ok", "msg": "EMERGENCY LANDING TRIGGERED!"})

@app.route('/get_config')
def get_config_api(): return jsonify({"config": CONFIG, "desc": PARAM_DESCRIPTIONS})

@app.route('/save_config', methods=['POST'])
def save_config_api():
    global CONFIG
    try:
        CONFIG = request.json
        if save_config_to_file(): progress("CONFIG SAVED!"); return jsonify({"status": "ok"})
        else: return jsonify({"status": "error", "msg": "Write Failed"})
    except Exception as e: return jsonify({"status": "error", "msg": str(e)})

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>KingPhoenix GCS</title>
    <style>
        body { background: #121212; color: #eee; font-family: monospace; margin: 0; padding: 0; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .tabs { display: flex; border-bottom: 2px solid #333; margin-bottom: 20px; }
        .tab-btn { background: #1e1e1e; border: none; padding: 15px 30px; color: #888; font-weight: bold; cursor: pointer; font-size: 16px; }
        .tab-btn.active { background: #333; color: #00d2ff; border-top: 3px solid #00d2ff; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .grid { display: grid; grid-template-columns: 3fr 1fr; gap: 20px; }
        .video-box { border: 2px solid #333; background: #000; }
        .video-box img { width: 100%; display: block; }
        .panel { background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; }
        .stat-row { display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        .btn-start { width: 100%; padding: 15px; background: linear-gradient(45deg, #009900, #00cc00); color: white; border: none; font-weight: bold; cursor: pointer; border-radius:5px;}
        .btn-start:disabled { background: #444; color: #888; cursor: not-allowed; }
        .btn-stop { width: 100%; padding: 15px; background: #cc0000; color: white; border: none; font-weight: bold; cursor: pointer; border-radius:5px; margin-top: 10px;}
        .btn-stop:hover { background: #ff0000; }
        .console { height: 120px; background: #000; color: #0f0; padding: 10px; overflow-y: auto; border: 1px solid #333; font-size:11px; }
        .settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .section-box { background: #1e1e1e; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #333; }
        .section-title { color: #00d2ff; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 15px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; color: #aaa; margin-bottom: 5px; font-size: 12px; }
        .form-group input, .form-group select { width: 95%; padding: 8px; background: #111; border: 1px solid #444; color: white; }
        .btn-save { padding: 15px; background: #007bff; color: white; border: none; cursor: pointer; font-weight: bold; width: 100%; margin-top:20px;}
        .big-stat { font-size: 18px; color: #fff; }
        .big-label { color: #00d2ff; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('hud')">HUD MONITOR</button>
            <button class="tab-btn" onclick="showTab('settings')">SETTINGS</button>
        </div>
        <div id="hud" class="tab-content active">
            <div class="grid">
                <div class="video-box"><img src="/video_feed"></div>
                <div class="panel">
                    <div style="text-align:center; margin-bottom:10px; font-weight:bold; font-size:18px;" id="top-status">DISARMED</div>
                    
                    <button id="btn-start" class="btn-start" onclick="startMission()">START MISSION</button>
                    <button id="btn-stop" class="btn-stop" onclick="stopMission()">STOP / ABORT</button>
                    
                    <br><br>
                    <div class="big-label">STATUS MISI</div>
                    <div id="state" class="big-stat" style="color:yellow; margin-bottom:10px;">INIT</div>
                    <div class="stat-row"><span>ALTITUDE</span><span id="alt" class="stat-value">-- m</span></div>
                    <div class="stat-row"><span>BATTERY</span><span id="bat" class="stat-value">-- V</span></div>
                    <div class="stat-row"><span>MODE</span><span id="mode" class="stat-value">--</span></div>
                    <div class="stat-row"><span>T265</span><span id="t265" class="stat-value">--</span></div>
                    <div class="console" id="msg-box">System Initialized.</div>
                </div>
            </div>
        </div>
        <div id="settings" class="tab-content">
            <div id="settings-container" class="settings-grid"></div>
            <button class="btn-save" onclick="saveConfig()">SAVE CONFIGURATION (APPLY LIVE)</button>
        </div>
    </div>
    <script>
        let currentConfig = {};
        function showTab(t) {
            document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(e=>e.classList.remove('active'));
            document.getElementById(t).classList.add('active'); event.target.classList.add('active');
            if(t==='settings') loadSettings();
        }
        function startMission() {
            fetch('/start_mission').then(r=>r.json()).then(d=>{
                if(d.status==="ok"){document.getElementById('btn-start').disabled=true; document.getElementById('btn-start').innerText="RUNNING...";}
                else alert(d.msg);
            });
        }
        function stopMission() {
            if(confirm("YAKIN STOP & LANDING SEKARANG?")) {
                fetch('/stop_mission').then(r=>r.json()).then(d=>{
                    alert(d.msg);
                    document.getElementById('btn-start').innerText = "STOPPED";
                });
            }
        }
        setInterval(()=>{
            if(document.getElementById('hud').classList.contains('active')){
                fetch('/status').then(r=>r.json()).then(d=>{
                    document.getElementById('mode').innerText=d.mode;
                    document.getElementById('bat').innerText=d.bat.toFixed(1);
                    document.getElementById('alt').innerText=d.alt.toFixed(2);
                    document.getElementById('t265').innerText=d.t265;
                    document.getElementById('state').innerText=d.state;
                    let b=document.getElementById('top-status');
                    if(d.armed){b.innerText="ARMED";b.style.color="red";}else{b.innerText="DISARMED";b.style.color="#555";}
                    let m=document.getElementById('msg-box');
                    if(m.lastChild.innerText!=="> "+d.msg){let n=document.createElement('div');n.innerText="> "+d.msg;m.appendChild(n);m.scrollTop=m.scrollHeight;}
                    
                    // Logic tombol Start kembali aktif jika sudah Disarmed setelah STOP
                    if(d.state.includes("MENUNGGU") && document.getElementById('btn-start').innerText!=="START MISSION") {
                        document.getElementById('btn-start').disabled=false;
                        document.getElementById('btn-start').innerText="START MISSION";
                    }
                });
            }
        },500);
        function loadSettings(){
            fetch('/get_config').then(r=>r.json()).then(d=>{
                currentConfig=d.config; const desc=d.desc; const c=document.getElementById('settings-container'); c.innerHTML="";
                for(const s in currentConfig){
                    let h=`<div class="section-box"><div class="section-title">${s.toUpperCase()}</div>`;
                    for(const k in currentConfig[s]){
                        let v=currentConfig[s][k]; let ds=desc[k]||"";
                        h+=`<div class="form-group"><label>${k.toUpperCase()}</label>`;
                        if(k==="scan_mode") {
                            h+=`<select id="${s}-${k}">
                                <option value="NONE" ${v==="NONE"?"selected":""}>NONE (DIAM)</option>
                                <option value="FIXED" ${v==="FIXED"?"selected":""}>FIXED (GESER SATU ARAH)</option>
                                <option value="SWEEP" ${v==="SWEEP"?"selected":""}>SWEEP (KIRI-KANAN)</option>
                            </select>`;
                        } else if(typeof v==="boolean") {
                            h+=`<input type="checkbox" id="${s}-${k}" ${v?"checked":""}>`;
                        } else if(typeof v==="string") {
                            h+=`<input type="text" id="${s}-${k}" value="${v}">`;
                        } else {
                            h+=`<input type="number" step="0.001" id="${s}-${k}" value="${v}">`;
                        }
                        h+=`<div style="font-size:10px;color:#666;">${ds}</div></div>`;
                    }
                    h+="</div>"; c.innerHTML+=h;
                }
            });
        }
        function saveConfig(){
            for(const s in currentConfig){
                for(const k in currentConfig[s]){
                    const el=document.getElementById(`${s}-${k}`);
                    if(el.type==="checkbox") currentConfig[s][k]=el.checked;
                    else if(el.type==="number") currentConfig[s][k]=parseFloat(el.value);
                    else currentConfig[s][k]=el.value;
                }
            }
            fetch('/save_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentConfig)})
            .then(r=>r.json()).then(d=>{if(d.status==="ok")alert("SAVED!");else alert("Error: "+d.msg);});
        }
    </script>
</body>
</html>
    ''')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def run_flask(): app.run(host='0.0.0.0', port=CONFIG["system"]["http_port"], threaded=True)

# ==========================================
# 7. MAIN
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--connect', default=CONFIG["system"]["mavlink_connect"])
args = parser.parse_args()

progress("BOOTING SYSTEM V12...")
conn = mavutil.mavlink_connection(args.connect, autoreconnect=True, source_system=1, source_component=191)
callbacks = {'HEARTBEAT': heartbeat_cb, 'STATUSTEXT': statustext_cb, 'SYS_STATUS': sys_status_cb, 'GLOBAL_POSITION_INT': global_pos_cb}
threading.Thread(target=mavlink_loop, args=(conn, callbacks)).start()

try: pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.pose); pipe.start(cfg)
except: progress("T265 ERROR"); sys.exit(1)

threading.Thread(target=run_flask, daemon=True).start()
sched = BackgroundScheduler()
sched.add_job(send_vision_msg, 'interval', seconds=1/30.0)
sched.add_job(send_vel_cmd, 'interval', seconds=1/10.0)
sched.start()
threading.Thread(target=mission_logic_thread, daemon=True).start()

progress("SYSTEM READY.")

H_aeroRef_T265Ref = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
H_T265body_aeroBody = np.linalg.inv(H_aeroRef_T265Ref)

try:
    while True:
        frames = pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        if pose:
            with lock:
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