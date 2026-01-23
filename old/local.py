#!/usr/bin/env python3

###########################################################
##   KING PHOENIX V38: ADVANCED MAP & MISSION PLANNER    ##
##   (Level Advanced: Dynamic Routing & Web Mission Tab) ##
###########################################################

import sys, os, time, threading, math as m, argparse
import numpy as np
import cv2
import json
import copy
from flask import Flask, Response, render_template_string, jsonify, request

os.environ["MAVLINK20"] = "1"
import pyrealsense2 as rs
import transformations as tf
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
from pyzbar.pyzbar import decode

# ==========================================
# 1. CONFIGURATION & MAP ENGINE
# ==========================================
DEFAULT_CONFIG = {
    "flight": {
        "takeoff_alt": 1.0,         
        "forward_speed": 0.1,       
        "turn_speed": 0.1,          # Kecepatan maju saat manuver belok (jika perlu)
        "max_lat_vel": 0.3,         
        "yaw_speed": 20,            
        "pre_turn_delay": 3.0,      
        "post_turn_delay": 2.0,     
        "blind_fwd_time": 3.0,      
        "alt_source": "T265"        
    },
    "control": {
        "pid_kp": 0.008, "pid_ki": 0.0001, "pid_kd": 0.003,    
        "search_vel": 0.08, "search_fwd_vel": 0.05      
    },
    "vision": {
        "is_black_line": True, "threshold_val": 80, "roi_height_ratio": 0.30,    
        "min_area": 1000, "min_aspect_ratio": 0.8, "lost_timeout": 3.0,         
        "camera_offset_x": 0, "qr_confirm_count": 1        
    },
    "t265": {
        "scale_factor": 1.0, "confidence_threshold": 3, "ignore_quality": False      
    },
    "system": {
        "http_port": 5000,
        "mavlink_connect": "udpin:0.0.0.0:14550"
    },
    # --- LEVEL ADVANCED: MAP & MISI ---
    "mission": {
        "loop": False,
        "sequence": ["LANDING", "ToSouth", "LANDING", "ToEast"] # Default Mission
    },
    "map_topology": {
        # Format: "FROM_NODE": { "TO_NODE": "ACTION" }
        # ACTION: "STRAIGHT", "TURN_RIGHT_80", "TURN_LEFT_80", "U_TURN_180"
        
        # Dari Barat (ToWest)
        "ToWest": { "LANDING": "STRAIGHT" }, 
        
        # Dari Selatan (ToSouth)
        "ToSouth": { "LANDING": "STRAIGHT" },

        # Dari Timur (ToEast)
        "ToEast": { "LANDING": "STRAIGHT" },

        # Dari Tengah (LANDING) - Logika Percabangan
        # Ini tricky, tergantung "datang dari mana". 
        # Untuk V38, kita asumsikan routing absolut sederhana:
        "LANDING": {
            "ToWest": "TURN_LEFT_80",   # Asumsi hadap Utara
            "ToEast": "TURN_RIGHT_80",  # Asumsi hadap Utara
            "ToSouth": "U_TURN_180"     # Asumsi hadap Utara
        }
    }
}

# Advanced Routing Logic (Simplifikasi untuk T-Shape)
# Kita butuh tahu orientasi terakhir untuk menentukan belok kiri/kanan yang benar.
# Namun untuk V38 ini kita pakai "Action Mapping" manual di UI jika perlu,
# atau Auto-Logic sederhana.

CONFIG_FILE = "config.json"
CONFIG = {}

def load_config():
    global CONFIG
    CONFIG = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                # Recursive update simple
                for k, v in saved.items():
                    if k in CONFIG: CONFIG[k].update(v)
            print("CONFIG LOADED")
        except: print("CONFIG ERROR, USING DEFAULT")
    
    # Safety Check
    if "map_topology" not in CONFIG: CONFIG["map_topology"] = DEFAULT_CONFIG["map_topology"]
    if "mission" not in CONFIG: CONFIG["mission"] = DEFAULT_CONFIG["mission"]

def save_config_to_file():
    try:
        with open(CONFIG_FILE, 'w') as f: json.dump(CONFIG, f, indent=4)
        return True
    except: return False

load_config()

# ==========================================
# 2. GLOBAL STATE & NAVIGATION
# ==========================================
STATE_INIT = 0
STATE_WAIT_USER = 1
STATE_TAKEOFF = 2
STATE_NAVIGATING = 10   # Line Following mencari QR selanjutnya
STATE_ARRIVED = 11      # Sampai di QR, Ngerem
STATE_EXECUTING = 12    # Sedang melakukan aksi (Belok/U-Turn)
STATE_RECOVERY = 13     # Blind Forward setelah belok
STATE_LANDING = 99
STATE_SCANNING = 100

mission_state = STATE_INIT
mission_start_command = False
line_detected = False
last_line_time = time.time()
qr_data = ""
fcu_altitude = 0.0

# Navigation Memory
current_node = "UNKNOWN"    # Lokasi QR terakhir
target_node = None          # Tujuan berikutnya dari list misi
mission_index = 0           # Indeks urutan misi saat ini
leg_start_time = 0.0        # Timer untuk blind forward
active_action = "NONE"      # Aksi yang sedang dilakukan (misal "TURN_RIGHT_80")

# PID Vars
prev_error = 0.0; integral_error = 0.0; last_known_direction = 0 
web_data = {"mode": "INIT", "armed": False, "alt": 0.0, "bat": 0.0, "msg": "Booting...", "t265": "Wait...", "state": "INIT", "next_wp": "NONE"}

global_frame = None; frame_lock = threading.Lock()
mav_lock = threading.Lock()
pipe, data, prev_data, H_aeroRef_aeroBody = None, None, None, None
reset_counter = 1; current_time_us = 0
active_cam_index = None
app = Flask(__name__)

def progress(s): 
    print(s, file=sys.stdout); sys.stdout.flush()
    web_data["msg"] = s

# ==========================================
# 3. ADVANCED ROUTING ENGINE
# ==========================================
def get_action_for_route(start_node, end_node):
    """
    Mencari aksi berdasarkan Peta Topology.
    Contoh: Jika start="ToSouth" dan end="LANDING" -> return "STRAIGHT"
    """
    topology = CONFIG["map_topology"]
    
    # Cek Direct Link
    if start_node in topology and end_node in topology[start_node]:
        return topology[start_node][end_node]
    
    # Jika tidak ada di map, Default Fallback
    return "STRAIGHT" # Dangerous, but better than crash

# ==========================================
# 4. HARDWARE & MAVLINK
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

def mavlink_loop(conn):
    while True:
        conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0)
        msg = conn.recv_match(type=['HEARTBEAT', 'SYS_STATUS', 'GLOBAL_POSITION_INT', 'STATUSTEXT'], timeout=1, blocking=True)
        if msg:
            if msg.get_type() == 'HEARTBEAT':
                web_data["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                for name, id in conn.mode_mapping().items():
                    if msg.custom_mode == id: web_data["mode"] = name; break
            elif msg.get_type() == 'SYS_STATUS': web_data["bat"] = msg.voltage_battery / 1000.0
            elif msg.get_type() == 'GLOBAL_POSITION_INT': 
                global fcu_altitude; fcu_altitude = msg.relative_alt / 1000.0
            elif msg.get_type() == 'STATUSTEXT': web_data["msg"] = msg.text.upper()

def send_vision_msg(conn):
    global current_time_us, H_aeroRef_aeroBody, reset_counter, data, fcu_altitude
    with mav_lock:
        if H_aeroRef_aeroBody is not None and data is not None:
            rpy = np.array(tf.euler_from_matrix(H_aeroRef_aeroBody, 'sxyz'))
            cov = 0.01 * pow(10, 3 - int(data.tracker_confidence))
            conn.mav.vision_position_estimate_send(current_time_us, 
                H_aeroRef_aeroBody[0][3], H_aeroRef_aeroBody[1][3], H_aeroRef_aeroBody[2][3],
                rpy[0], rpy[1], rpy[2], [cov]*21, reset_counter)
            conf_str = ["FAILED", "LOW", "MED", "HIGH"]
            web_data["t265"] = f"{conf_str[data.tracker_confidence]} (X:{H_aeroRef_aeroBody[0][3]:.1f})"
            if CONFIG["flight"].get("alt_source", "T265") == "FCU": web_data["alt"] = fcu_altitude
            else: web_data["alt"] = -H_aeroRef_aeroBody[2][3]

def perform_rotation(conn, action_str):
    # Parsing Action String: "TURN_RIGHT_80", "U_TURN_180", "TURN_LEFT_90"
    angle = 0
    direction = 1 # 1=Right, -1=Left
    
    if "180" in action_str: angle = 180; direction = 1
    elif "90" in action_str: angle = 90
    elif "80" in action_str: angle = 80
    else: angle = 90 # Default
    
    if "LEFT" in action_str: direction = -1
    
    yaw_spd = CONFIG["flight"].get("yaw_speed", 20)
    progress(f"EXEC: {action_str} ({angle}deg)")
    
    for i in range(3):
        conn.mav.command_long_send(conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, angle, yaw_spd, direction, 1, 0, 0, 0)
        time.sleep(0.1)
    
    return (angle / yaw_spd) + 1.5

# ==========================================
# 5. VISION & CONTROL
# ==========================================
current_vx = 0.0; current_vy = 0.0

def vision_thread_func(conn):
    global current_vx, current_vy, line_detected, last_line_time, qr_data, prev_error, integral_error, last_known_direction, global_frame, mission_state, current_node, active_action
    
    cap = find_working_camera()
    while not cap: time.sleep(1); cap = find_working_camera()
    cap.set(3, 320); cap.set(4, 240)

    while True:
        success, frame = cap.read()
        if not success: 
            cap.release(); time.sleep(0.5); cap = find_working_camera(); continue
        
        # Image Processing
        h, w, _ = frame.shape
        cut_y = int(h * CONFIG["vision"]["roi_height_ratio"])
        roi = frame[cut_y:h, 0:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        t_type = cv2.THRESH_BINARY_INV if CONFIG["vision"]["is_black_line"] else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blur, CONFIG["vision"]["threshold_val"], 255, t_type)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Line Detection
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
        
        if best_cnt is not None:
            line_detected = True; last_line_time = time.time()
            M = cv2.moments(best_cnt); cx = int(M['m10'] / M['m00'])
            cv2.drawContours(roi, [best_cnt], -1, (0,255,255), 2)
            cv2.line(frame, (cx_scr, cut_y), (cx, cut_y), (0,0,255), 2)
            
            error_x = cx - cx_scr
            if abs(error_x) > 10: last_known_direction = 1 if error_x > 0 else -1
            
            # PID Calculation
            kp = CONFIG["control"].get("pid_kp", 0.005); ki = CONFIG["control"].get("pid_ki", 0.000); kd = CONFIG["control"].get("pid_kd", 0.002)
            integral_error += error_x; integral_error = max(min(integral_error, 500), -500) 
            derivative = error_x - prev_error; raw_vy = (error_x * kp) + (integral_error * ki) + (derivative * kd)
            prev_error = error_x; max_lat = CONFIG["flight"]["max_lat_vel"]
            current_vy = max(min(raw_vy, max_lat), -max_lat)
            current_vx = CONFIG["flight"]["forward_speed"]
        else:
            line_detected = False; integral_error = 0.0; prev_error = 0.0; current_vx = 0.0; current_vy = 0.0

        # QR Processing & Logic Interrupt
        qr_objects = decode(frame)
        for obj in qr_objects:
            qr_text = obj.data.decode("utf-8")
            cv2.polylines(frame, [np.array([obj.polygon], np.int32)], True, (255,0,255), 2)
            cv2.putText(frame, qr_text, (obj.rect.left, obj.rect.top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

            # === MASTER MISSION TRIGGER ===
            # Hanya trigger jika sedang Navigating/Searching dan ketemu QR baru
            if mission_state in [STATE_NAVIGATING, STATE_SCANNING, STATE_TAKEOFF, STATE_WAIT_USER]:
                # Update Lokasi
                current_node = qr_text 
                progress(f"LOCATED AT: {current_node}")
                
                # Check apakah ini Target kita?
                if target_node and qr_text == target_node:
                    print(f">>> ARRIVED AT WAYPOINT: {qr_text} <<<")
                    
                    # STOP DRONE INSTANTLY
                    with mav_lock: conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0,0,0, 0,0,0, 0,0)
                    current_vx = 0; current_vy = 0
                    mission_state = STATE_ARRIVED # Trigger logic di main thread
                
                # Jika bukan target, tapi kita di persimpangan (Pass-Through)
                # Misalnya: Target ToEast, tapi ketemu LANDING (Intersection)
                elif target_node and qr_text != target_node:
                    # Cek Map: Dari Sini (qr_text) ke Target (target_node) harus ngapain?
                    action = get_action_for_route(qr_text, target_node)
                    if action == "STRAIGHT":
                        progress(f"INTERSECTION {qr_text}: GOING STRAIGHT")
                        # Reset blind forward timer agar "tembus" persimpangan
                        leg_start_time = time.time()
                    else:
                        # Harus belok di intersection ini
                        print(f">>> INTERSECTION TURN: {action} <<<")
                        with mav_lock: conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0,0,0, 0,0,0, 0,0)
                        current_vx = 0; current_vy = 0
                        active_action = action # Simpan aksi untuk dieksekusi
                        mission_state = STATE_ARRIVED # Trigger logic

        # Display Frame Update
        with frame_lock: global_frame = frame.copy()
        time.sleep(0.01)

def send_vel_cmd(conn):
    global mission_state, current_vx, current_vy, leg_start_time
    
    # STOP TOTAL (Rotating/Arrived)
    if mission_state in [STATE_ARRIVED, STATE_EXECUTING]: 
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0, 0, 0, 0,0,0, 0,0)
        return

    # NAVIGATING (LINE FOLLOW)
    if mission_state == STATE_NAVIGATING and line_detected:
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, current_vx, current_vy, 0, 0,0,0, 0,0)
    
    # RECOVERY (BLIND FORWARD)
    elif mission_state == STATE_RECOVERY:
        elapsed = time.time() - leg_start_time
        if elapsed < CONFIG["flight"]["blind_fwd_time"]:
            # Maju buta lurus
            fwd = CONFIG["control"].get("search_fwd_vel", 0.05)
            conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, fwd, 0, 0, 0,0,0, 0,0)
        else:
            # Waktu habis, pindah ke scanning biasa
            mission_state = STATE_SCANNING

    # SCANNING (LINE LOST)
    elif (mission_state == STATE_NAVIGATING or mission_state == STATE_SCANNING) and not line_detected:
        search_spd = CONFIG["control"].get("search_vel", 0.08)
        search_vy = search_spd if last_known_direction < 0 else -search_spd
        if last_known_direction == 0: search_vy = -0.02
        # Cek apakah perlu blind forward (Pass through intersection)
        elapsed = time.time() - leg_start_time
        vx = 0.0
        if elapsed < CONFIG["flight"]["blind_fwd_time"]: vx = CONFIG["control"].get("search_fwd_vel", 0.05); search_vy = 0.0 # Force Straight
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, vx, search_vy, 0, 0,0,0, 0,0)

def mission_logic_thread(conn):
    global mission_state, mission_index, target_node, active_action, leg_start_time, current_node
    
    while True:
        time.sleep(0.1)
        web_data["state"] = f"S:{mission_state} T:{target_node}"
        
        if mission_state == STATE_INIT:
            if data and data.tracker_confidence >= 1: progress("T265 OK"); mission_state = STATE_WAIT_USER

        elif mission_state == STATE_WAIT_USER:
            if mission_start_command: 
                progress("TAKEOFF..."); mission_state = STATE_TAKEOFF
                conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                time.sleep(1)
                conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, CONFIG["flight"]["takeoff_alt"])
                time.sleep(5)
                
                # LOAD MISSION PERTAMA
                seq = CONFIG["mission"]["sequence"]
                if len(seq) > 0:
                    mission_index = 0
                    target_node = seq[0]
                    progress(f"MISSION START -> {target_node}")
                    mission_state = STATE_NAVIGATING
                    leg_start_time = time.time()
                else:
                    progress("NO MISSION"); mission_state = STATE_LANDING

        elif mission_state == STATE_ARRIVED:
            # Drone sudah ngerem karena deteksi QR target atau Intersection Turn
            time.sleep(CONFIG["flight"]["pre_turn_delay"])
            
            # Cek apakah ini Target Akhir dari step ini?
            if current_node == target_node:
                progress(f"REACHED {target_node}")
                mission_index += 1
                seq = CONFIG["mission"]["sequence"]
                
                if mission_index >= len(seq):
                    if CONFIG["mission"]["loop"]:
                        mission_index = 0; target_node = seq[0]
                        progress("LOOPING MISSION")
                    else:
                        progress("MISSION FINISHED"); mission_state = STATE_LANDING
                        continue
                else:
                    # Next Waypoint
                    old_target = target_node
                    target_node = seq[mission_index]
                    progress(f"NEXT TARGET: {target_node}")
                    
                    # Tentukan Aksi dari Node sekarang ke Next Target
                    active_action = get_action_for_route(current_node, target_node)
            
            # Eksekusi Aksi (Entah karena sampai target, atau intersection)
            if active_action == "STRAIGHT":
                progress("ACTION: STRAIGHT")
                mission_state = STATE_RECOVERY # Maju lurus
                leg_start_time = time.time()
            else:
                mission_state = STATE_EXECUTING
                dur = perform_rotation(conn, active_action)
                time.sleep(dur)
                progress("TURN DONE")
                mission_state = STATE_RECOVERY # Blind Forward setelah putar
                leg_start_time = time.time()

        elif mission_state == STATE_RECOVERY:
            # Logic diurus send_vel_cmd, hanya tunggu waktu
            elapsed = time.time() - leg_start_time
            if elapsed > CONFIG["flight"]["blind_fwd_time"]:
                mission_state = STATE_NAVIGATING

        elif mission_state == STATE_LANDING:
            conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()["LAND"])
            mission_start_command = False
            mission_state = STATE_WAIT_USER

# ==========================================
# 6. FLASK WEB
# ==========================================
def get_display_frame():
    while True:
        with frame_lock:
            if global_frame is None: encoded = cv2.imencode('.jpg', np.zeros((240,320,3),np.uint8))[1].tobytes()
            else: encoded = cv2.imencode('.jpg', global_frame)[1].tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n'); time.sleep(0.1)

@app.route('/status')
def status(): 
    web_data["next_wp"] = target_node if target_node else "NONE"
    return jsonify(web_data)
@app.route('/start_mission')
def start_cmd():
    global mission_start_command
    if (web_data["t265"].startswith("HIGH") or CONFIG["t265"]["ignore_quality"]): 
        mission_start_command = True; return jsonify({"status": "ok"})
    return jsonify({"status": "error", "msg": "T265 Not Ready"})
@app.route('/stop_mission')
def stop_cmd(): global mission_state; mission_state = STATE_LANDING; return jsonify({"status": "ok"})
@app.route('/get_config')
def get_config_api(): return jsonify({"config": CONFIG, "desc": PARAM_DESCRIPTIONS})
@app.route('/save_config', methods=['POST'])
def save_config_api(): global CONFIG; CONFIG = request.json; save_config_to_file(); return jsonify({"status": "ok"})
@app.route('/')
def index():
    return render_template_string('''<!DOCTYPE html><html><head><title>PHOENIX V38 ADVANCED</title><meta name="viewport" content="width=device-width,initial-scale=1.0"><style>body{background:#121212;color:#eee;font-family:monospace}.container{max-width:1200px;margin:20px auto}.btn{padding:10px;width:100%;margin:5px 0;cursor:pointer;font-weight:bold}.btn-g{background:green;color:white}.btn-r{background:red;color:white}.input-dark{background:#222;color:white;border:1px solid #444;width:95%;padding:5px}.grid{display:grid;grid-template-columns:3fr 1fr;gap:10px}</style></head><body><div class="container"><h2>PHOENIX V38 MISSION PLANNER</h2><div class="grid"><img src="/video_feed" style="width:100%"><div style="background:#222;padding:10px"><h3>STATUS</h3><div id="st">--</div><div id="nxt">Target: --</div><button class="btn btn-g" onclick="act('/start_mission')">START</button><button class="btn btn-r" onclick="act('/stop_mission')">LAND</button></div></div><hr><h3>MISSION EDITOR</h3><textarea id="json-cfg" rows="20" class="input-dark"></textarea><button class="btn btn-g" onclick="save()">SAVE CONFIG & MISSION</button></div><script>function act(u){fetch(u).then(r=>r.json()).then(d=>alert(d.msg||d.status));}function load(){fetch('/get_config').then(r=>r.json()).then(d=>{document.getElementById('json-cfg').value=JSON.stringify(d.config,null,4);});}function save(){fetch('/save_config',{method:'POST',headers:{'Content-Type':'application/json'},body:document.getElementById('json-cfg').value}).then(r=>r.json()).then(d=>alert(d.status));}setInterval(()=>{fetch('/status').then(r=>r.json()).then(d=>{document.getElementById('st').innerText=d.state;document.getElementById('nxt').innerText="Target: "+d.next_wp;});},500);load();</script></body></html>''')

@app.route('/video_feed')
def video_feed(): return Response(get_display_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
def run_flask(): app.run(host='0.0.0.0', port=5000, threaded=True)

# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--connect', default=CONFIG["system"]["mavlink_connect"])
    args = parser.parse_args()

    progress("BOOTING PHOENIX V38 (ADVANCED MAP)...")
    conn = mavutil.mavlink_connection(args.connect, autoreconnect=True, source_system=1, source_component=191)
    
    threading.Thread(target=mavlink_loop, args=(conn,)).start()
    
    try: pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.pose); pipe.start(cfg)
    except: progress("T265 FAIL"); sys.exit(1)

    threading.Thread(target=run_flask, daemon=True).start()
    sched = BackgroundScheduler()
    sched.add_job(send_vision_msg, 'interval', seconds=1/30.0, args=(conn,))
    sched.add_job(send_vel_cmd, 'interval', seconds=1/10.0, args=(conn,))
    sched.start()

    threading.Thread(target=vision_thread_func, args=(conn,), daemon=True).start()
    threading.Thread(target=mission_logic_thread, args=(conn,), daemon=True).start()

    progress("SYSTEM READY")
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
