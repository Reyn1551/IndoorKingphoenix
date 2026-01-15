#!/usr/bin/env python3

#####################################################
##    KING PHOENIX V4: AUTO MISSION + DRIFT FIX    ##
#####################################################

import sys, os, time, threading, math as m, argparse
import numpy as np
import cv2
import json
from flask import Flask, Response, render_template_string, jsonify

os.environ["MAVLINK20"] = "1"
import pyrealsense2 as rs
import transformations as tf
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil
from pyzbar.pyzbar import decode

# ==========================================
# BAGIAN KONFIGURASI (TUNING DISINI)
# ==========================================

# 1. KONEKSI PIXHAWK
# Gunakan /dev/ttyTHS0 jika lewat GPIO (Pin 8 & 10)
# Gunakan /dev/ttyACM0 jika lewat kabel USB
DEFAULT_CONNECT = 'udp:127.0.0.1:14550' 
DEFAULT_BAUD    = 921600 

# 2. TUNING GERAK
FORWARD_SPEED = 0.1     # Kecepatan Maju (m/s)
GAIN_YAW      = 0.005   # Kekuatan Belok (Makin besar, makin tajam)
TAKEOFF_ALT   = 1.0     # Tinggi terbang (meter)

# 3. TUNING DRIFT (PENTING!)
# Jika drone miring/drift ke KIRI -> Isi Angka POSITIF (misal 20)
# Jika drone miring/drift ke KANAN -> Isi Angka NEGATIF (misal -20)
CENTER_OFFSET = 0       

# 4. VISION
IS_BLACK_LINE = True    # True = Lakban Hitam
LOST_LINE_TIMEOUT = 3.0 # Detik toleransi garis hilang

# ==========================================
# STATE MACHINE & GLOBAL VARS
# ==========================================
STATE_INIT, STATE_TAKEOFF, STATE_OUTBOUND = 0, 1, 2
STATE_SCANNING, STATE_ROTATING, STATE_INBOUND, STATE_LANDING = 3, 4, 5, 6

mission_state = STATE_INIT
command_vx, command_vy = 0.0, 0.0
line_detected = False
last_line_time = time.time()
qr_data = ""

web_data = {
    "mode": "INIT", "armed": False, "alt": 0.0, "bat": 0.0,
    "msg": "Booting...", "t265": "Wait...", "state": "INIT"
}

scale_factor = 1.0
pipe, data, prev_data, H_aeroRef_aeroBody = None, None, None, None
reset_counter = 1
current_time_us = 0
mavlink_thread_should_exit = False
lock = threading.Lock()
active_cam_index = None

app = Flask(__name__)

def progress(s): 
    print(s, file=sys.stdout)
    sys.stdout.flush()
    web_data["msg"] = s

# --- FUNGSI KAMERA ROBUST ---
def find_working_camera():
    global active_cam_index
    # Cek kamera terakhir
    if active_cam_index is not None:
        cap = cv2.VideoCapture(active_cam_index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: return cap
            else: cap.release()
    
    # Scan ulang
    for index in range(10):
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    active_cam_index = index
                    return cap
                else: cap.release()
        except: pass
    return None

# --- MAVLINK LOOP ---
def mavlink_loop(conn, callbacks):
    interesting = list(callbacks.keys())
    while not mavlink_thread_should_exit:
        try:
            conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                    mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0)
            msg = conn.recv_match(type=interesting, timeout=1, blocking=True)
            if msg: callbacks[msg.get_type()](msg)
        except: time.sleep(0.1)

# --- CALLBACKS ---
def heartbeat_cb(msg):
    mode_map = conn.mode_mapping()
    for name, id in mode_map.items():
        if msg.custom_mode == id:
            web_data["mode"] = name; break
    web_data["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

def statustext_cb(msg): web_data["msg"] = msg.text.upper()
def sys_status_cb(msg): web_data["bat"] = msg.voltage_battery / 1000.0

# --- SEND DATA TO FCU ---
def send_vision_msg():
    global current_time_us, H_aeroRef_aeroBody, reset_counter, data
    with lock:
        if H_aeroRef_aeroBody is not None and data is not None:
            rpy = np.array(tf.euler_from_matrix(H_aeroRef_aeroBody, 'sxyz'))
            cov = 0.01 * pow(10, 3 - int(data.tracker_confidence))
            cov_arr = [cov] * 21 
            conn.mav.vision_position_estimate_send(current_time_us, 
                H_aeroRef_aeroBody[0][3], H_aeroRef_aeroBody[1][3], H_aeroRef_aeroBody[2][3],
                rpy[0], rpy[1], rpy[2], cov_arr, reset_counter)
            
            conf_str = ["FAIL", "LOW", "MED", "HIGH"]
            web_data["t265"] = f"{conf_str[data.tracker_confidence]}"
            web_data["alt"] = -H_aeroRef_aeroBody[2][3]

def set_mode(mode):
    if mode not in conn.mode_mapping(): return
    conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()[mode])

def perform_rotation():
    conn.mav.command_long_send(conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 180, 0, 1, 1, 0, 0, 0)

def send_vel_cmd():
    global command_vx, command_vy, mission_state
    if mission_state in [STATE_OUTBOUND, STATE_INBOUND]:
        vx = command_vx if line_detected else 0.0
        vy = command_vy if line_detected else 0.0
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, vx, vy, 0, 0,0,0, 0,0)
    elif mission_state == STATE_SCANNING:
        conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0, 0, 0, 0,0,0, 0,0)

# --- LOGIC MISI ---
def mission_logic_thread():
    global mission_state, qr_data, last_line_time
    while True:
        time.sleep(0.1)
        web_data["state"] = ["INIT", "TAKEOFF", "OUTBOUND", "SCAN QR", "ROTATING", "INBOUND", "LANDING"][mission_state]

        if mission_state == STATE_INIT:
            if data and data.tracker_confidence == 3:
                progress("T265 READY. Syncing..."); time.sleep(5); mission_state = STATE_TAKEOFF

        elif mission_state == STATE_TAKEOFF:
            set_mode("GUIDED")
            conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            while not web_data["armed"]: time.sleep(0.5)
            progress("TAKING OFF...")
            conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, TAKEOFF_ALT)
            while web_data["alt"] < (TAKEOFF_ALT - 0.2): time.sleep(0.5)
            progress("STARTING OUTBOUND..."); mission_state = STATE_OUTBOUND

        elif mission_state == STATE_OUTBOUND:
            if not line_detected and (time.time() - last_line_time > LOST_LINE_TIMEOUT):
                progress("END OF LINE. SCANNING..."); mission_state = STATE_SCANNING; qr_data = ""

        elif mission_state == STATE_SCANNING:
            if "RETURN" in qr_data:
                progress("QR RETURN DETECTED! ROTATING..."); mission_state = STATE_ROTATING
                perform_rotation(); time.sleep(6); last_line_time = time.time()
                progress("RETURNING HOME..."); mission_state = STATE_INBOUND

        elif mission_state == STATE_INBOUND:
            if not line_detected and (time.time() - last_line_time > LOST_LINE_TIMEOUT):
                progress("HOME REACHED. LANDING..."); mission_state = STATE_LANDING

        elif mission_state == STATE_LANDING:
            set_mode("LAND"); time.sleep(5)

# --- VISION PROCESSING (FIXED CROP & OFFSET) ---
def gen_frames():
    global command_vx, command_vy, line_detected, last_line_time, qr_data
    
    while True:
        cap = find_working_camera()
        if not cap:
            blank = np.zeros((240,640,3),np.uint8)
            cv2.putText(blank, "SEARCHING CAMERA...", (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', blank)[1].tobytes() + b'\r\n')
            time.sleep(1); continue

        # FIX 1: Minta resolusi LEBAR (640x480) ke driver agar tidak di-crop
        cap.set(3, 640); cap.set(4, 480)

        while True:
            success, raw_frame = cap.read()
            if not success: cap.release(); break

            # FIX 2: Resize manual ke 320x240 (Ringan & Sudut Pandang Luas)
            frame = cv2.resize(raw_frame, (320, 240))

            h, w, _ = frame.shape
            roi = frame[int(h/2):h, 0:w] # ROI Setengah Bawah
            
            # AI VISION
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # STEP 1: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This equalizes light so dark areas (shadows) and bright areas (glare) are leveled out.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # STEP 2: Heavy Blur to reduce noise
            blur = cv2.GaussianBlur(gray, (9, 9), 0)

            # STEP 3: Adaptive Thresholding (Better than Global Threshold)
            # Instead of one number (80), it calculates the threshold for every small region.
            if IS_BLACK_LINE:
                # THRESH_BINARY_INV because we want the Black Line to become White (255) in the mask
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                             cv2.THRESH_BINARY_INV, 21, 5)
            else:
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                             cv2.THRESH_BINARY, 21, 5)

            # STEP 4: MORPHOLOGICAL CLOSING (The Glare Fix)
            # This connects broken parts of the line caused by reflection
            kernel = np.ones((15, 15), np.uint8) 
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # OPTIONAL: Erode slightly to remove random noise dots on the floor
            thresh = cv2.erode(thresh, np.ones((3,3), np.uint8), iterations=1)

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            best_cnt = None; max_area = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 1000: continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if (float(bh)/bw) > 0.8: 
                    if area > max_area: max_area = area; best_cnt = cnt
            
            line_detected = False
            command_vx = 0.0; command_vy = 0.0
            
            # FIX 3: TARGET OFFSET (Geser Garis Biru untuk lawan Drift)
            target_x = int(w/2) + CENTER_OFFSET
            cv2.line(frame, (target_x, int(h/2)), (target_x, h), (255, 0, 0), 1) # Garis Biru (Target)

            if best_cnt is not None:
                line_detected = True
                last_line_time = time.time()
                M = cv2.moments(best_cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    real_cy = int(h/2) + int(M['m01'] / M['m00'])
                    
                    cv2.drawContours(roi, [best_cnt], -1, (0,255,255), 2)
                    cv2.line(frame, (target_x, real_cy), (cx, real_cy), (0,0,255), 2) # Garis Merah (Error)
                    
                    # LOGIKA GERAK
                    # Error = Posisi Garis (CX) - Posisi Target (Target X)
                    # Jika Garis di KIRI (misal 100) dan Target di TENGAH (160)
                    # Error = 100 - 160 = -60
                    # Drone harus ke KIRI (Vy Negatif).
                    # Jadi Rumus: Vy = Error * Gain
                    error_x = cx - target_x 
                    
                    command_vx = FORWARD_SPEED
                    command_vy = max(min((error_x * GAIN_YAW), 0.2), -0.2)

            # QR CODE
            qr_objects = decode(frame)
            for obj in qr_objects:
                qr_text = obj.data.decode("utf-8")
                qr_data = qr_text 
                pts = np.array([obj.polygon], np.int32)
                cv2.polylines(frame, [pts], True, (255,0,255), 2)
                cv2.putText(frame, qr_text, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

            # SPLIT SCREEN
            thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            right_panel = np.zeros_like(frame)
            right_panel[int(h/2):h, 0:w] = thresh_color
            cv2.putText(right_panel, "AI VISION", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            combined = cv2.hconcat([frame, right_panel])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', combined)[1].tobytes() + b'\r\n')

# --- WEB UI ---
@app.route('/status')
def status(): return jsonify(web_data)

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>KING PHOENIX V4</title>
    <style>
        body { background-color: #111; color: #eee; font-family: monospace; margin: 0; padding: 10px; }
        .grid { display: grid; grid-template-columns: 2.5fr 1fr; gap: 10px; }
        .video-box img { width: 100%; border: 2px solid #444; }
        .panel { background: #222; padding: 10px; border: 1px solid #444; }
        .stat-row { display: flex; justify-content: space-between; border-bottom: 1px solid #333; padding: 5px 0; }
        .val { font-weight: bold; color: #0f0; }
        .console { height: 150px; background: #000; color: #0f0; overflow-y: auto; padding: 5px; border: 1px solid #333; margin-top: 10px;}
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <h2 style="margin:0; color:#00d2ff;">KING PHOENIX V4</h2>
        <span id="badge" style="background:#333; padding:5px 10px; font-weight:bold;">DISARMED</span>
    </div>
    <div class="grid">
        <div class="video-box"><img src="/video_feed"></div>
        <div class="panel">
            <div class="stat-row"><span>MODE</span><span class="val" id="mode">--</span></div>
            <div class="stat-row"><span>BATTERY</span><span class="val" id="bat">--</span></div>
            <div class="stat-row"><span>ALTITUDE</span><span class="val" id="alt">--</span></div>
            <div class="stat-row"><span>T265</span><span class="val" id="t265">--</span></div>
            <div class="stat-row"><span>STATE</span><span class="val" id="state" style="color:yellow">--</span></div>
            <div class="console" id="log">System Ready.</div>
        </div>
    </div>
    <script>
        setInterval(() => {
            fetch('/status').then(r=>r.json()).then(d=>{
                document.getElementById('mode').innerText = d.mode;
                document.getElementById('bat').innerText = d.bat.toFixed(1) + "V";
                document.getElementById('alt').innerText = d.alt.toFixed(2) + "m";
                document.getElementById('t265').innerText = d.t265;
                document.getElementById('state').innerText = d.state;
                let b = document.getElementById('badge');
                if(d.armed){ b.innerText="ARMED"; b.style.background="red"; }
                else{ b.innerText="DISARMED"; b.style.background="#333"; }
                let l = document.getElementById('log');
                if(l.lastChild.innerText !== "> "+d.msg){
                    let n = document.createElement('div'); n.innerText="> "+d.msg;
                    l.appendChild(n); l.scrollTop=l.scrollHeight;
                }
            });
        }, 500);
    </script>
</body></html>
    ''')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def run_flask(): app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True)

# --- MAIN ---
parser = argparse.ArgumentParser()
parser.add_argument('--connect', default=DEFAULT_CONNECT) 
parser.add_argument('--baudrate', type=int, default=DEFAULT_BAUD)
args = parser.parse_args()

progress("BOOTING V4...")
os.system("pkill -f mavros") # Matikan mavros otomatis

try:
    conn = mavutil.mavlink_connection(args.connect, autoreconnect=True, baud=args.baudrate)
except Exception as e:
    progress(f"CONN ERROR: {e}"); sys.exit(1)

threading.Thread(target=mavlink_loop, args=(conn, {'HEARTBEAT': heartbeat_cb, 'STATUSTEXT': statustext_cb, 'SYS_STATUS': sys_status_cb})).start()

try: pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.pose); pipe.start(cfg)
except: progress("T265 ERROR!"); sys.exit(1)

threading.Thread(target=run_flask, daemon=True).start()
sched = BackgroundScheduler()
sched.add_job(send_vision_msg, 'interval', seconds=1/30.0)
sched.add_job(send_vel_cmd, 'interval', seconds=1/10.0)
sched.start()
threading.Thread(target=mission_logic_thread, daemon=True).start()

progress("SYSTEM READY. Buka Browser.")

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
                H_265[0][3] = data.translation.x * scale_factor
                H_265[1][3] = data.translation.y * scale_factor
                H_265[2][3] = data.translation.z * scale_factor
                H_aeroRef_aeroBody = H_aeroRef_T265Ref.dot(H_265.dot(H_T265body_aeroBody))
                if prev_data and m.sqrt((data.translation.x-prev_data.translation.x)**2) > 0.1: reset_counter+=1
                prev_data = data
except KeyboardInterrupt: pipe.stop(); mavlink_thread_should_exit=True