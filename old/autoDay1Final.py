#!/usr/bin/env python3

#####################################################
##    FULL AUTO: FILTERED LINE + AI HUD PIP        ##
##    FIXED VERSION (Scale Factor Added)           ##
#####################################################

import sys, os, time, threading, math as m, argparse
import numpy as np
import cv2
from flask import Flask, Response, render_template_string

os.environ["MAVLINK20"] = "1"
import pyrealsense2 as rs
import transformations as tf
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil

# --- KONFIGURASI FILTER GARIS (PENTING) ---
MIN_AREA = 1000        # Abaikan objek kecil (bintik/bayangan)
MIN_ASPECT_RATIO = 0.8 # Rasio Panjang/Lebar. (Saya turunkan jadi 0.8 biar lebih toleran)

# --- KONFIGURASI DRONE ---
HTTP_PORT = 5000       
FORWARD_SPEED = 0.1    
GAIN_YAW = 0.005       
LOST_LINE_TIMEOUT = 2.0 
TAKEOFF_ALT = 0.6      
IS_BLACK_LINE = True   

# --- KONFIGURASI T265 ---
scale_factor = 1.0     # <--- INI YANG TADI ERROR (Sudah ditambahkan)

# --- GLOBAL VARS ---
command_vx = 0.0
command_vy = 0.0
line_detected = False
last_line_time = time.time()
landing_triggered = False
mission_started = False
drone_mode = "UNKNOWN"
drone_armed = False
last_status_text = "Booting..."
battery_voltage = 0.0
current_alt = 0.0
t265_confidence = 0

active_cam_index = None
app = Flask(__name__)
lock = threading.Lock()
mavlink_thread_should_exit = False

# T265 Data
pipe, data, prev_data, H_aeroRef_aeroBody = None, None, None, None
reset_counter = 1
current_time_us = 0

def progress(s): print(s, file=sys.stdout); sys.stdout.flush()

# --- FUNGSI KAMERA ---
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

# --- MAVLINK ---
def mavlink_loop(conn, callbacks):
    interesting = list(callbacks.keys())
    while not mavlink_thread_should_exit:
        conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0)
        msg = conn.recv_match(type=interesting, timeout=1, blocking=True)
        if msg: callbacks[msg.get_type()](msg)

def heartbeat_cb(msg):
    global drone_mode, drone_armed
    mode_map = conn.mode_mapping()
    for name, id in mode_map.items():
        if msg.custom_mode == id:
            drone_mode = name; break
    drone_armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

def statustext_cb(msg): global last_status_text; last_status_text = msg.text.upper()
def sys_status_cb(msg): global battery_voltage; battery_voltage = msg.voltage_battery / 1000.0

def send_vision_msg():
    global current_time_us, H_aeroRef_aeroBody, reset_counter, data
    with lock:
        if H_aeroRef_aeroBody is not None and data is not None:
            rpy = np.array(tf.euler_from_matrix(H_aeroRef_aeroBody, 'sxyz'))
            cov_p = 0.01 * pow(10, 3 - int(data.tracker_confidence))
            cov_t = 0.01 * pow(10, 1 - int(data.tracker_confidence))
            cov = [cov_p, 0, 0, 0, 0, 0, cov_p, 0, 0, 0, 0, cov_p, 0, 0, 0, cov_t, 0, 0, cov_t, 0, cov_t]
            conn.mav.vision_position_estimate_send(current_time_us, 
                H_aeroRef_aeroBody[0][3], H_aeroRef_aeroBody[1][3], H_aeroRef_aeroBody[2][3],
                rpy[0], rpy[1], rpy[2], cov, reset_counter)

def set_mode(mode):
    conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()[mode])

def send_vel_cmd():
    global command_vx, command_vy, line_detected, landing_triggered, drone_mode, mission_started
    if not mission_started: return
    
    if (not line_detected) and (time.time() - last_line_time > LOST_LINE_TIMEOUT) and (drone_mode == "GUIDED") and (not landing_triggered):
        progress("!!! LINE LOST - LANDING !!!")
        set_mode("LAND"); landing_triggered = True; return

    if drone_mode == "LAND": return
    vx = command_vx if line_detected else 0.0
    vy = command_vy if line_detected else 0.0
    conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, vx, vy, 0, 0,0,0, 0,0)

# --- MISSION LOGIC (AUTO TAKEOFF) ---
def mission_logic():
    global mission_started, last_status_text, current_alt, t265_confidence
    progress("MISSION: Waiting for T265..."); last_status_text = "WAIT T265..."
    while t265_confidence < 1: time.sleep(1)
    
    progress("MISSION: T265 READY. Wait EKF Sync..."); last_status_text = "T265 OK. SYNC..."
    time.sleep(5)

    progress("MISSION: GUIDED & ARM..."); last_status_text = "ARMING..."
    set_mode("GUIDED"); time.sleep(1)
    conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    
    while not drone_armed: time.sleep(0.5)
    progress("MISSION: TAKEOFF..."); last_status_text = "TAKING OFF..."
    conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, TAKEOFF_ALT)
    
    while current_alt > -0.90: time.sleep(0.5) # Wait altitude
    progress("MISSION: START TRACKING"); last_status_text = "AUTO: LINE FOLLOW"
    mission_started = True

# --- VISION PROCESSING (FILTERED) ---
def gen_frames():
    global command_vx, command_vy, line_detected, last_line_time
    cap = find_working_camera()
    if not cap:
        while True: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((240,320,3),np.uint8))[1].tobytes() + b'\r\n'); time.sleep(1)
    
    cap.set(3, 320); cap.set(4, 240)
    
    while True:
        success, frame = cap.read()
        if not success: cap.release(); cap = find_working_camera(); continue
        
        h, w, _ = frame.shape
        roi = frame[int(h/2):h, 0:w] # ROI Setengah Bawah
        
        # 1. Pre-Processing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if IS_BLACK_LINE:
            _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)
            
        # 2. Contour Filtering (CARI GARIS SAJA)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA: continue 
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bh) / bw 
            
            if aspect_ratio > MIN_ASPECT_RATIO:
                if area > max_area:
                    max_area = area
                    best_cnt = cnt
        
        line_detected = False
        command_vx = 0.0; command_vy = 0.0
        cx_scr = int(w/2)
        cv2.line(frame, (cx_scr, int(h/2)), (cx_scr, h), (255, 0, 0), 1)

        # Jika garis valid ditemukan
        if best_cnt is not None:
            line_detected = True
            last_line_time = time.time()
            
            M = cv2.moments(best_cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                real_cy = int(h/2) + int(M['m01'] / M['m00']) 
                
                # Logic Gerak
                error_x = cx_scr - cx
                command_vx = FORWARD_SPEED
                command_vy = (error_x * GAIN_YAW)
                command_vy = max(min(command_vy, 0.2), -0.2)
                
                # Visualisasi
                cv2.drawContours(roi, [best_cnt], -1, (0,255,255), 2)
                cv2.circle(frame, (cx, real_cy), 5, (0, 255, 0), -1)
                cv2.line(frame, (cx_scr, real_cy), (cx, real_cy), (0, 0, 255), 2)

        # --- PIP (PICTURE IN PICTURE) AI VISION ---
        thumb_w = 100
        thumb_h = int(thumb_w * (roi.shape[0] / roi.shape[1]))
        thumb_thresh = cv2.resize(thresh, (thumb_w, thumb_h))
        thumb_color = cv2.cvtColor(thumb_thresh, cv2.COLOR_GRAY2BGR)
        
        cv2.rectangle(thumb_color, (0,0), (thumb_w, thumb_h), (0,0,255), 1)
        y_offset = h - thumb_h - 10
        x_offset = w - thumb_w - 10
        frame[y_offset:y_offset+thumb_h, x_offset:x_offset+thumb_w] = thumb_color
        cv2.putText(frame, "AI VIEW", (x_offset, y_offset-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        # --- HUD TEXT ---
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        mode_color = (0, 255, 0) if drone_armed else (0, 255, 255)
        cv2.putText(frame, f"{drone_mode}", (w-100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1)
        cv2.putText(frame, f"Alt: {-current_alt:.1f}m", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        vis_txt = "WAITING TAKEOFF"
        if mission_started: vis_txt = "TRACKING LINE" if line_detected else "SEARCHING"
        if drone_mode == "LAND": vis_txt = "LANDING"
        cv2.putText(frame, vis_txt, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if line_detected else (0,0,255), 1)
        
        cv2.rectangle(frame, (0, h-25), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"MSG: {last_status_text}", (5, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template_string('<html><body style="background:#222; color:white; text-align:center;"><h2>AUTO MISSION v2 (FILTERED)</h2><img src="/video_feed" style="border:2px solid #555;width:90%"></body></html>')
@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def run_flask(): app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True)

# --- MAIN ---
parser = argparse.ArgumentParser(); parser.add_argument('--connect', default='udp:127.0.0.1:14550')
args = parser.parse_args()

progress("INIT SYSTEM...")
conn = mavutil.mavlink_connection(args.connect, autoreconnect=True, baud=57600)
threading.Thread(target=mavlink_loop, args=(conn, {'HEARTBEAT': heartbeat_cb, 'STATUSTEXT': statustext_cb, 'SYS_STATUS': sys_status_cb})).start()

try: pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.pose); pipe.start(cfg)
except: progress("T265 FAIL"); sys.exit(1)

threading.Thread(target=run_flask, daemon=True).start()
sched = BackgroundScheduler()
sched.add_job(send_vision_msg, 'interval', seconds=1/30.0)
sched.add_job(send_vel_cmd, 'interval', seconds=1/10.0)
sched.start()
threading.Thread(target=mission_logic, daemon=True).start()

progress("READY. Auto Takeoff Active.")
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
                t265_confidence = data.tracker_confidence
                H_265 = tf.quaternion_matrix([data.rotation.w, data.rotation.x, data.rotation.y, data.rotation.z])
                H_265[0][3] = data.translation.x * scale_factor
                H_265[1][3] = data.translation.y * scale_factor
                H_265[2][3] = data.translation.z * scale_factor
                H_aeroRef_aeroBody = H_aeroRef_T265Ref.dot(H_265.dot(H_T265body_aeroBody))
                current_alt = H_aeroRef_aeroBody[2][3]
                if prev_data and m.sqrt((data.translation.x-prev_data.translation.x)**2) > 0.1: reset_counter+=1
                prev_data = data
except KeyboardInterrupt: pipe.stop(); mavlink_thread_should_exit=True