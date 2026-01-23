#!/usr/bin/env python3

import sys, os, time, threading, math as m
import numpy as np
import pyrealsense2 as rs
import transformations as tf
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil

# --- CONFIG ---
mavlink_connect = "udpin:0.0.0.0:14550"
takeoff_alt = 1.0  # Increased to 1.0m (0.5m is sometimes rejected)
yaw_speed = 30     # degrees per second

# --- GLOBALS ---
conn = None
is_turning = False 
mav_lock = threading.Lock()
H_aeroRef_aeroBody = None
reset_counter = 1
current_time_us = 0
should_quit = False

# Vision & State variables
current_alt = 0.0
is_armed = False
current_mode = "UNKNOWN"

# ==========================================
# 1. HELPERS
# ==========================================
def progress(s): print(f">> {s}")

def heartbeat_listener():
    global is_armed, current_mode, current_alt, should_quit
    while not should_quit:
        msg = conn.recv_match(type=['HEARTBEAT', 'GLOBAL_POSITION_INT'], blocking=True, timeout=1)
        if not msg: continue
        
        if msg.get_type() == 'HEARTBEAT':
            # Decode Mode
            mode_map = conn.mode_mapping()
            for name, id in mode_map.items():
                if msg.custom_mode == id: current_mode = name; break
            # Decode Arming
            is_armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            
        if msg.get_type() == 'GLOBAL_POSITION_INT':
            current_alt = msg.relative_alt / 1000.0 # Convert mm to m

# ==========================================
# 2. T265 & MAVLINK SETUP
# ==========================================
def send_vision_msg():
    global current_time_us, H_aeroRef_aeroBody, reset_counter
    with mav_lock:
        if H_aeroRef_aeroBody is not None:
            rpy = np.array(tf.euler_from_matrix(H_aeroRef_aeroBody, 'sxyz'))
            # Send Vision Data to FCU
            conn.mav.vision_position_estimate_send(current_time_us, 
                H_aeroRef_aeroBody[0][3], H_aeroRef_aeroBody[1][3], H_aeroRef_aeroBody[2][3],
                rpy[0], rpy[1], rpy[2], [0.01]*21, reset_counter)

def t265_thread():
    global H_aeroRef_aeroBody, current_time_us, reset_counter, should_quit
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.pose)
    try:
        pipe.start(cfg)
        progress("T265 STARTED")
    except:
        progress("T265 FAILED")
        return

    H_aeroRef_T265Ref = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
    H_T265body_aeroBody = np.linalg.inv(H_aeroRef_T265Ref)
    prev_data = None

    while not should_quit:
        frames = pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        if pose:
            with mav_lock:
                current_time_us = int(round(time.time() * 1000000))
                data = pose.get_pose_data()
                H_265 = tf.quaternion_matrix([data.rotation.w, data.rotation.x, data.rotation.y, data.rotation.z])
                H_265[0][3] = data.translation.x
                H_265[1][3] = data.translation.y
                H_265[2][3] = data.translation.z
                H_aeroRef_aeroBody = H_aeroRef_T265Ref.dot(H_265.dot(H_T265body_aeroBody))
                if prev_data and m.sqrt((data.translation.x-prev_data.translation.x)**2) > 0.1: reset_counter+=1
                prev_data = data

# ==========================================
# 3. VELOCITY CONTROLLER
# ==========================================
def send_vel_cmd():
    global is_turning
    if is_turning: return 
    # Hold position (Brake)
    conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, 
        mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0, 0, 0, 0,0,0, 0,0)

# ==========================================
# 4. ROBUST MISSION LOGIC
# ==========================================
def perform_yaw_test(angle, direction):
    global is_turning
    dir_str = "RIGHT" if direction == 1 else "LEFT"
    progress(f"TESTING YAW: {angle} deg {dir_str}")
    
    is_turning = True 
    # Send Command 3 times
    for i in range(3):
        conn.mav.command_long_send(conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 
            angle, yaw_speed, direction, 1, 0, 0, 0)
        time.sleep(0.1)
    
    wait_time = (angle / yaw_speed) + 2.0
    time.sleep(wait_time)
    progress("TURN COMPLETE.")
    is_turning = False

def run_mission():
    global is_turning
    
    # 1. WARMUP
    progress("WAITING FOR T265 DATA...")
    time.sleep(2)
    
    # 2. SET MODE GUIDED
    progress("SWITCHING TO GUIDED...")
    while current_mode != "GUIDED":
        conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()['GUIDED'])
        time.sleep(0.5)
    progress("MODE: GUIDED")

    # 3. ARMING
    progress("ARMING...")
    conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    
    # Wait until actually armed
    timeout = 0
    while not is_armed:
        time.sleep(0.5)
        conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        timeout += 1
        if timeout > 10: progress("RETRY ARMING..."); timeout = 0
    progress("MOTORS ARMED!")
    
    time.sleep(2) # Give motors time to spin up

    # 4. TAKEOFF
    progress(f"TAKEOFF TO {takeoff_alt}m...")
    conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, takeoff_alt)
    
    # Wait until altitude reached
    while current_alt < (takeoff_alt - 0.2):
        print(f"Altitide: {current_alt:.2f}m")
        # Retry takeoff command if it's stuck on ground
        if current_alt < 0.2:
            conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, takeoff_alt)
        time.sleep(1)
    
    progress("TARGET ALTITUDE REACHED.")
    time.sleep(2) 
    
    # --- YAW TESTS ---
    perform_yaw_test(90, 1) # RIGHT
    time.sleep(1)
    perform_yaw_test(90, -1) # LEFT (Return to center)
    time.sleep(1)
    perform_yaw_test(180, 1) # 180 SPIN
    time.sleep(1)

    progress("LANDING...")
    conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()['LAND'])

# ==========================================
# STARTUP
# ==========================================
if __name__ == '__main__':
    progress("CONNECTING TO DRONE...")
    conn = mavutil.mavlink_connection(mavlink_connect)
    conn.wait_heartbeat()
    progress("CONNECTED")

    # Listener for Arm/Mode status
    t_status = threading.Thread(target=heartbeat_listener)
    t_status.daemon = True
    t_status.start()

    t_t265 = threading.Thread(target=t265_thread)
    t_t265.daemon = True
    t_t265.start()

    sched = BackgroundScheduler()
    sched.add_job(send_vision_msg, 'interval', seconds=1/30.0)
    sched.add_job(send_vel_cmd, 'interval', seconds=1/10.0) 
    sched.start()

    try:
        run_mission()
    except KeyboardInterrupt:
        progress("LANDING (USER INTERRUPT)")
        conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()['LAND'])
    finally:
        should_quit = True