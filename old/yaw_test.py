#!/usr/bin/env python3

import sys, os, time, threading, math as m
import numpy as np
import pyrealsense2 as rs
import transformations as tf
from apscheduler.schedulers.background import BackgroundScheduler
from pymavlink import mavutil

# --- CONFIG ---
mavlink_connect = "udpin:0.0.0.0:14550"
takeoff_alt = 0.5
yaw_speed = 30 # degrees per second

# --- GLOBALS ---
conn = None
is_turning = False # THE MAGIC FLAG
mav_lock = threading.Lock()
H_aeroRef_aeroBody = None
reset_counter = 1
current_time_us = 0
should_quit = False

# ==========================================
# 1. T265 & MAVLINK SETUP (Standard)
# ==========================================
def progress(s): print(f">> {s}")

def send_vision_msg():
    global current_time_us, H_aeroRef_aeroBody, reset_counter
    with mav_lock:
        if H_aeroRef_aeroBody is not None:
            rpy = np.array(tf.euler_from_matrix(H_aeroRef_aeroBody, 'sxyz'))
            # Send Vision Data to FCU so it knows where it is
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
# 2. THE CRITICAL VELOCITY CONTROLLER
# ==========================================
def send_vel_cmd():
    global is_turning
    
    # [THE FIX]
    # If we are turning, DO NOT send commands.
    # If we send velocity=0 while turning, the drone STOPS turning immediately.
    if is_turning:
        # print("Silent for Turn...") 
        return 

    # Otherwise, hold position (brake)
    conn.mav.set_position_target_local_ned_send(0, conn.target_system, conn.target_component, 
        mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111, 0,0,0, 0, 0, 0, 0,0,0, 0,0)

# ==========================================
# 3. MAIN TEST SEQUENCE
# ==========================================
def perform_yaw_test(angle, direction):
    global is_turning
    
    dir_str = "RIGHT" if direction == 1 else "LEFT"
    progress(f"TESTING YAW: {angle} deg {dir_str}")
    
    # 1. Engage Silence
    is_turning = True 
    
    # 2. Send Command (Repeat 3 times to ensure delivery)
    for i in range(3):
        conn.mav.command_long_send(conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, 
            angle,    # param1: target angle (0-360) or relative
            yaw_speed,# param2: speed deg/s
            direction,# param3: -1=CCW, 1=CW
            1,        # param4: 1=relative offset, 0=absolute
            0, 0, 0)
        time.sleep(0.05)
    
    # 3. Calculate Wait Time (Angle / Speed) + Buffer
    wait_time = (angle / yaw_speed) + 2.0
    time.sleep(wait_time)
    
    # 4. Resume Position Hold
    progress("TURN COMPLETE. HOLDING.")
    is_turning = False

def run_mission():
    global is_turning
    
    # Wait for T265 to warm up
    time.sleep(2)
    
    # Arm and Takeoff
    conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()['GUIDED'])
    conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    progress("ARMING...")
    time.sleep(2)
    conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, takeoff_alt)
    progress("TAKEOFF...")
    time.sleep(6) # Wait for ascent
    
    # --- TEST 1: TURN RIGHT 90 ---
    perform_yaw_test(90, 1) # 1 = CW (Right)
    time.sleep(2)
    
    # --- TEST 2: TURN LEFT 90 (Return to center) ---
    perform_yaw_test(90, -1) # -1 = CCW (Left)
    time.sleep(2)

    # --- TEST 3: 180 TURN ---
    perform_yaw_test(180, 1)
    time.sleep(2)

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

    # Start T265 Thread
    t = threading.Thread(target=t265_thread)
    t.daemon = True
    t.start()

    # Start Scheduler (Simulates the conflict source)
    sched = BackgroundScheduler()
    sched.add_job(send_vision_msg, 'interval', seconds=1/30.0)
    sched.add_job(send_vel_cmd, 'interval', seconds=1/10.0) # 10Hz Velocity Command
    sched.start()

    try:
        run_mission()
    except KeyboardInterrupt:
        progress("LANDING (USER INTERRUPT)")
        conn.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, conn.mode_mapping()['LAND'])
    finally:
        should_quit = True