import pyrealsense2 as rs
import numpy as np
import transformations as tf # You might need to install this: pip install transformations
import time
import threading

class T265Handler:
    def __init__(self, config_manager, drone_controller):
        self.cfg = config_manager
        self.drone = drone_controller
        self.pipe = None
        self.running = False
        
        # Coordinate transformation matrices (T265 -> ArduPilot)
        self.H_aeroRef_T265Ref = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]])
        self.H_T265body_aeroBody = np.linalg.inv(self.H_aeroRef_T265Ref)

    def start(self):
        try:
            self.pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.pose)
            self.pipe.start(cfg)
            self.running = True
            
            # Start the thread
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
            print("[T265] Service Started.")
        except Exception as e:
            print(f"[T265] Error starting: {e}")

    def stop(self):
        self.running = False
        if self.pipe:
            self.pipe.stop()

    def _loop(self):
        while self.running:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=1000)
                pose = frames.get_pose_frame()
                if pose:
                    data = pose.get_pose_data()
                    
                    # 1. Convert T265 Coordinate System to Mavlink (NED)
                    H_t265 = tf.quaternion_matrix([data.rotation.w, data.rotation.x, data.rotation.y, data.rotation.z])
                    H_t265[0][3] = data.translation.x * self.cfg.data["t265"]["scale_factor"]
                    H_t265[1][3] = data.translation.y * self.cfg.data["t265"]["scale_factor"]
                    H_t265[2][3] = data.translation.z * self.cfg.data["t265"]["scale_factor"]
                    
                    H_final = self.H_aeroRef_T265Ref.dot(H_t265.dot(self.H_T265body_aeroBody))
                    
                    # 2. Extract Euler Angles for Mavlink
                    rpy = tf.euler_from_matrix(H_final, 'sxyz')
                    
                    # 3. Send to Pixhawk via DroneController
                    current_time_us = int(round(time.time() * 1000000))
                    
                    # We access the raw connection inside drone_controller
                    if self.drone.conn:
                        self.drone.conn.mav.vision_position_estimate_send(
                            current_time_us,
                            H_final[0][3], H_final[1][3], H_final[2][3],
                            rpy[0], rpy[1], rpy[2],
                            [0.0]*21, 0  # Covariance and Reset counter
                        )
            except Exception as e:
                print(f"[T265] Loop Error: {e}")
                pass