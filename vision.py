import cv2
import numpy as np
from pyzbar.pyzbar import decode
import time

class VisionSystem:
    def __init__(self, config_manager):
        # We pass the entire config manager so we can read live updates
        self.cfg_mgr = config_manager
        
        # State Variables
        self.cap = None
        self.frame = None          # Raw frame
        self.display_frame = None  # Frame with drawings (for Web)
        
        # Output Data (The "Result" of vision)
        self.line_detected = False
        self.error_x = 0           # Distance from center (pixels)
        self.qr_data = ""
        self.last_qr_time = 0

    def start_camera(self):
        """Attempts to open any available camera port (0-9)."""
        if self.cap is not None and self.cap.isOpened():
            return True

        print("[VISION] Searching for camera...")
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    # Force low resolution for performance
                    self.cap.set(3, 320)
                    self.cap.set(4, 240)
                    print(f"[VISION] Camera found at index {index}")
                    return True
                cap.release()
        
        print("[VISION] CRITICAL: No Camera Found!")
        return False

    def get_jpeg(self):
        """Encodes the display frame to JPEG. Used by Flask."""
        if self.display_frame is None:
            # Return a blank black image if no frame available
            blank = np.zeros((240, 320, 3), np.uint8)
            return cv2.imencode('.jpg', blank)[1].tobytes()
        
        return cv2.imencode('.jpg', self.display_frame)[1].tobytes()

    def update(self):
        """
        Main loop: Reads frame, processes line, updates state.
        Call this function once per loop in your main script.
        """
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret: 
            print("[VISION] Frame read error")
            self.cap.release()
            return

        # 1. Get Settings (Cleanly from config)
        conf = self.cfg_mgr.data["vision"]
        
        h, w, _ = frame.shape
        self.frame = frame

        # 2. ROI (Region of Interest) - Look at the bottom part
        cut_y = int(h * conf["roi_height_ratio"])
        roi = frame[cut_y:h, 0:w]
        
        # 3. Pre-processing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Handle Black vs White line logic
        thresh_type = cv2.THRESH_BINARY_INV if conf["is_black_line"] else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blur, conf["threshold_val"], 255, thresh_type)

        # 4. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        max_score = -999999
        
        # Center of screen (with manual offset adjustment)
        cx_scr = int(w / 2) + conf["camera_offset_x"]
        
        # 5. Select the Best Line
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < conf["min_area"]: continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            if (float(bh) / bw) < conf["min_aspect_ratio"]: continue

            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            
            cx_cnt = int(M['m10'] / M['m00'])
            
            # Score = Size - Distance Penalty
            dist = abs(cx_cnt - cx_scr)
            score = area - (dist * conf["center_priority_weight"])
            
            if score > max_score:
                max_score = score
                best_cnt = cnt

        # 6. Update State based on result
        if best_cnt is not None:
            self.line_detected = True
            M = cv2.moments(best_cnt)
            cx = int(M['m10'] / M['m00'])
            self.error_x = cx - cx_scr
            
            # Draw visual debug on the ROI
            real_cy = cut_y + int(M['m01'] / M['m00'])
            cv2.line(frame, (cx_scr, real_cy), (cx, real_cy), (0, 0, 255), 3)
            cv2.drawContours(roi, [best_cnt], -1, (0, 255, 255), 2)
        else:
            self.line_detected = False
            # Don't reset error_x to 0 immediately! 
            # Keeping the last error helps the drone know which way it lost the line.

        # 7. QR Code Detection
        qr_objects = decode(frame)
        for obj in qr_objects:
            self.qr_data = obj.data.decode("utf-8")
            self.last_qr_time = time.time()
            # Draw QR box
            pts = np.array([obj.polygon], np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
            cv2.putText(frame, self.qr_data, (obj.rect.left, obj.rect.top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # 8. Create the "Display Frame" for the Web
        # Merging the threshold view into the main view for debugging
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        right_panel = np.zeros_like(frame)
        right_panel[cut_y:h, 0:w] = thresh_color
        
        # Helper line
        cv2.line(frame, (cx_scr, 0), (cx_scr, h), (255, 0, 0), 1)
        
        # Stack images side-by-side
        self.display_frame = cv2.hconcat([frame, right_panel])