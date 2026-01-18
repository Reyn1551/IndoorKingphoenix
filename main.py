import time
import threading
import signal
import sys
from flask import Flask, Response, jsonify, render_template_string, request

# Import our modular blocks
from config import ConfigManager
from vision import VisionSystem
from drone import DroneController
from t265 import T265Handler

# ================= SETUP =================
cfg_mgr = ConfigManager()
vision = VisionSystem(cfg_mgr)
drone = DroneController(cfg_mgr)
t265 = T265Handler(cfg_mgr, drone)
app = Flask(__name__)

# Global Mission State
mission_running = False
mission_state = "IDLE"

# ================= FLASK ROUTES =================
@app.route('/')
def index():
    # A simple HTML dashboard
    return render_template_string("""
    <html>
    <head>
        <title>King Phoenix Indoor GCS</title>
        <meta http-equiv="refresh" content="1"> <style>
            body { background: #111; color: #0f0; font-family: monospace; padding: 20px; }
            .btn { padding: 10px 20px; font-size: 20px; cursor: pointer; }
            .red { background: red; color: white; }
            .green { background: green; color: white; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            img { width: 100%; border: 2px solid #333; }
        </style>
    </head>
    <body>
        <h1>KING PHOENIX INDOOR GCS (MODULAR)</h1>
        <div class="grid">
            <div>
                <img src="/video_feed">
            </div>
            <div>
                <h2>Status: {{ state }}</h2>
                <p>Alt: {{ alt }}m | Bat: {{ bat }}V | Mode: {{ mode }}</p>
                <button class="btn green" onclick="fetch('/start')">START MISSION</button>
                <button class="btn red" onclick="fetch('/stop')">EMERGENCY STOP</button>
                <br><br>
                <h3>Vision: {{ line }} (Err: {{ err }})</h3>
            </div>
        </div>
    </body>
    </html>
    """, state=mission_state, alt=f"{drone.altitude:.2f}", bat=f"{drone.battery:.1f}", 
       mode=drone.flight_mode, line="LOCKED" if vision.line_detected else "SEARCHING", err=vision.error_x)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + vision.get_jpeg() + b'\r\n')
            time.sleep(0.04) # Limit to 25 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_mission():
    global mission_running, mission_state
    if not drone.armed:
        mission_running = True
        mission_state = "TAKEOFF"
        return jsonify({"msg": "Mission Started"})
    return jsonify({"msg": "Already Armed!"})

@app.route('/stop')
def stop_mission():
    global mission_running, mission_state
    mission_running = False
    mission_state = "ABORTED"
    drone.land()
    return jsonify({"msg": "Emergency Landing!"})

# ================= MISSION LOGIC =================
def mission_loop():
    global mission_state, mission_running
    
    print("[MAIN] Mission Loop Started")
    drone.connect()
    t265.start()
    
    while True:
        # 1. Update Vision
        vision.update()
        
        # 2. Update Telemetry
        drone.update_telemetry()
        
        # 3. Run State Machine
        if mission_running:
            
            if mission_state == "TAKEOFF":
                drone.set_mode("GUIDED")
                drone.arm()
                drone.takeoff()
                if drone.altitude >= cfg_mgr.data["flight"]["takeoff_alt"] - 0.2:
                    mission_state = "OUTBOUND"
            
            elif mission_state == "OUTBOUND":
                # Calculate PID
                if vision.line_detected:
                    vy = drone.compute_pid(vision.error_x)
                    vx = cfg_mgr.data["flight"]["forward_speed"]
                    drone.send_velocity(vx, vy, 0)
                else:
                    # Line lost recovery (simple sweep)
                    drone.send_velocity(0, 0.1 * drone.last_valid_direction, 0)
                
                # Check for QR Code to turn around
                if "TURN" in vision.qr_data: # Example keyword
                    mission_state = "RETURN"

            elif mission_state == "RETURN":
                # Similar logic but maybe different speed or height
                pass
                
        time.sleep(0.05) # 20Hz Control Loop

# ================= EXECUTION =================
if __name__ == '__main__':
    # Start the mission logic in a background thread
    logic_thread = threading.Thread(target=mission_loop, daemon=True)
    logic_thread.start()
    
    # Start Vision Camera
    if not vision.start_camera():
        print("[MAIN] WARNING: Camera not found, running in blind mode.")

    # Start Flask Web Server
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[MAIN] CTRL+C DETECTED: EMERGENCY SHUTDOWN")
    finally:
        print("[MAIN] Cleaning up...")
        t265.stop()
        if drone.armed:
            print("[MAIN] SENDING EMERGENCY LAND COMMAND...")
            drone.land() 
            time.sleep(1)
            
        print("[MAIN] System Halted.")