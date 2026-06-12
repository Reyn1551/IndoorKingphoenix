"""
Ground Control Station — Flask web app with dual camera feeds.

Routes:
- ``/``              Dashboard (dual cam, target grid, live status)
- ``/video_feed``    Bottom camera MJPEG
- ``/video_feed_front`` Front camera MJPEG
- ``/status``        JSON telemetry (comprehensive)
- ``/set_target``    POST a new grid target
- ``/start_mission`` Start autonomous flight
- ``/stop_mission``  Emergency land
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, render_template, request

if TYPE_CHECKING:
    from king_phoenix.drone import DroneController
    from king_phoenix.front_vision import FrontCamera
    from king_phoenix.mission import MissionController
    from king_phoenix.vision import BottomVision

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = str(Path(__file__).resolve().parent.parent / "templates")


def create_app(
    bottom: BottomVision,
    front: FrontCamera,
    drone: DroneController,
    mission: MissionController,
) -> Flask:
    """Build the Flask GCS application."""

    app = Flask(__name__, template_folder=_TEMPLATE_DIR)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        def gen():
            while True:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + bottom.get_jpeg() + b"\r\n"
                )
                time.sleep(0.1)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/video_feed_front")
    def video_feed_front():
        def gen():
            while True:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + front.get_jpeg() + b"\r\n"
                )
                time.sleep(0.1)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        tm = mission.telemetry.snapshot()
        return jsonify(
            {
                **tm,
                "battery": round(drone.battery, 1),
                "heading": round(drone.heading, 1),
                "flight_mode": drone.flight_mode,
                "line_detected": bottom.line_detected,
                "error_x": round(bottom.error_x, 1),
                "line_angle": round(bottom.line_angle_error, 1),
                "gate_alt": front.gate_altitude_request,
                "victims": front.found_victims,
                "wall_qr": front.wall_qr_center is not None,
                "bottom_cam": bottom.is_active,
                "front_cam": front.is_active,
            }
        )

    @app.route("/set_target", methods=["POST"])
    def set_target_api():
        t = request.json.get("target", "")
        if mission.set_target(t):
            return jsonify({"status": "ok", "msg": f"Target: {t}"})
        return jsonify({"status": "error", "msg": "Invalid QR"})

    @app.route("/start_mission")
    def start_mission():
        mission.start_mission()
        return jsonify({"status": "ok"})

    @app.route("/stop_mission")
    def stop_mission():
        mission.stop_mission()
        return jsonify({"status": "ok"})

    return app
