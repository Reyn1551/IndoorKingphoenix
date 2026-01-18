import json
import os

class ConfigManager:
    def __init__(self, filename="config.json"):
        self.filename = filename
        # 1. Load hardcoded defaults first (Safety net)
        self.data = self._get_defaults()
        # 2. Try to override with file data
        self.load()

    def _get_defaults(self):
        """
        Returns the hardcoded default settings.
        These are used if config.json is missing or corrupted.
        """
        return {
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

    def load(self):
        """
        Loads settings from JSON file. 
        Uses a 'Section Update' strategy to preserve defaults if keys are missing.
        """
        if not os.path.exists(self.filename):
            print(f"[CONFIG] File {self.filename} not found. Using Defaults.")
            return

        try:
            with open(self.filename, 'r') as f:
                saved_conf = json.load(f)
                
                # We iterate through sections (flight, control, etc.)
                # This ensures that if the file has extra junk, we ignore it,
                # and if the file is missing a key, we keep the default.
                for section, settings in saved_conf.items():
                    if section in self.data:
                        self.data[section].update(settings)
                        
            print(f"[CONFIG] Loaded successfully from {self.filename}")
        except Exception as e:
            print(f"[CONFIG] Error loading file: {e}. Reverting to Defaults.")
            # We do not overwrite self.data here, so it stays as Defaults

    def save(self, new_config=None):
        """
        Saves the current configuration to the JSON file.
        Returns True if successful, False otherwise.
        """
        if new_config:
            self.data = new_config

        try:
            with open(self.filename, 'w') as f:
                json.dump(self.data, f, indent=4)
            print("[CONFIG] Saved to disk.")
            return True
        except Exception as e:
            print(f"[CONFIG] Save failed: {e}")
            return False

    def get(self, section, key):
        """Helper to safely get a value without crashing if it doesn't exist"""
        return self.data.get(section, {}).get(key)