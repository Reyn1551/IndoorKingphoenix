"""
Configuration management with JSON file persistence.

Loads hardcoded defaults first, then overlays user settings from config.json.
Any key missing in the JSON file is seamlessly inherited from defaults.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded defaults (safety net when config.json is missing or corrupt)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "flight": {
        "takeoff_alt": 0.5,
        "forward_speed": 0.1,
        "max_lat_vel": 0.3,
        "yaw_speed": 30,
        "pre_turn_delay": 2.0,
        "post_turn_delay": 2.0,
        "blind_fwd_time": 2.5,
        "alt_source": "T265",
    },
    "control": {
        "pid_kp": 0.015,
        "pid_ki": 0.0001,
        "pid_kd": 0.005,
        "search_vel": 0.08,
        "search_fwd_vel": 0.05,
        "yaw_kp": 0.015,
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
        "qr_confirm_count": 1,
        "gates": {
            "red_lower1": [0, 148, 102],
            "red_upper1": [179, 255, 255],
            "red_lower2": [170, 120, 70],
            "red_upper2": [180, 255, 255],
            "yellow_lower": [20, 100, 100],
            "yellow_upper": [30, 255, 255],
            "min_gate_area": 10000,
            "gate_hold_alt": 0.5,
            "gate_avoid_alt": 1.3,
        },
    },
    "t265": {
        "scale_factor": 1.0,
        "confidence_threshold": 3,
        "ignore_quality": False,
    },
    "system": {
        "http_port": 5000,
        "mavlink_connect": "udpin:0.0.0.0:14550",
        "start_heading": "NORTH",
        "bottom_cam_index": 4,
        "front_cam_index": 6,
    },
}


class ConfigManager:
    """Manages application configuration backed by a JSON file.

    Usage::

        cfg = ConfigManager("config.json")
        value = cfg.get("vision", "threshold_val")  # safe access
        cfg.data["flight"]["forward_speed"] = 0.15   # direct mutation
        cfg.save()
    """

    def __init__(self, filename: str = "config.json") -> None:
        self.filename = filename
        self.data: Dict[str, Any] = {}
        self._load_defaults()
        self.load()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def load(self) -> bool:
        """Load settings from the JSON file.

        Uses a *section-update* strategy: only sections present in the
        file are merged, preserving defaults for any missing keys.
        """
        if not os.path.exists(self.filename):
            logger.warning(
                "Config file '%s' not found — using defaults.", self.filename
            )
            return False

        try:
            with open(self.filename, "r") as fh:
                saved = json.load(fh)

            for section, settings in saved.items():
                if section in self.data:
                    self.data[section].update(settings)

            logger.info("Configuration loaded from '%s'.", self.filename)
            return True
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load config: %s — using defaults.", exc)
            return False

    def save(self, new_config: Optional[Dict[str, Any]] = None) -> bool:
        """Persist the current (or supplied) configuration to disk."""
        if new_config is not None:
            self.data = new_config

        try:
            with open(self.filename, "w") as fh:
                json.dump(self.data, fh, indent=4)
            logger.info("Configuration saved to '%s'.", self.filename)
            return True
        except OSError as exc:
            logger.error("Failed to save config: %s", exc)
            return False

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Safely retrieve a nested configuration value."""
        return self.data.get(section, {}).get(key, default)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _load_defaults(self) -> None:
        """Populate ``self.data`` from hardcoded defaults (deep copy)."""
        self.data = copy.deepcopy(_DEFAULT_CONFIG)
