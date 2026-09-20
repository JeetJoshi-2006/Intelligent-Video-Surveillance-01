import copy
import os
from pathlib import Path


class ConfigError(ValueError):
    """Raised when the surveillance service cannot be started safely."""

DEFAULT_CONFIG = {
    "stream": {
        "source": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "buffer_seconds": 5
    },
    "motion": {
        "enabled": True,
        "min_motion_area": 1200,
        "history": 500,
        "var_threshold": 25
    },
    "detector": {
        "model_type": "yolo",
        "model_path": "yolo11n.pt",
        "confidence_threshold": 0.45,
        "target_classes": [0, 1, 2, 3, 5, 7, 24, 26, 28],
        "idle_scan_seconds": 1.0,
        "min_detection_area": 1500,
        "min_person_size": 40,
        "min_person_area": 2000,
    },
    "tracker": {
        "max_disappeared": 30,
        "min_iou": 0.3
    },
    "analytics": {
        "tripwire": {
            "enabled": True,
            "line_start": [100, 360],
            "line_end": [1180, 360],
            "direction": "both"
        },
        "intrusion": {
            "enabled": True,
            "polygon": [[700, 150], [1150, 150], [1150, 600], [700, 600]]
        },
        "loitering": {
            "enabled": True,
            "max_dwell_seconds": 8.0,
            "radius_pixels": 60.0
        }
    },
    "alerts": {
        "macos_banner": True,
        "macos_sound": "Hero",
        "telegram": {
            "enabled": False,
            "token_env": "ISS_TELEGRAM_BOT_TOKEN",
            "chat_id_env": "ISS_TELEGRAM_CHAT_ID"
        }
    },
    "recording": {
        "enabled": True,
        "output_dir": "recordings",
        "post_event_seconds": 8,
        "codec": "mp4v",
        "max_pending_incidents": 1,
        "max_age_days": 14,
        "max_total_mb": 2048
    },
    "web_ui": {
        "enabled": True,
        "port": 8080
    }
}

def _merge(defaults, supplied):
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if key not in result:
            raise ConfigError(f"Unknown configuration key: {key}")
        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _require(condition, message):
    if not condition:
        raise ConfigError(message)


def validate_config(config):
    stream = config["stream"]
    _require(isinstance(stream["source"], (int, str)), "stream.source must be a camera index, file path, or RTSP URL")
    for key in ("width", "height", "fps", "buffer_seconds"):
        _require(isinstance(stream[key], (int, float)) and stream[key] > 0, f"stream.{key} must be positive")

    motion = config["motion"]
    _require(isinstance(motion["enabled"], bool), "motion.enabled must be boolean")
    detector = config["detector"]
    _require(detector["model_type"] in {"yolo", "mock"}, "detector.model_type must be 'yolo' or 'mock'")
    _require(0 < float(detector["confidence_threshold"]) <= 1, "detector.confidence_threshold must be between 0 and 1")
    _require(float(detector["idle_scan_seconds"]) > 0, "detector.idle_scan_seconds must be positive")
    _require(isinstance(detector["target_classes"], list) and all(isinstance(x, int) and x >= 0 for x in detector["target_classes"]), "detector.target_classes must be a list of non-negative integers")
    _require(isinstance(detector["min_detection_area"], (int, float)) and detector["min_detection_area"] >= 0, "detector.min_detection_area must be non-negative")
    _require(isinstance(detector["min_person_size"], (int, float)) and detector["min_person_size"] >= 0, "detector.min_person_size must be non-negative")
    _require(isinstance(detector["min_person_area"], (int, float)) and detector["min_person_area"] >= 0, "detector.min_person_area must be non-negative")
    if detector["model_type"] == "yolo":
        _require(Path(detector["model_path"]).is_file(), f"YOLO model not found: {detector['model_path']}")

    tracker = config["tracker"]
    _require(isinstance(tracker["max_disappeared"], int) and tracker["max_disappeared"] >= 0, "tracker.max_disappeared must be a non-negative integer")
    _require(0 < float(tracker["min_iou"]) <= 1, "tracker.min_iou must be between 0 and 1")
    tripwire = config["analytics"]["tripwire"]
    _require(tripwire["direction"] in {"both", "forward", "reverse"}, "analytics.tripwire.direction must be both, forward, or reverse")
    _require(len(tripwire["line_start"]) == 2 and len(tripwire["line_end"]) == 2, "tripwire points must be [x, y]")
    intrusion = config["analytics"]["intrusion"]
    _require(not intrusion["enabled"] or len(intrusion["polygon"]) >= 3, "enabled intrusion polygon needs at least three points")
    recording = config["recording"]
    _require(len(recording["codec"]) == 4, "recording.codec must be a four-character codec")
    _require(float(recording["post_event_seconds"]) >= 0, "recording.post_event_seconds must be non-negative")
    _require(isinstance(recording["max_pending_incidents"], int) and recording["max_pending_incidents"] >= 1, "recording.max_pending_incidents must be at least 1")
    web_ui = config["web_ui"]
    _require(isinstance(web_ui["enabled"], bool), "web_ui.enabled must be boolean")
    _require(isinstance(web_ui["port"], int) and 1 <= web_ui["port"] <= 65535, "web_ui.port must be an integer between 1 and 65535")
    return config


def load_config(config_path="config/settings.yaml"):
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        import yaml
        with path.open() as file:
            supplied = yaml.safe_load(file) or {}
    except Exception as exc:
        raise ConfigError(f"Could not read configuration: {exc}") from exc
    if not isinstance(supplied, dict):
        raise ConfigError("Configuration root must be a mapping")
    return validate_config(_merge(DEFAULT_CONFIG, supplied))


def resolve_telegram_credentials(alert_config, environ=None):
    telegram = alert_config["telegram"]
    if not telegram["enabled"]:
        return None, None
    environ = os.environ if environ is None else environ
    token = environ.get(telegram["token_env"])
    chat_id = environ.get(telegram["chat_id_env"])
    if not token or not chat_id:
        raise ConfigError("Telegram is enabled but its token or chat ID environment variable is missing")
    return token, chat_id
