import os
import json

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
        "target_classes": [0, 1, 2, 3, 5, 7, 24, 26, 28]
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
            "bot_token": "",
            "chat_id": ""
        }
    },
    "recording": {
        "output_dir": "recordings",
        "post_event_seconds": 8,
        "codec": "avc1"
    }
}

def load_config(config_path="config/settings.yaml"):
    """Loads configuration from YAML or returns default config if yaml parser is unavailable."""
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG
