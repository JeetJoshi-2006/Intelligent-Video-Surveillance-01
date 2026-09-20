# Category mapping and color coding for real-world surveillance operations
# Colors are in BGR format for OpenCV

SURVEILLANCE_CATEGORIES = {
    # 1. Threats & Weapons -> RED
    "knife": {"category": "THREAT", "color": (0, 0, 255), "priority": 1},
    "scissors": {"category": "THREAT", "color": (0, 0, 255), "priority": 1},

    # 2. Human Entities -> GREEN
    "person": {"category": "HUMAN", "color": (0, 255, 0), "priority": 2},

    # 3. High-Value Tech Assets -> CYAN
    "laptop": {"category": "TECH_ASSET", "color": (255, 255, 0), "priority": 3},
    "cell phone": {"category": "TECH_ASSET", "color": (255, 255, 0), "priority": 3},
    "keyboard": {"category": "TECH_ASSET", "color": (200, 200, 0), "priority": 4},
    "mouse": {"category": "TECH_ASSET", "color": (200, 200, 0), "priority": 4},
    "tv": {"category": "TECH_ASSET", "color": (200, 200, 0), "priority": 4},

    # 4. Baggage & Left Luggage -> YELLOW / ORANGE
    "backpack": {"category": "BAGGAGE", "color": (0, 215, 255), "priority": 3},
    "handbag": {"category": "BAGGAGE", "color": (0, 215, 255), "priority": 3},
    "suitcase": {"category": "BAGGAGE", "color": (0, 165, 255), "priority": 3},
    "umbrella": {"category": "BAGGAGE", "color": (0, 215, 255), "priority": 4},

    # 5. Vehicles & Transport -> PURPLE / MAGENTA
    "car": {"category": "VEHICLE", "color": (255, 0, 255), "priority": 3},
    "motorcycle": {"category": "VEHICLE", "color": (255, 0, 255), "priority": 3},
    "truck": {"category": "VEHICLE", "color": (255, 0, 200), "priority": 3},
    "bus": {"category": "VEHICLE", "color": (255, 0, 200), "priority": 3},
    "bicycle": {"category": "VEHICLE", "color": (200, 0, 200), "priority": 4},

    # 6. Animals / False-Alarm Suppression -> GRAY
    "dog": {"category": "ANIMAL", "color": (160, 160, 160), "priority": 5},
    "cat": {"category": "ANIMAL", "color": (160, 160, 160), "priority": 5},
    "bird": {"category": "ANIMAL", "color": (140, 140, 140), "priority": 5},
}

def get_class_metadata(label):
    """Returns category name, display color (BGR), and priority for a detected object label."""
    return SURVEILLANCE_CATEGORIES.get(label.lower(), {
        "category": "GENERAL",
        "color": (0, 255, 128),
        "priority": 4
    })
