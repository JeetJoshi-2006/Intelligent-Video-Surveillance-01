import time
import math
import numpy as np

def ccw(A, B, C):
    """Checks whether points A, B, and C are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    """Determines whether line segment AB intersects line segment CD."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

def line_side(line_start, line_end, point):
    """Signed side of a directed line; positive is its forward/left side."""
    return ((line_end[0] - line_start[0]) * (point[1] - line_start[1])
            - (line_end[1] - line_start[1]) * (point[0] - line_start[0]))

def point_in_polygon(point, polygon):
    """
    Ray-casting algorithm to determine if a 2D point (x, y) lies inside an arbitrary polygon.
    polygon: list of [x, y] coordinates
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y

    return inside

class AnalyticsEngine:
    """
    Evaluates geometric and temporal surveillance rules:
      1. Virtual Tripwire (directional line crossing)
      2. Perimeter Intrusion (polygon geo-fencing)
      3. Loitering Dwell Time (target lingering in a restricted area)
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.tripwire_cfg = self.config.get("tripwire", {})
        self.intrusion_cfg = self.config.get("intrusion", {})
        self.loitering_cfg = self.config.get("loitering", {})

        # Alert debounce registries {rule_key_track_id: timestamp_last_fired}
        self.alert_history = {}
        self.debounce_cooldown = 15.0 # seconds before re-alerting same track ID

        # Loitering spatial anchor registry {track_id: {'first_time': t, 'anchor_pos': (x,y)}}
        self.loiter_anchors = {}

    def _can_fire(self, alert_key):
        now = time.time()
        if alert_key in self.alert_history:
            if now - self.alert_history[alert_key] < self.debounce_cooldown:
                return False
        self.alert_history[alert_key] = now
        return True

    def evaluate_tracklets(self, tracklets):
        """
        Evaluates active tracklets against configured rules.
        Returns:
            list of alert events: [{'rule': str, 'track_id': int, 'label': str, 'details': str, 'timestamp': float}]
        """
        events = []
        now = time.time()

        for t in tracklets:
            curr_pos = t.centroid
            prev_pos = t.prev_centroid
            t_id = t.track_id
            label = t.label

            # 1. Virtual Tripwire Evaluation
            if self.tripwire_cfg.get("enabled", False):
                l_start = tuple(self.tripwire_cfg.get("line_start", [100, 360]))
                l_end = tuple(self.tripwire_cfg.get("line_end", [1180, 360]))
                
                direction = self.tripwire_cfg.get("direction", "both")
                prev_side = line_side(l_start, l_end, prev_pos)
                curr_side = line_side(l_start, l_end, curr_pos)
                crossed = curr_pos != prev_pos and segments_intersect(l_start, l_end, prev_pos, curr_pos)
                direction_matches = (
                    direction == "both"
                    or (direction == "forward" and prev_side < 0 <= curr_side)
                    or (direction == "reverse" and prev_side > 0 >= curr_side)
                )
                if crossed and direction_matches:
                    alert_key = f"tripwire_{t_id}"
                    if self._can_fire(alert_key):
                        events.append({
                            "rule": "VIRTUAL_TRIPWIRE",
                            "track_id": t_id,
                            "label": label,
                            "details": f"{label.capitalize()} (ID #{t_id}) crossed virtual tripwire boundary.",
                            "timestamp": now,
                            "position": curr_pos
                        })

            # 2. Perimeter Intrusion Evaluation
            if self.intrusion_cfg.get("enabled", False):
                polygon = self.intrusion_cfg.get("polygon", [])
                if len(polygon) >= 3 and point_in_polygon(curr_pos, polygon):
                    alert_key = f"intrusion_{t_id}"
                    if self._can_fire(alert_key):
                        events.append({
                            "rule": "PERIMETER_INTRUSION",
                            "track_id": t_id,
                            "label": label,
                            "details": f"Unauthorized {label} (ID #{t_id}) breached restricted zone.",
                            "timestamp": now,
                            "position": curr_pos
                        })

            # 3. Loitering Evaluation
            if self.loitering_cfg.get("enabled", False):
                max_dwell = self.loitering_cfg.get("max_dwell_seconds", 8.0)
                radius = self.loitering_cfg.get("radius_pixels", 60.0)

                if t_id not in self.loiter_anchors:
                    self.loiter_anchors[t_id] = {"first_time": now, "anchor": curr_pos}
                else:
                    anchor = self.loiter_anchors[t_id]["anchor"]
                    dist = math.hypot(curr_pos[0] - anchor[0], curr_pos[1] - anchor[1])
                    if dist <= radius:
                        dwell = now - self.loiter_anchors[t_id]["first_time"]
                        if dwell >= max_dwell:
                            alert_key = f"loitering_{t_id}"
                            if self._can_fire(alert_key):
                                events.append({
                                    "rule": "LOITERING_DETECTED",
                                    "track_id": t_id,
                                    "label": label,
                                    "details": f"{label.capitalize()} (ID #{t_id}) loitering for {int(dwell)}s (threshold: {max_dwell}s).",
                                    "timestamp": now,
                                    "position": curr_pos
                                })
                    else:
                        # Reset anchor if target relocated outside radius
                        self.loiter_anchors[t_id] = {"first_time": now, "anchor": curr_pos}

        # Cleanup expired anchors
        active_ids = {t.track_id for t in tracklets}
        self.loiter_anchors = {k: v for k, v in self.loiter_anchors.items() if k in active_ids}

        return events
