import numpy as np
import time

def calculate_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0.0, xB - xA)
    interHeight = max(0.0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    unionArea = boxAArea + boxBArea - interArea

    if unionArea <= 0:
        return 0.0
    return float(interArea / unionArea)


class KalmanBoxFilter:
    """
    Constant-velocity 8-state Kalman Filter for 2D bounding boxes:
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """
    def __init__(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))

        # State vector
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # State covariance P
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0])

        # State transition F
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        # Process noise Q
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0])

        # Measurement noise R
        self.R = np.diag([2.0, 2.0, 4.0, 4.0])

    def predict(self):
        """Predicts state vector and covariance by 1 step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Clamp dimensions
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return self.get_bbox()

    def update(self, bbox):
        """Corrects state using measurement bbox [x1, y1, x2, y2]."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        z = np.array([cx, cy, w, h], dtype=np.float64)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I = np.eye(8, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return self.get_bbox()

    def get_bbox(self):
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return [
            round(cx - w / 2.0, 1),
            round(cy - h / 2.0, 1),
            round(cx + w / 2.0, 1),
            round(cy + h / 2.0, 1),
        ]


class Tracklet:
    """Represents a persistent tracked entity in the surveillance scene."""
    def __init__(self, track_id, bbox, class_id, label, confidence):
        self.track_id = track_id
        self.bbox = [float(v) for v in bbox] # [x1, y1, x2, y2]
        self.class_id = class_id
        self.label = label
        self.confidence = float(confidence)
        self.disappeared = 0
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.hit_streak = 1
        self.age = 1

        self.kf = KalmanBoxFilter(self.bbox)

        # Centroid history for trajectory & tripwire analytics
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.trajectory = [(cx, cy)]
        self.max_history = 60

    def predict(self):
        """Propagates state estimate when a frame has no new detections."""
        self.bbox = self.kf.predict()
        self.age += 1
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > self.max_history:
            self.trajectory.pop(0)
        return self.bbox

    def update(self, bbox, confidence):
        """Updates tracklet with a confirmed detection."""
        self.kf.update(bbox)
        self.bbox = [float(v) for v in bbox]
        self.confidence = float(confidence)
        self.disappeared = 0
        self.hit_streak += 1
        self.age += 1
        self.last_seen = time.time()

        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > self.max_history:
            self.trajectory.pop(0)

    @property
    def centroid(self):
        return self.trajectory[-1]

    @property
    def prev_centroid(self):
        if len(self.trajectory) >= 2:
            return self.trajectory[-2]
        return self.trajectory[-1]

    @property
    def dwell_time(self):
        return self.last_seen - self.first_seen


class MultiObjectTracker:
    """
    Two-Stage ByteTrack-inspired Multi-Object Tracker with Kalman State Estimation.
    Associates high-confidence detections first, then recovers occluded tracks with
    second-stage association, and predicts smooth motion on frames between inference scans.
    """
    def __init__(self, max_disappeared=30, min_iou=0.25, high_conf_thresh=0.40):
        self.next_track_id = 1
        self.tracklets = {} # {track_id: Tracklet}
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.high_conf_thresh = high_conf_thresh

    def predict(self):
        """Predicts next state for all active tracklets (call on frames without detection)."""
        for tracklet in self.tracklets.values():
            tracklet.predict()
        return list(self.tracklets.values())

    def _associate(self, track_keys, detections, min_iou_threshold):
        """Greedy IoU association between selected tracks and detections."""
        if not track_keys or not detections:
            return set(), set(), track_keys, list(range(len(detections)))

        iou_matrix = np.zeros((len(track_keys), len(detections)), dtype=np.float32)
        for i, t_id in enumerate(track_keys):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = calculate_iou(self.tracklets[t_id].bbox, det["bbox"])

        matched_tracks = set()
        matched_detections = set()

        while True:
            max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0
            if max_val < min_iou_threshold:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_idx, det_idx = idx[0], idx[1]

            t_id = track_keys[track_idx]
            det = detections[det_idx]
            self.tracklets[t_id].update(det["bbox"], det["confidence"])

            matched_tracks.add(t_id)
            matched_detections.add(det_idx)

            iou_matrix[track_idx, :] = -1
            iou_matrix[:, det_idx] = -1

        unmatched_tracks = [t_id for t_id in track_keys if t_id not in matched_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_detections]

        return matched_tracks, matched_detections, unmatched_tracks, unmatched_dets

    def update(self, detections):
        """
        Updates the tracker with detections from current frame using two-stage ByteTrack association.
        detections: list of dicts [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int, 'label': str}]
        Returns: list of active Tracklet objects
        """
        if len(detections) == 0:
            to_delete = []
            for t_id, tracklet in self.tracklets.items():
                tracklet.predict()
                tracklet.disappeared += 1
                if tracklet.disappeared > self.max_disappeared:
                    to_delete.append(t_id)
            for t_id in to_delete:
                del self.tracklets[t_id]
            return list(self.tracklets.values())

        if len(self.tracklets) == 0:
            for det in detections:
                t = Tracklet(self.next_track_id, det["bbox"], det["class_id"], det["label"], det["confidence"])
                self.tracklets[self.next_track_id] = t
                self.next_track_id += 1
            return list(self.tracklets.values())

        # Split detections into high-confidence and low-confidence
        high_dets = [d for d in detections if d.get("confidence", 0.0) >= self.high_conf_thresh]
        low_dets = [d for d in detections if d.get("confidence", 0.0) < self.high_conf_thresh]

        # If high_dets is empty, treat all detections as high to avoid dropping
        if not high_dets:
            high_dets = detections
            low_dets = []

        all_track_ids = list(self.tracklets.keys())

        # Stage 1: Match high-confidence detections
        m_t1, m_d1, u_t1, u_d1 = self._associate(all_track_ids, high_dets, self.min_iou)

        # Stage 2: Match remaining unmatched tracks with low-confidence detections (occlusion recovery)
        m_t2, m_d2, u_t2, _ = self._associate(u_t1, low_dets, self.min_iou)

        # Register remaining unmatched high-confidence detections as new tracks
        for d_idx in u_d1:
            det = high_dets[d_idx]
            t = Tracklet(self.next_track_id, det["bbox"], det["class_id"], det["label"], det["confidence"])
            self.tracklets[self.next_track_id] = t
            self.next_track_id += 1

        # Age and remove tracks that were not matched in either stage
        to_delete = []
        for t_id in u_t2:
            self.tracklets[t_id].predict()
            self.tracklets[t_id].disappeared += 1
            if self.tracklets[t_id].disappeared > self.max_disappeared:
                to_delete.append(t_id)
        for t_id in to_delete:
            del self.tracklets[t_id]

        return list(self.tracklets.values())
