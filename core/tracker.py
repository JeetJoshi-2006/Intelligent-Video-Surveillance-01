import numpy as np
import time

def calculate_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    unionArea = boxAArea + boxBArea - interArea

    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

class Tracklet:
    """Represents a single persistent tracked entity in the surveillance scene."""
    def __init__(self, track_id, bbox, class_id, label, confidence):
        self.track_id = track_id
        self.bbox = bbox # [x1, y1, x2, y2]
        self.class_id = class_id
        self.label = label
        self.confidence = confidence
        self.disappeared = 0
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        
        # History of centroid coordinates for trajectory analysis
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.trajectory = [(cx, cy)]
        self.max_history = 60

    def update(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.disappeared = 0
        self.last_seen = time.time()
        
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
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
    Centroid & IoU association tracker for continuous surveillance identification.
    Associates incoming detections with existing tracklets using optimal matching.
    """
    def __init__(self, max_disappeared=30, min_iou=0.25):
        self.next_track_id = 1
        self.tracklets = {} # {track_id: Tracklet}
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou

    def update(self, detections):
        """
        Updates the tracker with detections from current frame.
        detections: list of dicts [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int, 'label': str}]
        Returns: list of active Tracklet objects
        """
        if len(detections) == 0:
            # Mark all active tracklets as missing this frame
            to_delete = []
            for t_id, tracklet in self.tracklets.items():
                tracklet.disappeared += 1
                if tracklet.disappeared > self.max_disappeared:
                    to_delete.append(t_id)
            for t_id in to_delete:
                del self.tracklets[t_id]
            return list(self.tracklets.values())

        if len(self.tracklets) == 0:
            # Initialize tracklets for all incoming detections
            for det in detections:
                t = Tracklet(self.next_track_id, det["bbox"], det["class_id"], det["label"], det["confidence"])
                self.tracklets[self.next_track_id] = t
                self.next_track_id += 1
            return list(self.tracklets.values())

        # Build IoU association matrix
        track_ids = list(self.tracklets.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

        for i, t_id in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = calculate_iou(self.tracklets[t_id].bbox, det["bbox"])

        matched_tracks = set()
        matched_detections = set()

        # Greedy match based on highest IoU
        while True:
            max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0
            if max_val < self.min_iou:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_idx, det_idx = idx[0], idx[1]
            
            t_id = track_ids[track_idx]
            det = detections[det_idx]
            self.tracklets[t_id].update(det["bbox"], det["confidence"])
            
            matched_tracks.add(t_id)
            matched_detections.add(det_idx)
            
            # Mask out this track and detection from subsequent matches
            iou_matrix[track_idx, :] = -1
            iou_matrix[:, det_idx] = -1

        # Register unmatched detections as new tracks
        for j, det in enumerate(detections):
            if j not in matched_detections:
                t = Tracklet(self.next_track_id, det["bbox"], det["class_id"], det["label"], det["confidence"])
                self.tracklets[self.next_track_id] = t
                self.next_track_id += 1

        # Increment disappeared count for unmatched existing tracks
        to_delete = []
        for t_id, tracklet in self.tracklets.items():
            if t_id not in matched_tracks:
                tracklet.disappeared += 1
                if tracklet.disappeared > self.max_disappeared:
                    to_delete.append(t_id)
        for t_id in to_delete:
            del self.tracklets[t_id]

        return list(self.tracklets.values())
