import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.tracker import MultiObjectTracker, Tracklet, KalmanBoxFilter, calculate_iou
from core.detector import EdgeDetector

class TestAdvancedTrackingAndDetector(unittest.TestCase):

    def test_kalman_prediction_smoothness(self):
        """Kalman filter should propagate bounding box velocity on frames without measurement."""
        kf = KalmanBoxFilter([100, 100, 200, 200])
        # Give two frames of constant rightward movement (+10 px per frame)
        kf.update([110, 100, 210, 200])
        kf.update([120, 100, 220, 200])

        # Predict next position without measurement
        predicted_box = kf.predict()
        # Should predict cx moving rightward
        pred_cx = (predicted_box[0] + predicted_box[2]) / 2.0
        self.assertGreater(pred_cx, 160.0)

    def test_bytetrack_two_stage_recovery(self):
        """Low-confidence detection in stage 2 should maintain track identity across occlusion."""
        tracker = MultiObjectTracker(max_disappeared=5, min_iou=0.2, high_conf_thresh=0.4)

        # Frame 1: High confidence detection
        det1 = [{"bbox": [100, 100, 200, 200], "confidence": 0.85, "class_id": 0, "label": "person"}]
        tracks1 = tracker.update(det1)
        self.assertEqual(len(tracks1), 1)
        track_id = tracks1[0].track_id

        # Frame 2: Person partially occluded / blurred, confidence drops to 0.25 (below high_conf_thresh 0.4)
        det2 = [{"bbox": [105, 102, 205, 202], "confidence": 0.25, "class_id": 0, "label": "person"}]
        tracks2 = tracker.update(det2)

        # Stage 2 should have matched the low-confidence detection to the existing track
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, track_id)
        self.assertEqual(tracks2[0].disappeared, 0)

    def test_tracker_predict_cadence(self):
        """Tracker predict() maintains active track state at 30 FPS between inference scans."""
        tracker = MultiObjectTracker(max_disappeared=10, min_iou=0.2)
        det = [{"bbox": [50, 50, 100, 100], "confidence": 0.9, "class_id": 0, "label": "person"}]
        tracker.update(det)

        # 3 intermediate video frames without neural inference
        for _ in range(3):
            tracks = tracker.predict()
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].track_id, 1)

    def test_detector_class_specific_thresholds(self):
        """Class confidences should allow lower threshold for knife (threat) while rejecting low-conf chair."""
        detector = EdgeDetector(
            mode="mock",
            conf_threshold=0.45,
            class_confidences={
                43: 0.25,  # knife threat
                56: 0.50   # chair
            }
        )

        # Knife at 0.30 should pass (above 0.25)
        self.assertTrue(detector._passes_filters(43, "knife", 0, 0, 100, 100, 0.30))

        # Knife at 0.20 should fail (below 0.25)
        self.assertFalse(detector._passes_filters(43, "knife", 0, 0, 100, 100, 0.20))

        # Chair at 0.40 should fail (below 0.50)
        self.assertFalse(detector._passes_filters(56, "chair", 0, 0, 100, 100, 0.40))

        # Unspecified class (e.g. car class 2) uses default conf_threshold 0.45
        self.assertFalse(detector._passes_filters(2, "car", 0, 0, 100, 100, 0.40))
        self.assertTrue(detector._passes_filters(2, "car", 0, 0, 100, 100, 0.46))

if __name__ == "__main__":
    unittest.main(verbosity=2)
