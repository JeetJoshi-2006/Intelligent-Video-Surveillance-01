import sys
import os
import time
import unittest
import tempfile
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config_loader import ConfigError, load_config
from core.tracker import MultiObjectTracker, Tracklet, calculate_iou
from core.analytics import AnalyticsEngine, segments_intersect, point_in_polygon
from core.detector import EdgeDetector, COCO_SURVEILLANCE_LABELS
from alerts.recorder import EventRecorder

try:
    import cv2
    from core.motion_gater import MotionGater
    from core.stream_manager import StreamManager
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class TestSurveillancePipeline(unittest.TestCase):

    def test_config_loader(self):
        cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml"))
        self.assertIn("stream", cfg)
        self.assertIn("motion", cfg)
        self.assertIn("detector", cfg)
        self.assertIn("analytics", cfg)
        print("✓ ConfigLoader passed.")

    def test_invalid_config_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "invalid.yaml"
            config_file.write_text("detector:\n  idle_scan_seconds: 0\n")
            with self.assertRaises(ConfigError):
                load_config(config_file)

    def test_iou_calculation(self):
        iou = calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
        self.assertAlmostEqual(iou, 1.0 / 7.0, places=3)
        print("✓ IoU calculation passed.")

    def test_tracker_association(self):
        tracker = MultiObjectTracker(max_disappeared=5, min_iou=0.2)
        
        # Frame 1 detections
        det1 = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9, "class_id": 0, "label": "person"},
            {"bbox": [500, 500, 600, 600], "confidence": 0.85, "class_id": 2, "label": "car"}
        ]
        tracks1 = tracker.update(det1)
        self.assertEqual(len(tracks1), 2)
        id_person = tracks1[0].track_id
        
        # Frame 2: person moves slightly to [105, 105, 205, 205]
        det2 = [
            {"bbox": [105, 105, 205, 205], "confidence": 0.92, "class_id": 0, "label": "person"}
        ]
        tracks2 = tracker.update(det2)
        matched = [t for t in tracks2 if t.track_id == id_person]
        self.assertEqual(len(matched), 1)
        self.assertAlmostEqual(matched[0].centroid[0], 155.0)
        print("✓ MultiObjectTracker passed.")

    def test_analytics_tripwire(self):
        line_start = (100, 360)
        line_end = (1180, 360)
        
        # Path crosses line vertically from (500, 300) to (500, 420)
        p_prev = (500, 300)
        p_curr = (500, 420)
        self.assertTrue(segments_intersect(line_start, line_end, p_prev, p_curr))

        # Path does not cross line
        p_no_cross = (500, 340)
        self.assertFalse(segments_intersect(line_start, line_end, p_prev, p_no_cross))
        print("✓ Analytics Virtual Tripwire geometry passed.")

    def test_analytics_intrusion(self):
        polygon = [[700, 150], [1150, 150], [1150, 600], [700, 600]]
        
        inside_pt = (800, 300)
        outside_pt = (400, 300)
        
        self.assertTrue(point_in_polygon(inside_pt, polygon))
        self.assertFalse(point_in_polygon(outside_pt, polygon))
        print("✓ Analytics Polygon Intrusion geometry passed.")

    def test_analytics_loitering(self):
        cfg = {
            "loitering": {
                "enabled": True,
                "max_dwell_seconds": 0.1,
                "radius_pixels": 50.0
            }
        }
        engine = AnalyticsEngine(config=cfg)
        
        track = Tracklet(track_id=42, bbox=[100, 100, 200, 200], class_id=0, label="person", confidence=0.9)
        
        # Initial frame
        events = engine.evaluate_tracklets([track])
        self.assertEqual(len(events), 0)
        
        # Wait until dwell time exceeds threshold
        time.sleep(0.12)
        track.update([102, 102, 202, 202], 0.91)
        events = engine.evaluate_tracklets([track])
        
        loitering_alerts = [e for e in events if e["rule"] == "LOITERING_DETECTED"]
        self.assertEqual(len(loitering_alerts), 1)
        print("✓ Analytics Loitering detection passed.")

    def test_detector_filters_hand_false_positives(self):
        detector = EdgeDetector(mode="mock", min_person_size=40, min_person_area=2000)

        # Hand-sized person bbox (e.g. 35x35) should be filtered out
        self.assertFalse(
            detector._passes_filters(0, "person", 0, 0, 35, 35, 0.9),
            "Hand-size person detection should be rejected"
        )

        # Normal person bbox should pass
        self.assertTrue(
            detector._passes_filters(0, "person", 0, 0, 150, 300, 0.9),
            "Normal person detection should pass filters"
        )
        print("✓ Detector hand-false-positive filter passed.")

    def test_detector_extended_labels(self):
        self.assertIn(63, COCO_SURVEILLANCE_LABELS)
        self.assertEqual(COCO_SURVEILLANCE_LABELS[63], "laptop")
        self.assertEqual(COCO_SURVEILLANCE_LABELS[67], "cell phone")
        self.assertIn(44, COCO_SURVEILLANCE_LABELS)
        print("✓ Detector extended class labels passed.")

    def test_motion_gater_and_stream(self):
        if not CV2_AVAILABLE:
            print("⚠ Skipping OpenCV MotionGater & StreamManager tests (cv2 not installed in this Python environment).")
            return

        gater = MotionGater(min_motion_area=500, history=10, var_threshold=16)
        static_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        for _ in range(5):
            gater.evaluate(static_frame)

        moving_frame = static_frame.copy()
        moving_frame[200:400, 300:500] = 255
        has_motion, _, boxes = gater.evaluate(moving_frame)
        self.assertTrue(has_motion)
        print("✓ MotionGater passed.")

    @unittest.skipUnless(CV2_AVAILABLE, "OpenCV is required for saved-video integration testing")
    def test_saved_person_video_detection_tracking_alert_and_recording(self):
        """Exercise the pipeline path with a saved clip and deterministic person detector."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "person_fixture.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            frames = []
            for y in range(25, 185, 40):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                # A simple moving person silhouette makes the fixture human-readable.
                cv2.circle(frame, (160, y + 20), 16, (255, 255, 255), -1)
                cv2.rectangle(frame, (148, y + 37), (172, y + 110), (255, 255, 255), -1)
                writer.write(frame)
                frames.append(frame)
            writer.release()
            self.assertTrue(video_path.is_file())

            cap = cv2.VideoCapture(str(video_path))
            tracker = MultiObjectTracker(max_disappeared=3, min_iou=0.1)
            analytics = AnalyticsEngine({"tripwire": {"enabled": True, "line_start": [0, 120], "line_end": [319, 120], "direction": "both"}})
            all_events, tracks = [], []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # Deterministic detector boundary: production YOLO is covered by
                # smoke testing, while this fixture validates the full app logic.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ys, xs = np.where(gray > 200)
                if len(xs):
                    detection = {"bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "confidence": 0.99, "class_id": 0, "label": "person"}
                    tracks = tracker.update([detection])
                    all_events.extend(analytics.evaluate_tracklets(tracks))
            cap.release()
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].label, "person")
            self.assertTrue(any(event["rule"] == "VIRTUAL_TRIPWIRE" for event in all_events))

            class StaticStream:
                def get_latest_frame(self):
                    return frames[-1], time.time()
            recorder = EventRecorder(temp_path / "clips", post_event_seconds=0, fps=10, max_pending_incidents=1, max_age_days=1, max_total_mb=10)
            self.assertTrue(recorder.record_incident("PERSON_TEST", [(time.time(), frame) for frame in frames], StaticStream()))
            deadline = time.time() + 2
            clips = []
            while time.time() < deadline:
                clips = list((temp_path / "clips").glob("*.mp4"))
                if clips and clips[0].stat().st_size > 0:
                    break
                time.sleep(0.02)
            self.assertTrue(clips, "incident recording was not saved")

if __name__ == "__main__":
    unittest.main(verbosity=2)
