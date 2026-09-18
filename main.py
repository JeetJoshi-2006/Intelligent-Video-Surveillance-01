import sys
import os
import time
import cv2
import numpy as np

from core.config_loader import load_config
from core.stream_manager import StreamManager
from core.motion_gater import MotionGater
from core.detector import EdgeDetector
from core.tracker import MultiObjectTracker
from core.analytics import AnalyticsEngine
from alerts.dispatcher import AlertDispatcher
from alerts.recorder import EventRecorder

def draw_hud(frame, tracklets, events, has_motion, fps, config):
    """Renders visual heads-up display (HUD), zones, and tracking trails on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Draw Intrusion Polygon
    intrusion_cfg = config.get("analytics", {}).get("intrusion", {})
    if intrusion_cfg.get("enabled", False):
        poly_pts = np.array(intrusion_cfg.get("polygon", []), np.int32)
        if len(poly_pts) >= 3:
            cv2.polylines(frame, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=2)
            # Subtle red fill
            cv2.fillPoly(overlay, [poly_pts], color=(0, 0, 180))

    # 2. Draw Virtual Tripwire
    tripwire_cfg = config.get("analytics", {}).get("tripwire", {})
    if tripwire_cfg.get("enabled", False):
        p1 = tuple(tripwire_cfg.get("line_start", [100, 360]))
        p2 = tuple(tripwire_cfg.get("line_end", [1180, 360]))
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        cv2.circle(frame, p1, 5, (0, 255, 255), -1)
        cv2.circle(frame, p2, 5, (0, 255, 255), -1)
        cv2.putText(frame, "TRIPWIRE", (p1[0], max(20, p1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Blend polygon overlay
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # 3. Draw Tracklets & Trajectories
    for t in tracklets:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tag background & text
        tag = f"#{t.track_id} {t.label} ({t.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, tag, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Trajectory trail
        for i in range(1, len(t.trajectory)):
            ptA = (int(t.trajectory[i - 1][0]), int(t.trajectory[i - 1][1]))
            ptB = (int(t.trajectory[i][0]), int(t.trajectory[i][1]))
            cv2.line(frame, ptA, ptB, (255, 200, 0), 2)

    # 4. Status Bar Header
    status_text = "STATUS: ACTIVE INFERENCE" if has_motion else "STATUS: STANDBY (THERMAL GUARD)"
    status_color = (0, 255, 0) if has_motion else (200, 200, 200)
    
    cv2.rectangle(frame, (10, 10), (450, 65), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS: {fps:.1f} | Objects: {len(tracklets)}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, status_text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    # 5. Alert Banner Overlay
    if events:
        for idx, ev in enumerate(events[-2:]):
            alert_bar_y = h - 50 - (idx * 40)
            cv2.rectangle(frame, (20, alert_bar_y), (w - 20, alert_bar_y + 35), (0, 0, 220), -1)
            cv2.putText(frame, f"ALERT: {ev['details']}", (35, alert_bar_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

def run_pipeline():
    config_path = os.path.join(os.path.dirname(__file__), "config/settings.yaml")
    config = load_config(config_path)

    stream_cfg = config.get("stream", {})
    source = stream_cfg.get("source", 0)
    
    print("=" * 70)
    print("  INTELLIGENT SURVEILLANCE SYSTEM (ISS) - APPLE SILICON WORKSTATION")
    print(f"  Source: {source} | Buffer: {stream_cfg.get('buffer_seconds')}s")
    print("=" * 70)

    # Initialize Modules
    stream_mgr = StreamManager(
        source=source,
        width=stream_cfg.get("width", 1280),
        height=stream_cfg.get("height", 720),
        fps=stream_cfg.get("fps", 30),
        buffer_seconds=stream_cfg.get("buffer_seconds", 5)
    )

    motion_cfg = config.get("motion", {})
    motion_gater = MotionGater(
        min_motion_area=motion_cfg.get("min_motion_area", 1200),
        history=motion_cfg.get("history", 500),
        var_threshold=motion_cfg.get("var_threshold", 25)
    )

    det_cfg = config.get("detector", {})
    detector = EdgeDetector(
        model_path=det_cfg.get("model_path", "yolo11n.pt"),
        conf_threshold=det_cfg.get("confidence_threshold", 0.45),
        target_classes=det_cfg.get("target_classes")
    )

    trk_cfg = config.get("tracker", {})
    tracker = MultiObjectTracker(
        max_disappeared=trk_cfg.get("max_disappeared", 30),
        min_iou=trk_cfg.get("min_iou", 0.3)
    )

    analytics = AnalyticsEngine(config=config.get("analytics", {}))

    alert_cfg = config.get("alerts", {})
    dispatcher = AlertDispatcher(
        macos_banner=alert_cfg.get("macos_banner", True),
        sound_name=alert_cfg.get("macos_sound", "Hero"),
        telegram_token=alert_cfg.get("telegram", {}).get("bot_token"),
        telegram_chat_id=alert_cfg.get("telegram", {}).get("chat_id")
    )

    rec_cfg = config.get("recording", {})
    recorder = EventRecorder(
        output_dir=rec_cfg.get("output_dir", "recordings"),
        post_event_seconds=rec_cfg.get("post_event_seconds", 8),
        fps=stream_cfg.get("fps", 30)
    )

    # Start stream capture thread
    try:
        stream_mgr.start()
    except Exception as e:
        print(f"[ERROR] Failed to start video capture: {e}")
        print("Tip: Check camera permissions in System Settings > Privacy & Security > Camera")
        sys.exit(1)

    print("[SYSTEM READY] Press 'q' in the video window or Ctrl+C in terminal to exit.")

    fps_history = []
    last_loop_time = time.time()
    active_events = []

    try:
        while True:
            frame, ts = stream_mgr.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 1. Motion Gating Stage
            has_motion, fg_mask, motion_boxes = motion_gater.evaluate(frame)

            # 2. Neural Detection (Triggered on motion or periodically)
            detections = []
            if has_motion:
                detections = detector.detect(frame)

            # 3. Multi-Object Tracking
            tracklets = tracker.update(detections)

            # 4. Spatial Analytics Evaluation
            events = analytics.evaluate_tracklets(tracklets)
            if events:
                active_events.extend(events)
                # Keep active alerts visible for 3 seconds
                active_events = [e for e in active_events if time.time() - e["timestamp"] < 3.0]
                
                for ev in events:
                    # Dispatch notifications
                    dispatcher.dispatch(ev, snapshot_frame=frame)
                    # Trigger incident video recording
                    pre_buffer = stream_mgr.get_pre_event_clip()
                    recorder.record_incident(ev["rule"], pre_buffer, stream_mgr)

            # Compute actual FPS
            now = time.time()
            dt = now - last_loop_time
            last_loop_time = now
            if dt > 0:
                fps_history.append(1.0 / dt)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 30.0

            # 5. Render HUD Display
            annotated_frame = draw_hud(frame, tracklets, active_events, has_motion, avg_fps, config)

            # Display window (if GUI environment available)
            cv2.imshow("Intelligent Surveillance System (Edge AI)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping pipeline...")
    finally:
        stream_mgr.stop()
        cv2.destroyAllWindows()
        print("[SYSTEM] Clean shutdown complete.")

if __name__ == "__main__":
    run_pipeline()
