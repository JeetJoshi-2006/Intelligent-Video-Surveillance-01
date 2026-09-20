import sys
import os
import time
import cv2
import numpy as np

from core.config_loader import ConfigError, load_config, resolve_telegram_credentials
from core.stream_manager import StreamManager
from core.motion_gater import MotionGater
from core.detector import EdgeDetector
from core.tracker import MultiObjectTracker
from core.analytics import AnalyticsEngine
from alerts.dispatcher import AlertDispatcher
from alerts.recorder import EventRecorder
from core.web_stream import start_web_server, update_dashboard, update_web_frame

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
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}")
        sys.exit(2)

    stream_cfg = config.get("stream", {})
    source = stream_cfg.get("source", 0)
    
    print("=" * 70)
    print("  INTELLIGENT SURVEILLANCE SYSTEM (ISS) - APPLE SILICON WORKSTATION")
    print(f"  Source: {source} | Buffer: {stream_cfg.get('buffer_seconds')}s")
    print("=" * 70)

    # Initialize lightweight modules (no heavy I/O or model loading)
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

    trk_cfg = config.get("tracker", {})
    tracker = MultiObjectTracker(
        max_disappeared=trk_cfg.get("max_disappeared", 30),
        min_iou=trk_cfg.get("min_iou", 0.3)
    )

    analytics = AnalyticsEngine(config=config.get("analytics", {}))

    alert_cfg = config.get("alerts", {})
    try:
        telegram_token, telegram_chat_id = resolve_telegram_credentials(alert_cfg)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}")
        sys.exit(2)
    dispatcher = AlertDispatcher(
        macos_banner=alert_cfg.get("macos_banner", True),
        sound_name=alert_cfg.get("macos_sound", "Hero"),
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id
    )

    rec_cfg = config.get("recording", {})
    recorder = EventRecorder(
        output_dir=rec_cfg["output_dir"], post_event_seconds=rec_cfg["post_event_seconds"],
        fps=stream_cfg["fps"], codec=rec_cfg["codec"],
        max_pending_incidents=rec_cfg["max_pending_incidents"],
        max_age_days=rec_cfg["max_age_days"], max_total_mb=rec_cfg["max_total_mb"],
    )

    # Build static metadata for the dashboard
    camera_info = {
        "source": str(source),
        "width": stream_cfg.get("width", 1280),
        "height": stream_cfg.get("height", 720),
        "fps": stream_cfg.get("fps", 30),
    }
    zones_init = {
        "tripwire": {"enabled": analytics.tripwire_cfg.get("enabled", False), "triggered": False},
        "intrusion": {"enabled": analytics.intrusion_cfg.get("enabled", False), "triggered": False},
        "loitering": {"enabled": analytics.loitering_cfg.get("enabled", False), "triggered": False},
    }

    # Start web dashboard FIRST so it is available during heavy initialization
    web_server = None
    web_cfg = config["web_ui"]
    if web_cfg["enabled"]:
        try:
            web_server = start_web_server(web_cfg["port"])
            update_dashboard("Starting", False, 0.0, [], [],
                             camera_info=camera_info,
                             stream_health={"frame_loss": 0, "dropped_frames": 0, "last_frame_ts": 0.0},
                             zones=zones_init,
                             detection_stats={"total_detections": 0, "total_alerts": 0},
                             recording={"active": False, "pending": 0})
        except OSError as exc:
            print(f"[WebUI] Disabled: could not start dashboard on port {web_cfg['port']}: {exc}")

    # Load detector (heavy — YOLO model loading can take 10-15s on first run)
    print("[INIT] Loading YOLO model...")
    det_cfg = config.get("detector", {})
    detector = EdgeDetector(
        model_path=det_cfg.get("model_path", "yolo11n.pt"),
        conf_threshold=det_cfg.get("confidence_threshold", 0.45),
        target_classes=det_cfg.get("target_classes"),
        mode="mock" if det_cfg.get("model_type") == "mock" else "auto",
        min_detection_area=det_cfg.get("min_detection_area", 1500),
        min_person_size=det_cfg.get("min_person_size", 40),
        min_person_area=det_cfg.get("min_person_area", 2000),
    )
    idle_scan_seconds = float(det_cfg["idle_scan_seconds"])

    # Start stream capture thread (can be slow on macOS camera init)
    print("[INIT] Opening camera stream...")
    try:
        stream_mgr.start()
    except Exception as e:
        print(f"[ERROR] Failed to start video capture: {e}")
        print("Tip: Check camera permissions in System Settings > Privacy & Security > Camera")
        if web_server:
            web_server.shutdown()
            web_server.server_close()
        sys.exit(1)

    print("[SYSTEM READY] Press 'q' in the video window or Ctrl+C in terminal to exit.")

    fps_history = []
    last_loop_time = time.time()
    last_inference_time = 0.0
    active_events = []
    tracklets = []
    total_detections = 0
    total_alerts = 0

    try:
        while True:
            frame, ts = stream_mgr.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 1. Motion Gating Stage
            if motion_cfg["enabled"]:
                has_motion, fg_mask, motion_boxes = motion_gater.evaluate(frame)
            else:
                has_motion, fg_mask, motion_boxes = True, None, []

            # 2. Neural Detection (Triggered on motion or periodically)
            should_infer = has_motion or (time.time() - last_inference_time >= idle_scan_seconds)
            if should_infer:
                detections = detector.detect(frame)
                total_detections += len(detections)
                last_inference_time = time.time()
                # Do not age tracks when inference is intentionally skipped.
                tracklets = tracker.update(detections)

            # 4. Spatial Analytics Evaluation
            events = analytics.evaluate_tracklets(tracklets)
            if events:
                active_events.extend(events)
                # Keep active alerts visible for 3 seconds
                active_events = [e for e in active_events if time.time() - e["timestamp"] < 3.0]
                
                for ev in events:
                    # Dispatch notifications
                    total_alerts += 1
                    dispatcher.dispatch(ev, snapshot_frame=frame)
                    if rec_cfg["enabled"]:
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
            update_web_frame(annotated_frame)

            # Publish enhanced state to web dashboard
            zones = {
                "tripwire": {"enabled": analytics.tripwire_cfg.get("enabled", False),
                             "triggered": any(e["rule"] == "VIRTUAL_TRIPWIRE" for e in active_events)},
                "intrusion": {"enabled": analytics.intrusion_cfg.get("enabled", False),
                              "triggered": any(e["rule"] == "PERIMETER_INTRUSION" for e in active_events)},
                "loitering": {"enabled": analytics.loitering_cfg.get("enabled", False),
                              "triggered": any(e["rule"] == "LOITERING_DETECTED" for e in active_events)},
            }
            stream_health = {
                "frame_loss": stream_mgr.dropped_frames,
                "dropped_frames": stream_mgr.dropped_frames,
                "last_frame_ts": stream_mgr.latest_timestamp,
            }
            recording_status = {
                "active": recorder._active > 0,
                "pending": recorder._active,
            }
            detection_stats = {
                "total_detections": total_detections,
                "total_alerts": total_alerts,
            }
            update_dashboard("Active", has_motion, avg_fps, tracklets, events,
                             camera_info=camera_info, stream_health=stream_health,
                             zones=zones, detection_stats=detection_stats,
                             recording=recording_status)

            # Display window (if GUI environment available)
            cv2.imshow("Intelligent Surveillance System (Edge AI)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping pipeline...")
    finally:
        stream_mgr.stop()
        if web_server:
            web_server.shutdown()
            web_server.server_close()
        cv2.destroyAllWindows()
        print("[SYSTEM] Clean shutdown complete.")

if __name__ == "__main__":
    run_pipeline()
