import os
import cv2
import time
import threading

class EventRecorder:
    """
    Records incidents with pre-event rolling buffer + post-event footage.
    Exports hardware-accelerated MP4 clips to disk asynchronously.
    """
    def __init__(self, output_dir="recordings", post_event_seconds=8, fps=30):
        self.output_dir = output_dir
        self.post_event_seconds = post_event_seconds
        self.fps = fps
        os.makedirs(self.output_dir, exist_ok=True)
        self.active_sessions = []
        self.lock = threading.Lock()

    def record_incident(self, event_name, pre_event_frames, stream_manager):
        """
        Spawns a thread to record the full incident (pre-event buffer + upcoming live frames).
        """
        thread = threading.Thread(
            target=self._record_worker,
            args=(event_name, pre_event_frames, stream_manager),
            daemon=True
        )
        thread.start()

    def _record_worker(self, event_name, pre_event_frames, stream_manager):
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{event_name}_{timestamp_str}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        frames_to_write = [frame for ts, frame in pre_event_frames]
        
        # Collect post-event frames for post_event_seconds
        post_frames_needed = int(self.post_event_seconds * self.fps)
        collected = 0
        last_ts = 0.0

        while collected < post_frames_needed:
            frame, ts = stream_manager.get_latest_frame()
            if frame is not None and ts != last_ts:
                frames_to_write.append(frame)
                last_ts = ts
                collected += 1
            time.sleep(1.0 / (self.fps * 2))

        if len(frames_to_write) == 0:
            return

        h, w = frames_to_write[0].shape[:2]
        # Use 'mp4v' or 'avc1' for macOS QuickTime compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))

        for f in frames_to_write:
            # Resize if dimensions differ
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.write(f)

        writer.release()
        print(f"[EventRecorder] Incident clip saved: {filepath} ({len(frames_to_write)} frames)")
