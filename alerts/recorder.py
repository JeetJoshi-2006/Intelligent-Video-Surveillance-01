"""Bounded incident recorder with retention management."""

import os
import time
import threading
from pathlib import Path

import cv2


class EventRecorder:
    """Records only one incident at a time and writes frames incrementally."""

    def __init__(self, output_dir="recordings", post_event_seconds=8, fps=30,
                 codec="mp4v", max_pending_incidents=1, max_age_days=14,
                 max_total_mb=2048):
        self.output_dir = Path(output_dir)
        self.post_event_seconds = post_event_seconds
        self.fps = fps
        self.codec = codec
        self.max_pending_incidents = max_pending_incidents
        self.max_age_days = max_age_days
        self.max_total_bytes = max_total_mb * 1024 * 1024
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._active = 0
        self._lock = threading.Lock()
        self._cleanup_retention()

    def record_incident(self, event_name, pre_event_frames, stream_manager):
        """Start an incident only when capacity is available; coalesce duplicates."""
        with self._lock:
            if self._active >= self.max_pending_incidents:
                print("[EventRecorder] Incident already recording; event coalesced.")
                return False
            self._active += 1
        thread = threading.Thread(
            target=self._record_worker,
            args=(event_name, pre_event_frames, stream_manager), daemon=True,
        )
        thread.start()
        return True

    def _record_worker(self, event_name, pre_event_frames, stream_manager):
        writer = None
        try:
            frames = [frame for _, frame in pre_event_frames]
            if not frames:
                return
            height, width = frames[0].shape[:2]
            timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
            filename = f"{event_name}_{timestamp}.mp4"
            filepath = self.output_dir / filename
            writer = cv2.VideoWriter(str(filepath), cv2.VideoWriter_fourcc(*self.codec), self.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Video writer failed to open with codec {self.codec}")

            # Write history one frame at a time rather than duplicating an entire
            # incident in memory.
            for frame in frames:
                writer.write(self._fit(frame, width, height))

            needed = int(self.post_event_seconds * self.fps)
            written = 0
            last_ts = 0.0
            deadline = time.monotonic() + self.post_event_seconds + 3
            while written < needed and time.monotonic() < deadline:
                frame, timestamp = stream_manager.get_latest_frame()
                if frame is not None and timestamp > last_ts:
                    writer.write(self._fit(frame, width, height))
                    last_ts = timestamp
                    written += 1
                time.sleep(1 / max(self.fps * 2, 1))
            print(f"[EventRecorder] Incident clip saved: {filepath}")
        except Exception as exc:
            print(f"[EventRecorder] Recording failed: {exc}")
        finally:
            if writer is not None:
                writer.release()
            with self._lock:
                self._active -= 1
            self._cleanup_retention()

    @staticmethod
    def _fit(frame, width, height):
        return frame if frame.shape[:2] == (height, width) else cv2.resize(frame, (width, height))

    def _cleanup_retention(self):
        files = sorted(self.output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
        expiry = time.time() - (self.max_age_days * 86400)
        for path in list(files):
            if path.stat().st_mtime < expiry:
                path.unlink(missing_ok=True)
                files.remove(path)
        total = sum(path.stat().st_size for path in files)
        while files and total > self.max_total_bytes:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
