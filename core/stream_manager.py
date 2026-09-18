import cv2
import time
import threading
from collections import deque
import numpy as np

class StreamManager:
    """
    High-performance, multi-threaded camera / RTSP stream ingest manager.
    Maintains a thread-safe circular ring buffer for pre-event alert extraction.
    """
    def __init__(self, source=0, width=1280, height=720, fps=30, buffer_seconds=5):
        self.source = source
        self.target_width = width
        self.target_height = height
        self.fps = fps
        self.max_buffer_size = int(buffer_seconds * fps)
        
        # Thread-safe circular deque: drops oldest frames automatically when full
        self.ring_buffer = deque(maxlen=self.max_buffer_size)
        self.lock = threading.Lock()
        
        self.cap = None
        self.running = False
        self.worker_thread = None
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.frame_count = 0
        self.dropped_frames = 0

    def start(self):
        """Initializes hardware capture backend and starts the background worker thread."""
        backend = cv2.CAP_ANY
        # On macOS, use AVFOUNDATION for built-in FaceTime HD camera
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            backend = cv2.CAP_AVFOUNDATION
            self.source = int(self.source)
        elif isinstance(self.source, str) and self.source.startswith("rtsp://"):
            backend = cv2.CAP_FFMPEG

        self.cap = cv2.VideoCapture(self.source, backend)
        
        # Set camera stream properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"[StreamManager] Unable to open video source: {self.source}")

        self.running = True
        self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker_thread.start()
        return self

    def _capture_loop(self):
        """Continuously pulls frames from device into the circular ring buffer."""
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.dropped_frames += 1
                time.sleep(0.01)
                continue

            now = time.time()
            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = now
                self.frame_count += 1
                # Store (timestamp, frame_copy) in circular buffer
                self.ring_buffer.append((now, frame.copy()))

    def get_latest_frame(self):
        """Returns the most recent frame captured without blocking the grabber thread."""
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_timestamp

    def get_pre_event_clip(self):
        """Retrieves all buffered historical frames for incident replay."""
        with self.lock:
            return [(ts, f.copy()) for ts, f in self.ring_buffer]

    def stop(self):
        """Terminates thread and releases camera hardware."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
