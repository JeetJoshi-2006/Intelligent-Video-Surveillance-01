# Intelligent Surveillance System (ISS)

Edge AI surveillance pipeline optimized for Apple Silicon macOS (arm64). Features high-accuracy YOLO11m detection, asynchronous 30 FPS decoupled inference, Kalman-filtered ByteTrack multi-object tracking, motion-gated thermal guards, tripwire/intrusion/loitering analytics, and incident recording.

## Run

```bash
./run.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
OPENCV_AVFOUNDATION_SKIP_AUTH=1 python -m unittest discover -s tests -v
python main.py
```

Grant terminal camera permissions in **System Settings → Privacy & Security → Camera**. Configure video source, detector model, and analytics rules in `config/settings.yaml`.

## Model Scaling & Hardware Acceleration

The system is configured with **YOLO11m** (Medium, ~20.1M parameters) as default, providing a major accuracy leap (+12.0 mAP over nano) on security threats (knives, scissors), tech assets (smartphones, laptops), and distant persons:

- **Apple Silicon Acceleration**: Prioritizes CoreML `.mlpackage` on the Apple Neural Engine (ANE) for 0% CPU compute and sustained cool operation. Falls back to PyTorch Metal Performance Shaders (MPS).
- **Asynchronous 30 FPS Decoupled Ingestion**: Camera acquisition and UI rendering run at 30 FPS. Deep neural inference runs on an asynchronous worker thread without stalling the video feed.
- **Kalman & ByteTrack Tracking**: Kinematic state estimation propagates bounding boxes smoothly across intermediate frames and maintains track identities through occlusions and motion blur.
- **Class-Specific Sensitivity**: Configured in `config/settings.yaml` under `detector.class_confidences` (e.g. higher recall for knife/scissors threat detection, higher precision for ambient objects).

## Utility Scripts

- **Benchmark & Hardware Profiler**:
  ```bash
  ./venv/bin/python scripts/benchmark_models.py --models yolo11n.pt yolo11s.pt yolo11m.pt
  ```
  Profiles latency (ms), FPS, and disk/memory footprint across model tiers on Apple Silicon.

- **One-Click CoreML Exporter**:
  ```bash
  ./venv/bin/python scripts/export_coreml.py --model yolo11m.pt
  ```
  Exports PyTorch weights to an optimized Apple CoreML `.mlpackage` with baked-in NMS and FP16 precision for ANE.

## Monitoring Dashboard

When the pipeline is running, open [http://localhost:8080](http://localhost:8080). The dashboard displays the live video stream, real-time FPS, active tracked objects, spatial zone status, and recent security alerts.

## Secrets and Alerts

Telegram is off by default. Export credentials via environment variables:
```bash
export ISS_TELEGRAM_BOT_TOKEN='…'
export ISS_TELEGRAM_CHAT_ID='…'
```

## Recording and Archival

The recorder allows bounded concurrent incident recording with circular 5-second pre-event buffering and auto-pruning based on `max_age_days` and `max_total_mb`.
