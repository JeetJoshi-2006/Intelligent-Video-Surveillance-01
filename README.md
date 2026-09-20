# Intelligent Surveillance System

Local macOS surveillance pipeline for a camera or RTSP source. It uses YOLO detection, motion-triggered inference with a periodic idle scan, IoU tracking, tripwire/intrusion/loitering analytics, desktop alerts, and bounded incident recording.

## Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py
```

Grant the terminal camera permission in **System Settings → Privacy & Security → Camera**. Configure camera source, rules, and retention in `config/settings.yaml`.

## Monitoring dashboard

When the pipeline is running, open [http://localhost:8080](http://localhost:8080). The local dashboard shows the annotated live stream, pipeline FPS, active tracked objects, and the latest analytics alerts. Set `web_ui.enabled: false` or change `web_ui.port` in `config/settings.yaml` to disable it or use a different port.

## Detection behavior

YOLO runs immediately when motion is present and also at `detector.idle_scan_seconds` while a scene is still. Tracks are retained when inference is deliberately skipped, so boxes do not disappear between scans.

`motion.enabled: false` performs continuous inference. Use `detector.model_type: mock` only for deterministic non-model tests.

## Secrets and alerts

Telegram is off by default. Never put a token or chat ID in YAML. When enabling it, export:

```bash
export ISS_TELEGRAM_BOT_TOKEN='…'
export ISS_TELEGRAM_CHAT_ID='…'
```

The values are read only when `alerts.telegram.enabled` is true. Do not commit `.env` files.

## Recording and retention

The recorder allows only `recording.max_pending_incidents` concurrent/coalesced incidents and writes frames incrementally. `max_age_days` and `max_total_mb` prune the oldest clips. Set `recording.enabled: false` to disable local recording.

## Health checks

Startup performs strict configuration validation and exits with an actionable `[CONFIG ERROR]` message. Runtime logs include camera failures, alert dispatches, recorder failures, saved incident paths, and the local dashboard address.
