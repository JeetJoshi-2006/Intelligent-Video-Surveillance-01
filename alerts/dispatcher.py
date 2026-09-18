import os
import subprocess
import threading
import cv2
import requests

class AlertDispatcher:
    """
    Handles event notification dispatch across multiple channels:
      - Native macOS notification banners (AppleScript)
      - Audio chime ('Hero', 'Ping', 'Glass')
      - Telegram Bot snapshot push
    """
    def __init__(self, macos_banner=True, sound_name="Hero", telegram_token=None, telegram_chat_id=None):
        self.macos_banner = macos_banner
        self.sound_name = sound_name
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id

    def dispatch(self, event, snapshot_frame=None):
        """Dispatches an alert event asynchronously to prevent pipeline lag."""
        threading.Thread(target=self._dispatch_worker, args=(event, snapshot_frame), daemon=True).start()

    def _dispatch_worker(self, event, snapshot_frame):
        rule = event.get("rule", "SECURITY_ALERT")
        details = event.get("details", "")
        
        print(f"\n[ALERT DISPATCHED] >>> {rule}: {details}")

        # 1. Native macOS desktop banner & alert sound
        if self.macos_banner:
            try:
                # Sanitize text for AppleScript
                safe_title = rule.replace('"', '\\"')
                safe_msg = details.replace('"', '\\"')
                sound_part = f'sound name "{self.sound_name}"' if self.sound_name else ''
                cmd = f'display notification "{safe_msg}" with title "🚨 {safe_title}" {sound_part}'
                subprocess.run(["osascript", "-e", cmd], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[AlertDispatcher] macOS banner error: {e}")

        # 2. Telegram Bot Push with Snapshot
        if self.telegram_token and self.telegram_chat_id:
            try:
                caption = f"🚨 *SECURITY ALERT*\n*Rule:* {rule}\n*Details:* {details}"
                if snapshot_frame is not None:
                    _, buffer = cv2.imencode(".jpg", snapshot_frame)
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                    files = {"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")}
                    data = {"chat_id": self.telegram_chat_id, "caption": caption, "parse_mode": "Markdown"}
                    requests.post(url, files=files, data=data, timeout=5)
                else:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    data = {"chat_id": self.telegram_chat_id, "text": caption, "parse_mode": "Markdown"}
                    requests.post(url, data=data, timeout=5)
            except Exception as e:
                print(f"[AlertDispatcher] Telegram dispatch error: {e}")
