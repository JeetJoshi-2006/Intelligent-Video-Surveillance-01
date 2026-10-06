"""Dependency-free web dashboard for the local surveillance pipeline.

Provides:
  * MJPEG live feed at /stream.mjpg
  * JSON status snapshot at /api/status
  * Server-Sent Events stream at /api/events for real-time dashboard updates
"""

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


LATEST_FRAME_JPEG = None
FRAME_LOCK = threading.Lock()
FRAME_META = {"width": 0, "height": 0, "last_update": 0.0}

DASHBOARD_LOCK = threading.Lock()
DASHBOARD_STATE = {
    "status": "Starting",
    "motion": False,
    "fps": 0.0,
    "objects": [],
    "events": [],
    "updated_at": 0.0,
    "camera": {"source": "0", "width": 1280, "height": 720, "fps": 30},
    "stream_health": {"frame_loss": 0, "dropped_frames": 0, "last_frame_ts": 0.0},
    "zones": {"tripwire": False, "intrusion": False, "loitering": False},
    "detection_stats": {"total_detections": 0, "total_alerts": 0},
    "recording": {"active": False, "pending": 0},
}

RECENT_EVENTS = deque(maxlen=12)
_SSE_CLIENTS = deque()
_SSE_LOCK = threading.Lock()


def _notify_sse_clients():
    """Push current state to all connected SSE clients."""
    with DASHBOARD_LOCK:
        payload = json.dumps({**DASHBOARD_STATE, "events": list(RECENT_EVENTS)})
    message = f"data: {payload}\n\n"
    dead = []
    with _SSE_LOCK:
        for client in list(_SSE_CLIENTS):
            try:
                client.wfile.write(message.encode())
                client.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead.append(client)
    for client in dead:
        with _SSE_LOCK:
            if client in _SSE_CLIENTS:
                _SSE_CLIENTS.remove(client)


PAGE = r"""<!DOCTYPE html>
<html class="dark" lang="en">
<head>
  <meta charset="utf-8"/>
  <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
  <meta content="web_dashboard" name="shell-type"/>
  <title>Sentinel Monitor | Intelligent Surveillance System</title>
  <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com" rel="preconnect"/>
  <link crossorigin="" href="https://fonts.gstatic.com" rel="preconnect"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet"/>
  <style>
    @layer base {
      html, body { margin: 0; padding: 0; }
      body { overscroll-behavior: none; }
      main > :first-child { margin-top: 0 !important; }
      main > :last-child { margin-bottom: 0 !important; }
    }
    ::-webkit-scrollbar { display: none; }
  </style>
  <script src="https://cdn.tailwindcss.com"></script>
  <script id="tailwind-config">
    tailwind.config = {
      darkMode: "class",
      theme: {
        extend: {
          colors: {
            "on-surface-variant": "#bac9cc",
            "surface-variant": "#2b3547",
            "surface-container-lowest": "#040e1e",
            "surface-container-highest": "#2b3547",
            "surface-tint": "#00daf3",
            "surface-container-high": "#202a3b",
            "on-secondary-fixed": "#001a42",
            "inverse-surface": "#d8e3fa",
            "surface-dim": "#081424",
            "surface-container": "#152030",
            "surface-bright": "#2f3a4b",
            "tertiary-fixed": "#6ffbbe",
            "on-primary-fixed": "#001f24",
            "on-secondary-container": "#00275d",
            "error": "#ffb4ab",
            "secondary": "#aec6ff",
            "secondary-container": "#4e8eff",
            "on-primary-container": "#00626e",
            "on-primary-fixed-variant": "#004f58",
            "on-tertiary": "#003824",
            "surface-container-low": "#111c2c",
            "on-tertiary-fixed-variant": "#005236",
            "inverse-on-surface": "#263142",
            "secondary-fixed": "#d8e2ff",
            "primary": "#c3f5ff",
            "primary-fixed": "#9cf0ff",
            "secondary-fixed-dim": "#aec6ff",
            "on-primary": "#00363d",
            "tertiary-fixed-dim": "#4edea3",
            "outline-variant": "#3b494c",
            "error-container": "#93000a",
            "outline": "#849396",
            "on-secondary": "#002e6a",
            "surface": "#081424",
            "background": "#081424",
            "on-tertiary-container": "#006645",
            "tertiary": "#a8ffd2",
            "on-secondary-fixed-variant": "#004395",
            "on-surface": "#d8e3fa",
            "tertiary-container": "#5be9ad",
            "on-tertiary-fixed": "#002113",
            "on-background": "#d8e3fa",
            "primary-container": "#00e5ff",
            "inverse-primary": "#006875",
            "on-error": "#690005",
            "on-error-container": "#ffdad6",
            "primary-fixed-dim": "#00daf3"
          },
          borderRadius: {
            "DEFAULT": "0.125rem",
            "lg": "0.25rem",
            "xl": "0.5rem",
            "full": "0.75rem"
          },
          spacing: {
            "gutter-desktop": "1rem",
            "space-xs": "0.25rem",
            "gutter": "0.75rem",
            "margin": "1rem",
            "space-md": "0.75rem",
            "space-lg": "1.25rem",
            "margin-desktop": "1.5rem",
            "space-xl": "2rem",
            "space-sm": "0.5rem"
          },
          fontFamily: {
            "label-md": ["JetBrains Mono"],
            "label-sm": ["JetBrains Mono"],
            "display": ["Space Grotesk"],
            "body-sm": ["Inter"],
            "headline-lg": ["Space Grotesk"],
            "headline-sm": ["Space Grotesk"],
            "body-md": ["Inter"],
            "headline-md": ["Space Grotesk"],
            "body-lg": ["Inter"],
            "label-lg": ["JetBrains Mono"],
            "display-mobile": ["Space Grotesk"],
            "headline-lg-mobile": ["Space Grotesk"]
          }
        }
      }
    };
  </script>
</head>
<body class="bg-background font-body-md text-on-surface antialiased selection:bg-primary-container selection:text-on-primary-container">

  <!-- LEFT SIDEBAR -->
  <aside class="fixed left-0 top-0 h-full w-72 bg-surface-container-lowest flex flex-col z-50 shadow-[0_1px_8px_rgba(0,0,0,0.4)]">
    <div class="p-space-md pb-space-xs flex flex-col gap-space-xs">
      <div class="flex items-center gap-space-sm">
        <div class="w-8 h-8 rounded bg-primary-container flex items-center justify-center text-on-primary-container font-headline-md font-bold shadow-[0_0_12px_rgba(0,229,255,0.4)]">
          <span class="material-symbols-outlined text-xl">shield</span>
        </div>
        <div class="flex flex-col">
          <span class="font-headline-sm text-headline-sm uppercase tracking-wider text-primary">Sentinel Monitor</span>
          <span class="font-label-sm text-label-sm text-on-surface-variant uppercase">Intelligent Surveillance</span>
        </div>
      </div>
      <div class="mt-space-xs flex items-center justify-between px-space-sm py-space-xs bg-surface-container-low rounded">
        <div class="flex items-center gap-space-xs">
          <span class="relative flex h-2 w-2">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-tertiary-container opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2 w-2 bg-tertiary"></span>
          </span>
          <span class="font-label-sm text-label-sm text-tertiary uppercase">Edge AI Node</span>
        </div>
        <span id="sidebarStatusBadge" class="font-label-sm text-label-sm text-on-surface-variant">ONLINE</span>
      </div>
    </div>

    <!-- Navigation Links -->
    <nav class="flex-1 overflow-y-auto px-space-xs py-space-xs flex flex-col gap-0.5">
      <a aria-current="page" class="flex items-center justify-between px-space-sm py-space-xs rounded transition-all bg-surface-container-high text-primary font-label-md" href="#">
        <div class="flex items-center gap-space-sm">
          <span class="material-symbols-outlined text-base">dashboard</span>
          <span class="font-label-md text-label-md">Overview</span>
        </div>
      </a>
      <a class="flex items-center justify-between px-space-sm py-space-xs rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-all" href="#">
        <div class="flex items-center gap-space-sm">
          <span class="material-symbols-outlined text-base">videocam</span>
          <span class="font-label-md text-label-md">Live Stream</span>
        </div>
        <span id="sidebarCamBadge" class="font-label-sm text-label-sm px-1.5 py-0.5 bg-surface-container text-primary rounded">CAM-01</span>
      </a>
      <a class="flex items-center justify-between px-space-sm py-space-xs rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-all" href="#">
        <div class="flex items-center gap-space-sm">
          <span class="material-symbols-outlined text-base">cognition</span>
          <span class="font-label-md text-label-md">AI Detection</span>
        </div>
        <span class="font-label-sm text-label-sm px-1.5 py-0.5 bg-surface-container-high text-tertiary rounded">YOLO11m</span>
      </a>
      <a class="flex items-center justify-between px-space-sm py-space-xs rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-all" href="#">
        <div class="flex items-center gap-space-sm">
          <span class="material-symbols-outlined text-base">warning</span>
          <span class="font-label-md text-label-md">Security Alerts</span>
        </div>
        <span id="sidebarAlertCount" class="font-label-sm text-label-sm px-1.5 py-0.5 bg-surface-container text-on-surface-variant rounded font-semibold">0 Live</span>
      </a>
      <a class="flex items-center justify-between px-space-sm py-space-xs rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-all" href="#">
        <div class="flex items-center gap-space-sm">
          <span class="material-symbols-outlined text-base">group</span>
          <span class="font-label-md text-label-md">Tracking (ByteTrack)</span>
        </div>
      </a>
      <a class="flex items-center justify-between px-space-sm py-space-xs rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-all" href="#">
        <div class="flex items-center gap-space-sm">
          <span class="material-symbols-outlined text-base">radar</span>
          <span class="font-label-md text-label-md">Spatial Zones</span>
        </div>
      </a>
      <a class="flex items-center justify-between px-space-sm py-space-xs rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-all" href="#">
        <div class="flex items-center gap-space-sm">
          <span class="material-symbols-outlined text-base">insights</span>
          <span class="font-label-md text-label-md">Analytics & Dwell</span>
        </div>
      </a>
      <a class="flex items-center justify-between px-space-sm py-space-xs rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-all" href="#">
        <div class="flex items-center gap-space-sm">
          <span class="material-symbols-outlined text-base">history</span>
          <span class="font-label-md text-label-md">Event Log & History</span>
        </div>
      </a>
    </nav>

    <!-- Sidebar Bottom Telemetry -->
    <div class="p-space-sm bg-surface-container-low flex flex-col gap-space-xs">
      <div class="px-space-xs py-space-xs bg-surface-container-lowest rounded flex flex-col gap-0.5">
        <div class="flex items-center justify-between">
          <span class="font-label-sm text-label-sm text-on-surface-variant">ACCELERATION</span>
          <span class="font-label-sm text-label-sm text-primary" id="sidebarHardwareBadge">Apple Silicon</span>
        </div>
        <div class="flex items-center justify-between font-label-sm text-label-sm">
          <span class="text-on-surface-variant">ANE / MPS</span>
          <span class="text-tertiary" id="sidebarFpsText">30.0 FPS</span>
          <span class="text-on-surface-variant">Decoupled</span>
        </div>
      </div>
      <div class="flex items-center gap-space-sm px-space-xs pt-space-xs">
        <div class="w-8 h-8 rounded-full bg-surface-container-high flex items-center justify-center text-primary font-bold">
          <span class="material-symbols-outlined text-lg">security</span>
        </div>
        <div class="flex flex-col min-w-0 flex-1">
          <span class="font-label-md text-label-md text-on-surface truncate">Security Operator</span>
          <span class="font-label-sm text-label-sm text-secondary truncate">Station Active</span>
        </div>
        <span class="material-symbols-outlined text-on-surface-variant text-base cursor-pointer hover:text-primary transition-colors" title="System Locked">shield_lock</span>
      </div>
    </div>
  </aside>

  <!-- MAIN BODY -->
  <div class="pl-72">
    <!-- TOP HEADER -->
    <header class="fixed top-0 left-72 right-0 h-16 bg-surface/90 backdrop-blur-xl z-40 flex items-center justify-between px-space-md shadow-[0_1px_8px_rgba(0,0,0,0.3)]">
      <div class="flex items-center gap-space-md">
        <div class="flex flex-col">
          <div class="flex items-center gap-space-xs">
            <span class="font-label-sm text-label-sm text-primary uppercase">LOCAL NODE</span>
            <span class="material-symbols-outlined text-sm text-on-surface-variant">chevron_right</span>
            <span class="font-label-sm text-label-sm text-on-surface-variant uppercase">WORKSTATION</span>
          </div>
          <span class="font-label-md text-label-md text-on-surface font-semibold">ISS Intelligent Surveillance Pipeline [Live]</span>
        </div>
      </div>
      <div class="flex-1 max-w-md mx-space-md">
        <div class="relative flex items-center">
          <span class="material-symbols-outlined absolute left-3 text-base text-on-surface-variant">search</span>
          <input id="searchInput" class="w-full pl-9 pr-space-md py-1.5 bg-surface-container-lowest text-on-surface font-label-sm text-label-sm rounded placeholder:text-outline-variant focus:outline-none focus:bg-surface-container" placeholder="Search Target ID, Class, Zone, Timestamp..." type="text"/>
        </div>
      </div>
      <div class="flex items-center gap-space-md">
        <div class="hidden xl:flex items-center gap-space-xs px-space-sm py-1 bg-surface-container rounded">
          <span class="relative flex h-2 w-2">
            <span id="headerStatusDot" class="relative inline-flex rounded-full h-2 w-2 bg-tertiary"></span>
          </span>
          <span id="headerStatusText" class="font-label-sm text-label-sm text-tertiary uppercase">ALL SYSTEMS NOMINAL</span>
        </div>
        <div class="hidden lg:flex items-center gap-space-xs text-on-surface-variant font-label-sm text-label-sm bg-surface-container-low px-space-sm py-1 rounded">
          <span class="material-symbols-outlined text-sm">schedule</span>
          <span id="headerUtcClock">--:--:-- UTC</span>
        </div>
        <div class="flex items-center gap-1 bg-surface-container-lowest p-0.5 rounded">
          <button class="p-1 text-on-surface-variant hover:text-primary transition-colors rounded" title="Single View"><span class="material-symbols-outlined text-sm">crop_square</span></button>
          <button class="p-1 bg-surface-container text-primary rounded" title="Tactical Grid"><span class="material-symbols-outlined text-sm">grid_view</span></button>
        </div>
        <button id="alertNotificationBtn" class="relative p-1.5 text-on-surface-variant hover:text-on-surface transition-colors" title="Alerts">
          <span class="material-symbols-outlined text-lg">notifications</span>
          <span id="notificationBadge" class="hidden absolute top-1 right-1 flex h-2 w-2 rounded-full bg-error"></span>
        </button>
        <button id="fullscreenToggleBtn" class="p-1.5 text-on-surface-variant hover:text-on-surface transition-colors" title="Toggle Fullscreen">
          <span class="material-symbols-outlined text-lg">fullscreen</span>
        </button>
      </div>
    </header>

    <!-- CONTENT WRAPPER -->
    <main class="w-full pt-16 bg-surface min-h-screen">
      <div class="flex flex-col w-full p-space-md gap-space-md text-on-surface">

        <!-- TOP STATS & KPI TELEMETRY STRIP -->
        <section class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-space-sm">
          <!-- Active Camera -->
          <div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between shadow-sm relative overflow-hidden">
            <div class="flex items-center justify-between">
              <span class="font-label-sm text-label-sm text-on-surface-variant uppercase">Video Stream</span>
              <span class="flex h-2 w-2 relative">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-tertiary-container opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-tertiary"></span>
              </span>
            </div>
            <div class="mt-space-xs flex items-baseline gap-space-xs">
              <span class="font-headline-lg text-headline-lg font-bold text-primary">01</span>
              <span class="font-label-md text-label-md text-on-surface-variant">/ 01</span>
            </div>
            <div class="mt-space-xs flex items-center gap-1 font-label-sm text-label-sm text-tertiary">
              <span class="material-symbols-outlined text-xs">videocam</span>
              <span id="streamResText">AVFoundation · 720p 30FPS</span>
            </div>
          </div>

          <!-- People Detected -->
          <div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between shadow-sm relative overflow-hidden">
            <div class="flex items-center justify-between">
              <span class="font-label-sm text-label-sm text-on-surface-variant uppercase">Persons Detected</span>
              <span class="material-symbols-outlined text-sm text-secondary">group</span>
            </div>
            <div class="mt-space-xs flex items-baseline gap-space-xs">
              <span id="kpiPersonsCount" class="font-headline-lg text-headline-lg font-bold text-on-surface">0</span>
              <span class="font-label-sm text-label-sm text-tertiary flex items-center">Live</span>
            </div>
            <div class="mt-space-xs font-label-sm text-label-sm text-on-surface-variant truncate">
              Kalman MOT State Synced
            </div>
          </div>

          <!-- Active Alerts -->
          <div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between shadow-sm relative overflow-hidden">
            <div class="flex items-center justify-between">
              <span class="font-label-sm text-label-sm text-on-surface-variant uppercase">Active Alerts</span>
              <span id="kpiAlertBadge" class="px-1.5 py-0.5 rounded bg-surface-container text-tertiary font-label-sm text-label-sm font-semibold">CLEAR</span>
            </div>
            <div class="mt-space-xs flex items-baseline gap-space-xs">
              <span id="kpiAlertCount" class="font-headline-lg text-headline-lg font-bold text-on-surface">0</span>
              <span class="font-label-sm text-label-sm text-on-surface-variant">Events</span>
            </div>
            <div id="kpiAlertDetails" class="mt-space-xs font-label-sm text-label-sm text-tertiary truncate">
              All Perimeters Secure
            </div>
          </div>

          <!-- Objects Tracked -->
          <div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between shadow-sm relative overflow-hidden">
            <div class="flex items-center justify-between">
              <span class="font-label-sm text-label-sm text-on-surface-variant uppercase">Objects Tracked</span>
              <span class="material-symbols-outlined text-sm text-primary">filter_center_focus</span>
            </div>
            <div class="mt-space-xs flex items-baseline gap-space-xs">
              <span id="kpiObjectsCount" class="font-headline-lg text-headline-lg font-bold text-secondary-fixed">0</span>
              <span class="font-label-sm text-label-sm text-primary">Entities</span>
            </div>
            <div class="mt-space-xs font-label-sm text-label-sm text-on-surface-variant truncate">
              ByteTrack Multi-Stage MOT
            </div>
          </div>

          <!-- AI Inference Telemetry -->
          <div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between shadow-sm relative overflow-hidden col-span-2 md:col-span-1">
            <div class="flex items-center justify-between">
              <span class="font-label-sm text-label-sm text-on-surface-variant uppercase">Neural Engine</span>
              <span class="px-1.5 py-0.5 rounded bg-surface-container text-primary font-label-sm text-label-sm">YOLO11m</span>
            </div>
            <div class="mt-space-xs flex items-baseline gap-space-xs">
              <span id="kpiPipelineFps" class="font-headline-lg text-headline-lg font-bold text-tertiary">30.0</span>
              <span class="font-label-sm text-label-sm text-on-surface-variant">FPS</span>
            </div>
            <div class="mt-space-xs font-label-sm text-label-sm text-on-surface-variant truncate" id="kpiLatencyText">
              Decoupled Async Pipeline
            </div>
          </div>
        </section>

        <!-- MAIN OPERATIONAL GRID (70% STAGE / 30% TELEMETRY) -->
        <div class="grid grid-cols-1 xl:grid-cols-12 gap-space-md items-start">
          <!-- LEFT / CENTER TACTICAL STAGE (xl:col-span-8) -->
          <div class="xl:col-span-8 flex flex-col gap-space-md">

            <!-- PRIMARY VIDEO FEED MODULE -->
            <div id="videoContainer" class="bg-surface-container-low rounded p-space-sm flex flex-col shadow-md">
              <!-- Feed Control Header -->
              <div class="flex flex-wrap items-center justify-between gap-space-sm px-space-xs pb-space-sm">
                <div class="flex items-center gap-space-sm">
                  <div class="flex items-center gap-space-xs">
                    <span class="relative flex h-2.5 w-2.5">
                      <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-error opacity-75"></span>
                      <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-error"></span>
                    </span>
                    <span class="font-label-md text-label-md font-semibold text-primary">CAM-01_LIVE_FEED</span>
                  </div>
                  <span class="font-label-sm text-label-sm px-1.5 py-0.5 bg-surface-container text-on-surface-variant rounded">HD 720p</span>
                  <span id="liveFpsBadge" class="font-label-sm text-label-sm px-1.5 py-0.5 bg-surface-container text-tertiary rounded">30.0 FPS</span>
                  <span id="motionPill" class="font-label-sm text-label-sm px-1.5 py-0.5 bg-surface-container text-on-surface-variant rounded">STANDBY</span>
                </div>
                <!-- Mode Selectors -->
                <div class="flex items-center gap-1 bg-surface-container-lowest p-1 rounded font-label-sm text-label-sm">
                  <button class="px-space-sm py-0.5 bg-surface-container text-primary rounded font-semibold transition-colors">Tactical View</button>
                  <div class="flex items-center gap-1 pl-space-xs">
                    <span class="w-1.5 h-1.5 rounded-full bg-primary"></span>
                    <span class="text-primary font-label-sm text-label-sm">YOLO11m BBoxes ON</span>
                  </div>
                </div>
              </div>

              <!-- Featured Viewport with Live Video Stream -->
              <div class="relative w-full aspect-video bg-black rounded overflow-hidden select-none">
                <!-- LIVE MJPEG STREAM IMAGE -->
                <img id="mainLiveFeed" src="/stream.mjpg" alt="Surveillance Stream" class="w-full h-full object-contain bg-black" />

                <!-- CCTV Aesthetic HUD Vignette & Grid Lines -->
                <div class="absolute inset-0 pointer-events-none bg-gradient-to-t from-surface-container-lowest/80 via-transparent to-surface-container-lowest/40"></div>

                <!-- Stream Meta Top Bar -->
                <div class="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
                  <div class="flex items-center gap-space-sm bg-surface-container-lowest/85 backdrop-blur-md px-space-sm py-1 rounded">
                    <span class="font-label-sm text-label-sm text-error font-bold tracking-widest animate-pulse">● LIVE STREAM</span>
                    <span id="streamTimestampText" class="font-label-sm text-label-sm text-on-surface-variant font-mono">--:--:--</span>
                  </div>
                  <div class="flex items-center gap-space-xs bg-surface-container-lowest/85 backdrop-blur-md px-space-sm py-1 rounded font-label-sm text-label-sm text-primary">
                    <span class="material-symbols-outlined text-xs">memory</span>
                    <span id="streamModelBadge">YOLO11m · MPS</span>
                  </div>
                </div>

                <!-- Bottom Tactical Overlay Control Strip -->
                <div class="absolute bottom-0 inset-x-0 bg-surface-container-lowest/95 backdrop-blur-md px-space-sm py-space-xs flex items-center justify-between">
                  <div class="flex items-center gap-space-xs">
                    <button id="pauseStreamBtn" class="p-1 rounded hover:bg-surface-container text-primary transition-colors" title="Pause / Resume Stream">
                      <span id="pauseBtnIcon" class="material-symbols-outlined text-base">pause</span>
                    </button>
                    <button id="snapshotBtn" class="p-1 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface transition-colors" title="Save Snapshot">
                      <span class="material-symbols-outlined text-base">photo_camera</span>
                    </button>
                    <div id="recordingPill" class="flex items-center gap-1 px-space-sm py-0.5 rounded bg-surface-container text-on-surface-variant font-label-sm text-label-sm font-semibold">
                      <span class="w-2 h-2 rounded-full bg-secondary"></span>
                      <span id="recText">BUFFERING 5s</span>
                    </div>
                  </div>
                  <!-- Optical Zoom Display -->
                  <div class="flex items-center gap-space-sm font-label-sm text-label-sm text-on-surface-variant">
                    <span>ANALYTICS:</span>
                    <span class="text-tertiary">Tripwire / Intrusion / Loitering</span>
                  </div>
                  <!-- Layer Controls -->
                  <div class="flex items-center gap-space-xs font-label-sm text-label-sm">
                    <button id="toggleFullscreenBtn" class="p-1 text-on-surface-variant hover:text-on-surface transition-colors" title="Fullscreen">
                      <span class="material-symbols-outlined text-base">fullscreen</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <!-- DETECTION ZONES & ARCHITECTURAL RADAR FACILITY MAP -->
            <div class="bg-surface-container-low rounded p-space-md shadow-sm">
              <div class="flex items-center justify-between pb-space-sm">
                <div class="flex items-center gap-space-xs">
                  <span class="material-symbols-outlined text-primary text-base">radar</span>
                  <span class="font-headline-sm text-headline-sm text-on-surface">Spatial Radar & Facility Zones</span>
                </div>
                <div class="flex items-center gap-space-sm font-label-sm text-label-sm">
                  <span class="text-on-surface-variant">RADAR SWEEP: ACTIVE</span>
                  <span class="px-1.5 py-0.5 bg-surface-container text-primary rounded">3 ANALYTICS ZONES</span>
                </div>
              </div>

              <div class="grid grid-cols-1 lg:grid-cols-12 gap-space-md items-center">
                <!-- Vector Blueprint Radar Map Canvas -->
                <div class="lg:col-span-7 bg-surface-container-lowest rounded p-space-sm relative aspect-[16/10] overflow-hidden flex items-center justify-center">
                  <!-- Blueprint Grid Lines -->
                  <svg class="absolute inset-0 w-full h-full opacity-20 pointer-events-none" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                      <pattern height="30" id="grid" patternunits="userSpaceOnUse" width="30">
                        <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#aec6ff" stroke-width="0.5"></path>
                      </pattern>
                    </defs>
                    <rect fill="url(#grid)" height="100%" width="100%"></rect>
                  </svg>

                  <!-- Vector Architectural Schematic -->
                  <svg class="w-full h-full max-h-72" fill="none" viewbox="0 0 500 320">
                    <rect fill="#0b1626" height="260" rx="4" stroke="#2b3547" stroke-width="2" width="420" x="40" y="30"></rect>
                    <line stroke="#2b3547" stroke-dasharray="3 3" stroke-width="1.5" x1="200" x2="200" y1="30" y2="290"></line>
                    <line stroke="#2b3547" stroke-width="1.5" x1="200" x2="460" y1="160" y2="160"></line>
                    <line stroke="#2b3547" stroke-width="1.5" x1="40" x2="200" y1="180" y2="180"></line>

                    <!-- ZONE A: Virtual Tripwire -->
                    <rect id="radarTripwireZone" fill="rgba(0, 229, 255, 0.08)" height="140" stroke="#00daf3" stroke-dasharray="4 2" stroke-width="1" width="150" x="45" y="35"></rect>
                    <text fill="#c3f5ff" font-family="Space Grotesk" font-size="11" font-weight="700" x="55" y="55">ZONE A - TRIPWIRE</text>
                    <text fill="#849396" font-family="JetBrains Mono" font-size="9" x="55" y="70">Line Crossing Axis</text>

                    <!-- Camera FOV cone CAM-01 -->
                    <path d="M 50 40 L 180 80 L 140 170 Z" fill="rgba(0, 218, 243, 0.15)" stroke="#00daf3" stroke-width="0.75"></path>
                    <circle cx="50" cy="40" fill="#00e5ff" r="4"></circle>

                    <!-- ZONE B: Perimeter Intrusion Polygon -->
                    <rect id="radarIntrusionZone" fill="rgba(255, 180, 171, 0.12)" height="120" stroke="#ffb4ab" stroke-width="1.5" width="250" x="205" y="35"></rect>
                    <text fill="#ffb4ab" font-family="Space Grotesk" font-size="11" font-weight="700" x="215" y="55">ZONE B - INTRUSION PERIMETER</text>
                    <text id="radarIntrusionText" fill="#ffb4ab" font-family="JetBrains Mono" font-size="9" x="215" y="70">Polygon Geo-fence Active</text>

                    <!-- ZONE C: Loitering Zone -->
                    <rect id="radarLoiterZone" fill="rgba(78, 142, 255, 0.08)" height="100" stroke="#4e8eff" stroke-width="1" width="150" x="45" y="185"></rect>
                    <text fill="#aec6ff" font-family="Space Grotesk" font-size="11" font-weight="700" x="55" y="205">ZONE C - DWELL AREA</text>
                    <text fill="#849396" font-family="JetBrains Mono" font-size="9" x="55" y="220">Loitering Guard &gt;8s</text>

                    <!-- Sweeping Target Indicator -->
                    <circle class="animate-ping" cx="280" cy="95" fill="#00daf3" r="4"></circle>
                    <circle cx="280" cy="95" fill="#00daf3" r="3"></circle>
                  </svg>

                  <!-- Sweeping Radar Beam Animation SVG Overlay -->
                  <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <svg class="w-48 h-48 animate-spin" style="animation-duration: 8s;" viewbox="0 0 100 100">
                      <circle cx="50" cy="50" fill="none" opacity="0.3" r="46" stroke="#00daf3" stroke-dasharray="2 4" stroke-width="0.5"></circle>
                      <path d="M 50 50 L 50 4 A 46 46 0 0 1 96 50 Z" fill="url(#radarGradient)" opacity="0.25"></path>
                      <defs>
                        <lineargradient id="radarGradient" x1="0%" x2="100%" y1="0%" y2="100%">
                          <stop offset="0%" stop-color="#00daf3" stop-opacity="0.8"></stop>
                          <stop offset="100%" stop-color="#00daf3" stop-opacity="0"></stop>
                        </lineargradient>
                      </defs>
                    </svg>
                  </div>
                </div>

                <!-- Zone Breakdown Metrics Table -->
                <div class="lg:col-span-5 flex flex-col gap-space-xs">
                  <div class="bg-surface-container p-space-xs rounded flex items-center justify-between">
                    <div class="flex items-center gap-space-xs">
                      <span id="zoneTripwireDot" class="w-2 h-2 rounded-full bg-primary"></span>
                      <span class="font-label-md text-label-md font-semibold">Virtual Tripwire</span>
                    </div>
                    <span id="zoneTripwireStatus" class="font-label-sm text-label-sm px-1.5 py-0.5 rounded bg-surface-container-high text-primary">Armed</span>
                  </div>

                  <div class="bg-surface-container p-space-xs rounded flex items-center justify-between">
                    <div class="flex items-center gap-space-xs">
                      <span id="zoneIntrusionDot" class="w-2 h-2 rounded-full bg-secondary"></span>
                      <span class="font-label-md text-label-md font-semibold">Perimeter Intrusion</span>
                    </div>
                    <span id="zoneIntrusionStatus" class="font-label-sm text-label-sm px-1.5 py-0.5 rounded bg-surface-container-high text-secondary">Armed</span>
                  </div>

                  <div class="bg-surface-container p-space-xs rounded flex items-center justify-between">
                    <div class="flex items-center gap-space-xs">
                      <span id="zoneLoiteringDot" class="w-2 h-2 rounded-full bg-tertiary"></span>
                      <span class="font-label-md text-label-md font-semibold">Loitering Dwell Guard</span>
                    </div>
                    <span id="zoneLoiteringStatus" class="font-label-sm text-label-sm px-1.5 py-0.5 rounded bg-surface-container-high text-tertiary">Armed</span>
                  </div>

                  <div class="bg-surface-container-lowest p-space-xs rounded mt-space-xs font-label-sm text-label-sm text-on-surface-variant flex items-center justify-between">
                    <span>PRE-EVENT ROLLING BUFFER</span>
                    <span class="text-tertiary">150 FRAMES (5 SEC)</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- REAL-TIME OBJECT TRACKING TABLE -->
            <div class="bg-surface-container-low rounded p-space-md shadow-sm">
              <div class="flex items-center justify-between pb-space-sm">
                <div class="flex items-center gap-space-xs">
                  <span class="material-symbols-outlined text-primary text-base">view_timeline</span>
                  <span class="font-headline-sm text-headline-sm text-on-surface">Live Entity Tracking Matrix</span>
                </div>
                <div class="flex items-center gap-space-xs">
                  <span class="font-label-sm text-label-sm text-on-surface-variant">AUTO-REFRESH: REALTIME</span>
                  <span class="w-1.5 h-1.5 rounded-full bg-tertiary animate-pulse"></span>
                </div>
              </div>
              <div class="overflow-x-auto">
                <table class="w-full text-left font-label-sm text-label-sm">
                  <thead class="bg-surface-container-lowest text-on-surface-variant uppercase">
                    <tr>
                      <th class="py-space-xs px-space-sm font-semibold">Track ID</th>
                      <th class="py-space-xs px-space-sm font-semibold">Entity Type</th>
                      <th class="py-space-xs px-space-sm font-semibold">Node Origin</th>
                      <th class="py-space-xs px-space-sm font-semibold">AI Conf.</th>
                      <th class="py-space-xs px-space-sm font-semibold">Dwell Time</th>
                      <th class="py-space-xs px-space-sm font-semibold text-right">Status</th>
                    </tr>
                  </thead>
                  <tbody id="trackingTableBody" class="divide-y-0">
                    <!-- Populated dynamically via SSE /status -->
                    <tr>
                      <td colspan="6" class="py-space-md px-space-sm text-center text-on-surface-variant">
                        No active entities tracked in camera field.
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

          </div>

          <!-- RIGHT PANEL: SOC COMMAND & TELEMETRY (xl:col-span-4) -->
          <div class="xl:col-span-4 flex flex-col gap-space-md">

            <!-- INCIDENT NOTIFICATIONS & CRITICAL ALERTS -->
            <div class="bg-surface-container-low rounded p-space-md shadow-sm flex flex-col gap-space-sm">
              <div class="flex items-center justify-between pb-space-xs">
                <div class="flex items-center gap-space-xs">
                  <span class="material-symbols-outlined text-error text-base">emergency</span>
                  <span class="font-headline-sm text-headline-sm text-on-surface">Incident Telemetry</span>
                </div>
                <span id="incidentHeaderBadge" class="font-label-sm text-label-sm px-1.5 py-0.5 rounded bg-surface-container text-tertiary font-bold">ALL CLEAR</span>
              </div>

              <!-- Alert Items Feed -->
              <div id="incidentList" class="flex flex-col gap-space-xs">
                <div class="bg-surface-container rounded p-space-sm flex flex-col gap-0.5">
                  <span class="font-label-sm text-label-sm text-tertiary flex items-center gap-1">
                    <span class="w-1.5 h-1.5 rounded-full bg-tertiary"></span>
                    SYSTEM NORMAL
                  </span>
                  <p class="font-label-md text-label-md text-on-surface">No security incidents detected.</p>
                  <span class="font-label-sm text-label-sm text-on-surface-variant">Autonomous analytics running continuously.</span>
                </div>
              </div>

              <!-- Incident Quick Action Controls -->
              <div class="flex items-center gap-space-xs mt-space-xs pt-space-xs border-t border-surface-container">
                <button id="lockdownBtn" class="flex-1 py-1.5 px-space-xs bg-error-container text-error rounded font-label-sm text-label-sm font-bold hover:bg-error hover:text-on-error transition-all">
                  Manual Alert Trigger
                </button>
                <button id="clearAlertsBtn" class="py-1.5 px-space-sm bg-surface-container text-primary rounded font-label-sm text-label-sm hover:bg-surface-container-high transition-colors">
                  Acknowledge
                </button>
              </div>
            </div>

            <!-- AI DETECTION ENGINE TELEMETRY -->
            <div class="bg-surface-container-low rounded p-space-md shadow-sm flex flex-col gap-space-sm">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-space-xs">
                  <span class="material-symbols-outlined text-primary text-base">psychology</span>
                  <span class="font-headline-sm text-headline-sm text-on-surface">Neural Engine Analytics</span>
                </div>
                <span class="font-label-sm text-label-sm text-tertiary">YOLO11m</span>
              </div>
              <div class="flex flex-wrap gap-1 font-label-sm text-label-sm">
                <span class="px-1.5 py-0.5 bg-surface-container text-primary rounded">YOLOv11-Medium</span>
                <span class="px-1.5 py-0.5 bg-surface-container text-secondary rounded">Kalman ByteTrack</span>
                <span class="px-1.5 py-0.5 bg-surface-container text-tertiary rounded">Metal MPS / ANE</span>
              </div>
              <div class="grid grid-cols-3 gap-space-xs bg-surface-container-lowest p-space-xs rounded text-center">
                <div>
                  <div class="font-label-sm text-label-sm text-on-surface-variant">Inference</div>
                  <div id="sideInferenceLatency" class="font-label-md text-label-md text-primary font-bold">-- ms</div>
                </div>
                <div>
                  <div class="font-label-sm text-label-sm text-on-surface-variant">Pipeline FPS</div>
                  <div id="sidePipelineFps" class="font-label-md text-label-md text-tertiary font-bold">30.0</div>
                </div>
                <div>
                  <div class="font-label-sm text-label-sm text-on-surface-variant">Model Size</div>
                  <div class="font-label-md text-label-md text-secondary font-bold">20.1M</div>
                </div>
              </div>

              <!-- Inline Neural Precision Chart -->
              <div class="bg-surface-container-lowest rounded p-space-xs flex flex-col gap-1">
                <div class="flex items-center justify-between font-label-sm text-label-sm text-on-surface-variant">
                  <span>REAL-TIME INFERENCE STATUS</span>
                  <span id="motionStatusBadge" class="text-tertiary">STANDBY</span>
                </div>
                <div class="w-full bg-surface-container h-1.5 rounded overflow-hidden">
                  <div id="fpsBar" class="bg-primary h-full transition-all duration-300" style="width: 100%"></div>
                </div>
              </div>
            </div>

            <!-- HARDWARE DIAGNOSTICS -->
            <div class="bg-surface-container-low rounded p-space-md shadow-sm flex flex-col gap-space-sm">
              <div class="flex items-center justify-between pb-space-xs">
                <div class="flex items-center gap-space-xs">
                  <span class="material-symbols-outlined text-primary text-base">memory</span>
                  <span class="font-headline-sm text-headline-sm text-on-surface">Hardware Diagnostics</span>
                </div>
                <span class="font-label-sm text-label-sm text-tertiary font-mono">NOMINAL</span>
              </div>
              <div class="flex flex-col gap-space-xs font-label-sm text-label-sm">
                <div>
                  <div class="flex justify-between text-on-surface-variant mb-1">
                    <span>Apple Silicon Unified Memory</span>
                    <span class="text-primary font-mono">Shared GPU/ANE Pool</span>
                  </div>
                  <div class="w-full h-1.5 rounded bg-surface-container-lowest overflow-hidden">
                    <div class="h-full bg-primary" style="width: 45%"></div>
                  </div>
                </div>
                <div>
                  <div class="flex justify-between text-on-surface-variant mb-1">
                    <span>Decoupled Ingest Thread</span>
                    <span class="text-secondary font-mono">30 FPS Constant</span>
                  </div>
                  <div class="w-full h-1.5 rounded bg-surface-container-lowest overflow-hidden">
                    <div class="h-full bg-secondary" style="width: 100%"></div>
                  </div>
                </div>
                <div>
                  <div class="flex justify-between text-on-surface-variant mb-1">
                    <span>Storage Retention Guard</span>
                    <span class="text-tertiary font-mono">Max 2GB Auto-Pruned</span>
                  </div>
                  <div class="w-full h-1.5 rounded bg-surface-container-lowest overflow-hidden">
                    <div class="h-full bg-tertiary" style="width: 25%"></div>
                  </div>
                </div>
              </div>
              <div class="bg-surface-container-lowest p-space-xs rounded flex items-center justify-between font-label-sm text-label-sm text-on-surface-variant mt-space-xs">
                <div class="flex items-center gap-1">
                  <span class="material-symbols-outlined text-xs text-tertiary">lock</span>
                  <span>100% On-Device · Zero Cloud</span>
                </div>
                <span class="text-primary font-mono">SECURE</span>
              </div>
            </div>

          </div>
        </div>
      </div>
    </main>
  </div>

  <!-- CLIENT-SIDE TELEMETRY SCRIPT -->
  <script>
    (function initSurveillanceTerminal() {
      // Live UTC Clock
      function updateClock() {
        const now = new Date();
        const utcStr = now.toISOString().slice(11, 19) + ' UTC';
        const el = document.getElementById('headerUtcClock');
        if (el) el.textContent = utcStr;
        const tsEl = document.getElementById('streamTimestampText');
        if (tsEl) tsEl.textContent = now.toISOString().replace('T', ' ').slice(0, 19);
      }
      setInterval(updateClock, 1000);
      updateClock();

      // Controls
      const pauseBtn = document.getElementById('pauseStreamBtn');
      const pauseIcon = document.getElementById('pauseBtnIcon');
      const img = document.getElementById('mainLiveFeed');
      let isPaused = false;
      if (pauseBtn && img) {
        pauseBtn.addEventListener('click', () => {
          isPaused = !isPaused;
          if (isPaused) {
            img.src = '';
            pauseIcon.textContent = 'play_arrow';
            pauseBtn.classList.replace('text-primary', 'text-error');
          } else {
            img.src = '/stream.mjpg?' + Date.now();
            pauseIcon.textContent = 'pause';
            pauseBtn.classList.replace('text-error', 'text-primary');
          }
        });
      }

      // Snapshot download
      const snapBtn = document.getElementById('snapshotBtn');
      if (snapBtn && img) {
        snapBtn.addEventListener('click', () => {
          const a = document.createElement('a');
          a.href = img.src;
          a.download = 'surveillance_snapshot_' + Date.now() + '.jpg';
          a.click();
        });
      }

      // Fullscreen
      const fsBtn = document.getElementById('toggleFullscreenBtn');
      const vidCont = document.getElementById('videoContainer');
      if (fsBtn && vidCont) {
        fsBtn.addEventListener('click', () => {
          if (!document.fullscreenElement) {
            vidCont.requestFullscreen().catch(() => {});
          } else {
            document.exitFullscreen().catch(() => {});
          }
        });
      }

      // Lockdown / Alert trigger simulation
      const alertBtn = document.getElementById('lockdownBtn');
      if (alertBtn) {
        alertBtn.addEventListener('click', () => {
          alert('DEFENSE ALERT: Operator initiated perimeter verification.');
        });
      }

      // Render Dynamic Telemetry from Pipeline
      function render(data) {
        if (!data) return;

        // FPS & Status
        const fps = (typeof data.fps === 'number') ? data.fps : 30.0;
        const fpsStr = fps.toFixed(1);
        const fpsBadge = document.getElementById('liveFpsBadge');
        if (fpsBadge) fpsBadge.textContent = fpsStr + ' FPS';
        const sideFps = document.getElementById('sidePipelineFps');
        if (sideFps) sideFps.textContent = fpsStr;
        const sideFpsText = document.getElementById('sidebarFpsText');
        if (sideFpsText) sideFpsText.textContent = fpsStr + ' FPS';
        const kpiFps = document.getElementById('kpiPipelineFps');
        if (kpiFps) kpiFps.textContent = fpsStr;

        // Motion status
        const motionPill = document.getElementById('motionPill');
        const motionBadge = document.getElementById('motionStatusBadge');
        if (data.motion) {
          if (motionPill) { motionPill.textContent = 'MOTION DETECTED'; motionPill.className = 'font-label-sm text-label-sm px-1.5 py-0.5 bg-tertiary-container text-on-tertiary-container rounded font-bold'; }
          if (motionBadge) { motionBadge.textContent = 'ACTIVE INFERENCE'; motionBadge.className = 'text-tertiary font-bold'; }
        } else {
          if (motionPill) { motionPill.textContent = 'STANDBY (THERMAL GUARD)'; motionPill.className = 'font-label-sm text-label-sm px-1.5 py-0.5 bg-surface-container text-on-surface-variant rounded'; }
          if (motionBadge) { motionBadge.textContent = 'THERMAL GUARD'; motionBadge.className = 'text-on-surface-variant'; }
        }

        // Latency
        const latText = document.getElementById('sideInferenceLatency');
        const kpiLat = document.getElementById('kpiLatencyText');
        if (fps > 0) {
          const lat = (1000 / fps).toFixed(1) + 'ms';
          if (latText) latText.textContent = lat;
          if (kpiLat) kpiLat.textContent = 'Latency: ' + lat;
        }

        // Entities
        const objects = data.objects || [];
        const objCount = document.getElementById('kpiObjectsCount');
        if (objCount) objCount.textContent = objects.length;

        const persons = objects.filter(o => o.label === 'person');
        const personCount = document.getElementById('kpiPersonsCount');
        if (personCount) personCount.textContent = persons.length;

        // Render Tracking Table
        const tbody = document.getElementById('trackingTableBody');
        if (tbody) {
          if (objects.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="py-space-md px-space-sm text-center text-on-surface-variant">No active entities tracked in camera field.</td></tr>';
          } else {
            let html = '';
            objects.forEach(obj => {
              const confPct = (obj.confidence * 100).toFixed(1) + '%';
              const dwell = (obj.dwell_seconds || 0).toFixed(1) + 's';
              const icon = (obj.label === 'person') ? 'person' : (obj.label === 'car' ? 'directions_car' : 'devices');
              const isThreat = (obj.label === 'knife' || obj.label === 'scissors');
              const rowClass = isThreat ? 'bg-error-container/20 text-error' : 'hover:bg-surface-container';
              const badgeClass = isThreat ? 'bg-error text-on-error font-bold uppercase' : 'bg-surface-container text-tertiary';
              const statusLabel = isThreat ? 'THREAT' : 'Tracking';

              html += `<tr class="${rowClass} transition-colors">
                <td class="py-space-xs px-space-sm font-mono text-primary font-bold">#${obj.id}</td>
                <td class="py-space-xs px-space-sm">
                  <div class="flex items-center gap-1.5">
                    <span class="material-symbols-outlined text-xs">${icon}</span>
                    <span class="capitalize">${obj.label}</span>
                  </div>
                </td>
                <td class="py-space-xs px-space-sm text-on-surface-variant font-mono">CAM-01</td>
                <td class="py-space-xs px-space-sm text-tertiary font-mono">${confPct}</td>
                <td class="py-space-xs px-space-sm font-mono">${dwell}</td>
                <td class="py-space-xs px-space-sm text-right">
                  <span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded ${badgeClass}">
                    ${statusLabel}
                  </span>
                </td>
              </tr>`;
            });
            tbody.innerHTML = html;
          }
        }

        // Zones & Alerts
        const events = data.events || [];
        const alertBadge = document.getElementById('kpiAlertBadge');
        const alertCount = document.getElementById('kpiAlertCount');
        const alertDetails = document.getElementById('kpiAlertDetails');
        const sideAlertBadge = document.getElementById('sidebarAlertCount');
        const notifDot = document.getElementById('notificationBadge');

        if (events.length > 0) {
          if (alertBadge) { alertBadge.textContent = 'ALERT'; alertBadge.className = 'px-1.5 py-0.5 rounded bg-error-container text-error font-label-sm text-label-sm font-bold animate-pulse'; }
          if (alertCount) { alertCount.textContent = events.length; alertCount.className = 'font-headline-lg text-headline-lg font-bold text-error'; }
          if (alertDetails) { alertDetails.textContent = events[0].details || 'Perimeter activity'; alertDetails.className = 'mt-space-xs font-label-sm text-label-sm text-error truncate'; }
          if (sideAlertBadge) { sideAlertBadge.textContent = events.length + ' Live'; sideAlertBadge.className = 'font-label-sm text-label-sm px-1.5 py-0.5 bg-error-container text-error rounded font-semibold'; }
          if (notifDot) notifDot.classList.remove('hidden');

          // Render Incidents Feed
          const incList = document.getElementById('incidentList');
          if (incList) {
            let incHtml = '';
            events.slice(0, 4).forEach((ev, idx) => {
              const rule = ev.rule || 'INCIDENT';
              const details = ev.details || 'Event logged';
              incHtml += `<div class="bg-error-container/30 rounded p-space-sm shadow-[0_0_12px_rgba(255,180,171,0.2)] flex flex-col gap-0.5 border border-error/20">
                <div class="flex items-center justify-between">
                  <span class="font-label-sm text-label-sm text-error font-bold tracking-wider flex items-center gap-1">
                    <span class="w-1.5 h-1.5 rounded-full bg-error animate-ping"></span>
                    ${rule}
                  </span>
                  <span class="font-label-sm text-label-sm text-error/80">Active</span>
                </div>
                <p class="font-headline-sm text-headline-sm text-on-surface font-bold">${details}</p>
              </div>`;
            });
            incList.innerHTML = incHtml;
          }
        } else {
          if (alertBadge) { alertBadge.textContent = 'CLEAR'; alertBadge.className = 'px-1.5 py-0.5 rounded bg-surface-container text-tertiary font-label-sm text-label-sm font-semibold'; }
          if (alertCount) { alertCount.textContent = '0'; alertCount.className = 'font-headline-lg text-headline-lg font-bold text-on-surface'; }
          if (alertDetails) { alertDetails.textContent = 'All Perimeters Secure'; alertDetails.className = 'mt-space-xs font-label-sm text-label-sm text-tertiary truncate'; }
          if (sideAlertBadge) { sideAlertBadge.textContent = '0 Live'; sideAlertBadge.className = 'font-label-sm text-label-sm px-1.5 py-0.5 bg-surface-container text-on-surface-variant rounded font-semibold'; }
          if (notifDot) notifDot.classList.add('hidden');
        }

        // Zones Status
        const zones = data.zones || {};
        if (zones.tripwire && zones.tripwire.triggered) {
          const zt = document.getElementById('zoneTripwireStatus');
          if (zt) { zt.textContent = 'TRIGGERED'; zt.className = 'font-label-sm text-label-sm px-1.5 py-0.5 rounded bg-error text-on-error font-bold animate-pulse'; }
        }
        if (zones.intrusion && zones.intrusion.triggered) {
          const zi = document.getElementById('zoneIntrusionStatus');
          if (zi) { zi.textContent = 'BREACH'; zi.className = 'font-label-sm text-label-sm px-1.5 py-0.5 rounded bg-error text-on-error font-bold animate-pulse'; }
        }
        if (zones.loitering && zones.loitering.triggered) {
          const zl = document.getElementById('zoneLoiteringStatus');
          if (zl) { zl.textContent = 'LOITERING'; zl.className = 'font-label-sm text-label-sm px-1.5 py-0.5 rounded bg-error text-on-error font-bold animate-pulse'; }
        }
      }

      // Connect Server-Sent Events (SSE)
      let es = null;
      function connectSSE() {
        es = new EventSource('/api/events');
        es.onmessage = function(e) {
          try {
            const data = JSON.parse(e.data);
            render(data);
          } catch(err) {}
        };
        es.onerror = function() {
          es.close();
          setTimeout(connectSSE, 3000);
        };
      }
      connectSSE();

      // Polling fallback
      setInterval(function() {
        fetch('/api/status', { cache: 'no-store' })
          .then(r => r.json())
          .then(render)
          .catch(() => {});
      }, 2000);

    })();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif path == "/api/status":
            with DASHBOARD_LOCK:
                payload = json.dumps({**DASHBOARD_STATE, "events": list(RECENT_EVENTS)}).encode()
            self._send(200, "application/json; charset=utf-8", payload)
        elif path == "/api/events":
            self._handle_sse()
        elif path == "/stream.mjpg":
            self._stream_frames()
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        client = self
        try:
            initial = json.dumps({**DASHBOARD_STATE, "events": list(RECENT_EVENTS)})
            self.wfile.write(f"data: {initial}\n\n".encode())
            self.wfile.flush()
            with _SSE_LOCK:
                _SSE_CLIENTS.append(client)
            while True:
                time.sleep(15)
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _SSE_LOCK:
                if client in _SSE_CLIENTS:
                    _SSE_CLIENTS.remove(client)

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_frames(self):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        try:
            while True:
                with FRAME_LOCK:
                    frame_bytes = LATEST_FRAME_JPEG
                if frame_bytes:
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode())
                    self.wfile.write(frame_bytes + b"\r\n")
                else:
                    time.sleep(1 / 10)
                time.sleep(1 / 30)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        return


def update_web_frame(frame):
    """Encode the current annotated frame for MJPEG viewers."""
    global LATEST_FRAME_JPEG
    if frame is None:
        return
    h, w = frame.shape[:2]
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if ok:
        with FRAME_LOCK:
            LATEST_FRAME_JPEG = jpeg.tobytes()
            FRAME_META.update({"width": w, "height": h, "last_update": time.time()})


def update_dashboard(status, has_motion, fps, tracklets, events=(),
                     camera_info=None, stream_health=None, zones=None,
                     detection_stats=None, recording=None):
    """Publish lightweight pipeline state used by the dashboard status endpoint."""
    objects = [{"id": t.track_id, "label": t.label, "confidence": round(float(t.confidence), 3),
                "dwell_seconds": round(float(t.dwell_time), 1)} for t in tracklets]
    with DASHBOARD_LOCK:
        DASHBOARD_STATE.update({
            "status": status,
            "motion": bool(has_motion),
            "fps": round(float(fps), 1),
            "objects": objects,
            "updated_at": time.time(),
        })
        if camera_info is not None:
            DASHBOARD_STATE["camera"] = camera_info
        if stream_health is not None:
            DASHBOARD_STATE["stream_health"] = stream_health
        if zones is not None:
            DASHBOARD_STATE["zones"] = zones
        if detection_stats is not None:
            DASHBOARD_STATE["detection_stats"] = detection_stats
        if recording is not None:
            DASHBOARD_STATE["recording"] = recording
        for event in events:
            RECENT_EVENTS.appendleft({"rule": event["rule"], "details": event["details"],
                                      "timestamp": event["timestamp"]})
    _notify_sse_clients()


def start_web_server(port=8080):
    """Start the local dashboard in a daemon thread and return its server."""
    server = ThreadingHTTPServer(("0.0.0.0", int(port)), DashboardHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[WebUI] Live dashboard: http://localhost:{port}")
    print(f"[WebUI] SSE events: http://localhost:{port}/api/events")
    return server
