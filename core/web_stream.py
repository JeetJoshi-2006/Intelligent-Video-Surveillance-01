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


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel | Surveillance Monitor</title>
<style>
:root{
  --bg:#07111f; --panel:#0c1a2d; --panel2:#091627; --border:#20354e;
  --muted:#8fa4bd; --text:#eef6ff; --cyan:#46d9ff; --green:#4ee29a;
  --red:#ff6b76; --amber:#ffc43b; --blue:#5c84ff; --purple:#a78af9;
  --danger:#ff6b76; --warn:#ffc43b; --info:#46d9ff;
}
*{box-sizing:border-box;-webkit-font-smoothing:antialiased}
body{
  margin:0; min-height:100vh;
  background:radial-gradient(circle at 75% -20%,#153b5b 0,transparent 38%),var(--bg);
  color:var(--text); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  overflow-x:hidden;
}
.shell{max-width:1600px;margin:auto;padding:24px}
header{
  display:flex;align-items:center;justify-content:space-between;gap:16px;
  padding-bottom:16px;border-bottom:1px solid var(--border);margin-bottom:22px;
}
.brand{display:flex;gap:12px;align-items:center}
.mark{
  width:42px;height:42px;display:grid;place-items:center;
  background:linear-gradient(135deg,#29cae9,#3d64dc);border-radius:11px;
  box-shadow:0 0 28px #38d8ff45;font-size:22px;font-weight:700;
}
h1{font-size:18px;margin:0;letter-spacing:.03em}
.sub{color:var(--muted);font-size:12px;margin-top:3px}
.right-header{display:flex;align-items:center;gap:14px}
.live-pill{
  display:flex;align-items:center;gap:7px;
  font-size:12px;font-weight:700;letter-spacing:.08em;
  padding:6px 14px;border-radius:20px;background:var(--panel);border:1px solid var(--border);
  transition:border-color .3s,box-shadow .3s;
}
.live-pill.on{border-color:var(--green);box-shadow:0 0 14px #4ee29a33}
.live-pill.on::before{
  content:"";display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--green);box-shadow:0 0 10px var(--green);
  animation:pulse 1.5s infinite;
}
.live-pill.off::before{
  content:"";display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--muted);
}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.connection-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px}
.dot-connected{background:var(--green);box-shadow:0 0 12px var(--green)}
.dot-connecting{background:var(--amber);box-shadow:0 0 10px var(--amber)}
.dot-disconnected{background:var(--red);box-shadow:0 0 10px var(--red)}

.grid{
  display:grid;
  grid-template-columns:1fr 340px;
  gap:18px;
}
.panel{
  background:linear-gradient(145deg,var(--panel2),var(--panel));
  border:1px solid var(--border);border-radius:15px;box-shadow:0 18px 44px #0003;
  overflow:hidden;
}
.video{position:relative}
.video-head,.side-head,.mini-head{
  display:flex;justify-content:space-between;align-items:center;
  padding:13px 18px;border-bottom:1px solid var(--border);
}
.camera-name{font-weight:650;font-size:14px}
.tag{
  font-size:11px;color:var(--cyan);border:1px solid #23718a;
  padding:4px 10px;border-radius:999px;
}
.feed{
  position:relative;min-height:380px;background:#02070d;display:grid;place-items:center;
}
.feed img{width:100%;display:block;max-height:72vh;object-fit:contain;z-index:1}
.feed .corner{
  position:absolute;top:14px;left:14px;z-index:2;
  border-left:2px solid var(--cyan);border-top:2px solid var(--cyan);
  width:26px;height:26px;opacity:.8;pointer-events:none;
}
.zone-overlay{
  position:absolute;inset:0;pointer-events:none;z-index:0;
}
.zone-dot{
  position:absolute;width:10px;height:10px;border-radius:50%;
  border:1px solid var(--text);opacity:.85;transform:translate(-50%,-50%);
  box-shadow:0 0 8px currentColor;
}
.zone-dot.inactive{opacity:.25;background:var(--muted);box-shadow:none}
.zone-dot.warn{background:var(--amber);color:var(--amber)}
.zone-dot.crit{background:var(--red);color:var(--red);animation:pulse 1.2s infinite}

.empty{
  position:absolute;color:var(--muted);z-index:0;
  font-style:italic;text-align:center;max-width:80%;
}

.stats-grid{
  display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
  background:var(--border);border-radius:0 0 15px 15px;overflow:hidden;
}
.stat{
  background:var(--panel);padding:14px 16px;text-align:center;
  display:flex;flex-direction:column;gap:3px;
}
.stat small{
  color:var(--muted);font-size:11px;text-transform:uppercase;
  letter-spacing:.06em;font-weight:600;
}
.stat strong{
  font-size:20px;font-weight:700;
  display:flex;align-items:center;justify-content:center;gap:5px;
}
.stat.ok strong{color:var(--green)}
.stat.warn strong{color:var(--amber)}
.stat.crit strong{color:var(--red)}
.stat .trend{
  font-size:11px;color:var(--muted);margin-top:2px;
  display:flex;align-items:center;gap:3px;
}
.trend.up{color:var(--green)}
.trend.down{color:var(--red)}

.side-section{
  padding:0 0 16px 0;
}
.side-section:last-child{border-top:1px solid var(--border)}
.side-head{border-top:1px solid var(--border)}
.side-head:first-child{border-top:none}
.count{
  font-size:11px;color:var(--muted);font-weight:600;
}
.object-list,.events{
  max-height:280px;overflow-y:auto;padding:8px 0;
}
.object-item{
  display:flex;justify-content:space-between;align-items:center;
  padding:9px 16px;border-bottom:1px solid var(--border);
}
.object-item:last-child{border-bottom:none}
.object-item .obj-left{display:flex;align-items:center;gap:10px}
.obj-dot{width:10px;height:10px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);flex-shrink:0}
.obj-dot.threat{background:var(--red);box-shadow:0 0 8px var(--red)}
.obj-dot.tech{background:var(--blue);box-shadow:0 0 8px var(--blue)}
.obj-dot.baggage{background:var(--amber);box-shadow:0 0 8px var(--amber)}
.obj-label{font-weight:600;font-size:13px}
.obj-sub{color:var(--muted);font-size:11px;margin-top:2px}
.obj-conf{
  font-size:11px;color:var(--cyan);font-weight:600;
  background:var(--panel);border:1px solid var(--border);
  padding:2px 8px;border-radius:6px;
}
.confidence-bar{
  width:60px;height:4px;background:var(--border);border-radius:2px;
  overflow:hidden;margin-top:2px;
}
.confidence-fill{height:100%;background:var(--green);transition:width .3s}
.obj-dwell{margin-left:auto;text-align:right;font-size:11px;color:var(--muted)}
.event-item{
  padding:12px 16px;border-bottom:1px solid var(--border);
  animation:slideIn .3s ease;
}
.event-item:last-child{border-bottom:none}
.event-item .evt-head{
  display:flex;justify-content:space-between;align-items:center;gap:8px;
}
.event-badge{
  font-size:10px;font-weight:700;text-transform:uppercase;
  padding:2px 8px;border-radius:5px;letter-spacing:.05em;
}
.badge-tripwire{background:#23718a33;color:var(--cyan);border:1px solid #23718a}
.badge-intrusion{background:#ff6b7633;color:var(--red);border:1px solid var(--red)}
.badge-loitering{background:#ffc43b33;color:var(--amber);border:1px solid var(--amber)}
.event-details{
  color:var(--muted);font-size:12px;margin-top:4px;line-height:1.5;
}
.event-time{
  font-size:11px;color:var(--muted);margin-top:6px;text-align:right;
}
@keyframes slideIn{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}
.none{color:var(--muted);text-align:center;padding:24px;font-style:italic;font-size:13px}

.zone-indicator{
  display:flex;gap:6px;flex-wrap:wrap;padding:8px 0;
}
.zone-chip{
  display:flex;align-items:center;gap:5px;
  font-size:11px;color:var(--muted);padding:5px 10px;
  border-radius:6px;background:var(--panel);border:1px solid var(--border);
}
.zone-chip.enabled{color:var(--text);background:rgba(78,226,154,.08);border-color:rgba(78,226,154,.3)}
.zone-chip.triggered{
  color:var(--text);background:rgba(255,107,118,.15);
  border-color:var(--red);font-weight:600;
  box-shadow:0 0 10px #ff6b7644;
}

@media(max-width:1000px){
  .grid{grid-template-columns:1fr}
  header{flex-direction:column;align-items:flex-start;gap:12px}
}
</style>
</head>
<body>
<main class="shell">
  <header>
    <div class="brand">
      <div class="mark"><b>S</b></div>
      <div>
        <h1>SENTINEL MONITOR</h1>
        <div class="sub">Edge AI surveillance console</div>
      </div>
    </div>
    <div class="right-header">
      <div class="live-pill" id="livePill">
        <span class="connection-dot dot-connecting" id="connDot"></span>
        <span id="connText">CONNECTING</span>
      </div>
    </div>
  </header>

  <div class="grid">
    <div class="panel video">
      <div class="video-head">
        <span class="camera-name">Camera 01 · Live feed</span>
        <span class="tag" id="modeTag">INITIALIZING</span>
      </div>
      <div class="feed">
        <div class="corner"></div>
        <div class="zone-overlay" id="zoneOverlay"></div>
        <span class="empty" id="emptyState">Waiting for camera frames…</span>
        <img src="/stream.mjpg" alt="Live surveillance feed" id="feedImg"
             onload="document.getElementById('emptyState').style.display='none'">
      </div>
      <div class="stats-grid">
        <div class="stat" id="statPipeline">
          <small>Pipeline</small><strong class="ok">—</strong>
        </div>
        <div class="stat" id="statFps">
          <small>Inference rate</small><strong>—</strong>
          <div class="trend" id="fpsTrend"></div>
        </div>
        <div class="stat" id="statObjects">
          <small>Tracked objects</small><strong>—</strong>
        </div>
        <div class="stat" id="statAlerts">
          <small>Total alerts</small><strong>—</strong>
          <div class="trend" id="alertTrend"></div>
        </div>
      </div>
    </div>

    <aside class="panel side">
      <div class="side-head">
        <span>Active detections</span><span class="count" id="detectionsCount">0 visible</span>
      </div>
      <div class="object-list" id="objects">
        <div class="none">No objects currently tracked</div>
      </div>

      <div class="side-head mini-head">
        <span>Detection zones</span><span class="count" id="zoneSummary">—</span>
      </div>
      <div class="zone-indicator" id="zoneIndicator">
        <span class="none">No zones configured</span>
      </div>

      <div class="side-section">
        <div class="side-head"><span>Recent alerts</span><span class="count">latest 12</span></div>
        <div class="events" id="events">
          <div class="none">No alerts recorded</div>
        </div>
      </div>
    </aside>
  </div>
</main>
<script>
const esc=v=>String(v).replace(/[&<>"']/g,c=>({
  '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));

let fpsHistory=[];
let alertHistory=[];
let lastFps=0;

function categoryMeta(label){
  const map={
    person:'human',bicycle:'vehicle',car:'vehicle',motorcycle:'vehicle',bus:'vehicle',
    truck:'vehicle',backpack:'baggage',handbag:'baggage',suitcase:'baggage',
    knife:'threat',scissors:'threat',
    laptop:'tech',cell phone:'tech',keyboard:'tech',mouse:'tech',tv:'tech'
  };
  return map[(label||'').toLowerCase()]||'general';
}
function dotClass(cat){
  const m={threat:'threat',tech:'tech',baggage:'baggage'};
  return m[cat]||'';
}

function setStatusPill(status, motion){
  const pill=document.getElementById('livePill');
  const dot=document.getElementById('connDot');
  const text=document.getElementById('connText');
  const tag=document.getElementById('modeTag');
  if(status==='Active'){
    dot.className='connection-dot dot-connected';
    text.textContent='LIVE';
    pill.classList.remove('off');pill.classList.add('on');
  }else if(status==='Starting'||status==='Reconnecting'){
    dot.className='connection-dot dot-connecting';
    text.textContent=status.toUpperCase();
    pill.classList.remove('on','off');
  }else{
    dot.className='connection-dot dot-disconnected';
    text.textContent='OFFLINE';
    pill.classList.remove('on');pill.classList.add('off');
  }
  tag.textContent=motion?'MOTION DETECTED':'SCENE CLEAR';
  tag.style.borderColor=motion?'#ff6b76':'#23718a';
  tag.style.color=motion?'var(--red)':'var(--cyan)';
}

function renderStats(d){
  document.querySelector('#statPipeline strong').textContent=d.status||'—';
  document.querySelector('#statPipeline .stat').className='stat '+((d.status==='Active')?'ok':(d.status==='Error'?'crit':'warn'));
  const fpsEl=document.querySelector('#statFps strong');
  const fpsTrend=document.getElementById('fpsTrend');
  if(d.fps&&d.fps>0){
    fpsEl.textContent=d.fps.toFixed(1)+' FPS';
    fpsHistory.push(d.fps);
    if(fpsHistory.length>10)fpsHistory.shift();
    const trend=fpsHistory.length>1?fpsHistory[0]-fpsHistory[1]:0;
    fpsTrend.textContent=trend>=0?'▲ '+(Math.abs(trend).toFixed(1)):'▼ '+(Math.abs(trend).toFixed(1));
    fpsTrend.className='trend'+(trend>=0?' up':' down');
  }else{
    fpsEl.textContent='—';fpsTrend.textContent='';
  }
  lastFps=d.fps;
  document.querySelector('#statObjects strong').textContent=d.objects?.length||0;
  const alertCount=d.detection_stats?.total_alerts||0;
  document.querySelector('#statAlerts strong').textContent=alertCount;
  const aTrend=alertHistory.length?(alertCount-alertHistory[0]):0;
  const aEl=document.getElementById('alertTrend');
  aEl.textContent=aTrend>0?'● +'+(aTrend):'';
  aEl.className='trend'+(aTrend>=0?' up':' down');
  alertHistory.push(alertCount);
  if(alertHistory.length>10)alertHistory.shift();
}

function renderObjects(o){
  const list=document.getElementById('objects');
  document.getElementById('detectionsCount').textContent=(o||[]).length+' visible';
  if(!o||!o.length){
    list.innerHTML='<div class="none">No objects currently tracked</div>';
    return;
  }
  list.innerHTML=o.map(x=>{
    const cat=categoryMeta(x.label);
    const dot=dotClass(cat);
    const confBar=x.confidence?Math.round(x.confidence*100):0;
    return `<div class="object-item">
      <div class="obj-left">
        <span class="obj-dot ${dot}"></span>
        <div>
          <div class="obj-label">${esc(x.label)} <span style="color:var(--muted)">#${x.id}</span></div>
          <div class="obj-sub">${cat.toUpperCase()} · Confidence ${confBar}%</div>
          <div class="confidence-bar"><div class="confidence-fill" style="width:${confBar}%"></div></div>
        </div>
      </div>
      <div class="obj-dwell">
        <div>${Math.round(x.dwell_seconds)}s</div>
        <div class="obj-conf">${confBar}%</div>
      </div>
    </div>`;
  }).join('');
}

function renderEvents(e){
  const list=document.getElementById('events');
  if(!e||!e.length){
    list.innerHTML='<div class="none">No alerts recorded</div>';
    return;
  }
  list.innerHTML=e.map(x=>{
    const rule=x.rule||'';
    const badge=rule.includes('TRIPWIRE')?'badge-tripwire':
      rule.includes('INTRUSION')?'badge-intrusion':
      rule.includes('LOITERING')?'badge-loitering':'badge-tripwire';
    const when=new Date(x.timestamp*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});
    return `<div class="event-item">
      <div class="evt-head">
        <span class="event-badge ${badge}">${esc(rule.replace(/_/g,' '))}</span>
      </div>
      <div class="event-details">${esc(x.details)}</div>
      <div class="event-time">${when}</div>
    </div>`;
  }).join('');
  list.scrollTop=0;
}

function renderZones(zones){
  const el=document.getElementById('zoneIndicator');
  if(!zones){el.innerHTML='<span class="none">No zones configured</span>';return;}
  let html='';
  Object.entries(zones).forEach(([name,info])=>{
    if(!info||!info.enabled)return;
    const cls=info.triggered?'triggered':'enabled';
    const labels={'tripwire':'Virtual Tripwire','intrusion':'Intrusion Polygon','loitering':'Loitering Monitor'};
    html+=`<div class="zone-chip ${cls}">${labels[name]||name}</div>`;
  });
  const trigCount=Object.values(zones).filter(v=>v&&v.triggered).length;
  const enCount=Object.values(zones).filter(v=>v&&v.enabled).length;
  document.getElementById('zoneSummary').textContent=trigCount+'/'+enCount+' triggered';
  el.innerHTML=html||'<span class="none">No zones configured</span>';
}

function renderOverlay(d){
  // No extra overlay dots needed; zones are baked into the frame via the HUD
}

function render(d){
  setStatusPill(d.status,d.motion);
  renderStats(d);
  renderObjects(d.objects||[]);
  renderEvents(d.events||[]);
  renderZones(d.zones);
  renderOverlay(d);
}

let es=null;
function connectSSE(){
  es=new EventSource('/api/events');
  es.onmessage=function(e){
    try{render(JSON.parse(e.data))}catch(_){}
  };
  es.onerror=function(){
    es.close();
    const pill=document.getElementById('livePill');
    const dot=document.getElementById('connDot');
    dot.className='connection-dot dot-connecting';
    document.getElementById('connText').textContent='RECONNECTING';
    setTimeout(connectSSE,2000);
  };
}
connectSSE();

// Fallback: also poll once in case SSE fails entirely
setInterval(function(){
  if(es&&es.readyState===2){
    fetch('/api/status',{cache:'no-store'})
      .then(r=>r.json())
      .then(render)
      .catch(()=>{});
  }
},3000);
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
        self.send_response(status); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def _stream_frames(self):
        self.send_response(200); self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME"); self.end_headers()
        try:
            while True:
                with FRAME_LOCK:
                    frame_bytes = LATEST_FRAME_JPEG
                    meta = dict(FRAME_META)
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
