import sys, time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting diagnostic...")
log(f"Python: {sys.version}")

log("Importing torch...")
t0 = time.time()
import torch
log(f"Torch imported in {time.time()-t0:.2f}s, version: {torch.__version__}")

log(f"MPS available: {torch.backends.mps.is_available()}")
log(f"MPS built: {torch.backends.mps.is_built()}")

log("Importing torchvision...")
t0 = time.time()
import torchvision
log(f"Torchvision imported in {time.time()-t0:.2f}s, version: {torchvision.__version__}")

log("Importing ultralytics...")
t0 = time.time()
import ultralytics
log(f"Ultralytics imported in {time.time()-t0:.2f}s, version: {ultralytics.__version__}")

log("Done with imports!")
