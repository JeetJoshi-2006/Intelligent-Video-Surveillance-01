import os
import numpy as np

COCO_SURVEILLANCE_LABELS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}

class EdgeDetector:
    """
    Hardware-accelerated Object Detector for Apple Silicon.
    Automatically prioritizes:
      1. CoreML (.mlpackage) on Apple Neural Engine (ANE)
      2. PyTorch YOLO on Metal Performance Shaders (MPS)
      3. Synthetic Mock detector for offline verification & tests
    """
    def __init__(self, model_path="yolo11n.pt", conf_threshold=0.45, target_classes=None, mode="auto"):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or list(COCO_SURVEILLANCE_LABELS.keys())
        self.mode = mode
        self.model = None
        self.device = "cpu"
        self._init_backend()

    def _init_backend(self):
        if self.mode == "mock":
            print("[EdgeDetector] Initialized in synthetic MOCK mode.")
            return

        try:
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

        try:
            from ultralytics import YOLO
            # If a CoreML package exists, prioritize it for the Apple Neural Engine
            coreml_path = self.model_path.replace(".pt", ".mlpackage")
            if os.path.exists(coreml_path):
                print(f"[EdgeDetector] Loading CoreML engine: {coreml_path}")
                self.model = YOLO(coreml_path)
            else:
                print(f"[EdgeDetector] Loading model on device='{self.device}': {self.model_path}")
                self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"[EdgeDetector] Deep learning backend unavailable ({e}). Falling back to test mode.")
            self.mode = "mock"

    def detect(self, frame):
        """
        Runs object detection on a BGR image frame.
        Returns:
            list of dicts: [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int, 'label': str}]
        """
        if frame is None:
            return []

        if self.mode == "mock" or self.model is None:
            return []

        try:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                classes=self.target_classes,
                device=self.device,
                verbose=False
            )
            
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = COCO_SURVEILLANCE_LABELS.get(cls_id, f"obj_{cls_id}")
                    
                    detections.append({
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "confidence": round(conf, 3),
                        "class_id": cls_id,
                        "label": label
                    })
            return detections
        except Exception as e:
            print(f"[EdgeDetector] Inference error: {e}")
            return []
