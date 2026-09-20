import os
import numpy as np

COCO_SURVEILLANCE_LABELS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    44: "knife",
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    73: "book",
    76: "scissors",
    77: "teddy bear",
}

PERSON_MIN_SIZE = 40
PERSON_MIN_AREA = 2000

class EdgeDetector:
    """
    Hardware-accelerated Object Detector for Apple Silicon.
    Automatically prioritizes:
      1. CoreML (.mlpackage) on Apple Neural Engine (ANE)
      2. PyTorch YOLO on Metal Performance Shaders (MPS)
      3. Synthetic Mock detector for offline verification & tests
    """
    def __init__(self, model_path="yolo11n.pt", conf_threshold=0.45, target_classes=None, mode="auto",
                 min_detection_area=1500, min_person_size=40, min_person_area=2000):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or list(COCO_SURVEILLANCE_LABELS.keys())
        self.mode = mode
        self.min_detection_area = min_detection_area
        self.min_person_size = min_person_size
        self.min_person_area = min_person_area
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

    def _passes_filters(self, cls_id, label, x1, y1, x2, y2, conf):
        """Filter out false-positive detections based on class-specific size thresholds."""
        w = x2 - x1
        h = y2 - y1
        area = w * h

        if label == "person":
            if w < self.min_person_size or h < self.min_person_size:
                return False
            if area < self.min_person_area:
                return False

        if area < self.min_detection_area:
            if label == "person":
                return False

        return True

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

                    if not self._passes_filters(cls_id, label, x1, y1, x2, y2, conf):
                        continue

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
