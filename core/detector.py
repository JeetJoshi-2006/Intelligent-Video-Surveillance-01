import os
import time
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
    43: "knife",
    44: "spoon",
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
    65: "remote",
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
    def __init__(self, model_path="yolo11m.pt", conf_threshold=0.45, class_confidences=None,
                 target_classes=None, mode="auto", min_detection_area=1500,
                 min_person_size=40, min_person_area=2000, inference_size=640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.class_confidences = class_confidences or {}
        self.target_classes = target_classes or list(COCO_SURVEILLANCE_LABELS.keys())
        self.mode = mode
        self.min_detection_area = min_detection_area
        self.min_person_size = min_person_size
        self.min_person_area = min_person_area
        self.inference_size = inference_size
        self.model = None
        self.device = "cpu"
        self.backend_name = "Mock" if mode == "mock" else "Unknown"
        self.last_latency_ms = 0.0

        # Calculate lowest threshold needed for model predict()
        all_thresholds = [self.conf_threshold] + list(self.class_confidences.values())
        self._min_model_conf = max(0.1, min(all_thresholds)) if all_thresholds else self.conf_threshold

        self._init_backend()

    def _init_backend(self):
        if self.mode == "mock":
            self.backend_name = "Mock (Test Mode)"
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
            coreml_path = self.model_path if self.model_path.endswith(".mlpackage") else self.model_path.replace(".pt", ".mlpackage")
            if os.path.exists(coreml_path):
                print(f"[EdgeDetector] Loading CoreML engine: {coreml_path}")
                self.model = YOLO(coreml_path)
                self.backend_name = "CoreML (Apple Neural Engine)"
            else:
                print(f"[EdgeDetector] Loading model on device='{self.device}': {self.model_path}")
                self.model = YOLO(self.model_path)
                self.backend_name = f"PyTorch ({self.device.upper()})"
                # Warmup on MPS to avoid first-inference hang
                if self.device == "mps":
                    print(f"[EdgeDetector] Warming up Metal (MPS) shaders on Apple Silicon...")
                    try:
                        t_w = time.time()
                        import torch
                        dummy = torch.zeros(1, 3, self.inference_size, self.inference_size, device="mps")
                        self.model.predict(dummy, device="mps", verbose=False)
                        print(f"[EdgeDetector] Metal shader warmup complete ({time.time()-t_w:.2f}s)")
                    except Exception as ex:
                        print(f"[EdgeDetector] Warmup skipped ({ex})")
        except Exception as e:
            print(f"[EdgeDetector] Deep learning backend unavailable ({e}). Falling back to test mode.")
            self.mode = "mock"
            self.backend_name = "Mock (Fallback)"

    def _passes_filters(self, cls_id, label, x1, y1, x2, y2, conf):
        """Filter out false-positive detections based on class-specific size and confidence thresholds."""
        required_conf = self.class_confidences.get(cls_id, self.conf_threshold)
        if conf < required_conf:
            return False

        w = x2 - x1
        h = y2 - y1
        area = w * h

        if label == "person":
            if w < self.min_person_size or h < self.min_person_size:
                return False
            if area < self.min_person_area:
                return False
            # Humans are predominantly vertical; reject wide horizontal slabs (e.g. chairs, jackets, cushions)
            aspect_ratio = h / max(1.0, float(w))
            if aspect_ratio < 0.65 or aspect_ratio > 4.5:
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
            t0 = time.time()
            try:
                results = self.model.predict(
                    frame,
                    conf=self._min_model_conf,
                    iou=0.45,
                    imgsz=self.inference_size,
                    classes=self.target_classes,
                    device=self.device,
                    verbose=False
                )
            except Exception as pe:
                if self.device == "mps":
                    print(f"[EdgeDetector] MPS predict issue ({pe}), falling back to CPU...")
                    self.device = "cpu"
                    self.backend_name = "PyTorch (CPU)"
                    results = self.model.predict(
                        frame,
                        conf=self._min_model_conf,
                        iou=0.45,
                        imgsz=self.inference_size,
                        classes=self.target_classes,
                        device="cpu",
                        verbose=False
                    )
                else:
                    raise pe
            self.last_latency_ms = (time.time() - t0) * 1000.0

            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    if hasattr(self, "model") and hasattr(self.model, "names") and cls_id in self.model.names:
                        label = str(self.model.names[cls_id]).lower()
                    else:
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
