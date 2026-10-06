#!/usr/bin/env python3
"""Diagnostic & Optimization Tool for Knife / Threat Detection.

Usage:
    python scripts/test_knife_detection.py --image path/to/image.jpg
    python scripts/test_knife_detection.py --source 0
    python scripts/test_knife_detection.py --source recordings/sample.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import load_config
from core.detector import EdgeDetector


def test_knife_detection(source, config_path="config/settings.yaml", imgsz=None, min_conf=None):
    cfg = load_config(config_path)
    det_cfg = cfg.get("detector", {})

    if imgsz:
        det_cfg["inference_size"] = imgsz
    if min_conf is not None:
        det_cfg["class_confidences"][43] = min_conf

    print("=" * 60)
    print("      KNIFE / THREAT DETECTION DIAGNOSTIC SUITE")
    print("=" * 60)
    print(f"Model Path:         {det_cfg.get('model_path')}")
    print(f"Inference Size:     {det_cfg.get('inference_size')} px")
    print(f"Knife Class ID:     43")
    print(f"Knife Sensitivity:  {det_cfg.get('class_confidences', {}).get(43, 0.22)} threshold")
    print(f"Target Classes:     {det_cfg.get('target_classes')}")
    print("=" * 60)

    detector = EdgeDetector(
        model_path=det_cfg.get("model_path", "yolo11m.pt"),
        conf_threshold=det_cfg.get("confidence_threshold", 0.50),
        class_confidences=det_cfg.get("class_confidences", {}),
        target_classes=det_cfg.get("target_classes"),
        mode=det_cfg.get("model_type", "yolo"),
        min_person_size=det_cfg.get("min_person_size", 50),
        min_person_area=det_cfg.get("min_person_area", 3500),
        inference_size=det_cfg.get("inference_size", 640),
    )

    print(f"Active Backend:     {detector.backend_name}")
    print(f"Device:             {detector.device}")
    print("-" * 60)

    # Check if source is image file
    if source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        frame = cv2.imread(source)
        if frame is None:
            print(f"[Error] Failed to load image: {source}")
            return
        t0 = time.time()
        detections = detector.detect(frame)
        latency = (time.time() - t0) * 1000

        print(f"[Inference Complete] {latency:.1f}ms | Total Detections: {len(detections)}")
        knife_found = False
        for i, d in enumerate(detections):
            is_threat = d["label"] in ("knife", "scissors")
            prefix = ">>> THREAT DETECTED <<<" if is_threat else "   "
            print(f"{prefix} [{i+1}] {d['label'].upper()} (ID: {d['class_id']}) "
                  f"Conf: {d['confidence']:.3f} | BBox: {d['bbox']}")
            if d["label"] == "knife":
                knife_found = True

        if not knife_found:
            print("\n[Analysis] No knife detected above threshold.")
            print("Troubleshooting steps:")
            print("1. Increase inference resolution: try --imgsz 800 or 960")
            print("2. Lower knife confidence: try --min-conf 0.18")
            print("3. Ensure the knife blade is not heavily shadowed or motion blurred.")
        return

    # Video stream / webcam
    cap_src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[Error] Failed to open video source: {source}")
        return

    print("Starting video feed... Press 'q' to exit.")
    frame_count = 0
    threat_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            detections = detector.detect(frame)

            for d in detections:
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                lbl = d["label"]
                conf = d["confidence"]
                if lbl == "knife":
                    threat_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"THREAT: KNIFE {conf:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, f"{lbl} {conf:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            cv2.imshow("Knife Detection Diagnostic", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Finished. Total Frames: {frame_count}, Knife Detections: {threat_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Knife Detection")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID (0), video path, or image path")
    parser.add_argument("--image", type=str, default=None, help="Image file path (shortcut for --source)")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference resolution override (e.g. 640, 800, 960)")
    parser.add_argument("--min-conf", type=float, default=None, help="Knife confidence threshold override (e.g. 0.20)")

    args = parser.parse_args()
    src = args.image if args.image else args.source
    test_knife_detection(src, config_path=args.config, imgsz=args.imgsz, min_conf=args.min_conf)
