#!/usr/bin/env python3
"""
Apple Neural Engine (ANE) CoreML Exporter for Intelligent Video Surveillance.
Exports PyTorch YOLO weights (.pt) to Apple CoreML (.mlpackage) with baked-in NMS
and FP16 quantization for low-power, zero-thermal-throttling edge inference.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to Apple CoreML (.mlpackage)")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Path or name of YOLO model (e.g. yolo11m.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution (default: 640)")
    parser.add_argument("--half", action="store_true", default=True, help="Export FP16 half precision for ANE")
    parser.add_argument("--nms", action="store_true", default=True, help="Include Non-Maximum Suppression layer in CoreML model")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    model_path = Path(args.model)
    if not model_path.is_file():
        candidate = project_root / args.model
        if candidate.is_file():
            model_path = candidate
        else:
            print(f"[ERROR] Model file not found: {args.model}")
            sys.exit(1)

    print("=" * 70)
    print("  APPLE NEURAL ENGINE (ANE) COREML EXPORTER")
    print(f"  Source Model:     {model_path}")
    print(f"  Input Resolution: {args.imgsz}x{args.imgsz}")
    print(f"  FP16 Precision:   {args.half}")
    print(f"  Fused NMS Layer:  {args.nms}")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' is not installed in the current environment.")
        sys.exit(1)

    try:
        import coremltools
        print(f"[INFO] Found coremltools {coremltools.__version__}")
    except ImportError:
        print("[WARNING] 'coremltools' is not yet installed.")
        print("[INFO] Attempting install or export via ultralytics...")

    print(f"[EXPORT] Loading {model_path}...")
    model = YOLO(str(model_path))

    print("[EXPORT] Compiling to Apple CoreML format (.mlpackage)...")
    try:
        exported_path = model.export(
            format="coreml",
            imgsz=args.imgsz,
            half=args.half,
            nms=args.nms
        )
        print("=" * 70)
        print("  ✓ COREML EXPORT SUCCESSFUL!")
        print(f"  Compiled package: {exported_path}")
        print("  Update 'model_path' in config/settings.yaml to use this .mlpackage for 0% CPU ANE execution.")
        print("=" * 70)
    except Exception as e:
        print(f"[ERROR] CoreML export failed: {e}")
        print("Note: Run 'pip install coremltools' in your venv to enable CoreML compilation.")
        sys.exit(2)

if __name__ == "__main__":
    main()
