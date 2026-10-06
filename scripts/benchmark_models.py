#!/usr/bin/env python3
"""
Hardware Benchmark & Profiler for Apple Silicon Intelligent Surveillance.
Compares latency (ms), FPS, and resource utilization across YOLO model sizes
(YOLO11n, YOLO11s, YOLO11m) on Metal Performance Shaders (MPS) and CoreML (ANE).
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

def benchmark_single_model(model_path, iterations=50, imgsz=640):
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is required for benchmarking.")
        return None

    model_name = Path(model_path).name
    is_coreml = str(model_path).endswith(".mlpackage") or os.path.isdir(str(model_path))

    print(f"\n[BENCHMARK] Loading {model_name}...")
    t_load_start = time.time()
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"  [ERROR] Failed to load {model_name}: {e}")
        return None
    load_time = (time.time() - t_load_start) * 1000.0

    # Create dummy 1280x720 RGB surveillance frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warmup runs (5 frames)
    print(f"  Warming up {device.upper()} pipeline...")
    for _ in range(5):
        if is_coreml:
            model.predict(dummy_frame, imgsz=imgsz, verbose=False)
        else:
            model.predict(dummy_frame, imgsz=imgsz, device=device, verbose=False)

    if device == "mps":
        try:
            import torch
            torch.mps.synchronize()
        except Exception:
            pass

    # Timing iterations
    latencies = []
    print(f"  Running {iterations} timed inference iterations...")
    for _ in range(iterations):
        t0 = time.perf_counter()
        if is_coreml:
            model.predict(dummy_frame, imgsz=imgsz, verbose=False)
        else:
            model.predict(dummy_frame, imgsz=imgsz, device=device, verbose=False)
            if device == "mps":
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    min_lat = float(np.min(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    # Get file size in MB
    p = Path(model_path)
    if p.is_file():
        file_size_mb = p.stat().st_size / (1024 * 1024)
    elif p.is_dir():
        file_size_mb = sum(f.stat().st_size for f in p.glob('**/*') if f.is_file()) / (1024 * 1024)
    else:
        file_size_mb = 0.0

    return {
        "model": model_name,
        "backend": "CoreML (ANE)" if is_coreml else f"PyTorch ({device.upper()})",
        "size_mb": file_size_mb,
        "load_ms": load_time,
        "mean_ms": mean_lat,
        "min_ms": min_lat,
        "p95_ms": p95_lat,
        "fps": fps
    }

def main():
    parser = argparse.ArgumentParser(description="Profile and benchmark YOLO models on Apple Silicon.")
    parser.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                        help="List of model paths to benchmark")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark frames")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    print("=" * 80)
    print("  APPLE SILICON SURVEILLANCE INFERENCE PROFILER & BENCHMARK")
    print(f"  Target Resolution: {args.imgsz}x{args.imgsz} | Sample Size: {args.iterations} frames")
    print("=" * 80)

    results = []
    for model_name in args.models:
        model_path = Path(model_name)
        if not model_path.exists():
            candidate = project_root / model_name
            if candidate.exists():
                model_path = candidate
            else:
                print(f"[SKIP] Model not found: {model_name}")
                continue
        res = benchmark_single_model(model_path, iterations=args.iterations, imgsz=args.imgsz)
        if res:
            results.append(res)

    if not results:
        print("[ERROR] No models were successfully benchmarked.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("  HARDWARE BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<15} | {'Backend':<18} | {'Disk MB':<8} | {'Mean ms':<8} | {'P95 ms':<8} | {'Est FPS':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<15} | {r['backend']:<18} | {r['size_mb']:<8.1f} | {r['mean_ms']:<8.1f} | {r['p95_ms']:<8.1f} | {r['fps']:<8.1f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
