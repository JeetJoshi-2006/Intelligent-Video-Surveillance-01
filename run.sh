#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo "  INTELLIGENT SURVEILLANCE SYSTEM (ISS) - LAUNCHER"
echo "  Optimized for Apple Silicon macOS (arm64)"
echo "=========================================================="

# Create virtual environment if not already present
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating isolated Python virtual environment..."
    python3 -m venv venv
fi

echo "[SETUP] Activating virtual environment..."
source venv/bin/activate

# Install pinned dependency ranges only when imports are unavailable.
if ! python3 -c "import cv2, torch, ultralytics, yaml, requests" 2>/dev/null; then
    echo "[SETUP] Installing dependencies..."
    pip install -r requirements.txt
fi

if [ "${ISS_SKIP_TESTS:-0}" != "1" ]; then
    echo "[SYSTEM] Running unit verification tests..."
    python3 -m unittest discover -s tests -v
fi

echo "[SYSTEM] Launching Surveillance Pipeline..."
python3 main.py
