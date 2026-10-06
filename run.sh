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

# Use venv Python directly (source activate doesn't work in non-interactive shells)
VENV_PYTHON="venv/bin/python"

# Install pinned dependency ranges only when imports are unavailable.
if ! $VENV_PYTHON -c "import cv2, torch, ultralytics, yaml, requests" 2>/dev/null; then
    echo "[SETUP] Installing dependencies..."
    venv/bin/pip install -r requirements.txt
fi

if [ "$1" == "--test" ] || [ "${ISS_RUN_TESTS:-0}" == "1" ]; then
    echo "[SYSTEM] Running unit verification tests..."
    OPENCV_AVFOUNDATION_SKIP_AUTH=1 $VENV_PYTHON -m unittest discover -s tests -v
fi

echo "[SYSTEM] Launching Surveillance Pipeline..."
$VENV_PYTHON main.py
