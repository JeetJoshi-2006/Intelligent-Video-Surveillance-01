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

# Install required packages
echo "[SETUP] Checking and installing dependencies..."
pip install --upgrade pip
pip install numpy opencv-python pyyaml requests
pip install torch torchvision ultralytics

echo "[SYSTEM] Running unit verification tests..."
python3 tests/test_pipeline.py

echo "[SYSTEM] Launching Surveillance Pipeline..."
python3 main.py
