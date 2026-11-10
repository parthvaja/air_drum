#!/bin/bash
# Air Drum — Setup & Run Script
# Run this INSIDE the Docker container (jetson_labs_dev)

echo "========================================="
echo "  Air Drum Setup"
echo "========================================="

# Install dependencies
echo "[1/3] Installing Python packages..."
pip install flask ultralytics pygame "opencv-python<5" --break-system-packages -q 2>/dev/null || \
pip install flask ultralytics pygame "opencv-python<5" -q

# Generate drum sounds if missing
if [ ! -f sounds/kick.wav ]; then
    echo "[2/3] Generating 8 percussion sounds..."
    python generate_sounds.py
else
    echo "[2/3] Sounds already exist, skipping"
fi

# Download YOLOv8n-pose if not cached
echo "[3/3] Checking YOLOv8n-pose model..."
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')" 2>/dev/null

echo ""
echo "========================================="
echo "  Starting Air Drum!"
echo "  8 zones: Hi-Hat, Tom High, Tom Low,"
echo "  Cymbal, Clap, Snare, Kick, Open HH"
echo "  Open: http://<your-jetson-ip>:5555"
echo "========================================="
echo ""

python app.py
