"""
Air Drum — Real-time virtual drumming with YOLOv8 pose estimation.
Runs on NVIDIA Jetson Orin Nano inside Docker container.

Logic: Detect wrist keypoints → check if wrist is inside a percussion
zone → play that zone's sound. No velocity/movement tracking needed.

Usage:
    python app.py
Then open http://<jetson-ip>:5555 in your browser.
"""

import cv2
import numpy as np
import time
import threading
import os
import json
from collections import deque
from flask import Flask, Response, render_template, jsonify

# ── Audio setup ─────────────────────────────────────────────────────
try:
    os.environ['SDL_AUDIODRIVER'] = 'alsa'
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    # Allow multiple simultaneous sounds
    pygame.mixer.set_num_channels(16)
    AUDIO_OK = True
    print("[audio] pygame mixer ready")
except Exception as e:
    AUDIO_OK = False
    print(f"[audio] pygame unavailable ({e}), running in silent mode")

# ── Load YOLOv8 pose model ──────────────────────────────────────────
from ultralytics import YOLO
print("[model] Loading YOLOv8n-pose...")
model = YOLO("yolov8n-pose.pt")
print("[model] Ready")

# ── Configuration ───────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
CONFIDENCE_MIN = 0.4           # keypoint confidence cutoff
COOLDOWN_SEC = 0.25            # seconds before same zone re-triggers

# YOLO pose keypoint indices (COCO):  9 = left wrist, 10 = right wrist
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

# ── Percussion Zones ────────────────────────────────────────────────
# 8 zones arranged in 2 rows of 4 across the bottom half of the frame.
# Each: (id, label, sound_file, row, col, color_BGR)
#
#   Row 0 (middle band):  Hi-Hat(C) | Tom High | Tom Low | Cymbal
#   Row 1 (bottom band):  Clap      | Snare    | Kick    | Hi-Hat(O)
#
ZONE_COLS = 4
ZONE_ROWS = 2
ZONE_Y_START = 0.40   # zones occupy bottom 60% of frame

ZONES = [
    # Row 0 — top row of zones
    {"id": "A", "label": "Hi-Hat",    "sound": "hihat_closed.wav", "row": 0, "col": 0, "color": (255, 255, 0)},
    {"id": "B", "label": "Tom High",  "sound": "tom_high.wav",     "row": 0, "col": 1, "color": (0, 200, 255)},
    {"id": "C", "label": "Tom Low",   "sound": "tom_low.wav",      "row": 0, "col": 2, "color": (255, 150, 0)},
    {"id": "D", "label": "Cymbal",    "sound": "cymbal.wav",       "row": 0, "col": 3, "color": (200, 200, 0)},
    # Row 1 — bottom row of zones
    {"id": "E", "label": "Clap",      "sound": "clap.wav",         "row": 1, "col": 0, "color": (200, 0, 200)},
    {"id": "F", "label": "Snare",     "sound": "snare.wav",        "row": 1, "col": 1, "color": (0, 165, 255)},
    {"id": "G", "label": "Kick",      "sound": "kick.wav",         "row": 1, "col": 2, "color": (0, 80, 255)},
    {"id": "H", "label": "Open HH",   "sound": "hihat_open.wav",   "row": 1, "col": 3, "color": (0, 255, 200)},
]

def zone_rect(zone, fw, fh):
    """Return (x1, y1, x2, y2) pixel coords for a zone."""
    col_w = fw / ZONE_COLS
    row_h = (fh * (1.0 - ZONE_Y_START)) / ZONE_ROWS
    y_off = fh * ZONE_Y_START
    x1 = int(zone["col"] * col_w)
    y1 = int(y_off + zone["row"] * row_h)
    x2 = int((zone["col"] + 1) * col_w)
    y2 = int(y_off + (zone["row"] + 1) * row_h)
    return x1, y1, x2, y2

# ── Load sounds ─────────────────────────────────────────────────────
sounds = {}
if AUDIO_OK:
    sound_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
    for z in ZONES:
        path = os.path.join(sound_dir, z["sound"])
        if os.path.exists(path):
            sounds[z["id"]] = pygame.mixer.Sound(path)
            print(f"[audio] Loaded {z['sound']} → zone {z['id']} ({z['label']})")
        else:
            print(f"[audio] WARNING: {path} not found")

# ── Shared state ────────────────────────────────────────────────────
app = Flask(__name__)
lock = threading.Lock()

class DrumState:
    def __init__(self):
        self.hit_counts = {z["id"]: 0 for z in ZONES}
        self.total_hits = 0
        self.recent_hits = deque(maxlen=60)
        self.bpm = 0
        self.last_hit_zone = ""
        self.last_hit_time = 0
        self.fps = 0
        # Per-zone cooldown: zone_id → timestamp of last trigger
        self.zone_last_trigger = {z["id"]: 0.0 for z in ZONES}
        # Per-hand zone tracking: hand → zone_id or None
        self.hand_in_zone = {"left": None, "right": None}

state = DrumState()

# ── Zone detection ──────────────────────────────────────────────────
def point_in_zone(x, y, zone, fw, fh):
    """Check if point (x, y) is inside a zone."""
    x1, y1, x2, y2 = zone_rect(zone, fw, fh)
    return x1 <= x <= x2 and y1 <= y <= y2

def find_zone(x, y, fw, fh):
    """Return zone dict if (x, y) is in any zone, else None."""
    for z in ZONES:
        if point_in_zone(x, y, z, fw, fh):
            return z
    return None

def trigger_hit(zone):
    """Play sound and update counters for a zone hit."""
    now = time.time()
    zid = zone["id"]

    with lock:
        # Cooldown check — don't re-trigger same zone too fast
        if now - state.zone_last_trigger[zid] < COOLDOWN_SEC:
            return
        state.zone_last_trigger[zid] = now
        state.hit_counts[zid] += 1
        state.total_hits += 1
        state.recent_hits.append(now)
        state.last_hit_zone = zone["label"]
        state.last_hit_time = now
        # BPM from recent hits
        if len(state.recent_hits) >= 2:
            span = state.recent_hits[-1] - state.recent_hits[0]
            if span > 0:
                state.bpm = int((len(state.recent_hits) - 1) / span * 60)

    # Play sound
    if AUDIO_OK and zid in sounds:
        sounds[zid].play()

def check_wrist(hand_key, x, y, fw, fh):
    """Check if a wrist is in a zone. Trigger on entry (not while staying)."""
    zone = find_zone(x, y, fw, fh)

    with lock:
        prev_zone_id = state.hand_in_zone[hand_key]
        new_zone_id = zone["id"] if zone else None

    if zone is not None and new_zone_id != prev_zone_id:
        # Wrist just ENTERED a new zone → trigger
        trigger_hit(zone)

    with lock:
        state.hand_in_zone[hand_key] = new_zone_id

# ── Video processing thread ─────────────────────────────────────────
output_frame = None
frame_lock = threading.Lock()

def video_loop():
    global output_frame
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"[camera] Cannot open camera {CAMERA_INDEX}")
        print("[camera] Running in demo mode (no camera)")
        cap = None

    fps_counter = 0
    fps_time = time.time()

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
        else:
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera - Demo Mode", (130, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, "Connect a USB camera and restart", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        fh, fw = frame.shape[:2]

        # ── Draw percussion zones ──
        now = time.time()
        for z in ZONES:
            x1, y1, x2, y2 = zone_rect(z, fw, fh)
            color = z["color"]

            # Highlight if recently hit
            with lock:
                recent = now - state.zone_last_trigger[z["id"]] < 0.15

            # Semi-transparent fill
            overlay = frame.copy()
            fill_alpha = 0.3 if recent else 0.12
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

            # Border (thicker when active)
            border_w = 3 if recent else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, border_w)

            # Zone label and letter
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.putText(frame, z["id"], (cx - 8, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, z["label"], (cx - 30, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Hit count
            with lock:
                cnt = state.hit_counts[z["id"]]
            cv2.putText(frame, str(cnt), (x1 + 5, y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── Run pose estimation ──
        if cap is not None:
            results = model(frame, verbose=False, conf=CONFIDENCE_MIN)

            for r in results:
                if r.keypoints is None:
                    continue
                kps = r.keypoints.data
                for person in kps:
                    # Left wrist
                    lw = person[KP_LEFT_WRIST]
                    if lw[2] > CONFIDENCE_MIN:
                        lx, ly = int(lw[0]), int(lw[1])
                        cv2.circle(frame, (lx, ly), 14, (255, 100, 100), -1)
                        cv2.circle(frame, (lx, ly), 16, (255, 255, 255), 2)
                        cv2.putText(frame, "L", (lx - 5, ly + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        check_wrist("left", lx, ly, fw, fh)

                    # Right wrist
                    rw = person[KP_RIGHT_WRIST]
                    if rw[2] > CONFIDENCE_MIN:
                        rx, ry = int(rw[0]), int(rw[1])
                        cv2.circle(frame, (rx, ry), 14, (100, 100, 255), -1)
                        cv2.circle(frame, (rx, ry), 16, (255, 255, 255), 2)
                        cv2.putText(frame, "R", (rx - 5, ry + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        check_wrist("right", rx, ry, fw, fh)

                    # Arm skeleton (shoulder→elbow→wrist)
                    arm_pairs = [(5, 7), (7, 9), (6, 8), (8, 10)]
                    for a, b in arm_pairs:
                        pa, pb = person[a], person[b]
                        if pa[2] > CONFIDENCE_MIN and pb[2] > CONFIDENCE_MIN:
                            cv2.line(frame,
                                     (int(pa[0]), int(pa[1])),
                                     (int(pb[0]), int(pb[1])),
                                     (0, 255, 0), 2)

        # ── HUD ──
        with lock:
            if now - state.last_hit_time < 0.3 and state.last_hit_zone:
                cv2.putText(frame, state.last_hit_zone + "!",
                            (fw // 2 - 60, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            bpm_val = state.bpm
            total = state.total_hits

        # FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            with lock:
                state.fps = fps_counter
            fps_counter = 0
            fps_time = time.time()

        with lock:
            fps_display = state.fps

        cv2.putText(frame, f"FPS: {fps_display}", (fw - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"BPM: {bpm_val}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Hits: {total}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        with frame_lock:
            output_frame = frame.copy()

        time.sleep(0.001)

    if cap is not None:
        cap.release()

# ── Flask routes ────────────────────────────────────────────────────
def gen_frames():
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            ret, buf = cv2.imencode('.jpg', output_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html', zones=ZONES)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with lock:
        data = {
            "hit_counts": dict(state.hit_counts),
            "total_hits": state.total_hits,
            "bpm": state.bpm,
            "last_hit": state.last_hit_zone,
            "fps": state.fps,
        }
    return jsonify(data)

@app.route('/reset', methods=['POST'])
def reset():
    with lock:
        for k in state.hit_counts:
            state.hit_counts[k] = 0
        state.total_hits = 0
        state.recent_hits.clear()
        state.bpm = 0
        state.last_hit_zone = ""
    return jsonify({"status": "ok"})

# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t = threading.Thread(target=video_loop, daemon=True)
    t.start()
    print("\n" + "=" * 50)
    print("  AIR DRUM is running!")
    print(f"  {len(ZONES)} percussion zones: {', '.join(z['label'] for z in ZONES)}")
    print("  Open in browser: http://<jetson-ip>:5555")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5555, threaded=True)
