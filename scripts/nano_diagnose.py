#!/usr/bin/env python3
"""Quick diagnostic: tests camera open, frame capture, and face detection."""
from __future__ import annotations

import sys
import time

ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
SRC = __import__("os").path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from app.config import load_config
from capture.camera import CameraStream

print("=== Focus Monitor Diagnostics ===\n")

# 1. OpenCV
print("[1] Checking OpenCV...")
try:
    import cv2
    print(f"    OK  OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"    FAIL  {e}")
    sys.exit(1)

# 2. NumPy
print("[2] Checking NumPy...")
try:
    import numpy as np
    print(f"    OK  NumPy {np.__version__}")
except ImportError as e:
    print(f"    FAIL  {e}")

config = load_config()

# 3. Camera open
print("[3] Opening camera through CameraStream...")
camera = CameraStream(
    config.camera_index,
    config.camera_width,
    config.camera_height,
    backend=config.camera_backend,
    rotation=config.camera_rotation,
    allow_index_scan=config.camera_allow_index_scan,
    scan_max_index=config.camera_scan_max_index,
    recover_failures=config.camera_recover_failures,
)

# 4. Frame read
print("[4] Reading frames...")
frame = None
for attempt in range(10):
    capture = camera.read()
    if capture.ok and capture.frame is not None:
        frame = capture.frame
        break
    time.sleep(0.1)

if frame is None:
    print("    FAIL  Could not read a frame from camera")
    print(f"    Error: {capture.error if 'capture' in locals() else 'unknown'}")
    camera.close()
    sys.exit(1)

h, w = frame.shape[:2]
print(f"    OK  Frame received: {w}x{h} pixels")
print(f"    OK  Capture path: {camera.active_camera_label}")

# 5. MediaPipe
print("[5] Checking MediaPipe...")
try:
    import mediapipe as mp
    print(f"    OK  MediaPipe {mp.__version__}")
except ImportError as e:
    print(f"    FAIL  {e}")
    print("    MediaPipe is not installed. Run: pip install mediapipe")
    sys.exit(1)

# 6. FaceMesh detection on a live frame
print("[6] Testing face detection on 5 frames (look at the camera)...")
print(f"    (frames will be resized to 640x480 before detection)")
mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
detected = 0
frames_tried = 0
for _ in range(30):
    capture = camera.read()
    if not capture.ok or capture.frame is None:
        continue
    frame = capture.frame
    frames_tried += 1
    # Always resize to 640x480 — MediaPipe fails on very high-res frames
    fh, fw = frame.shape[:2]
    if fw != 640 or fh != 480:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)
    if result.multi_face_landmarks:
        detected += 1
        print(f"    DETECTED face on frame {frames_tried}")
        if detected >= 3:
            break
    time.sleep(0.1)

camera.close()
mesh.close()

if detected == 0:
    print("    FAIL  No face detected in 30 frames")
    print("    Possible causes:")
    print("      - Camera angle too wide or off-axis")
    print("      - Lighting too low")
    print("      - Camera flipped/mirrored (try moving closer)")
else:
    print(f"\n    OK  Face detected {detected} times across {frames_tried} frames")

print("\n=== Done ===")
