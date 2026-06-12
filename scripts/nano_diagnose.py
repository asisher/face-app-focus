#!/usr/bin/env python3
"""Quick diagnostic: tests camera open, frame capture, and face detection."""
from __future__ import annotations

import sys
import time

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

# 3. Camera open
print("[3] Opening camera (index 0)...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("    FAIL  Could not open camera index 0")
    print("    Try: v4l2-ctl --list-devices  to find your camera")
    # Try index 1
    print("    Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("    FAIL  Camera index 1 also failed. Check USB connection.")
        sys.exit(1)
    else:
        print("    OK  Camera opened at index 1  (set CAMERA_INDEX=1 in .env)")
else:
    print("    OK  Camera opened at index 0")

# 4. Frame read
print("[4] Reading frames...")
ok = False
frame = None
for attempt in range(10):
    ok, frame = cap.read()
    if ok and frame is not None:
        break
    time.sleep(0.1)

if not ok or frame is None:
    print("    FAIL  Could not read a frame from camera")
    cap.release()
    sys.exit(1)

h, w = frame.shape[:2]
print(f"    OK  Frame received: {w}x{h} pixels")
cap.release()

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
cap = cv2.VideoCapture(0)
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
    ok, frame = cap.read()
    if not ok or frame is None:
        continue
    frames_tried += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)
    if result.multi_face_landmarks:
        detected += 1
        print(f"    DETECTED face on frame {frames_tried}")
        if detected >= 3:
            break
    time.sleep(0.1)

cap.release()
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
