#!/usr/bin/env python3
"""Captures one frame, resizes it, and saves snapshot.jpg next to this script."""
from __future__ import annotations
import sys
import time
import os
import cv2

OUT = os.path.join(os.path.dirname(__file__), "snapshot.jpg")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("FAIL: could not open camera index 0")
    sys.exit(1)

# Warm up camera
for _ in range(5):
    cap.read()
    time.sleep(0.05)

ok, frame = cap.read()
cap.release()

if not ok or frame is None:
    print("FAIL: could not read frame")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Original frame: {w}x{h}")

# Resize to 640x480 for inspection
small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
cv2.imwrite(OUT, small)
print(f"Saved: {OUT}")
print("Copy to your Mac with:")
print(f"  scp cybernano@<NANO_IP>:{OUT} ~/Desktop/snapshot.jpg")
