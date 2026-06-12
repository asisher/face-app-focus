#!/usr/bin/env python3
"""Captures one frame, resizes it, and saves snapshot.jpg next to this script."""
from __future__ import annotations
import sys
import os
import cv2

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from app.config import load_config
from capture.camera import CameraStream

OUT = os.path.join(os.path.dirname(__file__), "snapshot.jpg")

config = load_config()
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

capture = camera.read()
camera.close()

if not capture.ok or capture.frame is None:
    print(f"FAIL: could not read frame ({capture.error})")
    sys.exit(1)

frame = capture.frame

h, w = frame.shape[:2]
print(f"Original frame: {w}x{h}")
print(f"Camera path: {camera.active_camera_label}")

# Resize to 640x480 for inspection
small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
cv2.imwrite(OUT, small)
print(f"Saved: {OUT}")
print("Copy to your Mac with:")
print(f"  scp cybernano@<NANO_IP>:{OUT} ~/Desktop/snapshot.jpg")
