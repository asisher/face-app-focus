from __future__ import annotations

from dataclasses import dataclass
import contextlib
import os
import platform
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class CameraFrame:
    ok: bool
    frame: Optional[np.ndarray]
    error: str = ""


def normalize_camera_frame(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if frame is None:
        return None

    if frame.ndim == 2:
        with contextlib.suppress(cv2.error):
            return cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim != 3:
        return frame

    channels = frame.shape[2]
    if channels == 1:
        mono = frame[:, :, 0]
        with contextlib.suppress(cv2.error):
            return cv2.cvtColor(mono, cv2.COLOR_BAYER_BG2BGR)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    if channels == 2:
        with contextlib.suppress(cv2.error):
            return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        return frame
    if channels == 4:
        with contextlib.suppress(cv2.error):
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    return frame


def apply_camera_rotation(frame: Optional[np.ndarray], rotation_deg: int) -> Optional[np.ndarray]:
    if frame is None:
        return None

    normalized = rotation_deg % 360
    if normalized == 0:
        return frame
    if normalized == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


class CameraStream:
    _JETSON_SHELL_BACKEND = "jetson-csi-shell"

    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        backend: str = "auto",
        rotation: int = 0,
        allow_index_scan: bool = False,
        scan_max_index: int = 5,
        recover_failures: int = 3,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._backend = backend
        self._rotation = rotation % 360
        self._allow_index_scan = allow_index_scan
        self._scan_max_index = scan_max_index
        self._recover_failures = max(1, recover_failures)

        self._capture: Optional[cv2.VideoCapture] = None
        self._active_index = -1
        self._active_backend_name = "none"
        self._consecutive_failures = 0
        self._last_error = ""
        self._last_open_attempt = 0.0
        self._retry_interval_sec = 1.5
        self._shell_index: Optional[int] = None
        self._opencv_has_gstreamer: Optional[bool] = None
        self._shell_capture_path = os.path.join(
            tempfile.gettempdir(),
            f"focus-monitor-camera-{os.getpid()}.jpg",
        )

    def _jetson_csi_pipeline(self, index: int) -> str:
        return (
            f"nvarguscamerasrc sensor-id={index} ! "
            f"video/x-raw(memory:NVMM), width=(int){self._width}, height=(int){self._height}, "
            "format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv ! video/x-raw, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! "
            "appsink drop=true sync=false"
        )

    def _is_jetson_linux(self) -> bool:
        return platform.system().lower() == "linux" and platform.machine().lower() in {"aarch64", "arm64"}

    def _has_opencv_gstreamer(self) -> bool:
        if self._opencv_has_gstreamer is not None:
            return self._opencv_has_gstreamer

        with contextlib.suppress(Exception):
            info = cv2.getBuildInformation()
            self._opencv_has_gstreamer = "GStreamer:                   YES" in info or "GStreamer: YES" in info
            return self._opencv_has_gstreamer

        self._opencv_has_gstreamer = False
        return False

    def _available_backends(self) -> List[Tuple[int, str]]:
        if self._backend in {"jetson", "jetson-csi", "csi"} and not self._has_opencv_gstreamer():
            return [(0, self._JETSON_SHELL_BACKEND)]

        if self._backend in {"jetson", "jetson-csi", "csi"} and hasattr(cv2, "CAP_GSTREAMER"):
            return [(int(cv2.CAP_GSTREAMER), "jetson-csi")]

        if self._backend != "auto":
            forced = self._backend.strip().lower()
            if forced == "avfoundation" and hasattr(cv2, "CAP_AVFOUNDATION"):
                return [(int(cv2.CAP_AVFOUNDATION), "avfoundation")]
            if forced == "gstreamer" and hasattr(cv2, "CAP_GSTREAMER"):
                return [(int(cv2.CAP_GSTREAMER), "gstreamer")]
            if forced == "any":
                return [(int(cv2.CAP_ANY), "any")]
            return [(int(cv2.CAP_ANY), "any")]

        if platform.system().lower() == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            return [
                (int(cv2.CAP_AVFOUNDATION), "avfoundation"),
                (int(cv2.CAP_ANY), "any"),
            ]
        if self._is_jetson_linux() and not self._has_opencv_gstreamer():
            return [
                (0, self._JETSON_SHELL_BACKEND),
                (int(cv2.CAP_ANY), "any"),
            ]
        if self._is_jetson_linux() and hasattr(cv2, "CAP_GSTREAMER"):
            return [
                (int(cv2.CAP_GSTREAMER), "jetson-csi"),
                (int(cv2.CAP_ANY), "any"),
            ]
        return [(int(cv2.CAP_ANY), "any")]

    def _capture_jetson_shell_frame(self, index: int) -> Optional[np.ndarray]:
        cmd = [
            "gst-launch-1.0",
            "-q",
            "nvarguscamerasrc",
            f"sensor-id={index}",
            "num-buffers=1",
            "!",
            f"video/x-raw(memory:NVMM), width=(int){self._width}, height=(int){self._height}, format=(string)NV12, framerate=(fraction)30/1",
            "!",
            "nvvidconv",
            "!",
            f"video/x-raw, width=(int){self._width}, height=(int){self._height}, format=(string)BGRx",
            "!",
            "videoconvert",
            "!",
            "video/x-raw, format=(string)BGR",
            "!",
            "jpegenc",
            "!",
            "filesink",
            f"location={self._shell_capture_path}",
        ]

        with contextlib.suppress(FileNotFoundError):
            result = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True,
            )
            if result.returncode != 0:
                self._last_error = (result.stderr or "gst_capture_failed").strip() or "gst_capture_failed"
                return None

            frame = cv2.imread(self._shell_capture_path)
            frame = normalize_camera_frame(frame)
            frame = apply_camera_rotation(frame, self._rotation)
            if frame is None:
                self._last_error = "gst_capture_empty"
                return None
            return frame

        self._last_error = "gst_launch_missing"
        return None

    def _candidate_indices(self) -> List[int]:
        if not self._allow_index_scan:
            return [self._index]

        candidates = [self._index]
        for idx in range(0, self._scan_max_index + 1):
            if idx != self._index:
                candidates.append(idx)
        return candidates

    def _open_capture(self, index: int, backend_flag: int) -> Optional[cv2.VideoCapture]:
        source: Union[int, str] = index
        if backend_flag == getattr(cv2, "CAP_GSTREAMER", -9999):
            source = self._jetson_csi_pipeline(index)

        cap = cv2.VideoCapture(source, backend_flag)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None

        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        ok = False
        frame = None
        # Some cameras need a few initial reads before delivering a frame.
        for _ in range(4):
            ok, frame = cap.read()
            frame = normalize_camera_frame(frame)
            frame = apply_camera_rotation(frame, self._rotation)
            if ok and frame is not None:
                break
        if not ok or frame is None:
            cap.release()
            return None
        return cap

    def _open_with_fallbacks(self) -> bool:
        now = time.monotonic()
        if (now - self._last_open_attempt) < self._retry_interval_sec:
            return False
        self._last_open_attempt = now

        if self._capture is not None and self._capture.isOpened():
            self._capture.release()
        self._shell_index = None

        for backend_flag, backend_name in self._available_backends():
            for index in self._candidate_indices():
                if backend_name == self._JETSON_SHELL_BACKEND:
                    frame = self._capture_jetson_shell_frame(index)
                    if frame is None:
                        continue
                    self._capture = None
                    self._shell_index = index
                    self._active_index = index
                    self._active_backend_name = backend_name
                    self._consecutive_failures = 0
                    self._last_error = ""
                    return True
                cap = self._open_capture(index, backend_flag)
                if cap is None:
                    continue
                self._capture = cap
                self._active_index = index
                self._active_backend_name = backend_name
                self._consecutive_failures = 0
                self._last_error = ""
                return True

        self._capture = None
        self._active_index = -1
        self._active_backend_name = "none"
        self._last_error = "camera_open_failed"
        return False

    def read(self) -> CameraFrame:
        if self._active_backend_name == self._JETSON_SHELL_BACKEND and self._shell_index is not None:
            frame = self._capture_jetson_shell_frame(self._shell_index)
            if frame is not None:
                self._consecutive_failures = 0
                return CameraFrame(ok=True, frame=frame)
            return CameraFrame(ok=False, frame=None, error=self._last_error or "camera_read_failed")

        if self._capture is None or not self._capture.isOpened():
            reopened = self._open_with_fallbacks()
            if not reopened:
                return CameraFrame(ok=False, frame=None, error=self._last_error or "camera_not_open")
            if self._active_backend_name == self._JETSON_SHELL_BACKEND and self._shell_index is not None:
                frame = self._capture_jetson_shell_frame(self._shell_index)
                if frame is not None:
                    self._consecutive_failures = 0
                    return CameraFrame(ok=True, frame=frame)
                return CameraFrame(ok=False, frame=None, error=self._last_error or "camera_read_failed")

        ok, frame = self._capture.read()
        frame = normalize_camera_frame(frame)
        frame = apply_camera_rotation(frame, self._rotation)
        if not ok or frame is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._recover_failures:
                reopened = self._open_with_fallbacks()
                if reopened and self._capture is not None:
                    ok, frame = self._capture.read()
                    frame = normalize_camera_frame(frame)
                    frame = apply_camera_rotation(frame, self._rotation)
                    if ok and frame is not None:
                        self._consecutive_failures = 0
                        return CameraFrame(ok=True, frame=frame)
            return CameraFrame(
                ok=False,
                frame=None,
                error=(
                    f"camera_read_failed(index={self._active_index},"
                    f"backend={self._active_backend_name})"
                ),
            )

        self._consecutive_failures = 0
        return CameraFrame(ok=True, frame=frame)

    @property
    def active_camera_label(self) -> str:
        return f"index={self._active_index},backend={self._active_backend_name}"

    def close(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            self._capture.release()
        with contextlib.suppress(OSError):
            os.remove(self._shell_capture_path)
