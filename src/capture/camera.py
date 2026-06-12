from __future__ import annotations

from dataclasses import dataclass
import platform
import time
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraFrame:
    ok: bool
    frame: Optional[np.ndarray]
    error: str = ""


class CameraStream:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        backend: str = "auto",
        allow_index_scan: bool = False,
        scan_max_index: int = 5,
        recover_failures: int = 3,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._backend = backend
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

    def _available_backends(self) -> list[tuple[int, str]]:
        if self._backend != "auto":
            forced = self._backend.strip().lower()
            if forced == "avfoundation" and hasattr(cv2, "CAP_AVFOUNDATION"):
                return [(int(cv2.CAP_AVFOUNDATION), "avfoundation")]
            if forced == "any":
                return [(int(cv2.CAP_ANY), "any")]
            return [(int(cv2.CAP_ANY), "any")]

        if platform.system().lower() == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            return [
                (int(cv2.CAP_AVFOUNDATION), "avfoundation"),
                (int(cv2.CAP_ANY), "any"),
            ]
        return [(int(cv2.CAP_ANY), "any")]

    def _candidate_indices(self) -> list[int]:
        if not self._allow_index_scan:
            return [self._index]

        candidates = [self._index]
        for idx in range(0, self._scan_max_index + 1):
            if idx != self._index:
                candidates.append(idx)
        return candidates

    def _open_capture(self, index: int, backend_flag: int) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(index, backend_flag)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        ok = False
        frame = None
        # Some cameras need a few initial reads before delivering a frame.
        for _ in range(4):
            ok, frame = cap.read()
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

        for backend_flag, backend_name in self._available_backends():
            for index in self._candidate_indices():
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
        if self._capture is None or not self._capture.isOpened():
            reopened = self._open_with_fallbacks()
            if not reopened:
                return CameraFrame(ok=False, frame=None, error=self._last_error or "camera_not_open")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._recover_failures:
                reopened = self._open_with_fallbacks()
                if reopened and self._capture is not None:
                    ok, frame = self._capture.read()
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
