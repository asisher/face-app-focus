from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


@dataclass
class FaceObservation:
    has_face: bool
    yaw_deg: float
    pitch_deg: float
    confidence: float
    eye_down_score: float
    nose_x_norm: float
    nose_y_norm: float
    left_pupil_x_norm: float
    left_pupil_y_norm: float
    right_pupil_x_norm: float
    right_pupil_y_norm: float


class FaceLandmarkDetector:
    def __init__(self) -> None:
        self._mesh = None
        self._init_attempted = False

    def _ensure_mesh(self) -> None:
        if self._mesh is not None or self._init_attempted or mp is None:
            return

        self._init_attempted = True
        try:
            with _suppress_native_stderr():
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
        except Exception:
            self._mesh = None

    # MediaPipe works best at 640x480. Very high-res frames cause face detection
    # to fail because the face occupies too small a fraction of the image.
    _INFER_W = 640
    _INFER_H = 480

    def infer(self, frame_bgr: np.ndarray) -> FaceObservation:
        self._ensure_mesh()
        if self._mesh is None:
            return FaceObservation(False, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

        h, w = frame_bgr.shape[:2]
        if w != self._INFER_W or h != self._INFER_H:
            frame_bgr = cv2.resize(frame_bgr, (self._INFER_W, self._INFER_H), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return FaceObservation(False, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

        landmarks = result.multi_face_landmarks[0].landmark
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        nose_tip = landmarks[1]
        chin = landmarks[152]

        eye_vec_x = right_eye_outer.x - left_eye_outer.x
        eye_vec_y = right_eye_outer.y - left_eye_outer.y
        yaw = -np.degrees(np.arctan2((nose_tip.x - 0.5), max(abs(eye_vec_x), 1e-6)))
        pitch = np.degrees(np.arctan2((chin.y - nose_tip.y) - 0.16, max(abs(eye_vec_x), 1e-6)))
        roll_mag = abs(np.degrees(np.arctan2(eye_vec_y, max(abs(eye_vec_x), 1e-6))))
        # Roll should reduce certainty, but not too aggressively for natural posture.
        confidence = max(0.0, min(1.0, 1.0 - (roll_mag / 65.0)))
        eye_down_score = self._estimate_eye_down_score(landmarks)
        nose_x = float(max(0.0, min(1.0, nose_tip.x)))
        nose_y = float(max(0.0, min(1.0, nose_tip.y)))

        try:
            left_pupil = landmarks[468]
            right_pupil = landmarks[473]
            left_pupil_x = float(max(0.0, min(1.0, left_pupil.x)))
            left_pupil_y = float(max(0.0, min(1.0, left_pupil.y)))
            right_pupil_x = float(max(0.0, min(1.0, right_pupil.x)))
            right_pupil_y = float(max(0.0, min(1.0, right_pupil.y)))
        except (IndexError, TypeError):
            left_pupil_x = -1.0
            left_pupil_y = -1.0
            right_pupil_x = -1.0
            right_pupil_y = -1.0

        return FaceObservation(
            True,
            float(yaw),
            float(pitch),
            float(confidence),
            float(eye_down_score),
            nose_x,
            nose_y,
            left_pupil_x,
            left_pupil_y,
            right_pupil_x,
            right_pupil_y,
        )

    def _estimate_eye_down_score(self, landmarks) -> float:
        # Eye top/bottom and iris center landmarks from MediaPipe Face Mesh.
        left_top = landmarks[159].y
        left_bottom = landmarks[145].y
        left_iris = landmarks[468].y

        right_top = landmarks[386].y
        right_bottom = landmarks[374].y
        right_iris = landmarks[473].y

        left_open = max(1e-6, left_bottom - left_top)
        right_open = max(1e-6, right_bottom - right_top)

        left_pos = (left_iris - left_top) / left_open
        right_pos = (right_iris - right_top) / right_open

        # Around 0.5 is centered in eye box; >0.6 trends downward gaze.
        avg_pos = (left_pos + right_pos) / 2.0
        return max(0.0, min(1.0, (avg_pos - 0.50) / 0.35))

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()


@contextlib.contextmanager
def _suppress_native_stderr():
    stderr_fd = None
    stderr_dup = None
    devnull = None
    try:
        stderr_fd = 2
        stderr_dup = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        yield
    except OSError:
        yield
    finally:
        if stderr_dup is not None and stderr_fd is not None:
            os.dup2(stderr_dup, stderr_fd)
            os.close(stderr_dup)
        if devnull is not None:
            os.close(devnull)
