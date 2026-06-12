from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QObject, QTimer

from app.config import AppConfig
from capture.camera import CameraStream
from domain.models import AttentionSample, AttentionState
from inference.attention_rules import AttentionClassifier
from inference.face_landmarks import FaceLandmarkDetector
from scoring.engine import ScoringEngine
from storage.repository import ScoreRepository
from ui.window import MainWindow


class AppController(QObject):
    def __init__(self, config: AppConfig, window: MainWindow) -> None:
        super().__init__()
        self._config = config
        self._window = window

        self._camera = CameraStream(
            config.camera_index,
            config.camera_width,
            config.camera_height,
            backend=config.camera_backend,
            rotation=config.camera_rotation,
            allow_index_scan=config.camera_allow_index_scan,
            scan_max_index=config.camera_scan_max_index,
            recover_failures=config.camera_recover_failures,
        )
        self._detector = FaceLandmarkDetector()
        self._classifier = AttentionClassifier(config)
        self._scoring = ScoringEngine()
        self._repo = ScoreRepository(config.db_path)

        restored = self._repo.restore_latest(datetime.now())
        self._scoring.restore(
            restored.total_score,
            restored.hour_score,
            restored.hour_key,
            restored.hour_focus_weight,
            restored.hour_possible_weight,
            restored.total_focus_weight,
            restored.total_possible_weight,
        )

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / config.sample_fps))

    def _tick(self) -> None:
        frame = self._camera.read()
        now = datetime.now()
        base_sample = AttentionSample(
            timestamp=now,
            state=AttentionState.UNKNOWN,
            confidence=0.0,
            reason="boot",
        )

        if not frame.ok or frame.frame is None:
            classified = AttentionSample(
                timestamp=now,
                state=AttentionState.UNKNOWN,
                confidence=0.0,
                reason=(
                    f"{frame.error or 'camera_unavailable'}"
                    f" ({self._camera.active_camera_label})"
                ),
                focus_prob=0.0,
                yaw_deg=0.0,
                pitch_deg=0.0,
                eye_down_score=0.0,
                nose_x_norm=-1.0,
                nose_y_norm=-1.0,
                calibration_progress=0.0,
                is_calibrated=False,
                alignment_score=0.0,
            )
            frame_bgr = None
        else:
            observation = self._detector.infer(frame.frame)
            classified = self._classifier.classify(observation, base_sample)
            frame_bgr = frame.frame

        score = self._scoring.update(classified)
        self._repo.append(classified, score)
        self._window.update_view(classified, score, frame_bgr)

    def close(self) -> None:
        self._timer.stop()
        self._camera.close()
        self._detector.close()
        self._repo.close()
