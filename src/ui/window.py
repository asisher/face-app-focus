from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from domain.models import AttentionSample, AttentionState, ScoreSnapshot


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Focus Monitor")
        self.resize(760, 760)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)

        self.preview_label = QLabel("Camera preview unavailable")
        self.preview_label.setFixedSize(720, 405)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background: #101010; color: #cfd8e3; border: 1px solid #2a3440;")
        layout.addWidget(self.preview_label)

        self.status_label = QLabel("Status: Booting")
        self.status_label.setAlignment(Qt.AlignLeft)
        self.score_label = QLabel("Current score: 0")
        self.hour_label = QLabel("Hour score: 0")
        self.total_label = QLabel("Total score: 0")
        self.reason_label = QLabel("Reason: n/a")
        self.metrics_label = QLabel("Metrics: yaw=0.0 pitch=0.0 focus_prob=0.00 conf=0.00")

        for widget in [
            self.status_label,
            self.score_label,
            self.hour_label,
            self.total_label,
            self.reason_label,
            self.metrics_label,
        ]:
            widget.setStyleSheet("font-size: 18px;")
            layout.addWidget(widget)

        self.privacy_label = QLabel("Local processing only. No frames stored.")
        self.privacy_label.setStyleSheet("font-size: 13px; color: #5f6b7a;")
        layout.addWidget(self.privacy_label)

        self._target_x_norm = 0.50
        self._target_y_norm = 0.45
        self._target_radius_norm = 0.16

    def update_view(
        self,
        sample: AttentionSample,
        score: ScoreSnapshot,
        frame_bgr: np.ndarray | None = None,
    ) -> None:
        state = sample.state
        if state == AttentionState.FOCUSED:
            color = "#1f8f4e"
        elif state == AttentionState.DISTRACTED:
            color = "#c0392b"
        else:
            color = "#a17b18"

        self.status_label.setText(f"Status: {state.value.title()}")
        self.status_label.setStyleSheet(f"font-size: 18px; font-weight: 600; color: {color};")
        self.score_label.setText(f"Current score: {score.current_score:.2f}")
        self.hour_label.setText(f"Hour score: {score.hour_score:.2f}")
        self.total_label.setText(f"Total score: {score.total_score:.2f}")
        self.reason_label.setText(f"Reason: {sample.reason}")
        self.metrics_label.setText(
            "Metrics: "
            f"yaw={sample.yaw_deg:.1f} "
            f"pitch={sample.pitch_deg:.1f} "
            f"eye_down={sample.eye_down_score:.2f} "
            f"focus_prob={sample.focus_prob:.2f} "
            f"conf={sample.confidence:.2f} "
            f"calib={sample.calibration_progress:.0%}"
        )

        if frame_bgr is not None:
            self._update_preview(frame_bgr, sample)

    def _update_preview(self, frame_bgr: np.ndarray, sample: AttentionSample) -> None:
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        status_text = f"{sample.state.value.upper()} | {sample.reason}"
        details_text = (
            f"yaw={sample.yaw_deg:.1f} pitch={sample.pitch_deg:.1f} "
            f"focus={sample.focus_prob:.2f} conf={sample.confidence:.2f} "
            f"calib={sample.calibration_progress:.0%}"
        )

        target_x = int(self._target_x_norm * w)
        target_y = int(self._target_y_norm * h)
        target_radius = max(12, int(self._target_radius_norm * min(w, h)))

        nose_visible = sample.nose_x_norm >= 0.0 and sample.nose_y_norm >= 0.0
        nose_x = int(sample.nose_x_norm * w) if nose_visible else -1
        nose_y = int(sample.nose_y_norm * h) if nose_visible else -1

        left_pupil_visible = sample.left_pupil_x_norm >= 0.0 and sample.left_pupil_y_norm >= 0.0
        right_pupil_visible = sample.right_pupil_x_norm >= 0.0 and sample.right_pupil_y_norm >= 0.0
        left_pupil_x = int(sample.left_pupil_x_norm * w) if left_pupil_visible else -1
        left_pupil_y = int(sample.left_pupil_y_norm * h) if left_pupil_visible else -1
        right_pupil_x = int(sample.right_pupil_x_norm * w) if right_pupil_visible else -1
        right_pupil_y = int(sample.right_pupil_y_norm * h) if right_pupil_visible else -1
        inside_target = False
        if nose_visible:
            dist_sq = (nose_x - target_x) ** 2 + (nose_y - target_y) ** 2
            inside_target = dist_sq <= (target_radius ** 2)

        is_green = sample.is_calibrated and inside_target and sample.alignment_score >= 0.70
        target_color = (60, 210, 95) if is_green else (60, 170, 240)

        cv2.circle(annotated, (target_x, target_y), target_radius, target_color, 2)
        cv2.circle(annotated, (target_x, target_y), 4, target_color, -1)
        cv2.putText(
            annotated,
            "Head target",
            (target_x - 45, max(20, target_y - target_radius - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            target_color,
            1,
        )

        if nose_visible:
            cv2.circle(annotated, (nose_x, nose_y), 6, (250, 210, 80), -1)
            cv2.circle(annotated, (nose_x, nose_y), 10, (30, 30, 30), 1)

        if left_pupil_visible:
            cv2.circle(annotated, (left_pupil_x, left_pupil_y), 4, (80, 250, 250), -1)
            cv2.circle(annotated, (left_pupil_x, left_pupil_y), 8, (20, 40, 40), 1)
            cv2.putText(
                annotated,
                "L",
                (left_pupil_x - 12, left_pupil_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 250, 250),
                1,
            )

        if right_pupil_visible:
            cv2.circle(annotated, (right_pupil_x, right_pupil_y), 4, (80, 250, 250), -1)
            cv2.circle(annotated, (right_pupil_x, right_pupil_y), 8, (20, 40, 40), 1)
            cv2.putText(
                annotated,
                "R",
                (right_pupil_x + 6, right_pupil_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 250, 250),
                1,
            )

        calib_text = (
            f"Calibrated {'YES' if sample.is_calibrated else 'NO'}  "
            f"Progress {sample.calibration_progress:.0%}"
        )
        dot_color = (60, 210, 95) if is_green else (80, 80, 220)
        cv2.circle(annotated, (18, 22), 7, dot_color, -1)
        cv2.putText(annotated, calib_text, (32, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)

        cv2.rectangle(annotated, (0, h - 58), (w, h), (10, 10, 10), thickness=-1)
        cv2.putText(annotated, status_text, (12, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (45, 220, 140), 2)
        cv2.putText(annotated, details_text, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
