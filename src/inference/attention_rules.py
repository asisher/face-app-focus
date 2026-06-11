from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

from app.config import AppConfig
from domain.models import AttentionSample, AttentionState
from inference.face_landmarks import FaceObservation


@dataclass(slots=True)
class RuleSignals:
    focus_prob: float
    reason: str


class AttentionClassifier:
    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._recent: Deque[AttentionState] = deque(maxlen=8)
        self._neutral_yaw = 0.0
        self._neutral_pitch = 0.0
        self._neutral_samples = 0

    def classify(self, observation: FaceObservation, sample: AttentionSample) -> AttentionSample:
        calibration_progress = min(1.0, self._neutral_samples / max(1, self._cfg.neutral_min_samples))
        is_calibrated = self._neutral_samples >= self._cfg.neutral_min_samples

        if not observation.has_face or observation.confidence < self._cfg.min_confidence:
            state = AttentionState.UNKNOWN
            reason = "face_not_reliable"
            confidence = max(0.0, observation.confidence)
            focus_prob = 0.35
            alignment_score = 0.0
        else:
            self._update_neutral(observation)
            calibration_progress = min(1.0, self._neutral_samples / max(1, self._cfg.neutral_min_samples))
            is_calibrated = self._neutral_samples >= self._cfg.neutral_min_samples
            signals = self._signals(observation)
            if signals.focus_prob >= self._cfg.focus_threshold:
                state = AttentionState.FOCUSED
            elif signals.focus_prob <= self._cfg.distracted_threshold:
                state = AttentionState.DISTRACTED
            else:
                state = AttentionState.UNKNOWN
            reason = signals.reason
            confidence = observation.confidence
            focus_prob = signals.focus_prob
            alignment_score = self._alignment_score(observation)

        self._recent.append(state)
        smoothed = self._smooth_state(state)
        return AttentionSample(
            timestamp=sample.timestamp,
            state=smoothed,
            confidence=confidence,
            reason=reason,
            focus_prob=focus_prob,
            yaw_deg=observation.yaw_deg,
            pitch_deg=observation.pitch_deg,
            eye_down_score=observation.eye_down_score,
            nose_x_norm=observation.nose_x_norm,
            nose_y_norm=observation.nose_y_norm,
            left_pupil_x_norm=observation.left_pupil_x_norm,
            left_pupil_y_norm=observation.left_pupil_y_norm,
            right_pupil_x_norm=observation.right_pupil_x_norm,
            right_pupil_y_norm=observation.right_pupil_y_norm,
            calibration_progress=calibration_progress,
            is_calibrated=is_calibrated,
            alignment_score=alignment_score,
        )

    def _signals(self, observation: FaceObservation) -> RuleSignals:
        calibration_progress = min(1.0, self._neutral_samples / max(1, self._cfg.neutral_min_samples))
        warmup = 1.0 - calibration_progress

        yaw_from_neutral = observation.yaw_deg - self._neutral_yaw
        pitch_from_neutral = observation.pitch_deg - self._neutral_pitch
        abs_yaw = abs(yaw_from_neutral)
        abs_pitch = abs(pitch_from_neutral)

        # During warm-up, keep thresholds looser to avoid false distraction labels.
        down_threshold = self._cfg.distract_down_pitch_deg * (1.0 + (0.45 * warmup))
        side_threshold = self._cfg.distract_side_yaw_deg * (1.0 + (0.40 * warmup))

        down = pitch_from_neutral > down_threshold
        side = abs_yaw > side_threshold
        eye_down = observation.eye_down_score >= self._cfg.phone_eye_down_threshold
        pupils_visible = self._pupils_visible(observation)

        face_toward_screen = (
            abs_yaw <= (0.65 * side_threshold)
            and abs_pitch <= (0.65 * down_threshold)
            and not eye_down
        )

        if pupils_visible and face_toward_screen:
            return RuleSignals(focus_prob=max(0.82, observation.confidence * 0.92), reason="pupils_forward")

        yaw_penalty = min(1.0, abs_yaw / max(1.0, side_threshold * 2.0))
        pitch_penalty = min(1.0, abs_pitch / max(1.0, down_threshold * 2.0))
        eye_penalty = min(1.0, observation.eye_down_score)
        direction_score = max(
            0.0,
            1.0 - ((0.55 * yaw_penalty) + (0.25 * pitch_penalty) + (0.20 * eye_penalty)),
        )
        focus_prob = max(0.0, min(1.0, direction_score * observation.confidence))

        if down and side:
            return RuleSignals(focus_prob=min(focus_prob, 0.18), reason="down_and_side")
        if down and eye_down:
            return RuleSignals(focus_prob=min(focus_prob, 0.15), reason="phone_glance_likely")
        if down:
            return RuleSignals(focus_prob=min(focus_prob, 0.30), reason="looking_down")
        if eye_down:
            return RuleSignals(focus_prob=min(focus_prob, 0.35), reason="eyes_down")
        if side:
            return RuleSignals(focus_prob=min(focus_prob, 0.40), reason="looking_away_from_screen")
        if direction_score < 0.70:
            return RuleSignals(focus_prob=focus_prob, reason="partially_off_center")
        if calibration_progress < 1.0:
            return RuleSignals(focus_prob=max(0.70, focus_prob), reason="eyes_forward_warmup")
        return RuleSignals(focus_prob=max(0.75, focus_prob), reason="eyes_forward")

    def _pupils_visible(self, observation: FaceObservation) -> bool:
        return (
            observation.left_pupil_x_norm >= 0.0
            and observation.left_pupil_y_norm >= 0.0
            and observation.right_pupil_x_norm >= 0.0
            and observation.right_pupil_y_norm >= 0.0
        )

    def _smooth_state(self, latest: AttentionState) -> AttentionState:
        if len(self._recent) < 4:
            return latest

        focused_votes = sum(1 for s in self._recent if s == AttentionState.FOCUSED)
        distracted_votes = sum(1 for s in self._recent if s == AttentionState.DISTRACTED)

        if focused_votes >= 5:
            return AttentionState.FOCUSED
        # Require stronger consistency for distraction to reduce false positives.
        if distracted_votes >= 6:
            return AttentionState.DISTRACTED
        return latest

    def _update_neutral(self, observation: FaceObservation) -> None:
        # Adaptive baseline for "center" to match camera placement and user posture.
        if observation.confidence < max(self._cfg.min_confidence, 0.60):
            return

        if self._neutral_samples == 0:
            # Bootstrap from any plausible forward-looking pose.
            if (
                abs(observation.yaw_deg) <= 35.0
                and abs(observation.pitch_deg) <= 35.0
                and observation.eye_down_score < self._cfg.phone_eye_down_threshold
            ):
                self._neutral_yaw = observation.yaw_deg
                self._neutral_pitch = observation.pitch_deg
                self._neutral_samples = 1
            return

        threshold_scale = 1.7 if self._neutral_samples < self._cfg.neutral_min_samples else 1.0

        centered_enough = (
            abs(observation.yaw_deg - self._neutral_yaw) <= (self._cfg.neutral_update_max_yaw * threshold_scale)
            and abs(observation.pitch_deg - self._neutral_pitch) <= (self._cfg.neutral_update_max_pitch * threshold_scale)
            and observation.eye_down_score < self._cfg.phone_eye_down_threshold
        )
        if not centered_enough:
            return

        alpha = self._cfg.neutral_ema_alpha
        if self._neutral_samples == 0:
            self._neutral_yaw = observation.yaw_deg
            self._neutral_pitch = observation.pitch_deg
        else:
            self._neutral_yaw = ((1.0 - alpha) * self._neutral_yaw) + (alpha * observation.yaw_deg)
            self._neutral_pitch = ((1.0 - alpha) * self._neutral_pitch) + (alpha * observation.pitch_deg)
        self._neutral_samples += 1

    def _alignment_score(self, observation: FaceObservation) -> float:
        yaw_delta = abs(observation.yaw_deg - self._neutral_yaw)
        pitch_delta = abs(observation.pitch_deg - self._neutral_pitch)
        yaw_score = max(0.0, 1.0 - (yaw_delta / max(1.0, self._cfg.neutral_update_max_yaw)))
        pitch_score = max(0.0, 1.0 - (pitch_delta / max(1.0, self._cfg.neutral_update_max_pitch)))
        return max(0.0, min(1.0, (0.55 * yaw_score) + (0.45 * pitch_score)))
