from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class AttentionState(StrEnum):
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class AttentionSample:
    timestamp: datetime
    state: AttentionState
    confidence: float
    reason: str
    focus_prob: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    eye_down_score: float = 0.0
    nose_x_norm: float = -1.0
    nose_y_norm: float = -1.0
    left_pupil_x_norm: float = -1.0
    left_pupil_y_norm: float = -1.0
    right_pupil_x_norm: float = -1.0
    right_pupil_y_norm: float = -1.0
    calibration_progress: float = 0.0
    is_calibrated: bool = False
    alignment_score: float = 0.0


@dataclass(slots=True)
class ScoreSnapshot:
    timestamp: datetime
    current_score: float
    hour_score: float
    total_score: float
    hour_focus_weight: float = 0.0
    hour_possible_weight: float = 0.0
    total_focus_weight: float = 0.0
    total_possible_weight: float = 0.0
