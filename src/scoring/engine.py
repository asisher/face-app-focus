from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from domain.models import AttentionSample, AttentionState, ScoreSnapshot


@dataclass(slots=True)
class ScoreState:
    current_score: float = 50.0
    hour_score: float = 50.0
    total_score: float = 50.0
    active_hour_key: str = ""
    hour_focus_weight: float = 0.0
    hour_possible_weight: float = 0.0
    total_focus_weight: float = 0.0
    total_possible_weight: float = 0.0


class ScoringEngine:
    def __init__(self) -> None:
        self._state = ScoreState()

    @property
    def state(self) -> ScoreState:
        return self._state

    def restore(
        self,
        total_score: float,
        hour_score: float,
        hour_key: str,
        hour_focus_weight: float,
        hour_possible_weight: float,
        total_focus_weight: float,
        total_possible_weight: float,
    ) -> None:
        self._state.total_score = max(0.0, min(100.0, total_score))
        self._state.hour_score = max(0.0, min(100.0, hour_score))
        self._state.active_hour_key = hour_key
        self._state.hour_focus_weight = max(0.0, hour_focus_weight)
        self._state.hour_possible_weight = max(0.0, hour_possible_weight)
        self._state.total_focus_weight = max(0.0, total_focus_weight)
        self._state.total_possible_weight = max(0.0, total_possible_weight)

    def update(self, sample: AttentionSample) -> ScoreSnapshot:
        hour_key = sample.timestamp.strftime("%Y-%m-%d %H")
        if self._state.active_hour_key and hour_key != self._state.active_hour_key:
            self._state.hour_score = 50.0
            self._state.hour_focus_weight = 0.0
            self._state.hour_possible_weight = 0.0
        self._state.active_hour_key = hour_key

        focus_value = self._focus_value(sample)
        weight = max(0.1, sample.confidence)
        unknown_factor = 0.4 if sample.state == AttentionState.UNKNOWN else 1.0
        possible = weight * unknown_factor
        focused = focus_value * possible

        self._state.hour_possible_weight += possible
        self._state.hour_focus_weight += focused
        self._state.total_possible_weight += possible
        self._state.total_focus_weight += focused

        self._state.hour_score = self._ratio_score(
            self._state.hour_focus_weight,
            self._state.hour_possible_weight,
        )
        self._state.total_score = self._ratio_score(
            self._state.total_focus_weight,
            self._state.total_possible_weight,
        )

        # Smooth the live score to prevent rapid jumps while preserving trend direction.
        instant_score = 100.0 * focus_value
        alpha = 0.18
        self._state.current_score = ((1.0 - alpha) * self._state.current_score) + (alpha * instant_score)

        return ScoreSnapshot(
            timestamp=sample.timestamp,
            current_score=round(self._state.current_score, 2),
            hour_score=round(self._state.hour_score, 2),
            total_score=round(self._state.total_score, 2),
            hour_focus_weight=round(self._state.hour_focus_weight, 5),
            hour_possible_weight=round(self._state.hour_possible_weight, 5),
            total_focus_weight=round(self._state.total_focus_weight, 5),
            total_possible_weight=round(self._state.total_possible_weight, 5),
        )

    def _focus_value(self, sample: AttentionSample) -> float:
        if sample.state == AttentionState.FOCUSED:
            return max(0.6, sample.focus_prob)
        if sample.state == AttentionState.DISTRACTED:
            return min(0.35, sample.focus_prob)
        return 0.45

    def _ratio_score(self, focus_weight: float, possible_weight: float) -> float:
        if possible_weight <= 1e-9:
            return 50.0
        return max(0.0, min(100.0, 100.0 * (focus_weight / possible_weight)))


def hour_key_for_timestamp(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H")
