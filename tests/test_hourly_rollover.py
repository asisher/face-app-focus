from datetime import datetime

from domain.models import AttentionSample, AttentionState
from scoring.engine import ScoringEngine


def test_hour_score_rolls_over_between_hours() -> None:
    engine = ScoringEngine()

    before = AttentionSample(datetime(2026, 6, 9, 10, 59, 59), AttentionState.FOCUSED, 1.0, "eyes_forward")
    after = AttentionSample(datetime(2026, 6, 9, 11, 0, 1), AttentionState.FOCUSED, 1.0, "eyes_forward")

    score_before = engine.update(before)
    score_after = engine.update(after)

    assert score_before.hour_possible_weight > 0.0
    assert score_after.hour_possible_weight > 0.0
    assert score_after.hour_focus_weight <= score_after.hour_possible_weight
