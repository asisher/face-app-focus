from datetime import datetime

from domain.models import AttentionSample, AttentionState
from scoring.engine import ScoringEngine


def test_scoring_engine_changes_by_state() -> None:
    engine = ScoringEngine()
    ts = datetime(2026, 6, 9, 10, 0, 0)

    focused = AttentionSample(ts, AttentionState.FOCUSED, 0.9, "eyes_forward")
    distracted = AttentionSample(ts, AttentionState.DISTRACTED, 0.9, "looking_down")

    score_1 = engine.update(focused)
    score_2 = engine.update(distracted)

    assert score_1.hour_score >= 60.0
    assert score_2.current_score < score_1.current_score
    assert 0.0 <= score_2.total_score <= 100.0
