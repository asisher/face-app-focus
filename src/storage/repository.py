from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from domain.models import AttentionSample, ScoreSnapshot
from scoring.engine import hour_key_for_timestamp


@dataclass(slots=True)
class RestoredScores:
    total_score: float
    hour_score: float
    hour_key: str
    hour_focus_weight: float
    hour_possible_weight: float
    total_focus_weight: float
    total_possible_weight: float


class ScoreRepository:
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples_v2 (
                ts TEXT NOT NULL,
                state TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                current_score REAL NOT NULL,
                hour_score REAL NOT NULL,
                total_score REAL NOT NULL,
                hour_key TEXT NOT NULL,
                hour_focus_weight REAL NOT NULL,
                hour_possible_weight REAL NOT NULL,
                total_focus_weight REAL NOT NULL,
                total_possible_weight REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def append(self, sample: AttentionSample, score: ScoreSnapshot) -> None:
        self._conn.execute(
            """
            INSERT INTO samples_v2 (
                ts, state, confidence, reason,
                current_score, hour_score, total_score, hour_key,
                hour_focus_weight, hour_possible_weight,
                total_focus_weight, total_possible_weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                score.timestamp.isoformat(),
                sample.state.value,
                sample.confidence,
                sample.reason,
                score.current_score,
                score.hour_score,
                score.total_score,
                hour_key_for_timestamp(score.timestamp),
                score.hour_focus_weight,
                score.hour_possible_weight,
                score.total_focus_weight,
                score.total_possible_weight,
            ),
        )
        self._conn.commit()

    def restore_latest(self, now: datetime) -> RestoredScores:
        row = self._conn.execute(
            """
            SELECT
                total_score,
                total_focus_weight,
                total_possible_weight
            FROM samples_v2
            ORDER BY ts DESC LIMIT 1
            """
        ).fetchone()
        total_score = float(row[0]) if row else 50.0
        total_focus_weight = float(row[1]) if row else 0.0
        total_possible_weight = float(row[2]) if row else 0.0
        current_hour = hour_key_for_timestamp(now)

        row_hour = self._conn.execute(
            """
            SELECT
                hour_score,
                hour_focus_weight,
                hour_possible_weight
            FROM samples_v2
            WHERE hour_key = ?
            ORDER BY ts DESC LIMIT 1
            """,
            (current_hour,),
        ).fetchone()
        hour_score = float(row_hour[0]) if row_hour else 50.0
        hour_focus_weight = float(row_hour[1]) if row_hour else 0.0
        hour_possible_weight = float(row_hour[2]) if row_hour else 0.0
        return RestoredScores(
            total_score=total_score,
            hour_score=hour_score,
            hour_key=current_hour,
            hour_focus_weight=hour_focus_weight,
            hour_possible_weight=hour_possible_weight,
            total_focus_weight=total_focus_weight,
            total_possible_weight=total_possible_weight,
        )

    def close(self) -> None:
        self._conn.close()
