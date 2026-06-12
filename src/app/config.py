from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class AppConfig:
    db_path: str
    camera_index: int
    camera_width: int
    camera_height: int
    camera_backend: str
    camera_allow_index_scan: bool
    camera_scan_max_index: int
    camera_recover_failures: int
    sample_fps: int
    focus_threshold: float
    distracted_threshold: float
    distract_down_pitch_deg: float
    distract_side_yaw_deg: float
    phone_eye_down_threshold: float
    neutral_ema_alpha: float
    neutral_update_max_yaw: float
    neutral_update_max_pitch: float
    neutral_min_samples: int
    min_confidence: float


def load_config() -> AppConfig:
    _load_dotenv()

    return AppConfig(
        db_path=os.getenv("FOCUS_DB_PATH", "focus_monitor.db"),
        camera_index=_env_int("CAMERA_INDEX", 0),
        camera_width=_env_int("CAMERA_WIDTH", 640),
        camera_height=_env_int("CAMERA_HEIGHT", 480),
        camera_backend=_env_str("CAMERA_BACKEND", "auto").lower(),
        camera_allow_index_scan=_env_bool("CAMERA_ALLOW_INDEX_SCAN", False),
        camera_scan_max_index=max(0, _env_int("CAMERA_SCAN_MAX_INDEX", 5)),
        camera_recover_failures=max(1, _env_int("CAMERA_RECOVER_FAILURES", 3)),
        sample_fps=max(1, _env_int("SAMPLE_FPS", 5)),
        focus_threshold=_env_float("FOCUS_THRESHOLD", 0.60),
        distracted_threshold=_env_float("DISTRACTED_THRESHOLD", 0.35),
        distract_down_pitch_deg=_env_float("DISTRACT_DOWN_PITCH_DEG", 18.0),
        distract_side_yaw_deg=_env_float("DISTRACT_SIDE_YAW_DEG", 20.0),
        phone_eye_down_threshold=_env_float("PHONE_EYE_DOWN_THRESHOLD", 0.56),
        neutral_ema_alpha=_env_float("NEUTRAL_EMA_ALPHA", 0.06),
        neutral_update_max_yaw=_env_float("NEUTRAL_UPDATE_MAX_YAW", 14.0),
        neutral_update_max_pitch=_env_float("NEUTRAL_UPDATE_MAX_PITCH", 12.0),
        neutral_min_samples=max(5, _env_int("NEUTRAL_MIN_SAMPLES", 25)),
        min_confidence=_env_float("MIN_CONFIDENCE", 0.45),
    )
