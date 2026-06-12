from __future__ import annotations

import signal
import time
from datetime import datetime

from app.config import load_config
from app.runtime import configure_runtime
from capture.camera import CameraStream
from domain.models import AttentionSample, AttentionState
from inference.attention_rules import AttentionClassifier
from inference.face_landmarks import FaceLandmarkDetector
from scoring.engine import ScoringEngine
from storage.repository import ScoreRepository


def main() -> int:
    configure_runtime()
    config = load_config()

    camera = CameraStream(
        config.camera_index,
        config.camera_width,
        config.camera_height,
        backend=config.camera_backend,
        allow_index_scan=config.camera_allow_index_scan,
        scan_max_index=config.camera_scan_max_index,
        recover_failures=config.camera_recover_failures,
    )
    detector = FaceLandmarkDetector()
    classifier = AttentionClassifier(config)
    scoring = ScoringEngine()
    repo = ScoreRepository(config.db_path)

    restored = repo.restore_latest(datetime.now())
    scoring.restore(
        restored.total_score,
        restored.hour_score,
        restored.hour_key,
        restored.hour_focus_weight,
        restored.hour_possible_weight,
        restored.total_focus_weight,
        restored.total_possible_weight,
    )

    running = True

    def _stop(sig, frame):  # type: ignore[no-untyped-def]
        nonlocal running
        running = False
        print("\n[headless] Stopping.")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    interval = 1.0 / max(1, config.sample_fps)
    print(f"[headless] Focus monitor running at {config.sample_fps} FPS. Ctrl+C to stop.")

    while running:
        tick_start = time.monotonic()
        now = datetime.now()

        frame = camera.read()
        base = AttentionSample(
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
                reason=frame.error or "camera_unavailable",
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
        else:
            observation = detector.infer(frame.frame)
            classified = classifier.classify(observation, base)

        score = scoring.update(classified)
        repo.append(classified, score)

        state_str = classified.state.value.upper()
        print(
            f"[{now.strftime('%H:%M:%S')}] "
            f"{state_str:<12} | "
            f"score={score.current_score:5.1f} "
            f"hour={score.hour_score:5.1f} "
            f"total={score.total_score:5.1f} | "
            f"reason={classified.reason:<28} "
            f"yaw={classified.yaw_deg:6.1f} "
            f"pitch={classified.pitch_deg:6.1f} "
            f"calib={classified.calibration_progress:.0%}"
        )

        elapsed = time.monotonic() - tick_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    camera.close()
    detector.close()
    repo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
