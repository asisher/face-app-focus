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
        rotation=config.camera_rotation,
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

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    interval = 1.0 / max(1, config.sample_fps)

    # ANSI helpers
    CLEAR = "\033[2J\033[H"
    BOLD  = "\033[1m"
    RESET = "\033[0m"
    GREEN = "\033[32m"
    RED   = "\033[31m"
    YELLOW = "\033[33m"
    CYAN  = "\033[36m"

    def _bar(value: float, width: int = 20) -> str:
        filled = int(round(value / 100.0 * width))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _state_color(state: AttentionState) -> str:
        if state == AttentionState.FOCUSED:
            return GREEN
        if state == AttentionState.DISTRACTED:
            return RED
        return YELLOW

    tick = 0
    while running:
        tick_start = time.monotonic()
        now = datetime.now()
        tick += 1

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

        sc = _state_color(classified.state)
        state_str = classified.state.value.upper()
        w = 44  # inner box width

        lines = [
            CLEAR,
            CYAN + "+" + "-" * w + "+" + RESET,
            CYAN + "|" + RESET + BOLD + "  FOCUS MONITOR".center(w) + RESET + CYAN + "|" + RESET,
            CYAN + "+" + "-" * w + "+" + RESET,
            CYAN + "|" + RESET + f"  Time   : {now.strftime('%H:%M:%S')}".ljust(w) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Status : {sc}{BOLD}{state_str}{RESET}".ljust(w + len(sc) + len(BOLD) + len(RESET)) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Reason : {classified.reason}".ljust(w) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + " " * w + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Score  : {score.current_score:5.1f}  {_bar(score.current_score)}".ljust(w) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Hour   : {score.hour_score:5.1f}  {_bar(score.hour_score)}".ljust(w) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Total  : {score.total_score:5.1f}  {_bar(score.total_score)}".ljust(w) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + " " * w + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Yaw    : {classified.yaw_deg:6.1f} deg".ljust(w) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Pitch  : {classified.pitch_deg:6.1f} deg".ljust(w) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Conf   : {classified.confidence:.2f}".ljust(w) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Calib  : {classified.calibration_progress:.0%}".ljust(w) + CYAN + "|" + RESET,
            CYAN + "|" + RESET + " " * w + CYAN + "|" + RESET,
            CYAN + "|" + RESET + f"  Tick   : {tick}   FPS target: {config.sample_fps}".ljust(w) + CYAN + "|" + RESET,
            CYAN + "+" + "-" * w + "+" + RESET,
            "  Ctrl+C to stop",
        ]
        print("\n".join(lines), flush=True)

        elapsed = time.monotonic() - tick_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\n[headless] Stopped.")



if __name__ == "__main__":
    raise SystemExit(main())
