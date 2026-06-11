from __future__ import annotations

import os


def configure_runtime() -> None:
    # Reduce TensorFlow/MediaPipe native logs and prefer CPU execution on desktop.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("GLOG_logtostderr", "1")
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("error")
    except Exception:
        # Keep startup resilient even if absl API changes.
        pass
