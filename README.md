# Focus Monitor (MVP)

Privacy-first Python desktop tool that estimates attention from your webcam and shows:

- Live camera preview (what the app sees)
- Live state (`Focused`, `Distracted`, `Unknown`)
- Current score
- Current hour score
- Total score for the day/session
- Diagnostic metrics (`yaw`, `pitch`, `focus_prob`, `confidence`) and reason label

No frames are stored. Only event and score summaries are written to SQLite.

## Quick Start (macOS)

1. Ensure Python 3.11+ is available. If needed:

```bash
brew install python@3.12
```

2. Create and activate a virtual environment:

```bash
/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -e .
```

4. Optional: copy config and edit thresholds:

```bash
cp .env.example .env
```

5. Run:

```bash
focus-monitor
```

## Notes

- Camera access permission is required on macOS.
- `mediapipe` is used for facial landmarks. If landmarks are unavailable, the app falls back to `Unknown` state.
- Scoring is ratio-based (weighted focus percentage), with a smoothed live score to avoid sudden collapses.

## Jetson Nano Quick Setup

This repository now includes a Nano bootstrap flow so you can clone and set up quickly.

1. Clone this repo on the Nano.
2. Run `./scripts/nano_setup.sh`.
3. Run `./scripts/nano_run.sh`.

Detailed instructions: `docs/jetson-nano-setup.md`.

## Camera Troubleshooting

If you see `camera preview unavailable` or `camera_read_failed`:

1. Confirm camera permission for Terminal/VS Code in macOS Privacy settings.
2. Close other apps using the camera (Zoom, Teams, browser tabs).
3. Try camera overrides in `.env`:

```bash
CAMERA_BACKEND=avfoundation
CAMERA_INDEX=0
CAMERA_ALLOW_INDEX_SCAN=false
CAMERA_SCAN_MAX_INDEX=5
CAMERA_RECOVER_FAILURES=3
```

4. Restart the app after changing `.env`.

The app reads `.env` automatically at startup.
Set `CAMERA_ALLOW_INDEX_SCAN=true` only if your default index is unknown.

## Startup Logs

MediaPipe/TFLite may print native startup logs on some macOS systems.
Most of these are informational and do not indicate a failure.
The app now minimizes this output at startup.

## Focus Detection Tuning

The model now uses a combined matrix of:

- Head pose (`yaw`, `pitch`)
- Eye-down signal (`eye_down`) for phone-like glances
- Adaptive neutral center calibration (to your posture)

If phone checks are still marked focused:

- Lower `PHONE_EYE_DOWN_THRESHOLD` (for example `0.50`).

If centered screen gaze is marked distracted:

- Raise `DISTRACTED_THRESHOLD` slightly (for example `0.38`).
- Raise `DISTRACT_SIDE_YAW_DEG` slightly (for example `24`).
