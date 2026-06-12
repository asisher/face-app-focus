# Jetson Nano Setup Guide

This project can be cloned directly on Jetson Nano and set up with two commands.

## 1. Clone

```bash
git clone <your-repo-url> data-finder
cd data-finder
```

## 2. Make scripts executable

```bash
chmod +x scripts/nano_setup.sh scripts/nano_run.sh
```

## 3. Install and prepare environment

```bash
./scripts/nano_setup.sh
```

## 4. Configure app tuning

Edit `.env` and use values tuned for Nano:

```env
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_BACKEND=jetson-csi
CAMERA_ROTATION=180
CAMERA_ALLOW_INDEX_SCAN=false
CAMERA_SCAN_MAX_INDEX=5
CAMERA_RECOVER_FAILURES=3
SAMPLE_FPS=5

FOCUS_THRESHOLD=0.60
DISTRACTED_THRESHOLD=0.34
DISTRACT_DOWN_PITCH_DEG=18
DISTRACT_SIDE_YAW_DEG=24
PHONE_EYE_DOWN_THRESHOLD=0.56
MIN_CONFIDENCE=0.40

NEUTRAL_EMA_ALPHA=0.08
NEUTRAL_UPDATE_MAX_YAW=16
NEUTRAL_UPDATE_MAX_PITCH=14
NEUTRAL_MIN_SAMPLES=18
```

## 5. Run

```bash
./scripts/nano_run.sh
```

## Notes

- Keep active cooling enabled on Nano.
- Use stable power supply (5V/4A recommended).
- If `mediapipe` fails to install on your JetPack version, pinning dependencies or using a Jetson-compatible wheel may be required.
