#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[nano_setup] Updating apt index"
sudo apt update

echo "[nano_setup] Installing system dependencies"
sudo apt install -y \
  git \
  build-essential \
  cmake \
  pkg-config \
  libopenblas-dev \
  liblapack-dev \
  libatlas-base-dev \
  libjpeg-dev \
  libtiff-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  software-properties-common

# Jetson Nano ships with Python 3.6 which is too old.
# Install Python 3.8 from deadsnakes PPA if not already present.
if ! command -v python3.8 &>/dev/null; then
  echo "[nano_setup] Installing Python 3.8 via deadsnakes PPA"
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install -y python3.8 python3.8-venv python3.8-dev
fi

PYTHON=python3.8
echo "[nano_setup] Using Python: $($PYTHON --version)"

if [[ ! -d .venv ]]; then
  echo "[nano_setup] Creating virtual environment with Python 3.8"
  $PYTHON -m venv .venv
fi

echo "[nano_setup] Installing Python dependencies"
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

if [[ ! -f .env && -f .env.example ]]; then
  echo "[nano_setup] Creating .env from .env.example"
  cp .env.example .env
fi

echo "[nano_setup] Done"
echo "[nano_setup] Next: run ./scripts/nano_run.sh"
