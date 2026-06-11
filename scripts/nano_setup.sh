#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[nano_setup] Updating apt index"
sudo apt update

echo "[nano_setup] Installing system dependencies"
sudo apt install -y \
  git \
  python3 \
  python3-venv \
  python3-pip \
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
  libswscale-dev

if [[ ! -d .venv ]]; then
  echo "[nano_setup] Creating virtual environment"
  python3 -m venv .venv
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
