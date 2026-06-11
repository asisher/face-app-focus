#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "[nano_run] Missing .venv. Run ./scripts/nano_setup.sh first."
  exit 1
fi

echo "[nano_run] Enabling Jetson max performance mode"
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

source .venv/bin/activate
exec focus-monitor
