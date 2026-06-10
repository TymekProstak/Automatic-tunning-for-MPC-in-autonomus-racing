#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Config
# ============================================================
WS="${TUNE_WS:-$HOME/Desktop/fs_control_ka_racing/TUNE_RANDOM_SEARCH}"
PY_SCRIPT_REL="${TUNE_PY_SCRIPT_REL:-src/random_search/random_search.py}"
SKIP_BUILD="${SKIP_BUILD:-0}"
ROS_SETUP="${TUNE_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
VENV_DIR="${TUNE_VENV_DIR:-$WS/.venv_tuning}"

RUN_LOG_DIR="${TUNE_RUN_LOG_DIR:-$WS/run_logs}"
mkdir -p "$RUN_LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$RUN_LOG_DIR/tuning_random_search_${STAMP}.log"

echo "[RUN] Workspace: $WS"
echo "[RUN] Python script rel: $PY_SCRIPT_REL"
echo "[RUN] ROS setup: $ROS_SETUP"
echo "[RUN] Log file: $RUN_LOG"

# ============================================================
# Basic checks
# ============================================================
if [ ! -d "$WS" ]; then
  echo "[FATAL] Workspace does not exist: $WS" >&2
  exit 1
fi

if [ ! -f "$ROS_SETUP" ]; then
  echo "[FATAL] ROS setup not found: $ROS_SETUP" >&2
  exit 2
fi

cd "$WS"

# ============================================================
# 0) Source ROS distro first
# ============================================================
# shellcheck disable=SC1090
source "$ROS_SETUP"

# ============================================================
# 1) Build catkin workspace
# ============================================================
if [ "$SKIP_BUILD" != "1" ]; then
  if command -v catkin >/dev/null 2>&1; then
    echo "[RUN] Building with: catkin build"
    catkin build
  elif command -v catkin_make >/dev/null 2>&1; then
    echo "[RUN] Building with: catkin_make"
    catkin_make
  else
    echo "[FATAL] Neither 'catkin' nor 'catkin_make' found in PATH." >&2
    exit 3
  fi
else
  echo "[RUN] SKIP_BUILD=1 -> skipping workspace build"
fi

# ============================================================
# 2) Source catkin env
# ============================================================
if [ -f "$WS/devel/setup.bash" ]; then
  # shellcheck disable=SC1090
  source "$WS/devel/setup.bash"
else
  echo "[FATAL] Missing: $WS/devel/setup.bash (build failed?)" >&2
  exit 4
fi

# ============================================================
# 3) Create + activate venv
# ============================================================
if [ ! -d "$VENV_DIR" ]; then
  echo "[RUN] Creating venv: $VENV_DIR"
  env PYTHONPATH="" python3 -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel >/dev/null

# ============================================================
# 4) Ensure Python deps
# ============================================================
python - <<'PY'
import sys
import subprocess
from typing import Optional

def ensure_import(import_name: str, pip_name: Optional[str] = None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
        mod = sys.modules[import_name]
        ver = getattr(mod, "__version__", "unknown")
        print(f"[OK] {import_name} available: {ver}")
    except ImportError:
        print(f"[RUN] Installing {pip_name} into venv...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        __import__(import_name)
        mod = sys.modules[import_name]
        ver = getattr(mod, "__version__", "unknown")
        print(f"[OK] {import_name} installed: {ver}")

ensure_import("numpy")
PY

# ============================================================
# 5) Resolve Python script
# ============================================================
export TUNE_WS="$WS"

PY_SCRIPT="$WS/$PY_SCRIPT_REL"
if [ ! -f "$PY_SCRIPT" ]; then
  echo "[FATAL] Python script not found: $PY_SCRIPT" >&2
  exit 5
fi

echo "[RUN] Resolved Python script: $PY_SCRIPT"

# ============================================================
# 6) Run tuning
# ============================================================
echo "[RUN] Starting tuning..."
echo "[RUN] Output will be tee'd to: $RUN_LOG"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL python -u "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
else
  python -u "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
fi
