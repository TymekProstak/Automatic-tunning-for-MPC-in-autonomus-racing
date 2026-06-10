#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Config
# ============================================================
WS="${TUNE_WS:-$HOME/Desktop/fs_control_agh_racing/TUNE_CMA_ES}"

# Poprawiona ścieżka:
PY_SCRIPT_REL="${TUNE_PY_SCRIPT_REL:-src/evolution_optimization/ev_op_fixed.py}"

SKIP_BUILD="${SKIP_BUILD:-0}"
ROS_SETUP="${TUNE_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
VENV_DIR="${TUNE_VENV_DIR:-$WS/.venv_tuning}"

RUN_LOG_DIR="${TUNE_RUN_LOG_DIR:-$WS/run_logs}"
mkdir -p "$RUN_LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$RUN_LOG_DIR/tuning_cma_${STAMP}.log"

echo "[RUN] Workspace         : $WS"
echo "[RUN] Python script rel : $PY_SCRIPT_REL"
echo "[RUN] ROS setup         : $ROS_SETUP"
echo "[RUN] Venv              : $VENV_DIR"
echo "[RUN] Log file          : $RUN_LOG"

# ============================================================
# Basic checks
# ============================================================
if [[ ! -d "$WS" ]]; then
  echo "[FATAL] Workspace does not exist: $WS" >&2
  exit 1
fi

if [[ ! -f "$ROS_SETUP" ]]; then
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
if [[ "$SKIP_BUILD" != "1" ]]; then
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
# 2) Source catkin environment
# ============================================================
if [[ -f "$WS/devel/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "$WS/devel/setup.bash"
elif [[ -f "$WS/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "$WS/install/setup.bash"
else
  echo "[FATAL] Missing both:" >&2
  echo "        $WS/devel/setup.bash" >&2
  echo "        $WS/install/setup.bash" >&2
  echo "        (build failed or wrong workspace?)" >&2
  exit 4
fi

# ============================================================
# 3) Create + activate venv
# ============================================================
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[RUN] Creating venv: $VENV_DIR"
  env PYTHONPATH="" python3 -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python3 -m pip install -U pip setuptools wheel >/dev/null

# ============================================================
# 4) Ensure Python deps
# ============================================================
python3 - <<'PY'
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
    except Exception:
        print(f"[RUN] Installing {pip_name} into venv...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        __import__(import_name)
        mod = sys.modules[import_name]
        ver = getattr(mod, "__version__", "unknown")
        print(f"[OK] {import_name} installed: {ver}")

ensure_import("numpy")
ensure_import("cma")
PY

# ============================================================
# 5) Resolve Python script
# ============================================================
export TUNE_WS="$WS"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

PY_SCRIPT="$WS/$PY_SCRIPT_REL"
if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[FATAL] Python script not found: $PY_SCRIPT" >&2
  exit 5
fi

echo "[RUN] Resolved Python script: $PY_SCRIPT"

# ============================================================
# 6) Run tuning
# ============================================================
echo "[RUN] Starting tuning: $PY_SCRIPT"
echo "[RUN] Output will be tee'd to: $RUN_LOG"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL python3 -u "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
else
  python3 -u "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
fi
