#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Config
# ============================================================

# MUSI być spójne z DEFAULT_CATKIN_WS z tunera Pythona
WS="${TUNE_WS:-$HOME/Desktop/fs_control_ka_racing/NO_STAGED/TUNE_BO_TPO}"

# Domyślny skrypt pod nowe custom-TPE 3-stage
PY_SCRIPT_REL="${TUNE_PY_SCRIPT_REL:-src/BO_TPO/tune_tpe_ka.py}"

# Jeśli nie chcesz przebudowy
SKIP_BUILD="${SKIP_BUILD:-0}"

# Setup ROS distro
ROS_SETUP="${TUNE_ROS_SETUP:-/opt/ros/noetic/setup.bash}"

# Lokalny venv
VENV_DIR="${TUNE_VENV_DIR:-$WS/.venv_tuning}"

# Dodatkowe pip pakiety opcjonalnie
TUNE_EXTRA_PIP="${TUNE_EXTRA_PIP:-}"

# Logi runów
RUN_LOG_DIR="${TUNE_RUN_LOG_DIR:-$WS/run_logs}"
mkdir -p "$RUN_LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$RUN_LOG_DIR/tuning_${STAMP}.log"

echo "[RUN] Workspace        : $WS"
echo "[RUN] Python script rel: $PY_SCRIPT_REL"
echo "[RUN] ROS setup        : $ROS_SETUP"
echo "[RUN] Venv             : $VENV_DIR"
echo "[RUN] Log file         : $RUN_LOG"

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
# 1) Source ROS distro first
# ============================================================
source "$ROS_SETUP"

# ============================================================
# 2) Build catkin workspace
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
# 3) Source catkin environment
# ============================================================
if [ -f "$WS/devel/setup.bash" ]; then
  source "$WS/devel/setup.bash"
elif [ -f "$WS/install/setup.bash" ]; then
  source "$WS/install/setup.bash"
else
  echo "[FATAL] Missing both devel/setup.bash and install/setup.bash" >&2
  exit 4
fi

# ============================================================
# 4) Create + activate venv
# ============================================================
if [ ! -d "$VENV_DIR" ]; then
  echo "[RUN] Creating venv: $VENV_DIR"
  # Dodane env PYTHONPATH="" i --system-site-packages aby środowisko było zdrowe
  env PYTHONPATH="" python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel >/dev/null

# ============================================================
# 5) Optional Python deps (Inline Python Block)
# ============================================================
python - <<'PY'
import sys
import subprocess
import os
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
        # Poprawka dodana tutaj: podanie "install" do pipa!
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        
        try:
            __import__(import_name)
            mod = sys.modules[import_name]
            ver = getattr(mod, "__version__", "unknown")
            print(f"[OK] {import_name} installed: {ver}")
        except ImportError as e:
            print(f"[ERROR] Failed to import {import_name} after installation: {e}")

# Sprawdzenie extra paczek zdefiniowanych w TUNE_EXTRA_PIP (Bash)
extra_pip = os.environ.get("TUNE_EXTRA_PIP", "").strip()
if extra_pip:
    print(f"[RUN] Installing extra pip packages from env: {extra_pip}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + extra_pip.split())

# Możesz tu dodawać kolejne zależności dla tunera, np:
ensure_import("skopt", "scikit-optimize")
PY

# ============================================================
# 6) Resolve Python script
# ============================================================
export TUNE_WS="$WS"
export PYTHONUNBUFFERED=1

PY_SCRIPT="$WS/$PY_SCRIPT_REL"
if [ ! -f "$PY_SCRIPT" ]; then
  echo "[FATAL] Python script not found: $PY_SCRIPT" >&2
  exit 5
fi

echo "[RUN] Resolved Python script: $PY_SCRIPT"

# ============================================================
# 7) Run tuning
# ============================================================
echo "[RUN] Starting tuning..."
echo "[RUN] Output will be tee'd to: $RUN_LOG"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL python "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
else
  python "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
fi
