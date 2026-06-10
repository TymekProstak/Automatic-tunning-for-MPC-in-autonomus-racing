#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Config
# ============================================================

# MUSI być spójne z DEFAULT_CATKIN_WS w tunerze Pythona
WS="${TUNE_WS:-$HOME/Desktop/fs_control_agh_racing/NO_STAGED/TUNE_RANDOM_SEARCH}"

# Domyślna ścieżka do staged random search
PY_SCRIPT_REL="${TUNE_PY_SCRIPT_REL:-src/random_search/random_search.py}"

# Jeśli nie chcesz przebudowy workspace
SKIP_BUILD="${SKIP_BUILD:-0}"

# ROS distro setup
ROS_SETUP="${TUNE_ROS_SETUP:-/opt/ros/noetic/setup.bash}"

# Lokalny venv
VENV_DIR="${TUNE_VENV_DIR:-$WS/.venv_tuning}"

# Opcjonalne dodatkowe pip pakiety:
# np. export TUNE_EXTRA_PIP="numpy matplotlib"
TUNE_EXTRA_PIP="${TUNE_EXTRA_PIP:-}"

# Logi runów
RUN_LOG_DIR="${TUNE_RUN_LOG_DIR:-$WS/run_logs}"
mkdir -p "$RUN_LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$RUN_LOG_DIR/random_search_tuning_${STAMP}.log"

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
# 1) Source ROS distro first
# ============================================================
# shellcheck disable=SC1090
source "$ROS_SETUP"

# ============================================================
# 2) Build catkin workspace
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
# 3) Source catkin environment
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
# 4) Create + activate venv
# ============================================================
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[RUN] Creating venv: $VENV_DIR"
  env PYTHONPATH="" python3 -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python3 -m pip install -U pip setuptools wheel >/dev/null

# ============================================================
# 5) Optional extra pip packages
# ============================================================
if [[ -n "$TUNE_EXTRA_PIP" ]]; then
  echo "[RUN] Installing extra pip packages: $TUNE_EXTRA_PIP"
  python3 -m pip install $TUNE_EXTRA_PIP
else
  echo "[RUN] No extra pip packages requested."
fi

# ============================================================
# 6) Resolve Python script
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
# 7) Run tuning
# ============================================================
echo "[RUN] Starting staged random-search tuning..."
echo "[RUN] Output will be tee'd to: $RUN_LOG"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL python3 -u "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
else
  python3 -u "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
fi
