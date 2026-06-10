#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Config
# ============================================================
WS="${EVAL_WS:-$HOME/Desktop/fs_control_agh_racing/TUNE_BO_TPO}"

# Wskazujemy na skrypt Pythona dla TPE z Torque Vectoringiem
PY_SCRIPT_REL="${EVAL_PY_SCRIPT_REL:-src/BO_TPO/eval.py}"

# Ścieżka do logów z tuningu TPE
RUN_LOG_JSONL="${EVAL_RUN_LOG_PATH:-$WS/src/BO_TPO/tuning_run_tpe_tv_night_4track.jsonl}"

SKIP_BUILD="${SKIP_BUILD:-0}"
ROS_SETUP="${TUNE_ROS_SETUP:-/opt/ros/noetic/setup.bash}"

VENV_DIR="${EVAL_VENV_DIR:-$WS/.venv_eval}"
RUN_LOG_DIR="${EVAL_RUNNER_LOG_DIR:-$WS/run_logs}"
mkdir -p "$RUN_LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$RUN_LOG_DIR/eval_tpe_test_tracks_tv_${STAMP}.log"

echo "[RUN] Workspace         : $WS"
echo "[RUN] Python script rel : $PY_SCRIPT_REL"
echo "[RUN] JSONL log path    : $RUN_LOG_JSONL"
echo "[RUN] ROS setup         : $ROS_SETUP"
echo "[RUN] Venv              : $VENV_DIR"
echo "[RUN] Log file          : $RUN_LOG"

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
# 1) Source ROS distro
# ============================================================
# shellcheck disable=SC1090
source "$ROS_SETUP"

# ============================================================
# 2) Build workspace
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
# 3) Source workspace env
# ============================================================
if [ -f "$WS/devel/setup.bash" ]; then
  # shellcheck disable=SC1090
  source "$WS/devel/setup.bash"
elif [ -f "$WS/install/setup.bash" ]; then
  # shellcheck disable=SC1090
  source "$WS/install/setup.bash"
else
  echo "[FATAL] Missing both:" >&2
  echo "        $WS/devel/setup.bash" >&2
  echo "        $WS/install/setup.bash" >&2
  exit 4
fi

# ============================================================
# 4) Create + activate venv
# ============================================================
if [ ! -d "$VENV_DIR" ]; then
  echo "[RUN] Creating venv: $VENV_DIR"
  env PYTHONPATH="" python3 -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python3 -m pip install -U pip setuptools wheel >/dev/null

# ============================================================
# 5) Resolve script
# ============================================================
PY_SCRIPT="$WS/$PY_SCRIPT_REL"
if [ ! -f "$PY_SCRIPT" ]; then
  echo "[FATAL] Python script not found: $PY_SCRIPT" >&2
  exit 5
fi

if [ ! -f "$RUN_LOG_JSONL" ]; then
  echo "[WARN] Evaluation source JSONL not found: $RUN_LOG_JSONL" >&2
fi

echo "[RUN] Resolved Python script: $PY_SCRIPT"

# ============================================================
# 6) Export env
# ============================================================
export TUNE_WS="$WS"
export EVAL_WS="$WS"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

# Podajemy skryptowi TPE ścieżkę do logów wejściowych (TPE Python korzysta z tej zmiennej)
export TEST_TUNING_RUN_JSONL="$RUN_LOG_JSONL"

# ============================================================
# 7) Run test pipeline
# ============================================================
echo "[RUN] Starting TPE TV test-eval pipeline..."
echo "[RUN] Output will be tee'd to: $RUN_LOG"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL python3 -u "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
else
  python3 -u "$PY_SCRIPT" 2>&1 | tee "$RUN_LOG"
fi
