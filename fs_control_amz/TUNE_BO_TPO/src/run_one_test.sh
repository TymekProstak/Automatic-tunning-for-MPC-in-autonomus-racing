#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Config
# ============================================================
WS="${EVAL_WS:-$HOME/Desktop/fs_control_amz/TUNE_BO_TPO}"
PY_SCRIPT_REL="${EVAL_PY_SCRIPT_REL:-src/manual_theta.py}"

ROS_SETUP="${EVAL_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
SKIP_BUILD="${SKIP_BUILD:-0}"

EVAL_PROFILE="${EVAL_PROFILE:-night_4track}"
VENV_DIR="${EVAL_VENV_DIR:-$WS/.venv_eval}"
RUN_LOG_DIR="${EVAL_RUNNER_LOG_DIR:-$WS/run_logs}"
mkdir -p "$RUN_LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUNNER_LOG="$RUN_LOG_DIR/manual_single_config_eval_${EVAL_PROFILE}_${STAMP}.log"

echo "[RUN] Workspace        : $WS"
echo "[RUN] Python script rel: $PY_SCRIPT_REL"
echo "[RUN] ROS setup        : $ROS_SETUP"
echo "[RUN] Venv             : $VENV_DIR"
echo "[RUN] EVAL_PROFILE     : $EVAL_PROFILE"
echo "[RUN] Runner log       : $RUNNER_LOG"

# ============================================================
# Checks
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
# 1) Source ROS
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
# 3) Source catkin env
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

echo "[RUN] Resolved Python script: $PY_SCRIPT"

# ============================================================
# 6) Export env
# ============================================================
export EVAL_WS="$WS"
export EVAL_PROFILE="$EVAL_PROFILE"

# opcjonalne override'y:
# export EVAL_TRACK_SET="8,9,10,11,12,13,14"
# export EVAL_SIM_TIME_S=60
# export EVAL_UCB_WEIGHT=1.0
# export EVAL_TEST_CRITICAL_CRASH_MULTIPLIER=1.0
# export EVAL_ROS_MASTER_PORT=11331
#
# Możesz też nadpisać nastawy bez edycji pythona:
# export MANUAL_THETA_JSON='{"velocity_planner.v_max":13.8,"velocity_planner.mux_acc":0.75,"velocity_planner.mux_dec":0.62,"velocity_planner.muy":0.80,"mpc.cost.Q_y":9.0,"mpc.cost.Q_psi":14.0,"mpc.cost.R_ddelta":2.5}'
#
# albo z pliku:
# export MANUAL_THETA_JSON_PATH="$WS/my_manual_theta.json"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

# ============================================================
# 7) Run
# ============================================================
echo "[RUN] Starting manual single-config evaluation..."
echo "[RUN] Output will be tee'd to: $RUNNER_LOG"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL python3 -u "$PY_SCRIPT" 2>&1 | tee "$RUNNER_LOG"
else
  python3 -u "$PY_SCRIPT" 2>&1 | tee "$RUNNER_LOG"
fi
