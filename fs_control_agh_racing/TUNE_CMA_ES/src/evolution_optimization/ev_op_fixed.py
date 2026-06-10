#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import signal
import socket
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np


# ============================================================
# Workspace
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_agh_racing/src/TUNE_CMA_ES")
CATKIN_WS = os.path.abspath(os.environ.get("TUNE_WS", DEFAULT_CATKIN_WS))
WS_SRC = os.path.join(CATKIN_WS, "src")

# ============================================================
# ROS launch paths
# ============================================================
SIM_LAUNCH = os.path.join(WS_SRC, "lem_simulator", "launch", "sim.launch")
CTRL_LAUNCH = os.path.join(WS_SRC, "dv_control", "launch", "control.launch")

CONTROL_PARAM_CANDIDATES = [
    os.path.join(WS_SRC, "dv_control", "config", "Params", "control_param.json"),
    os.path.join(WS_SRC, "dv_control", "config", "control_param.json"),
]


def _pick_first_existing(paths: List[str]) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    return paths[0]


CONTROL_PARAM_JSON = _pick_first_existing(CONTROL_PARAM_CANDIDATES)
METRICS_CSV = os.path.join(WS_SRC, "lem_simulator", "logs", "run_default_metrics.csv")

# ============================================================
# ACADOS
# ============================================================
DEFAULT_ACADOS_LIB = os.path.join(
    WS_SRC, "dv_control", "External", "acados", "install", "lib"
)
ACADOS_LIB = os.path.abspath(os.environ.get("TUNE_ACADOS_LIB", DEFAULT_ACADOS_LIB))

# ============================================================
# Evaluation setup
# ============================================================
GLOBAL_SEED = 123
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

DEFAULT_SIM_TIME_S = int(os.environ.get("TUNE_SIM_TIME_S", "60"))
UCB_WEIGHT = float(os.environ.get("TUNE_UCB_WEIGHT", "1.0"))

# ============================================================
# Profile / agreed staged budget
# ============================================================
PROFILE_NAME = os.environ.get("TUNE_PROFILE", "night_4track")

if PROFILE_NAME == "smoke_1track":
    TRACK_SET = [1]

    PHASE1_N_TRIALS = 12
    PHASE2_N_TRIALS = 24
    PHASE3_N_TRIALS = 24

    PHASE1_LAMBDA = 4
    PHASE1_MAXITER = 3

    PHASE2_LAMBDA = 6
    PHASE2_MAXITER = 4

    PHASE3_LAMBDA = 6
    PHASE3_MAXITER = 4

elif PROFILE_NAME == "night_4track":
    TRACK_SET = [1, 2, 3, 4]

    PHASE1_N_TRIALS = 48
    PHASE2_N_TRIALS = 64
    PHASE3_N_TRIALS = 96

    PHASE1_LAMBDA = 8
    PHASE1_MAXITER = 6

    PHASE2_LAMBDA = 8
    PHASE2_MAXITER = 8

    PHASE3_LAMBDA = 8
    PHASE3_MAXITER = 12

else:
    raise ValueError(f"Unknown PROFILE_NAME: {PROFILE_NAME}")


def _parse_track_set_env(var_name: str) -> Optional[List[int]]:
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return None
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError(f"{var_name} resolved to empty list")
    return out


def _override_cma_phase(phase: str, n_trials: int, lam: int, maxiter: int) -> Tuple[int, int, int]:
    lam_env = f"TUNE_{phase}_LAMBDA"
    maxiter_env = f"TUNE_{phase}_MAXITER"
    n_trials_env = f"TUNE_{phase}_N_TRIALS"

    lam2 = int(os.environ.get(lam_env, str(lam)))
    maxiter2 = int(os.environ.get(maxiter_env, str(maxiter)))
    n_trials2 = int(os.environ.get(n_trials_env, str(lam2 * maxiter2)))

    n_trials_was_set = n_trials_env in os.environ
    maxiter_was_set = maxiter_env in os.environ

    if n_trials_was_set and (lam2 * maxiter2) != n_trials2:
        if not maxiter_was_set:
            if n_trials2 % max(1, lam2) != 0:
                raise ValueError(
                    f"{n_trials_env}={n_trials2} must be divisible by {lam_env}={lam2} (or set {maxiter_env} explicitly)."
                )
            maxiter2 = n_trials2 // max(1, lam2)
        else:
            raise ValueError(
                f"Inconsistent overrides: {n_trials_env}={n_trials2} but {lam_env}*{maxiter_env}={lam2}*{maxiter2}={lam2*maxiter2}."
            )

    n_trials2 = lam2 * maxiter2
    return n_trials2, lam2, maxiter2


# Optional override: explicit training track list
_track_override = _parse_track_set_env("TUNE_TRACK_SET")
if _track_override is not None:
    TRACK_SET = _track_override

# Optional overrides: phase budgets (keeps CMA lambda/maxiter consistency)
PHASE1_N_TRIALS, PHASE1_LAMBDA, PHASE1_MAXITER = _override_cma_phase(
    "PHASE1", PHASE1_N_TRIALS, PHASE1_LAMBDA, PHASE1_MAXITER
)
PHASE2_N_TRIALS, PHASE2_LAMBDA, PHASE2_MAXITER = _override_cma_phase(
    "PHASE2", PHASE2_N_TRIALS, PHASE2_LAMBDA, PHASE2_MAXITER
)
PHASE3_N_TRIALS, PHASE3_LAMBDA, PHASE3_MAXITER = _override_cma_phase(
    "PHASE3", PHASE3_N_TRIALS, PHASE3_LAMBDA, PHASE3_MAXITER
)

assert PHASE1_LAMBDA * PHASE1_MAXITER == PHASE1_N_TRIALS
assert PHASE2_LAMBDA * PHASE2_MAXITER == PHASE2_N_TRIALS
assert PHASE3_LAMBDA * PHASE3_MAXITER == PHASE3_N_TRIALS

EPISODES_PER_TRIAL = len(TRACK_SET)
TOTAL_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS
TOTAL_EPISODES = TOTAL_TRIALS * EPISODES_PER_TRIAL
PURE_SIM_HOURS = TOTAL_EPISODES * DEFAULT_SIM_TIME_S / 3600.0

# ============================================================
# Critical crash multiplier
# ============================================================
PHASE1_CRITICAL_CRASH_MULTIPLIER = 1.00
PHASE2_CRITICAL_CRASH_MULTIPLIER = 1.00
PHASE3_CRITICAL_CRASH_MULTIPLIER = 1.00

# ============================================================
# Safe planner
# ============================================================
# Kept from the TV branch the user has been using.
SAFE_PLANNER = {
    "velocity_planner.v_max": 13.5,
    "velocity_planner.mux_acc": 0.7,
    "velocity_planner.mux_dec": 0.6,
    "velocity_planner.muy": 0.75,
}

# ============================================================
# Tuned params / bounds
# ============================================================
PLANNER_BOUNDS = {
    "velocity_planner.v_max": (8.0, 18.0),
    "velocity_planner.mux_acc": (0.3, 1.7),
    "velocity_planner.mux_dec": (0.3, 1.7),
    "velocity_planner.muy": (0.3, 1.7),
}

COST_BOUNDS = {
    "mpc.cost.Q_y": (1, 100.0),
    "mpc.cost.Q_psi": (1, 100.0),
    "mpc.cost.R_ddelta": (1, 100.0),
    "mpc.cost.R_tv": (1e-8, 1e-5),
}


ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(COST_BOUNDS)
ALL_BOUNDS.update(PLANNER_BOUNDS)

COST_KEYS = list(COST_BOUNDS.keys())
PLANNER_KEYS = list(PLANNER_BOUNDS.keys())
ALL_KEYS = COST_KEYS + PLANNER_KEYS
LOG_PARAMS = set(COST_BOUNDS.keys())

# ============================================================
# 3-stage architecture agreed with user
#   Stage 1: frozen planner, tune only costs
#   Stage 2: frozen best costs from stage 1, tune only planner
#   Stage 3: joint cost + planner, initialized from stage1/2 anchors
# ============================================================
STAGE1_COST_STD_FRAC = 0.30

if PROFILE_NAME == "smoke_1track":
    STAGE2_PLANNER_STD_FRAC_VMAX = 0.30
    STAGE2_PLANNER_STD_FRAC_MU = 0.28
    STAGE3_PLANNER_SAFE_BLEND = 0.25
elif PROFILE_NAME == "night_4track":
    STAGE2_PLANNER_STD_FRAC_VMAX = 0.35
    STAGE2_PLANNER_STD_FRAC_MU = 0.32
    STAGE3_PLANNER_SAFE_BLEND = 0.20
else:
    STAGE2_PLANNER_STD_FRAC_VMAX = 0.30
    STAGE2_PLANNER_STD_FRAC_MU = 0.28
    STAGE3_PLANNER_SAFE_BLEND = 0.25

STAGE3_COST_COV_ALPHA = 0.60
STAGE3_COST_COV_BETA = 1.50
STAGE3_COST_STD_MIN_FRAC = 0.05
STAGE3_COST_STD_MAX_FRAC = 0.10  # Sztywny limit 10% dla kosztów (log-space)

STAGE3_PLANNER_COV_ALPHA = 0.60
STAGE3_PLANNER_COV_BETA = 1.50
STAGE3_PLANNER_STD_MIN_FRAC = 0.10
STAGE3_PLANNER_STD_MAX_FRAC = 0.35

# ============================================================
# Cost function
# ============================================================
COST_WEIGHTS = {
    "vs_avg_mps": -2.0,
    "slip_ratio_metric": 175.0,
    "slip_angle_metric": 175.0,
}
CRASH_PENALTY = 30.0
CRASH_TIME_PENALTY = 0.05

SOFT_TRACK_DIV = 25.0
MEDIUM_TRACK_DIV = 2.0
HIGH_TRACK_MUL = 1.5

# ============================================================
# ROS / runtime
# ============================================================
SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("TUNE_ROS_MASTER_PORT", "11317"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_tune_cma_tv_{PROFILE_NAME}_{GLOBAL_SEED}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Outputs
# ============================================================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
TOP10_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_TOP10_PATH",
        os.path.join(SCRIPT_DIR, f"top10_so_far_tv_{PROFILE_NAME}.json"),
    )
)
SUMMARY_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_SUMMARY_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_summary_tv_{PROFILE_NAME}.json"),
    )
)
RUN_LOG_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_RUN_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_run_tv_{PROFILE_NAME}.jsonl"),
    )
)
DIST_LOG_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_DIST_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"cma_distribution_log_tv_{PROFILE_NAME}.jsonl"),
    )
)
BEST_JSON_PATH = os.path.join(
    os.path.dirname(CONTROL_PARAM_JSON),
    f"control_param_best_lti_3stage_cma_tv_{PROFILE_NAME}.json",
)

BEST_JSON_PATH = os.path.abspath(os.environ.get("TUNE_BEST_JSON_PATH", BEST_JSON_PATH))

# ============================================================
# Global state
# ============================================================
EPISODES_DONE = 0
TRIALS_DONE = 0
GLOBAL_CANDIDATE_ID = 0

BEST_OBJECTIVE_SO_FAR = float("inf")
BEST_THETA_SO_FAR: Dict[str, float] = {}

ALL_RESULTS: List[Dict[str, Any]] = []
PHASE1_RESULTS: List[Dict[str, Any]] = []
PHASE2_RESULTS: List[Dict[str, Any]] = []
PHASE3_RESULTS: List[Dict[str, Any]] = []

# ============================================================
# Helpers
# ============================================================
def _must_exist(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _ensure_ros_dirs() -> None:
    os.makedirs(ROS_HOME, exist_ok=True)
    os.makedirs(ROS_LOG_DIR, exist_ok=True)


def _ros_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    _ensure_ros_dirs()
    env = os.environ.copy()
    env["ROS_MASTER_URI"] = f"http://{ROS_MASTER_HOST}:{ROS_MASTER_PORT}"
    env["ROS_IP"] = "127.0.0.1"
    env["ROS_HOME"] = ROS_HOME
    env["ROS_LOG_DIR"] = ROS_LOG_DIR
    if extra:
        env.update(extra)
    return env


def _rosmaster_is_up(
    host: str = ROS_MASTER_HOST,
    port: int = ROS_MASTER_PORT,
    timeout: float = 0.3,
) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _popen_roscore() -> subprocess.Popen:
    cmd = ["roscore", "-p", str(ROS_MASTER_PORT)]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=_ros_env(),
        preexec_fn=os.setsid,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    p._cmd = cmd
    return p


def _set_json_path(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _save_json_atomic(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def apply_params_to_control_json(theta_x: Dict[str, float]) -> None:
    data = _load_json(CONTROL_PARAM_JSON)
    for path, val in theta_x.items():
        if path in ALL_BOUNDS:
            lo, hi = ALL_BOUNDS[path]
            _set_json_path(data, path, float(_clamp(float(val), lo, hi)))
    _save_json_atomic(CONTROL_PARAM_JSON, data)


def _parse_metrics_csv(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not os.path.exists(path):
        return out

    with open(path, "r") as f:
        lines = f.read().splitlines()

    if not lines:
        return out

    for line in lines[1:]:
        if not line.strip() or "," not in line:
            continue

        k, v = line.split(",", 1)
        k = k.strip()
        v = v.strip()

        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1]

        if v == "":
            out[k] = ""
            continue

        try:
            if "." in v or "e" in v or "E" in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v

    return out


def _wait_for_metrics_stable(path: str, timeout_s: float) -> bool:
    t0 = time.time()
    last_size = None
    stable_hits = 0

    while time.time() - t0 < timeout_s:
        if os.path.exists(path):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = None

            if size is not None:
                if last_size is not None and size == last_size and size > 0:
                    stable_hits += 1
                else:
                    stable_hits = 0
                last_size = size

                if stable_hits >= 2:
                    return True

        time.sleep(0.2)

    return False


def _metric_float_first(
    metrics: Dict[str, Any],
    names: List[str],
    default: float = 0.0,
) -> float:
    for name in names:
        if name in metrics:
            try:
                return float(metrics[name])
            except Exception:
                pass
    return float(default)


def compute_cost(metrics: Dict[str, Any]) -> float:
    crashed = int(metrics.get("crashed", 0)) == 1
    if crashed:
        crash_time = float(metrics.get("crash_time_s", -1.0))
        return CRASH_PENALTY + CRASH_TIME_PENALTY * max(
            0.0, DEFAULT_SIM_TIME_S - max(0.0, crash_time)
        )

    J = 0.0
    for k, w in COST_WEIGHTS.items():
        try:
            valf = float(metrics.get(k, 0.0))
        except Exception:
            valf = 0.0
        J += float(w) * valf

    soft_cnt = _metric_float_first(
        metrics,
        ["soft_track_violations_count", "soft_track_violation_count", "soft_track_violation_count_"],
        0.0,
    )
    med_cnt = _metric_float_first(
        metrics,
        ["medium_track_violations_count", "medium_track_violation_count", "medium_track_violation_count_"],
        0.0,
    )
    high_cnt = _metric_float_first(
        metrics,
        ["high_track_violations_count", "high_track_violation_count", "high_track_violation_count_"],
        0.0,
    )

    J += soft_cnt / SOFT_TRACK_DIV
    J += med_cnt / MEDIUM_TRACK_DIV
    J += high_cnt * HIGH_TRACK_MUL
    return float(J)


def _build_sim_launch_args(
    track_index: int,
    sim_time_s: int,
    critical_crash_multiplier: float,
) -> Dict[str, str]:
    return {
        "sim_time": str(sim_time_s),
        "track_id": str(track_index),
        "low_level_controlers": str(SIM_LOW_LEVEL_CONTROLERS),
        "ins_mode": str(SIM_INS_MODE),
        "critical_crash_multiplier": str(float(critical_crash_multiplier)),
    }


def _popen_roslaunch(
    launch_file: str,
    launch_args: Dict[str, str],
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    cmd = ["roslaunch", launch_file] + [f"{k}:={v}" for k, v in launch_args.items()]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=_ros_env(extra_env),
        preexec_fn=os.setsid,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    p._cmd = cmd
    return p


def _kill_process_group(
    p: Optional[subprocess.Popen],
    sig=signal.SIGINT,
    timeout_s: float = 5.0,
) -> None:
    if p is None:
        return

    try:
        pgid = os.getpgid(p.pid)
    except Exception:
        pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, sig)
        else:
            p.send_signal(sig)
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            break
        time.sleep(0.1)

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            p.kill()
    except Exception:
        pass


def _safe_cholesky(A: np.ndarray) -> np.ndarray:
    A = 0.5 * (A + A.T)
    eye = np.eye(A.shape[0])
    jitter = 1e-12
    for _ in range(10):
        try:
            return np.linalg.cholesky(A + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Could not compute Cholesky even after jitter.")


def _project_spd(A: np.ndarray, min_eig: float = 1e-12) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, min_eig)
    B = (V * w) @ V.T
    return 0.5 * (B + B.T)


def _z_bounds_for_key(k: str) -> Tuple[float, float]:
    lo, hi = ALL_BOUNDS[k]
    if k in LOG_PARAMS:
        return math.log(lo), math.log(hi)
    return float(lo), float(hi)


def _theta_to_search_coord(k: str, x: float) -> float:
    if k in LOG_PARAMS:
        return math.log(float(x))
    return float(x)


def _search_coord_to_theta(k: str, z: float) -> float:
    if k in LOG_PARAMS:
        return float(math.exp(z))
    return float(z)


def _merge_fixed_and_tuned(
    fixed_params: Dict[str, float],
    tuned: Dict[str, float],
) -> Dict[str, float]:
    out = dict(fixed_params)
    out.update(tuned)
    return out


def _extract_cost_theta(theta: Dict[str, float]) -> Dict[str, float]:
    return {k: float(theta[k]) for k in COST_KEYS}


def _extract_planner_theta(theta: Dict[str, float]) -> Dict[str, float]:
    return {k: float(theta[k]) for k in PLANNER_KEYS}


def _blend_planner_theta_with_safe(
    planner_theta: Dict[str, float],
    safe_planner: Dict[str, float],
    safe_blend: float,
) -> Dict[str, float]:
    safe_blend = float(_clamp(safe_blend, 0.0, 1.0))
    out: Dict[str, float] = {}
    for k in PLANNER_KEYS:
        lo, hi = PLANNER_BOUNDS[k]
        val = (1.0 - safe_blend) * float(planner_theta[k]) + safe_blend * float(safe_planner[k])
        out[k] = float(_clamp(val, lo, hi))
    return out


# ============================================================
# Episode / evaluation
# ============================================================
@dataclass
class EpisodeResult:
    cost: float
    metrics: Dict[str, Any]
    track_index: int
    crashed: bool
    crash_reason: str


def run_one_episode(
    theta_x: Dict[str, float],
    track_index: int,
    sim_time_s: int = DEFAULT_SIM_TIME_S,
    critical_crash_multiplier: float = 1.0,
) -> EpisodeResult:
    apply_params_to_control_json(theta_x)

    try:
        if os.path.exists(METRICS_CSV):
            os.remove(METRICS_CSV)
    except Exception:
        pass

    sim_args = _build_sim_launch_args(track_index, sim_time_s, critical_crash_multiplier)
    sim_p = _popen_roslaunch(SIM_LAUNCH, sim_args)

    ctrl_env = {
        "LD_LIBRARY_PATH": f"{ACADOS_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    }
    ctrl_p = _popen_roslaunch(CTRL_LAUNCH, {}, extra_env=ctrl_env)

    time.sleep(0.6)

    if sim_p.poll() is not None:
        rc = int(sim_p.returncode)
        metrics = {"crashed": 1, "crash_reason": f"sim_launch_failed_rc={rc}", "crash_time_s": -1}
        cost = compute_cost(metrics)
        _kill_process_group(sim_p)
        _kill_process_group(ctrl_p)
        return EpisodeResult(cost=cost, metrics=metrics, track_index=track_index, crashed=True, crash_reason=str(metrics["crash_reason"]))

    if ctrl_p.poll() is not None:
        rc = int(ctrl_p.returncode)
        metrics = {"crashed": 1, "crash_reason": f"ctrl_launch_failed_rc={rc}", "crash_time_s": -1}
        cost = compute_cost(metrics)
        _kill_process_group(sim_p)
        _kill_process_group(ctrl_p)
        return EpisodeResult(cost=cost, metrics=metrics, track_index=track_index, crashed=True, crash_reason=str(metrics["crash_reason"]))

    t0 = time.time()
    while True:
        if sim_p.poll() is not None:
            break
        if time.time() - t0 > sim_time_s + EPISODE_HARD_TIMEOUT_MARGIN_S:
            break
        time.sleep(0.2)

    _kill_process_group(sim_p)
    _kill_process_group(ctrl_p)

    ok = _wait_for_metrics_stable(METRICS_CSV, METRICS_WAIT_TIMEOUT_S)
    if not ok:
        metrics = {"crashed": 1, "crash_reason": "no_metrics_file", "crash_time_s": -1}
        cost = compute_cost(metrics)
        return EpisodeResult(cost=cost, metrics=metrics, track_index=track_index, crashed=True, crash_reason=str(metrics["crash_reason"]))

    metrics = _parse_metrics_csv(METRICS_CSV)
    cost = compute_cost(metrics)
    crashed = int(metrics.get("crashed", 0)) == 1
    reason = str(metrics.get("crash_reason", "")) if crashed else ""
    return EpisodeResult(cost=cost, metrics=metrics, track_index=track_index, crashed=crashed, crash_reason=reason)


def evaluate_theta(
    theta_x: Dict[str, float],
    tracks: List[int],
    sim_time_s: int = DEFAULT_SIM_TIME_S,
    critical_crash_multiplier: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    global EPISODES_DONE

    costs: List[float] = []
    per_track: List[Dict[str, Any]] = []
    crashes = 0

    for tid in tracks:
        res = run_one_episode(
            theta_x=theta_x,
            track_index=tid,
            sim_time_s=sim_time_s,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        EPISODES_DONE += 1

        costs.append(float(res.cost))
        if res.crashed:
            crashes += 1

        per_track.append(
            {
                "track_id": int(tid),
                "cost": float(res.cost),
                "crashed": bool(res.crashed),
                "crash_reason": str(res.crash_reason),
            }
        )

    mean_cost = sum(costs) / max(1, len(costs))
    var = sum((c - mean_cost) * (c - mean_cost) for c in costs) / max(1, len(costs))
    std_cost = math.sqrt(max(0.0, var))
    robust_cost = mean_cost + UCB_WEIGHT * std_cost

    info = {
        "tracks": list(tracks),
        "episodes": len(tracks),
        "crashes": int(crashes),
        "mean_cost": float(mean_cost),
        "std_cost": float(std_cost),
        "robust_cost": float(robust_cost),
        "objective_cost": float(robust_cost),
        "ucb_weight": float(UCB_WEIGHT),
        "per_track": per_track,
        "critical_crash_multiplier": float(critical_crash_multiplier),
    }
    return float(robust_cost), info


# ============================================================
# Search-space transform
# ============================================================
@dataclass
class SearchTransform:
    keys: List[str]
    mean_z: np.ndarray
    chol_z: np.ndarray

    def y_to_theta(self, y: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
        z = self.mean_z + self.chol_z @ np.asarray(y, dtype=float)

        theta: Dict[str, float] = {}
        clipped_dims = 0
        z_raw: Dict[str, float] = {}
        z_clipped: Dict[str, float] = {}

        for i, k in enumerate(self.keys):
            z_raw[k] = float(z[i])

            lo_z, hi_z = _z_bounds_for_key(k)
            z_i = _clamp(float(z[i]), lo_z, hi_z)
            if abs(z_i - float(z[i])) > 1e-12:
                clipped_dims += 1
            z_clipped[k] = float(z_i)

            x = _search_coord_to_theta(k, z_i)
            lo_x, hi_x = ALL_BOUNDS[k]
            x = _clamp(x, lo_x, hi_x)
            theta[k] = float(x)

        return theta, {
            "clipped_dims": int(clipped_dims),
            "z_raw": z_raw,
            "z_clipped": z_clipped,
        }


def _relax_block_cov(
    C_in: np.ndarray,
    keys: List[str],
    alpha: float,
    beta: float,
    std_min_frac: float,
    std_max_frac: float,
) -> np.ndarray:
    C_in = 0.5 * (C_in + C_in.T)
    diagC = np.diag(np.diag(C_in))
    C = beta * (alpha * C_in + (1.0 - alpha) * diagC)

    std_new = []
    for i, k in enumerate(keys):
        lo_z, hi_z = _z_bounds_for_key(k)
        full = max(1e-12, hi_z - lo_z)

        std_min = std_min_frac * full
        std_max = std_max_frac * full

        var_i = max(0.0, float(C[i, i]))
        std_i = math.sqrt(var_i) if var_i > 0.0 else std_min
        std_i = max(std_min, min(std_max, std_i))
        std_new.append(std_i)

    std_new = np.asarray(std_new, dtype=float)
    std_old = np.sqrt(np.maximum(np.diag(C), 1e-12))

    D_old_inv = np.diag(1.0 / std_old)
    Corr = D_old_inv @ C @ D_old_inv
    Corr = 0.5 * (Corr + Corr.T)
    Corr = np.clip(Corr, -0.95, 0.95)
    for i in range(Corr.shape[0]):
        Corr[i, i] = 1.0

    D_new = np.diag(std_new)
    C_new = D_new @ Corr @ D_new
    return _project_spd(C_new, min_eig=1e-10)


def _build_stage1_transform() -> Tuple[SearchTransform, np.ndarray]:
    mean_z = []
    stds = []

    for k in COST_KEYS:
        lo, hi = COST_BOUNDS[k]
        full = math.log(hi / lo)
        mean_z.append(math.log(math.sqrt(lo * hi)))
        stds.append(STAGE1_COST_STD_FRAC * full)

    mean_z_arr = np.asarray(mean_z, dtype=float)
    C0 = np.diag(np.asarray(stds, dtype=float) ** 2)
    L0 = _safe_cholesky(C0)
    return SearchTransform(keys=COST_KEYS, mean_z=mean_z_arr, chol_z=L0), C0


def _build_stage2_transform() -> Tuple[SearchTransform, np.ndarray]:
    mean_z = []
    stds = []

    for k in PLANNER_KEYS:
        mean_z.append(float(SAFE_PLANNER[k]))

        lo, hi = PLANNER_BOUNDS[k]
        frac = STAGE2_PLANNER_STD_FRAC_VMAX if k == "velocity_planner.v_max" else STAGE2_PLANNER_STD_FRAC_MU
        stds.append(frac * (hi - lo))

    mean_z_arr = np.asarray(mean_z, dtype=float)
    C0 = np.diag(np.asarray(stds, dtype=float) ** 2)
    L0 = _safe_cholesky(C0)
    return SearchTransform(keys=PLANNER_KEYS, mean_z=mean_z_arr, chol_z=L0), C0


def _build_stage3_transform(
    phase1_theta: Dict[str, float],
    phase2_theta: Dict[str, float],
    phase1_dist: Dict[str, Any],
    phase2_dist: Dict[str, Any],
) -> Tuple[SearchTransform, np.ndarray]:
    mean_cost_z = np.asarray(
        [_theta_to_search_coord(k, float(phase1_theta[k])) for k in COST_KEYS],
        dtype=float,
    )

    blended_planner_theta = _blend_planner_theta_with_safe(
        planner_theta=_extract_planner_theta(phase2_theta),
        safe_planner=SAFE_PLANNER,
        safe_blend=STAGE3_PLANNER_SAFE_BLEND,
    )
    mean_planner_z = np.asarray(
        [_theta_to_search_coord(k, float(blended_planner_theta[k])) for k in PLANNER_KEYS],
        dtype=float,
    )

    C1_cost = np.asarray(phase1_dist["distribution_cov_search_matrix"], dtype=float)
    C2_planner = np.asarray(phase2_dist["distribution_cov_search_matrix"], dtype=float)

    C3_cost = _relax_block_cov(
        C_in=C1_cost,
        keys=COST_KEYS,
        alpha=STAGE3_COST_COV_ALPHA,
        beta=STAGE3_COST_COV_BETA,
        std_min_frac=STAGE3_COST_STD_MIN_FRAC,
        std_max_frac=STAGE3_COST_STD_MAX_FRAC,
    )

    C3_planner = _relax_block_cov(
        C_in=C2_planner,
        keys=PLANNER_KEYS,
        alpha=STAGE3_PLANNER_COV_ALPHA,
        beta=STAGE3_PLANNER_COV_BETA,
        std_min_frac=STAGE3_PLANNER_STD_MIN_FRAC,
        std_max_frac=STAGE3_PLANNER_STD_MAX_FRAC,
    )

    mean_z = np.concatenate([mean_cost_z, mean_planner_z])

    C3 = np.zeros((len(ALL_KEYS), len(ALL_KEYS)), dtype=float)
    C3[:len(COST_KEYS), :len(COST_KEYS)] = C3_cost
    C3[len(COST_KEYS):, len(COST_KEYS):] = C3_planner
    C3 = _project_spd(C3, min_eig=1e-10)

    L3 = _safe_cholesky(C3)
    return SearchTransform(keys=ALL_KEYS, mean_z=mean_z, chol_z=L3), C3


# ============================================================
# CMA distribution logging
# ============================================================
def _extract_distribution_state(
    es: Any,
    transform: SearchTransform,
    fixed_params: Dict[str, float],
    phase_name: str,
    generation: int,
    lambda_: int,
) -> Dict[str, Any]:
    mean_y = np.asarray(es.mean, dtype=float)
    C_y = np.asarray(es.C, dtype=float)
    sigma = float(es.sigma)

    mean_z = transform.mean_z + transform.chol_z @ mean_y
    cov_z = transform.chol_z @ ((sigma ** 2) * C_y) @ transform.chol_z.T
    cov_z = 0.5 * (cov_z + cov_z.T)

    mean_search = {k: float(mean_z[i]) for i, k in enumerate(transform.keys)}
    center_theta = {k: float(_search_coord_to_theta(k, mean_z[i])) for i, k in enumerate(transform.keys)}
    full_center_theta = _merge_fixed_and_tuned(fixed_params, center_theta)
    cov_diag_search = {k: float(max(0.0, cov_z[i, i])) for i, k in enumerate(transform.keys)}

    std_theta_approx: Dict[str, float] = {}
    for i, k in enumerate(transform.keys):
        var_z = max(0.0, float(cov_z[i, i]))
        if k in LOG_PARAMS:
            mu = float(mean_z[i])
            var_x = (math.exp(var_z) - 1.0) * math.exp(2.0 * mu + var_z)
            std_theta_approx[k] = float(math.sqrt(max(0.0, var_x)))
        else:
            std_theta_approx[k] = float(math.sqrt(var_z))

    return {
        "phase": str(phase_name),
        "generation": int(generation),
        "lambda": int(lambda_),
        "sigma_internal": float(sigma),
        "distribution_mean_search": mean_search,
        "distribution_center_theta": center_theta,
        "distribution_center_theta_full": full_center_theta,
        "distribution_cov_diag_search": cov_diag_search,
        "distribution_std_theta_approx": std_theta_approx,
        "distribution_cov_search_matrix": cov_z.tolist(),
        "timestamp_unix": float(time.time()),
    }


def _make_initial_distribution_record(
    phase_name: str,
    transform: SearchTransform,
    fixed_params: Dict[str, float],
    C_init: np.ndarray,
) -> Dict[str, Any]:
    center_theta = {
        k: float(_search_coord_to_theta(k, transform.mean_z[i]))
        for i, k in enumerate(transform.keys)
    }
    full_center_theta = _merge_fixed_and_tuned(fixed_params, center_theta)

    mean_search = {
        k: float(transform.mean_z[i]) for i, k in enumerate(transform.keys)
    }
    cov_diag_search = {
        k: float(C_init[i, i]) for i, k in enumerate(transform.keys)
    }

    return {
        "phase": str(phase_name),
        "generation": 0,
        "lambda": 0,
        "sigma_internal": 1.0,
        "distribution_mean_search": mean_search,
        "distribution_center_theta": center_theta,
        "distribution_center_theta_full": full_center_theta,
        "distribution_cov_diag_search": cov_diag_search,
        "distribution_cov_search_matrix": C_init.tolist(),
        "timestamp_unix": float(time.time()),
    }


# ============================================================
# Logs / ranking
# ============================================================
def _result_objective(rec: Dict[str, Any]) -> float:
    return float(rec.get("objective_cost", rec.get("robust_cost", rec.get("mean_cost", float("inf")))))


def _update_top10_file() -> None:
    ranked = sorted(ALL_RESULTS, key=_result_objective)[:10]
    payload = {
        "profile_name": PROFILE_NAME,
        "episodes_done": int(EPISODES_DONE),
        "trials_done": int(TRIALS_DONE),
        "top10": [
            {
                "rank": i + 1,
                "candidate_id": int(r["candidate_id"]),
                "phase": str(r["phase"]),
                "phase_trial": int(r["phase_trial"]),
                "objective_cost": float(r["objective_cost"]),
                "mean_cost": float(r["mean_cost"]),
                "std_cost": float(r["std_cost"]),
                "robust_cost": float(r["robust_cost"]),
                "crashes": int(r["crashes"]),
                "critical_crash_multiplier": float(r["critical_crash_multiplier"]),
                "theta": {k: float(v) for k, v in r["theta"].items()},
            }
            for i, r in enumerate(ranked)
        ],
        "timestamp_unix": float(time.time()),
    }
    _save_json_atomic(TOP10_PATH, payload)


def _append_run_log(rec: Dict[str, Any]) -> None:
    r = dict(rec)
    r["timestamp_unix"] = float(time.time())
    _append_jsonl(RUN_LOG_PATH, r)


def _register_result(
    phase: str,
    phase_trial: int,
    theta_x: Dict[str, float],
    objective_cost: float,
    info: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    global GLOBAL_CANDIDATE_ID, TRIALS_DONE, BEST_OBJECTIVE_SO_FAR, BEST_THETA_SO_FAR

    GLOBAL_CANDIDATE_ID += 1
    TRIALS_DONE += 1

    rec = {
        "candidate_id": int(GLOBAL_CANDIDATE_ID),
        "phase": str(phase),
        "phase_trial": int(phase_trial),
        "objective_cost": float(objective_cost),
        "mean_cost": float(info.get("mean_cost", float("nan"))),
        "std_cost": float(info.get("std_cost", float("nan"))),
        "robust_cost": float(info.get("robust_cost", objective_cost)),
        "ucb_weight": float(info.get("ucb_weight", UCB_WEIGHT)),
        "crashes": int(info.get("crashes", 0)),
        "critical_crash_multiplier": float(info.get("critical_crash_multiplier", 1.0)),
        "theta": {k: float(v) for k, v in theta_x.items()},
        "tracks": list(info.get("tracks", [])),
    }
    if extra:
        rec["extra"] = dict(extra)

    ALL_RESULTS.append(rec)

    if phase == "phase1_cost_only_cma":
        PHASE1_RESULTS.append(rec)
    elif phase == "phase2_planner_only_cma":
        PHASE2_RESULTS.append(rec)
    elif phase == "phase3_joint_cma":
        PHASE3_RESULTS.append(rec)

    if objective_cost < BEST_OBJECTIVE_SO_FAR:
        BEST_OBJECTIVE_SO_FAR = float(objective_cost)
        BEST_THETA_SO_FAR = {k: float(v) for k, v in theta_x.items()}

    log_rec = {
        "candidate_id": int(rec["candidate_id"]),
        "phase": str(phase),
        "phase_trial": int(phase_trial),
        "objective_cost": float(objective_cost),
        "mean_cost": float(info.get("mean_cost", float("nan"))),
        "std_cost": float(info.get("std_cost", float("nan"))),
        "robust_cost": float(info.get("robust_cost", objective_cost)),
        "ucb_weight": float(info.get("ucb_weight", UCB_WEIGHT)),
        "crashes": int(info.get("crashes", 0)),
        "critical_crash_multiplier": float(info.get("critical_crash_multiplier", 1.0)),
        "theta": {k: float(v) for k, v in theta_x.items()},
        "best_objective_so_far": float(BEST_OBJECTIVE_SO_FAR),
        "episodes_done": int(EPISODES_DONE),
    }
    if extra:
        log_rec["extra"] = dict(extra)

    _append_run_log(log_rec)
    _update_top10_file()


# ============================================================
# Generic staged CMA phase
# ============================================================
def run_cma_phase(
    phase_name: str,
    transform: SearchTransform,
    fixed_params: Dict[str, float],
    lambda_: int,
    maxiter: int,
    critical_crash_multiplier: float,
    seed_offset: int,
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    try:
        import cma
    except ImportError:
        print("Brak biblioteki 'cma'. Zainstaluj: pip install cma numpy", file=sys.stderr)
        sys.exit(1)

    dim = len(transform.keys)

    es = cma.CMAEvolutionStrategy(
        [0.0] * dim,
        1.0,
        {
            "seed": GLOBAL_SEED + seed_offset,
            "popsize": lambda_,
            "maxiter": maxiter,
            "verbose": -9,
        },
    )

    best_theta = _merge_fixed_and_tuned(
        fixed_params,
        {k: _search_coord_to_theta(k, transform.mean_z[i]) for i, k in enumerate(transform.keys)},
    )
    best_obj = float("inf")
    generation = 0
    final_dist: Dict[str, Any] = {}

    while (not es.stop()) and generation < maxiter:
        generation += 1

        Y = es.ask()
        F: List[float] = []

        gen_best_theta: Dict[str, float] = {}
        gen_best_obj = float("inf")

        for i, yi in enumerate(Y):
            tuned_theta, dbg = transform.y_to_theta(np.asarray(yi, dtype=float))
            full_theta = _merge_fixed_and_tuned(fixed_params, tuned_theta)

            objective_cost, info = evaluate_theta(
                theta_x=full_theta,
                tracks=TRACK_SET,
                sim_time_s=DEFAULT_SIM_TIME_S,
                critical_crash_multiplier=critical_crash_multiplier,
            )

            _register_result(
                phase=phase_name,
                phase_trial=(generation - 1) * lambda_ + i,
                theta_x=full_theta,
                objective_cost=objective_cost,
                info=info,
                extra={
                    "generation": int(generation),
                    "candidate_in_generation": int(i),
                    "clipped_dims": int(dbg["clipped_dims"]),
                },
            )

            F.append(float(objective_cost))

            if objective_cost < gen_best_obj:
                gen_best_obj = float(objective_cost)
                gen_best_theta = dict(full_theta)

            if objective_cost < best_obj:
                best_obj = float(objective_cost)
                best_theta = dict(full_theta)

            print(
                f"[{phase_name}][gen {generation:02d} cand {i:02d}] "
                f"obj={objective_cost:.6g} mean={info.get('mean_cost', float('nan')):.6g} "
                f"std={info.get('std_cost', float('nan')):.6g} "
                f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)} "
                f"crit_mult={critical_crash_multiplier:.3f} "
                f"clip={dbg['clipped_dims']}"
            )

        es.tell(Y, F)

        final_dist = _extract_distribution_state(
            es=es,
            transform=transform,
            fixed_params=fixed_params,
            phase_name=phase_name,
            generation=generation,
            lambda_=lambda_,
        )
        final_dist["generation_best_theta"] = {k: float(v) for k, v in gen_best_theta.items()}
        final_dist["generation_best_objective"] = float(gen_best_obj)
        final_dist["global_best_objective_inside_phase"] = float(best_obj)

        _append_jsonl(DIST_LOG_PATH, final_dist)

    return best_theta, best_obj, final_dist


# ============================================================
# Save outputs
# ============================================================
def save_best_json(theta_x: Dict[str, float]) -> None:
    data = _load_json(CONTROL_PARAM_JSON)
    for path, val in theta_x.items():
        if path in ALL_BOUNDS:
            _set_json_path(data, path, float(val))
    _save_json_atomic(BEST_JSON_PATH, data)


def save_summary(
    phase1_theta: Dict[str, float],
    phase1_obj: float,
    phase2_theta: Dict[str, float],
    phase2_obj: float,
    phase3_theta: Dict[str, float],
    phase3_obj: float,
    phase1_dist: Dict[str, Any],
    stage2_init_dist: Dict[str, Any],
    phase2_dist: Dict[str, Any],
    stage3_init_dist: Dict[str, Any],
    phase3_dist: Dict[str, Any],
) -> None:
    payload = {
        "config": {
            "profile_name": PROFILE_NAME,
            "catkin_ws": CATKIN_WS,
            "ws_src": WS_SRC,
            "acados_lib": ACADOS_LIB,
            "sim_time_s": DEFAULT_SIM_TIME_S,
            "tracks": TRACK_SET,
            "episodes_per_trial": EPISODES_PER_TRIAL,
            "phase1_trials": PHASE1_N_TRIALS,
            "phase1_lambda": PHASE1_LAMBDA,
            "phase1_maxiter": PHASE1_MAXITER,
            "phase2_trials": PHASE2_N_TRIALS,
            "phase2_lambda": PHASE2_LAMBDA,
            "phase2_maxiter": PHASE2_MAXITER,
            "phase3_trials": PHASE3_N_TRIALS,
            "phase3_lambda": PHASE3_LAMBDA,
            "phase3_maxiter": PHASE3_MAXITER,
            "total_trials": TOTAL_TRIALS,
            "total_episodes": TOTAL_EPISODES,
            "pure_sim_hours": PURE_SIM_HOURS,
            "ucb_weight": UCB_WEIGHT,
            "safe_planner": SAFE_PLANNER,
            "planner_bounds": PLANNER_BOUNDS,
            "cost_bounds": COST_BOUNDS,
            "stage1_cost_std_frac": STAGE1_COST_STD_FRAC,
            "stage2_planner_std_frac_vmax": STAGE2_PLANNER_STD_FRAC_VMAX,
            "stage2_planner_std_frac_mu": STAGE2_PLANNER_STD_FRAC_MU,
            "stage3_planner_safe_blend": STAGE3_PLANNER_SAFE_BLEND,
            "stage3_cost_cov_alpha": STAGE3_COST_COV_ALPHA,
            "stage3_cost_cov_beta": STAGE3_COST_COV_BETA,
            "stage3_cost_std_min_frac": STAGE3_COST_STD_MIN_FRAC,
            "stage3_cost_std_max_frac": STAGE3_COST_STD_MAX_FRAC,
            "stage3_planner_cov_alpha": STAGE3_PLANNER_COV_ALPHA,
            "stage3_planner_cov_beta": STAGE3_PLANNER_COV_BETA,
            "stage3_planner_std_min_frac": STAGE3_PLANNER_STD_MIN_FRAC,
            "stage3_planner_std_max_frac": STAGE3_PLANNER_STD_MAX_FRAC,
            "phase1_critical_crash_multiplier": PHASE1_CRITICAL_CRASH_MULTIPLIER,
            "phase2_critical_crash_multiplier": PHASE2_CRITICAL_CRASH_MULTIPLIER,
            "phase3_critical_crash_multiplier": PHASE3_CRITICAL_CRASH_MULTIPLIER,
            "dist_log_path": DIST_LOG_PATH,
            "run_log_path": RUN_LOG_PATH,
        },
        "stage_distributions": {
            "phase1_final_distribution": phase1_dist,
            "stage2_initial_distribution": stage2_init_dist,
            "phase2_final_distribution": phase2_dist,
            "stage3_initial_distribution": stage3_init_dist,
            "phase3_final_distribution": phase3_dist,
        },
        "results": {
            "phase1_cost_only_cma": {
                "best_objective": float(phase1_obj),
                "best_theta": {k: float(v) for k, v in phase1_theta.items()},
            },
            "phase2_planner_only_cma": {
                "best_objective": float(phase2_obj),
                "best_theta": {k: float(v) for k, v in phase2_theta.items()},
            },
            "phase3_joint_cma": {
                "best_objective": float(phase3_obj),
                "best_theta": {k: float(v) for k, v in phase3_theta.items()},
            },
            "global_best": {
                "best_objective": float(BEST_OBJECTIVE_SO_FAR),
                "best_theta": {k: float(v) for k, v in BEST_THETA_SO_FAR.items()},
            },
        },
    }
    _save_json_atomic(SUMMARY_PATH, payload)


# ============================================================
# Main
# ============================================================
def main() -> None:
    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(ACADOS_LIB, "ACADOS_LIB")

    global BEST_OBJECTIVE_SO_FAR, BEST_THETA_SO_FAR, EPISODES_DONE, TRIALS_DONE

    BEST_OBJECTIVE_SO_FAR = float("inf")
    BEST_THETA_SO_FAR = {}
    EPISODES_DONE = 0
    TRIALS_DONE = 0
    ALL_RESULTS.clear()
    PHASE1_RESULTS.clear()
    PHASE2_RESULTS.clear()
    PHASE3_RESULTS.clear()

    for p in [RUN_LOG_PATH, SUMMARY_PATH, DIST_LOG_PATH]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    print("[CFG] PROFILE_NAME =", PROFILE_NAME)
    print("[CFG] CATKIN_WS    =", CATKIN_WS)
    print("[CFG] WS_SRC       =", WS_SRC)
    print("[CFG] ACADOS_LIB   =", ACADOS_LIB)
    print("[CFG] TRACK_SET    =", TRACK_SET)
    print("[CFG] SIM_TIME     =", DEFAULT_SIM_TIME_S, "s")
    print("[CFG] UCB_WEIGHT   =", UCB_WEIGHT)
    print("[CFG] TOTAL_TRIALS =", TOTAL_TRIALS)
    print("[CFG] TOTAL_EPISODES =", TOTAL_EPISODES)
    print(f"[CFG] PURE_SIM_HOURS ~= {PURE_SIM_HOURS:.3f}")
    print("[CFG] Phase1 = frozen planner / cost-only")
    print("[CFG] Phase2 = frozen best costs / planner-only")
    print("[CFG] Phase3 = joint from phase1+phase2 anchors")
    print("")

    _update_top10_file()

    roscore_p = None
    if START_ROSCORE_IF_NEEDED:
        if not _rosmaster_is_up():
            roscore_p = _popen_roscore()
            time.sleep(1.0)
            if not _rosmaster_is_up():
                print("[FATAL] roscore did not come up.", file=sys.stderr)
                _kill_process_group(roscore_p)
                sys.exit(3)
            print(f"[INFO] Started roscore on {ROS_MASTER_HOST}:{ROS_MASTER_PORT}.")
        else:
            print(f"[INFO] Reusing rosmaster on {ROS_MASTER_HOST}:{ROS_MASTER_PORT}.")

    try:
        # ====================================================
        # Phase 1
        # ====================================================
        print("\n================ PHASE 1: COST ONLY @ SAFE PLANNER (CMA-ES) ================\n")
        stage1_transform, _stage1_init_cov = _build_stage1_transform()

        phase1_theta, phase1_obj, phase1_dist = run_cma_phase(
            phase_name="phase1_cost_only_cma",
            transform=stage1_transform,
            fixed_params=SAFE_PLANNER,
            lambda_=PHASE1_LAMBDA,
            maxiter=PHASE1_MAXITER,
            critical_crash_multiplier=PHASE1_CRITICAL_CRASH_MULTIPLIER,
            seed_offset=11,
        )

        # ====================================================
        # Stage 2 init
        # ====================================================
        print("\n================ TRANSITION 1 -> 2: BUILD STAGE-2 DISTRIBUTION ================\n")
        fixed_costs = _extract_cost_theta(phase1_theta)
        stage2_transform, stage2_init_cov = _build_stage2_transform()
        stage2_init_dist = _make_initial_distribution_record(
            phase_name="phase2_planner_only_cma",
            transform=stage2_transform,
            fixed_params=fixed_costs,
            C_init=stage2_init_cov,
        )
        _append_jsonl(DIST_LOG_PATH, stage2_init_dist)

        print("[STAGE2 INIT] fixed costs from phase1 best:")
        for k in COST_KEYS:
            print(f"  {k}: {fixed_costs[k]:.6g}")
        print("[STAGE2 INIT] planner center:")
        for k in PLANNER_KEYS:
            print(f"  {k}: {stage2_init_dist['distribution_center_theta'][k]:.6g}")

        # ====================================================
        # Phase 2
        # ====================================================
        print("\n================ PHASE 2: PLANNER ONLY @ FIXED PHASE1 COSTS (CMA-ES) ================\n")
        phase2_theta, phase2_obj, phase2_dist = run_cma_phase(
            phase_name="phase2_planner_only_cma",
            transform=stage2_transform,
            fixed_params=fixed_costs,
            lambda_=PHASE2_LAMBDA,
            maxiter=PHASE2_MAXITER,
            critical_crash_multiplier=PHASE2_CRITICAL_CRASH_MULTIPLIER,
            seed_offset=22,
        )

        # ====================================================
        # Stage 3 init
        # ====================================================
        print("\n================ TRANSITION 2 -> 3: BUILD STAGE-3 JOINT DISTRIBUTION ================\n")
        stage3_transform, stage3_init_cov = _build_stage3_transform(
            phase1_theta=phase1_theta,
            phase2_theta=phase2_theta,
            phase1_dist=phase1_dist,
            phase2_dist=phase2_dist,
        )
        stage3_init_dist = _make_initial_distribution_record(
            phase_name="phase3_joint_cma",
            transform=stage3_transform,
            fixed_params={},
            C_init=stage3_init_cov,
        )
        _append_jsonl(DIST_LOG_PATH, stage3_init_dist)

        print("[STAGE3 INIT] joint center:")
        for k in ALL_KEYS:
            print(f"  {k}: {stage3_init_dist['distribution_center_theta'][k]:.6g}")

        # ====================================================
        # Phase 3
        # ====================================================
        print("\n================ PHASE 3: JOINT FROM STAGE1+STAGE2 ANCHORS (CMA-ES) ================\n")
        phase3_theta, phase3_obj, phase3_dist = run_cma_phase(
            phase_name="phase3_joint_cma",
            transform=stage3_transform,
            fixed_params={},
            lambda_=PHASE3_LAMBDA,
            maxiter=PHASE3_MAXITER,
            critical_crash_multiplier=PHASE3_CRITICAL_CRASH_MULTIPLIER,
            seed_offset=33,
        )

        save_best_json(BEST_THETA_SO_FAR)
        save_summary(
            phase1_theta=phase1_theta,
            phase1_obj=phase1_obj,
            phase2_theta=phase2_theta,
            phase2_obj=phase2_obj,
            phase3_theta=phase3_theta,
            phase3_obj=phase3_obj,
            phase1_dist=phase1_dist,
            stage2_init_dist=stage2_init_dist,
            phase2_dist=phase2_dist,
            stage3_init_dist=stage3_init_dist,
            phase3_dist=phase3_dist,
        )

    finally:
        if roscore_p is not None:
            _kill_process_group(roscore_p)

    print("\n================ DONE ================\n")
    print(f"Best global objective (robust cost): {BEST_OBJECTIVE_SO_FAR:.6g}")
    print("Best global params:")
    for k in sorted(BEST_THETA_SO_FAR.keys()):
        lo, hi = ALL_BOUNDS[k]
        print(f"  {k}: {BEST_THETA_SO_FAR[k]:.6g}   (bounds [{lo}, {hi}])")

    print(f"\nSaved best json: {BEST_JSON_PATH}")
    print(f"Saved top10 file: {TOP10_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved run log: {RUN_LOG_PATH}")
    print(f"Saved distribution log: {DIST_LOG_PATH}")
    print(f"\nTotal trials done: {TRIALS_DONE}")
    print(f"Total episodes done: {EPISODES_DONE}")
    print(f"Pure simulation hours target ~= {PURE_SIM_HOURS:.3f}")


if __name__ == "__main__":
    main()
