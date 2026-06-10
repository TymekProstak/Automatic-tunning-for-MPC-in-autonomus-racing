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
import copy
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np


# ============================================================
# Workspace
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_amz/TUNE_CMA_ES")
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
# Evaluation setup / profile presets
# ============================================================
GLOBAL_SEED = 123
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

DEFAULT_SIM_TIME_S = int(os.environ.get("TUNE_SIM_TIME_S", "60"))
UCB_WEIGHT = float(os.environ.get("TUNE_UCB_WEIGHT", "1.0"))

PROFILE_NAME = os.environ.get("TUNE_PROFILE", "night_4track").strip()

if PROFILE_NAME == "smoke_1track":
    TRACK_SET = [1]

    PHASE1_LAMBDA = 4
    PHASE1_MAXITER = 3   # 12 triali

    PHASE2_LAMBDA = 6
    PHASE2_MAXITER = 4   # 24 triale

    PHASE3_LAMBDA = 6
    PHASE3_MAXITER = 4   # 24 triale

elif PROFILE_NAME == "night_4track":
    TRACK_SET = [1, 2, 3, 4]

    PHASE1_LAMBDA = 8
    PHASE1_MAXITER = 8   # 64 triali

    PHASE2_LAMBDA = 8
    PHASE2_MAXITER = 8   # 64 triale

    PHASE3_LAMBDA = 8
    PHASE3_MAXITER = 12  # 96 triali

else:
    raise ValueError(f"Unknown PROFILE_NAME: {PROFILE_NAME}")

EPISODES_PER_TRIAL = len(TRACK_SET)

PHASE1_N_TRIALS = PHASE1_LAMBDA * PHASE1_MAXITER
PHASE2_N_TRIALS = PHASE2_LAMBDA * PHASE2_MAXITER
PHASE3_N_TRIALS = PHASE3_LAMBDA * PHASE3_MAXITER

if PROFILE_NAME == "smoke_1track":
    assert PHASE1_N_TRIALS == 12
    assert PHASE2_N_TRIALS == 24
    assert PHASE3_N_TRIALS == 24
elif PROFILE_NAME == "night_4track":
    assert PHASE1_N_TRIALS == 64
    assert PHASE2_N_TRIALS == 64
    assert PHASE3_N_TRIALS == 96

TOTAL_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS
TOTAL_EPISODES = TOTAL_TRIALS * EPISODES_PER_TRIAL
PURE_SIM_HOURS = TOTAL_EPISODES * DEFAULT_SIM_TIME_S / 3600.0

# ============================================================
# Crash multipliers
# ============================================================
PHASE1_CRITICAL_CRASH_MULTIPLIER = 1.00
PHASE2_CRITICAL_CRASH_MULTIPLIER = 1.00
PHASE3_CRITICAL_CRASH_MULTIPLIER = 1.00

# ============================================================
# Base control template
# ============================================================
BASE_CONTROL_TEMPLATE: Optional[Dict[str, Any]] = None

# ============================================================
# Safe planner / aggressiveness
# kept aligned with staged coupled TPE baseline
# ============================================================
SAFE_PLANNER = {
    "mpc.bounds.max_vx": 13.5,
    "mpc.model.mux": 0.6,
    "mpc.model.muy": 0.7,
    "mpc.cost.q_sdot": 0.1,
}

# ============================================================
# Tuned params / bounds
# kept aligned with staged coupled TPE baseline
#
# Stage 1:
#   true cost weights only
#   fixed: max_vx, mux, muy, q_sdot
#
# Stage 2:
#   planner/aggression variables:
#   max_vx, mux, muy, q_sdot
#   fixed: best Stage-1 true costs
#
# Stage 3:
#   all tuned params jointly
# ============================================================
COST_BOUNDS = {
    "mpc.cost.q_ey": (1.0, 100.0),
    "mpc.cost.Q_epsi": (1.0, 100.0),
    "mpc.cost.R_dT": (1.0, 100.0),
    "mpc.cost.R_u_ddelta_cmd": (1.0, 100.0),
    "mpc.cost.R_Mtv": (1e-8, 1e-5),
    "mpc.cost.Q_beta": (0.1, 10.0),
}

PLANNER_BOUNDS = {
    "mpc.bounds.max_vx": (8.0, 18.0),
    "mpc.model.mux": (0.4, 1.7),
    "mpc.model.muy": (0.4, 1.7),
    "mpc.cost.q_sdot": (0.1, 1.0),
}

COST_KEYS = list(COST_BOUNDS.keys())
PLANNER_KEYS = list(PLANNER_BOUNDS.keys())
ALL_KEYS = COST_KEYS + PLANNER_KEYS

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(COST_BOUNDS)
ALL_BOUNDS.update(PLANNER_BOUNDS)

# costs are log-space; q_sdot is explicitly linear
LOG_PARAMS = set(COST_BOUNDS.keys())

# ============================================================
# CMA initialization / staged carry-over
# architecture aligned with staged CMA baseline,
# parameterization aligned with staged coupled TPE baseline
# ============================================================
STAGE1_COST_STD_FRAC = 0.30

if PROFILE_NAME == "smoke_1track":
    STAGE2_PLANNER_STD_FRAC_VMAX = 0.30
    STAGE2_PLANNER_STD_FRAC_MU = 0.28
    STAGE2_PLANNER_STD_FRAC_QSDOT = 0.30
    STAGE3_PLANNER_SAFE_BLEND = 0.25
elif PROFILE_NAME == "night_4track":
    STAGE2_PLANNER_STD_FRAC_VMAX = 0.35
    STAGE2_PLANNER_STD_FRAC_MU = 0.32
    STAGE2_PLANNER_STD_FRAC_QSDOT = 0.30
    STAGE3_PLANNER_SAFE_BLEND = 0.20
else:
    STAGE2_PLANNER_STD_FRAC_VMAX = 0.30
    STAGE2_PLANNER_STD_FRAC_MU = 0.28
    STAGE2_PLANNER_STD_FRAC_QSDOT = 0.30
    STAGE3_PLANNER_SAFE_BLEND = 0.25

STAGE3_COST_COV_ALPHA = 0.60
STAGE3_COST_COV_BETA = 1.50
STAGE3_COST_STD_MIN_FRAC = 0.05
STAGE3_COST_STD_MAX_FRAC = 0.10

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
CRASH_PENALTY = 150.0
CRASH_TIME_PENALTY = 0.05

SOFT_TRACK_DIV = 20.0
MEDIUM_TRACK_DIV = 4.0
HIGH_TRACK_MUL = 1.5

# ============================================================
# ROS / runtime
# ============================================================
SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("TUNE_ROS_MASTER_PORT", "11321"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_tune_cma_full_coupled_{PROFILE_NAME}_{GLOBAL_SEED}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Outputs
# ============================================================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
TOP10_PATH = os.path.join(SCRIPT_DIR, f"top10_full_coupled_cma_{PROFILE_NAME}.json")
SUMMARY_PATH = os.path.join(SCRIPT_DIR, f"tuning_summary_full_coupled_cma_{PROFILE_NAME}.json")
RUN_LOG_PATH = os.path.join(SCRIPT_DIR, f"tuning_run_full_coupled_cma_{PROFILE_NAME}.jsonl")
DIST_LOG_PATH = os.path.join(SCRIPT_DIR, f"cma_distribution_log_full_coupled_{PROFILE_NAME}.jsonl")
BEST_JSON_PATH = os.path.join(
    os.path.dirname(CONTROL_PARAM_JSON),
    f"control_param_best_full_coupled_3stage_cma_{PROFILE_NAME}.json",
)

# ============================================================
# Global state
# ============================================================
EPISODES_DONE = 0
TRIALS_DONE = 0
GLOBAL_CANDIDATE_ID = 0

BEST_COST_SO_FAR = float("inf")
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


def _rosmaster_is_up(host: str = ROS_MASTER_HOST, port: int = ROS_MASTER_PORT, timeout: float = 0.3) -> bool:
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


def _materialize_control_json(theta_x: Dict[str, float]) -> Dict[str, Any]:
    global BASE_CONTROL_TEMPLATE

    if BASE_CONTROL_TEMPLATE is None:
        raise RuntimeError("BASE_CONTROL_TEMPLATE is not initialized")

    data = copy.deepcopy(BASE_CONTROL_TEMPLATE)
    for path, val in theta_x.items():
        if path in ALL_BOUNDS:
            lo, hi = ALL_BOUNDS[path]
            _set_json_path(data, path, float(max(lo, min(hi, float(val)))))
    return data


def apply_params_to_control_json(theta_x: Dict[str, float]) -> None:
    data = _materialize_control_json(theta_x)
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


def _metric_float_first(metrics: Dict[str, Any], names: List[str], default: float = 0.0) -> float:
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


def _build_sim_launch_args(track_index: int, sim_time_s: int, critical_crash_multiplier: float) -> Dict[str, str]:
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


def _kill_process_group(p: Optional[subprocess.Popen], sig=signal.SIGINT, timeout_s: float = 5.0) -> None:
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_cholesky(A: np.ndarray) -> np.ndarray:
    A = 0.5 * (A + A.T)
    eye = np.eye(A.shape[0])
    jitter = 1e-12
    for _ in range(10):
        try:
            return np.linalg.cholesky(A + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Nie mogę zrobić Cholesky nawet po jitterze.")


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


def _merge_fixed_and_tuned(fixed_params: Dict[str, float], tuned: Dict[str, float]) -> Dict[str, float]:
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
# Episode / Evaluation
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
    critical_crash_multiplier: float = PHASE1_CRITICAL_CRASH_MULTIPLIER,
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
    critical_crash_multiplier: float = PHASE1_CRITICAL_CRASH_MULTIPLIER,
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

        per_track.append({
            "track_id": int(tid),
            "cost": float(res.cost),
            "crashed": bool(res.crashed),
            "crash_reason": str(res.crash_reason),
        })

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

        dbg = {
            "clipped_dims": int(clipped_dims),
            "z_raw": z_raw,
            "z_clipped": z_clipped,
        }
        return theta, dbg


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
    C_new = _project_spd(C_new, min_eig=1e-10)
    return C_new


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
        if k == "mpc.bounds.max_vx":
            frac = STAGE2_PLANNER_STD_FRAC_VMAX
        elif k == "mpc.cost.q_sdot":
            frac = STAGE2_PLANNER_STD_FRAC_QSDOT
        else:
            frac = STAGE2_PLANNER_STD_FRAC_MU
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

    full_center_theta = _merge_fixed_and_tuned(fixed_params, center_theta)

    rec = {
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
    return rec


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
def _update_top10_file() -> None:
    ranked = sorted(ALL_RESULTS, key=lambda r: r["objective_cost"])[:10]
    payload = {
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
    J: float,
    info: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    global GLOBAL_CANDIDATE_ID, TRIALS_DONE, BEST_COST_SO_FAR, BEST_THETA_SO_FAR

    GLOBAL_CANDIDATE_ID += 1
    TRIALS_DONE += 1

    rec = {
        "candidate_id": int(GLOBAL_CANDIDATE_ID),
        "phase": str(phase),
        "phase_trial": int(phase_trial),
        "objective_cost": float(J),
        "mean_cost": float(info.get("mean_cost", J)),
        "std_cost": float(info.get("std_cost", 0.0)),
        "robust_cost": float(info.get("robust_cost", J)),
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
    elif phase == "phase2_planner_qsdot_cma":
        PHASE2_RESULTS.append(rec)
    elif phase == "phase3_joint_cma":
        PHASE3_RESULTS.append(rec)

    if J < BEST_COST_SO_FAR:
        BEST_COST_SO_FAR = float(J)
        BEST_THETA_SO_FAR = {k: float(v) for k, v in theta_x.items()}

    log_rec = {
        "candidate_id": int(rec["candidate_id"]),
        "phase": str(phase),
        "phase_trial": int(phase_trial),
        "objective_cost": float(J),
        "mean_cost": float(info.get("mean_cost", J)),
        "std_cost": float(info.get("std_cost", 0.0)),
        "robust_cost": float(info.get("robust_cost", J)),
        "crashes": int(info.get("crashes", 0)),
        "critical_crash_multiplier": float(info.get("critical_crash_multiplier", 1.0)),
        "theta": {k: float(v) for k, v in theta_x.items()},
        "best_cost_so_far": float(BEST_COST_SO_FAR),
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
    best_J = float("inf")

    generation = 0
    final_dist: Dict[str, Any] = {}

    while (not es.stop()) and generation < maxiter:
        generation += 1

        Y = es.ask()
        F: List[float] = []

        gen_best_theta = {}
        gen_best_J = float("inf")

        for i, yi in enumerate(Y):
            tuned_theta, dbg = transform.y_to_theta(np.asarray(yi, dtype=float))
            full_theta = _merge_fixed_and_tuned(fixed_params, tuned_theta)

            J, info = evaluate_theta(
                theta_x=full_theta,
                tracks=TRACK_SET,
                sim_time_s=DEFAULT_SIM_TIME_S,
                critical_crash_multiplier=critical_crash_multiplier,
            )

            _register_result(
                phase=phase_name,
                phase_trial=(generation - 1) * lambda_ + i,
                theta_x=full_theta,
                J=J,
                info=info,
                extra={
                    "generation": int(generation),
                    "candidate_in_generation": int(i),
                    "clipped_dims": int(dbg["clipped_dims"]),
                },
            )

            F.append(float(J))

            if J < gen_best_J:
                gen_best_J = float(J)
                gen_best_theta = dict(full_theta)

            if J < best_J:
                best_J = float(J)
                best_theta = dict(full_theta)

            print(
                f"[{phase_name}][gen {generation:02d} cand {i:02d}] "
                f"obj={J:.6g} mean={info.get('mean_cost', J):.6g} std={info.get('std_cost', 0.0):.6g} "
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
        final_dist["generation_best_cost"] = float(gen_best_J)
        final_dist["global_best_cost_inside_phase"] = float(best_J)

        _append_jsonl(DIST_LOG_PATH, final_dist)

    return best_theta, best_J, final_dist


# ============================================================
# Save outputs
# ============================================================
def save_best_json(theta_x: Dict[str, float]) -> None:
    data = _materialize_control_json(theta_x)
    _save_json_atomic(BEST_JSON_PATH, data)


def save_summary(
    phase1_theta: Dict[str, float],
    phase1_J: float,
    phase2_theta: Dict[str, float],
    phase2_J: float,
    phase3_theta: Dict[str, float],
    phase3_J: float,
    phase1_dist: Dict[str, Any],
    stage2_init_dist: Dict[str, Any],
    phase2_dist: Dict[str, Any],
    stage3_init_dist: Dict[str, Any],
    phase3_dist: Dict[str, Any],
) -> None:
    payload = {
        "config": {
            "mode": PROFILE_NAME,
            "catkin_ws": CATKIN_WS,
            "ws_src": WS_SRC,
            "acados_lib": ACADOS_LIB,
            "sim_time_s": DEFAULT_SIM_TIME_S,
            "tracks": TRACK_SET,
            "episodes_per_trial": EPISODES_PER_TRIAL,
            "phase1_lambda": PHASE1_LAMBDA,
            "phase1_maxiter": PHASE1_MAXITER,
            "phase1_trials": PHASE1_N_TRIALS,
            "phase2_lambda": PHASE2_LAMBDA,
            "phase2_maxiter": PHASE2_MAXITER,
            "phase2_trials": PHASE2_N_TRIALS,
            "phase3_lambda": PHASE3_LAMBDA,
            "phase3_maxiter": PHASE3_MAXITER,
            "phase3_trials": PHASE3_N_TRIALS,
            "total_trials": TOTAL_TRIALS,
            "total_episodes": TOTAL_EPISODES,
            "pure_sim_hours": PURE_SIM_HOURS,
            "safe_planner": SAFE_PLANNER,
            "planner_bounds": PLANNER_BOUNDS,
            "cost_bounds": COST_BOUNDS,
            "phase1_critical_crash_multiplier": PHASE1_CRITICAL_CRASH_MULTIPLIER,
            "phase2_critical_crash_multiplier": PHASE2_CRITICAL_CRASH_MULTIPLIER,
            "phase3_critical_crash_multiplier": PHASE3_CRITICAL_CRASH_MULTIPLIER,
            "stage1_cost_std_frac": STAGE1_COST_STD_FRAC,
            "stage2_planner_std_frac_vmax": STAGE2_PLANNER_STD_FRAC_VMAX,
            "stage2_planner_std_frac_mu": STAGE2_PLANNER_STD_FRAC_MU,
            "stage2_planner_std_frac_qsdot": STAGE2_PLANNER_STD_FRAC_QSDOT,
            "stage3_planner_safe_blend": STAGE3_PLANNER_SAFE_BLEND,
            "stage3_cost_cov_alpha": STAGE3_COST_COV_ALPHA,
            "stage3_cost_cov_beta": STAGE3_COST_COV_BETA,
            "stage3_cost_std_min_frac": STAGE3_COST_STD_MIN_FRAC,
            "stage3_cost_std_max_frac": STAGE3_COST_STD_MAX_FRAC,
            "stage3_planner_cov_alpha": STAGE3_PLANNER_COV_ALPHA,
            "stage3_planner_cov_beta": STAGE3_PLANNER_COV_BETA,
            "stage3_planner_std_min_frac": STAGE3_PLANNER_STD_MIN_FRAC,
            "stage3_planner_std_max_frac": STAGE3_PLANNER_STD_MAX_FRAC,
            "ucb_weight": UCB_WEIGHT,
            "crash_penalty": CRASH_PENALTY,
            "crash_time_penalty": CRASH_TIME_PENALTY,
            "dist_log_path": DIST_LOG_PATH,
            "run_log_path": RUN_LOG_PATH,
            "slack_tuning": False,
            "q_sdot_is_linear": True,
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
                "best_objective_cost": float(phase1_J),
                "best_theta": {k: float(v) for k, v in phase1_theta.items()},
            },
            "phase2_planner_qsdot_cma": {
                "best_objective_cost": float(phase2_J),
                "best_theta": {k: float(v) for k, v in phase2_theta.items()},
            },
            "phase3_joint_cma": {
                "best_objective_cost": float(phase3_J),
                "best_theta": {k: float(v) for k, v in phase3_theta.items()},
            },
            "global_best": {
                "best_objective_cost": float(BEST_COST_SO_FAR),
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

    global BEST_COST_SO_FAR, BEST_THETA_SO_FAR, EPISODES_DONE, TRIALS_DONE
    global BASE_CONTROL_TEMPLATE

    BASE_CONTROL_TEMPLATE = _load_json(CONTROL_PARAM_JSON)

    BEST_COST_SO_FAR = float("inf")
    BEST_THETA_SO_FAR = {}
    EPISODES_DONE = 0
    TRIALS_DONE = 0
    ALL_RESULTS.clear()
    PHASE1_RESULTS.clear()
    PHASE2_RESULTS.clear()
    PHASE3_RESULTS.clear()

    for p in [DIST_LOG_PATH, RUN_LOG_PATH, SUMMARY_PATH]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    print("[CFG] CATKIN_WS =", CATKIN_WS)
    print("[CFG] WS_SRC    =", WS_SRC)
    print("[CFG] ACADOS_LIB =", ACADOS_LIB)
    print("[CFG] MODE      =", PROFILE_NAME)
    print("[CFG] TRACK_SET =", TRACK_SET)
    print("[CFG] SIM_TIME  =", DEFAULT_SIM_TIME_S, "s")
    print("[CFG] TOTAL_TRIALS =", TOTAL_TRIALS)
    print("[CFG] TOTAL_EPISODES =", TOTAL_EPISODES)
    print(f"[CFG] PURE_SIM_HOURS ~= {PURE_SIM_HOURS:.3f}")
    print("[CFG] Phase1 =", PHASE1_N_TRIALS, "CMA trials")
    print("[CFG] Phase2 =", PHASE2_N_TRIALS, "CMA trials")
    print("[CFG] Phase3 =", PHASE3_N_TRIALS, "CMA trials")
    print(f"[CFG] Phase1 critical_crash_multiplier = {PHASE1_CRITICAL_CRASH_MULTIPLIER}")
    print(f"[CFG] Phase2 critical_crash_multiplier = {PHASE2_CRITICAL_CRASH_MULTIPLIER}")
    print(f"[CFG] Phase3 critical_crash_multiplier = {PHASE3_CRITICAL_CRASH_MULTIPLIER}")
    print(f"[CFG] SAFE_PLANNER = {SAFE_PLANNER}")
    print(f"[CFG] PLANNER_BOUNDS = {PLANNER_BOUNDS}")
    print(f"[CFG] COST_BOUNDS = {COST_BOUNDS}")
    print(f"[CFG] ALL_BOUNDS = {ALL_BOUNDS}")
    print(f"[CFG] UCB_WEIGHT = {UCB_WEIGHT}")
    print(f"[CFG] STAGE1_COST_STD_FRAC          = {STAGE1_COST_STD_FRAC}")
    print(f"[CFG] STAGE2_PLANNER_STD_FRAC_VMAX = {STAGE2_PLANNER_STD_FRAC_VMAX}")
    print(f"[CFG] STAGE2_PLANNER_STD_FRAC_MU   = {STAGE2_PLANNER_STD_FRAC_MU}")
    print(f"[CFG] STAGE2_PLANNER_STD_FRAC_QSDOT= {STAGE2_PLANNER_STD_FRAC_QSDOT}")
    print(f"[CFG] STAGE3_PLANNER_SAFE_BLEND    = {STAGE3_PLANNER_SAFE_BLEND}")
    print("[CFG] q_sdot jest liniowy i strojony od Stage 2.")
    print("[CFG] Slacki nie są strojone.")
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
        print("\n================ PHASE 1: COST ONLY @ SAFE PLANNER (CMA-ES) ================\n")
        print("[PHASE1] tune only cost weights")
        print("[PHASE1] fixed: max_vx, mux, muy, q_sdot")
        print("[PHASE1] slack penalties frozen from base JSON\n")

        stage1_transform, stage1_init_cov = _build_stage1_transform()
        _append_jsonl(
            DIST_LOG_PATH,
            _make_initial_distribution_record(
                phase_name="phase1_cost_only_cma",
                transform=stage1_transform,
                fixed_params=SAFE_PLANNER,
                C_init=stage1_init_cov,
            ),
        )

        phase1_theta, phase1_J, phase1_dist = run_cma_phase(
            phase_name="phase1_cost_only_cma",
            transform=stage1_transform,
            fixed_params=SAFE_PLANNER,
            lambda_=PHASE1_LAMBDA,
            maxiter=PHASE1_MAXITER,
            critical_crash_multiplier=PHASE1_CRITICAL_CRASH_MULTIPLIER,
            seed_offset=11,
        )

        print("\n================ TRANSITION 1 -> 2: BUILD STAGE-2 DISTRIBUTION ================\n")
        fixed_costs = _extract_cost_theta(phase1_theta)
        stage2_transform, stage2_init_cov = _build_stage2_transform()
        stage2_init_dist = _make_initial_distribution_record(
            phase_name="phase2_planner_qsdot_cma",
            transform=stage2_transform,
            fixed_params=fixed_costs,
            C_init=stage2_init_cov,
        )
        _append_jsonl(DIST_LOG_PATH, stage2_init_dist)

        print("[STAGE2 INIT] center theta full:")
        for k in sorted(stage2_init_dist["distribution_center_theta_full"].keys()):
            print(f"  {k}: {stage2_init_dist['distribution_center_theta_full'][k]:.6g}")

        print("\n================ PHASE 2: PLANNER+Q_SDOT @ FIXED BEST COSTS (CMA-ES) ================\n")
        print("[PHASE2] costs frozen to best from stage1")
        print("[PHASE2] free: max_vx, mux, muy, q_sdot")
        print("[PHASE2] slack penalties still frozen\n")

        phase2_theta, phase2_J, phase2_dist = run_cma_phase(
            phase_name="phase2_planner_qsdot_cma",
            transform=stage2_transform,
            fixed_params=fixed_costs,
            lambda_=PHASE2_LAMBDA,
            maxiter=PHASE2_MAXITER,
            critical_crash_multiplier=PHASE2_CRITICAL_CRASH_MULTIPLIER,
            seed_offset=22,
        )

        print("\n================ TRANSITION 2 -> 3: BUILD STAGE-3 DISTRIBUTION ================\n")
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

        print("[STAGE3 INIT] center theta full:")
        for k in sorted(stage3_init_dist["distribution_center_theta_full"].keys()):
            print(f"  {k}: {stage3_init_dist['distribution_center_theta_full'][k]:.6g}")

        print("\n================ PHASE 3: JOINT (CMA-ES) ================\n")
        print("[PHASE3] center = best costs from stage1 + blended planner from stage2")
        print("[PHASE3] covariance blocks relaxed from phase1/phase2")
        print("[PHASE3] no separate refine stage")
        print("[PHASE3] slack penalties still frozen\n")

        phase3_theta, phase3_J, phase3_dist = run_cma_phase(
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
            phase1_J=phase1_J,
            phase2_theta=phase2_theta,
            phase2_J=phase2_J,
            phase3_theta=phase3_theta,
            phase3_J=phase3_J,
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
    print(f"Best global objective cost: {BEST_COST_SO_FAR:.6g}")
    print("Best global params:")
    for k in sorted(BEST_THETA_SO_FAR.keys()):
        if k in ALL_BOUNDS:
            lo, hi = ALL_BOUNDS[k]
            print(f"  {k}: {BEST_THETA_SO_FAR[k]:.6g}   (bounds [{lo}, {hi}])")
        else:
            print(f"  {k}: {BEST_THETA_SO_FAR[k]:.6g}")

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
