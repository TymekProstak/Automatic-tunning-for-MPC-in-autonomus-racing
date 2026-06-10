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
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_agh_racing/NO_STAGED/TUNE_CMA_ES")
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
# Profile / base staged budget
# ============================================================
PROFILE_NAME = os.environ.get("TUNE_PROFILE", "night_4track")

if PROFILE_NAME == "smoke_1track":
    TRACK_SET = [1]

    PHASE1_N_TRIALS = 12
    PHASE2_N_TRIALS = 24
    PHASE3_N_TRIALS = 24

    # unstaged: preserve total budget
    UNSTAGED_N_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS  # 60
    UNSTAGED_LAMBDA = 6
    UNSTAGED_MAXITER = 10

elif PROFILE_NAME == "night_4track":
    TRACK_SET = [1, 2, 3, 4]

    PHASE1_N_TRIALS = 48
    PHASE2_N_TRIALS = 64
    PHASE3_N_TRIALS = 96

    # unstaged: preserve total budget
    UNSTAGED_N_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS  # 208
    UNSTAGED_LAMBDA = 8
    UNSTAGED_MAXITER = 26

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


def _override_unstaged_budget(n_trials: int, lam: int, maxiter: int) -> Tuple[int, int, int]:
    lam_env = "TUNE_UNSTAGED_LAMBDA"
    maxiter_env = "TUNE_UNSTAGED_MAXITER"
    n_trials_env = "TUNE_UNSTAGED_N_TRIALS"

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

# Optional overrides: unstaged budget (keeps lambda*maxiter consistency)
UNSTAGED_N_TRIALS, UNSTAGED_LAMBDA, UNSTAGED_MAXITER = _override_unstaged_budget(
    UNSTAGED_N_TRIALS, UNSTAGED_LAMBDA, UNSTAGED_MAXITER
)

assert UNSTAGED_LAMBDA * UNSTAGED_MAXITER == UNSTAGED_N_TRIALS

EPISODES_PER_TRIAL = len(TRACK_SET)
TOTAL_TRIALS = UNSTAGED_N_TRIALS
TOTAL_EPISODES = TOTAL_TRIALS * EPISODES_PER_TRIAL
PURE_SIM_HOURS = TOTAL_EPISODES * DEFAULT_SIM_TIME_S / 3600.0

UNSTAGED_PHASE_NAME = "unstaged_joint_cma"

# ============================================================
# Critical crash multiplier
# ============================================================
UNSTAGED_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("TUNE_UNSTAGED_CRITICAL_CRASH_MULTIPLIER", "1.0")
)

# ============================================================
# Safe planner
# ============================================================
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
# Unstaged joint CMA init distribution
# ============================================================
STAGE1_COST_STD_FRAC = 0.30

if PROFILE_NAME == "smoke_1track":
    UNSTAGED_PLANNER_STD_FRAC_VMAX = 0.30
    UNSTAGED_PLANNER_STD_FRAC_MU = 0.28
elif PROFILE_NAME == "night_4track":
    UNSTAGED_PLANNER_STD_FRAC_VMAX = 0.35
    UNSTAGED_PLANNER_STD_FRAC_MU = 0.32
else:
    UNSTAGED_PLANNER_STD_FRAC_VMAX = 0.30
    UNSTAGED_PLANNER_STD_FRAC_MU = 0.28

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
ROS_MASTER_PORT = int(os.environ.get("TUNE_ROS_MASTER_PORT", "11617"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_tune_cma_tv_unstaged_{PROFILE_NAME}_{GLOBAL_SEED}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Outputs
# ============================================================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
TOP10_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_TOP10_PATH",
        os.path.join(SCRIPT_DIR, f"top10_so_far_tv_unstaged_{PROFILE_NAME}.json"),
    )
)
SUMMARY_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_SUMMARY_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_summary_tv_unstaged_{PROFILE_NAME}.json"),
    )
)
RUN_LOG_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_RUN_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_run_tv_unstaged_{PROFILE_NAME}.jsonl"),
    )
)
DIST_LOG_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_DIST_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"cma_distribution_log_tv_unstaged_{PROFILE_NAME}.jsonl"),
    )
)
BEST_JSON_PATH = os.path.join(
    os.path.dirname(CONTROL_PARAM_JSON),
    f"control_param_best_lti_unstaged_cma_tv_{PROFILE_NAME}.json",
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
UNSTAGED_RESULTS: List[Dict[str, Any]] = []

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


def _build_unstaged_transform() -> Tuple[SearchTransform, np.ndarray]:
    mean_z = []
    stds = []

    # costs: center in log-space at geometric mean, like staged phase1
    for k in COST_KEYS:
        lo, hi = COST_BOUNDS[k]
        full = math.log(hi / lo)
        mean_z.append(math.log(math.sqrt(lo * hi)))
        stds.append(STAGE1_COST_STD_FRAC * full)

    # planners: center at SAFE_PLANNER, std from staged phase2 planner init
    for k in PLANNER_KEYS:
        lo, hi = PLANNER_BOUNDS[k]
        frac = UNSTAGED_PLANNER_STD_FRAC_VMAX if k == "velocity_planner.v_max" else UNSTAGED_PLANNER_STD_FRAC_MU
        mean_z.append(float(SAFE_PLANNER[k]))
        stds.append(frac * (hi - lo))

    mean_z_arr = np.asarray(mean_z, dtype=float)
    C0 = np.diag(np.asarray(stds, dtype=float) ** 2)
    L0 = _safe_cholesky(C0)

    return SearchTransform(keys=ALL_KEYS, mean_z=mean_z_arr, chol_z=L0), C0


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

    if phase == UNSTAGED_PHASE_NAME:
        UNSTAGED_RESULTS.append(rec)

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
# Generic CMA phase
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
    unstaged_theta: Dict[str, float],
    unstaged_obj: float,
    unstaged_init_dist: Dict[str, Any],
    unstaged_final_dist: Dict[str, Any],
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
            "unstaged_trials": UNSTAGED_N_TRIALS,
            "unstaged_lambda": UNSTAGED_LAMBDA,
            "unstaged_maxiter": UNSTAGED_MAXITER,
            "total_trials": TOTAL_TRIALS,
            "total_episodes": TOTAL_EPISODES,
            "pure_sim_hours": PURE_SIM_HOURS,
            "ucb_weight": UCB_WEIGHT,
            "safe_planner": SAFE_PLANNER,
            "planner_bounds": PLANNER_BOUNDS,
            "cost_bounds": COST_BOUNDS,
            "stage1_cost_std_frac": STAGE1_COST_STD_FRAC,
            "unstaged_planner_std_frac_vmax": UNSTAGED_PLANNER_STD_FRAC_VMAX,
            "unstaged_planner_std_frac_mu": UNSTAGED_PLANNER_STD_FRAC_MU,
            "unstaged_critical_crash_multiplier": UNSTAGED_CRITICAL_CRASH_MULTIPLIER,
            "dist_log_path": DIST_LOG_PATH,
            "run_log_path": RUN_LOG_PATH,
            "equivalent_staged_budget": {
                "phase1_trials": PHASE1_N_TRIALS,
                "phase2_trials": PHASE2_N_TRIALS,
                "phase3_trials": PHASE3_N_TRIALS,
            },
        },
        "stage_distributions": {
            "unstaged_initial_distribution": unstaged_init_dist,
            "unstaged_final_distribution": unstaged_final_dist,
        },
        "results": {
            "unstaged_joint_cma": {
                "best_objective": float(unstaged_obj),
                "best_theta": {k: float(v) for k, v in unstaged_theta.items()},
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
    UNSTAGED_RESULTS.clear()

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
    print("[CFG] UNSTAGED_PHASE =", UNSTAGED_PHASE_NAME)
    print("[CFG] UNSTAGED_N_TRIALS =", UNSTAGED_N_TRIALS)
    print("[CFG] UNSTAGED_LAMBDA   =", UNSTAGED_LAMBDA)
    print("[CFG] UNSTAGED_MAXITER  =", UNSTAGED_MAXITER)
    print("[CFG] TOTAL_EPISODES =", TOTAL_EPISODES)
    print(f"[CFG] PURE_SIM_HOURS ~= {PURE_SIM_HOURS:.3f}")
    print("[CFG] Joint CMA from the beginning over ALL_KEYS")
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
        print("\n================ UNSTAGED JOINT CMA-ES @ ALL PARAMS ================\n")

        unstaged_transform, unstaged_init_cov = _build_unstaged_transform()
        unstaged_init_dist = _make_initial_distribution_record(
            phase_name=UNSTAGED_PHASE_NAME,
            transform=unstaged_transform,
            fixed_params={},
            C_init=unstaged_init_cov,
        )
        _append_jsonl(DIST_LOG_PATH, unstaged_init_dist)

        print("[UNSTAGED INIT] center:")
        for k in ALL_KEYS:
            print(f"  {k}: {unstaged_init_dist['distribution_center_theta'][k]:.6g}")

        unstaged_theta, unstaged_obj, unstaged_final_dist = run_cma_phase(
            phase_name=UNSTAGED_PHASE_NAME,
            transform=unstaged_transform,
            fixed_params={},
            lambda_=UNSTAGED_LAMBDA,
            maxiter=UNSTAGED_MAXITER,
            critical_crash_multiplier=UNSTAGED_CRITICAL_CRASH_MULTIPLIER,
            seed_offset=101,
        )

        save_best_json(BEST_THETA_SO_FAR)
        save_summary(
            unstaged_theta=unstaged_theta,
            unstaged_obj=unstaged_obj,
            unstaged_init_dist=unstaged_init_dist,
            unstaged_final_dist=unstaged_final_dist,
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
