#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import random
import math
import signal
import subprocess
import socket
import copy
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List


# ============================================================
# Workspace
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_amz/TUNE_RANDOM_SEARCH")
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
# ACADOS lib needed by dv_control in this workspace
# ============================================================
DEFAULT_ACADOS_LIB = os.path.join(
    WS_SRC, "dv_control", "External", "acados", "install", "lib"
)
ACADOS_LIB = os.path.abspath(
    os.environ.get("TUNE_ACADOS_LIB", DEFAULT_ACADOS_LIB)
)

# ============================================================
# Evaluation setup / profile presets
# ============================================================
GLOBAL_SEED = int(os.environ.get("TUNE_GLOBAL_SEED", "123"))
random.seed(GLOBAL_SEED)

DEFAULT_SIM_TIME_S = int(os.environ.get("TUNE_SIM_TIME_S", "60"))
UCB_WEIGHT = float(os.environ.get("TUNE_UCB_WEIGHT", "1.0"))

PROFILE_NAME = os.environ.get("TUNE_PROFILE", "night_4track").strip()

if PROFILE_NAME == "smoke_1track":
    TRACK_SET = [1]

    PHASE1_N_TRIALS = 12
    PHASE1_STARTUP = 5

    PHASE2_N_TRIALS = 24
    PHASE2_STARTUP = 12

    PHASE3_N_TRIALS = 24
    PHASE3_STARTUP = 8

elif PROFILE_NAME == "night_4track":
    TRACK_SET = [1, 2, 3, 4]

    PHASE1_N_TRIALS = 64
    PHASE1_STARTUP = 24

    PHASE2_N_TRIALS = 64
    PHASE2_STARTUP = 16

    PHASE3_N_TRIALS = 96
    PHASE3_STARTUP = 24

else:
    raise ValueError(f"Unknown PROFILE_NAME: {PROFILE_NAME}")

EPISODES_PER_TRIAL = len(TRACK_SET)
TOTAL_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS
TOTAL_EPISODES = TOTAL_TRIALS * EPISODES_PER_TRIAL
PURE_SIM_HOURS = TOTAL_EPISODES * DEFAULT_SIM_TIME_S / 3600.0

# ============================================================
# Phase names
# ============================================================
PHASE1_NAME = "phase1_cost_only_random"
PHASE2_NAME = "phase2_planner_qsdot_random"
PHASE3_NAME = "phase3_joint_random"

# ============================================================
# Phase-dependent critical crash multiplier
# kept aligned with staged coupled baseline
# ============================================================
PHASE1_CRITICAL_CRASH_MULTIPLIER = float(os.environ.get("TUNE_PHASE1_CRASH_MULT", "1.0"))
PHASE2_CRITICAL_CRASH_MULTIPLIER = float(os.environ.get("TUNE_PHASE2_CRASH_MULT", "1.0"))
PHASE3_CRITICAL_CRASH_MULTIPLIER = float(os.environ.get("TUNE_PHASE3_CRASH_MULT", "1.0"))

# ============================================================
# Base control template
# ============================================================
BASE_CONTROL_TEMPLATE: Optional[Dict[str, Any]] = None

# ============================================================
# Safe planner / aggressiveness
# aligned with staged coupled baseline
# ============================================================
SAFE_PLANNER = {
    "mpc.bounds.max_vx": 13.5,
    "mpc.model.mux": 0.6,
    "mpc.model.muy": 0.7,
    "mpc.cost.q_sdot": 0.1,
}

# ============================================================
# Tuned params / bounds
# aligned with staged coupled baseline
# Stage 1: costs only
# Stage 2: planner + q_sdot only
# Stage 3: joint
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

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(PLANNER_BOUNDS)
ALL_BOUNDS.update(COST_BOUNDS)

LOG_PARAMS = set(COST_BOUNDS.keys())

# ============================================================
# Staged architecture knobs
# ============================================================
GOOD_FRAC = 0.30
MIN_G_SIZE = 4

PHASE2_SAFE_PLANNER_STARTUP_FRAC = 0.35
PHASE2_SAFE_PLANNER_SIGMA_FRAC_VMAX = 0.30
PHASE2_SAFE_PLANNER_SIGMA_FRAC_MU = 0.30
PHASE2_SAFE_PLANNER_SIGMA_FRAC_QSDOT = 0.30

TOP_K_COST_ANCHORS = 3
TOP_K_PLANNER_ANCHORS = 5

if PROFILE_NAME == "smoke_1track":
    STAGE3_PLANNER_SAFE_BLEND = 0.25
else:
    STAGE3_PLANNER_SAFE_BLEND = 0.20

# ============================================================
# Random-search hyperparams
# ============================================================
STAGE1_GLOBAL_RANDOM_FRACTION = 0.30
STAGE2_GLOBAL_RANDOM_FRACTION = 0.15
STAGE3_GLOBAL_RANDOM_FRACTION = 0.35

STAGE1_COST_SIGMA_INFLATE = 1.10
STAGE1_COST_LOG_SIGMA_MIN_FRAC = 0.10
STAGE1_COST_LOG_SIGMA_MAX_FRAC = 0.35

STAGE2_PLANNER_SIGMA_INFLATE = 1.10
STAGE2_PLANNER_SIGMA_MIN_FRAC = 0.10
STAGE2_PLANNER_SIGMA_MAX_FRAC = 0.30

STAGE3_PLANNER_SIGMA_INFLATE = 1.15
STAGE3_PLANNER_SIGMA_MIN_FRAC = 0.10
STAGE3_PLANNER_SIGMA_MAX_FRAC = 0.35

STAGE3_COST_SIGMA_INFLATE = 1.10
STAGE3_COST_LOG_SIGMA_MIN_FRAC = 0.06
STAGE3_COST_LOG_SIGMA_MAX_FRAC = 0.18

# ============================================================
# Cost function
# aligned with staged coupled baseline
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
ROS_MASTER_PORT = int(os.environ.get("TUNE_ROS_MASTER_PORT", "11322"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(
    f"~/.ros_tune_random_full_coupled_staged_{PROFILE_NAME}_{GLOBAL_SEED}"
)
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Outputs
# ============================================================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
TOP10_PATH = os.path.join(
    SCRIPT_DIR, f"top10_full_coupled_staged_random_search_{PROFILE_NAME}.json"
)
SUMMARY_PATH = os.path.join(
    SCRIPT_DIR, f"tuning_summary_full_coupled_staged_random_search_{PROFILE_NAME}.json"
)
RUN_LOG_PATH = os.path.join(
    SCRIPT_DIR, f"tuning_run_full_coupled_staged_random_search_{PROFILE_NAME}.jsonl"
)
BEST_JSON_PATH = os.path.join(
    os.path.dirname(CONTROL_PARAM_JSON),
    f"control_param_best_full_coupled_staged_random_search_{PROFILE_NAME}.json",
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


def _save_json_atomic(path: str, data: Dict[str, Any]) -> None:
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
            _set_json_path(data, path, float(_clamp(float(val), lo, hi)))
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


def _theta_key(theta: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, round(float(v), 12)) for k, v in theta.items()))


def _extract_cost_theta(theta: Dict[str, float]) -> Dict[str, float]:
    return {k: float(theta[k]) for k in COST_BOUNDS.keys()}


def _extract_planner_theta(theta: Dict[str, float]) -> Dict[str, float]:
    return {k: float(theta[k]) for k in PLANNER_BOUNDS.keys()}


def _merge_theta(planner_theta: Dict[str, float], cost_theta: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in PLANNER_BOUNDS.keys():
        out[k] = float(planner_theta[k])
    for k in COST_BOUNDS.keys():
        out[k] = float(cost_theta[k])
    return out


def _top_records(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    ranked = sorted(results, key=lambda r: r["objective_cost"])
    return ranked[:max(0, min(k, len(ranked)))]


def _blend_planner_theta_with_safe(
    planner_theta: Dict[str, float],
    safe_planner: Dict[str, float],
    safe_blend: float,
) -> Dict[str, float]:
    safe_blend = float(_clamp(safe_blend, 0.0, 1.0))
    out: Dict[str, float] = {}

    for k in PLANNER_BOUNDS.keys():
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
# Sampling helpers
# ============================================================
def _lhs_unit(rng: random.Random, n: int, d: int) -> List[List[float]]:
    u = [[0.0] * d for _ in range(n)]
    for j in range(d):
        strata = [(i + rng.random()) / float(n) for i in range(n)]
        rng.shuffle(strata)
        for i in range(n):
            u[i][j] = strata[i]
    return u


def _u_to_param(name: str, u: float, bounds: Dict[str, Tuple[float, float]]) -> float:
    lo, hi = bounds[name]
    u = _clamp(u, 0.0, 1.0)

    if name in LOG_PARAMS:
        t = math.log(lo) + u * (math.log(hi) - math.log(lo))
        return float(math.exp(t))

    return float(lo + u * (hi - lo))


def generate_lhs_trials(
    seed: int,
    n_trials: int,
    bounds: Dict[str, Tuple[float, float]],
) -> List[Dict[str, float]]:
    keys = list(bounds.keys())
    rng = random.Random(seed)
    U = _lhs_unit(rng, n_trials, len(keys))

    out: List[Dict[str, float]] = []
    for i in range(n_trials):
        th: Dict[str, float] = {}
        for j, k in enumerate(keys):
            th[k] = _u_to_param(k, U[i][j], bounds)
        out.append(th)

    return out


def sample_theta_global(
    rng: random.Random,
    bounds: Dict[str, Tuple[float, float]],
    fixed_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    th: Dict[str, float] = {}
    if fixed_params:
        th.update(fixed_params)

    for k, (lo, hi) in bounds.items():
        if k in LOG_PARAMS:
            u = rng.random()
            x = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            x = lo + (hi - lo) * rng.random()
        th[k] = float(_clamp(x, lo, hi))

    return th


def _weighted_choice_record(rng: random.Random, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(recs, key=lambda r: r["objective_cost"])
    weights = [1.0 / (i + 1) for i in range(len(ranked))]
    idx = rng.choices(range(len(ranked)), weights=weights, k=1)[0]
    return ranked[idx]


def _planner_sigmas_from_records(
    recs: List[Dict[str, Any]],
    inflate: float,
    min_frac: float,
    max_frac: float,
) -> Dict[str, float]:
    sigmas: Dict[str, float] = {}

    if len(recs) == 0:
        for k, (lo, hi) in PLANNER_BOUNDS.items():
            sigmas[k] = float(min_frac * (hi - lo))
        return sigmas

    ranked = sorted(recs, key=lambda r: r["objective_cost"])
    weights = [1.0 / (i + 1) for i in range(len(ranked))]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    for k, (lo, hi) in PLANNER_BOUNDS.items():
        xs = [float(rec["theta"][k]) for rec in ranked]
        mu = sum(w * x for w, x in zip(weights, xs))
        var = sum(w * (x - mu) * (x - mu) for w, x in zip(weights, xs))
        std_emp = math.sqrt(max(0.0, var))

        full = hi - lo
        std_min = min_frac * full
        std_max = max_frac * full

        sigma = inflate * std_emp
        sigma = max(std_min, min(std_max, sigma))
        sigmas[k] = float(sigma)

    return sigmas


def _cost_sigmas_from_records(
    recs: List[Dict[str, Any]],
    inflate: float,
    min_frac: float,
    max_frac: float,
) -> Dict[str, float]:
    sigmas: Dict[str, float] = {}

    if len(recs) == 0:
        for k, (lo, hi) in COST_BOUNDS.items():
            full = math.log(hi / lo)
            sigmas[k] = float(min_frac * full)
        return sigmas

    ranked = sorted(recs, key=lambda r: r["objective_cost"])
    weights = [1.0 / (i + 1) for i in range(len(ranked))]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    for k, (lo, hi) in COST_BOUNDS.items():
        logs = [math.log(float(rec["theta"][k])) for rec in ranked]
        mu = sum(w * x for w, x in zip(weights, logs))
        var = sum(w * (x - mu) * (x - mu) for w, x in zip(weights, logs))
        std_emp = math.sqrt(max(0.0, var))

        full = math.log(hi / lo)
        std_min = min_frac * full
        std_max = max_frac * full

        sigma = inflate * std_emp
        sigma = max(std_min, min(std_max, sigma))
        sigmas[k] = float(sigma)

    return sigmas


def _sample_local_from_center(
    rng: random.Random,
    center_theta: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    sigma_map: Dict[str, float],
    fixed_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    th: Dict[str, float] = {}
    if fixed_params:
        th.update(fixed_params)

    for k, (lo, hi) in bounds.items():
        x0 = float(center_theta[k])
        sigma = float(sigma_map[k])

        if k in LOG_PARAMS:
            z = math.log(x0) + rng.gauss(0.0, sigma)
            x = math.exp(z)
        else:
            x = x0 + rng.gauss(0.0, sigma)

        th[k] = float(_clamp(x, lo, hi))

    return th


# ============================================================
# G / L construction
# ============================================================
def build_G_L_from_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ranked = sorted(results, key=lambda r: r["objective_cost"])
    n = len(ranked)

    if n == 0:
        return [], []

    g_n = max(MIN_G_SIZE, int(math.ceil(GOOD_FRAC * n)))
    g_n = min(g_n, n)

    G = ranked[:g_n]
    L = ranked[g_n:]
    return G, L


def build_G_L_from_phase1() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_G_L_from_results(PHASE1_RESULTS)


def build_G_L_from_phase2() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_G_L_from_results(PHASE2_RESULTS)


def build_G_L_from_phase3() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_G_L_from_results(PHASE3_RESULTS)


# ============================================================
# Stage-specific startup builders
# ============================================================
def build_phase2_startup_planner_only(
    fixed_costs: Dict[str, float],
    n_points: int,
    seed: int,
) -> List[Dict[str, float]]:
    rng = random.Random(seed)

    out: List[Dict[str, float]] = []
    seen = set()

    def _push(planner_theta: Dict[str, float]) -> None:
        full = _merge_theta(planner_theta, fixed_costs)
        key = _theta_key(full)
        if key not in seen:
            out.append(full)
            seen.add(key)

    _push(dict(SAFE_PLANNER))

    target_safe_block = int(round(PHASE2_SAFE_PLANNER_STARTUP_FRAC * n_points))
    target_safe_block = max(1, min(target_safe_block, n_points))
    n_local_safe = max(0, target_safe_block - 1)

    for _ in range(n_local_safe):
        planner_theta: Dict[str, float] = {}
        for k, (lo, hi) in PLANNER_BOUNDS.items():
            if k == "mpc.bounds.max_vx":
                frac = PHASE2_SAFE_PLANNER_SIGMA_FRAC_VMAX
            elif k == "mpc.cost.q_sdot":
                frac = PHASE2_SAFE_PLANNER_SIGMA_FRAC_QSDOT
            else:
                frac = PHASE2_SAFE_PLANNER_SIGMA_FRAC_MU
            sigma = frac * (hi - lo)
            draw = SAFE_PLANNER[k] + rng.gauss(0.0, sigma)
            planner_theta[k] = float(_clamp(draw, lo, hi))
        _push(planner_theta)
        if len(out) >= n_points:
            return out[:n_points]

    remaining = n_points - len(out)
    if remaining > 0:
        planner_lhs = generate_lhs_trials(seed + 100, max(remaining * 3, remaining), PLANNER_BOUNDS)
        for th in planner_lhs:
            _push(th)
            if len(out) >= n_points:
                return out[:n_points]

    while len(out) < n_points:
        planner_theta: Dict[str, float] = {}
        for k, (lo, hi) in PLANNER_BOUNDS.items():
            planner_theta[k] = float(lo + (hi - lo) * rng.random())
        _push(planner_theta)

    return out[:n_points]


def build_phase3_startup_joint_anchors(
    best_stage1_theta: Dict[str, float],
    best_stage2_theta: Dict[str, float],
    n_points: int,
    seed: int,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rng = random.Random(seed)

    best_costs = _extract_cost_theta(best_stage1_theta)
    best_planner_raw = _extract_planner_theta(best_stage2_theta)
    best_planner = _blend_planner_theta_with_safe(
        planner_theta=best_planner_raw,
        safe_planner=SAFE_PLANNER,
        safe_blend=STAGE3_PLANNER_SAFE_BLEND,
    )
    center_theta = _merge_theta(best_planner, best_costs)

    anchors: List[Dict[str, float]] = []
    seen = set()

    def _push_anchor(theta: Dict[str, float]) -> None:
        key = _theta_key(theta)
        if key not in seen:
            anchors.append(dict(theta))
            seen.add(key)

    _push_anchor(center_theta)
    _push_anchor(_merge_theta(dict(SAFE_PLANNER), best_costs))

    for rec in _top_records(PHASE2_RESULTS, TOP_K_PLANNER_ANCHORS):
        planner_i_raw = _extract_planner_theta(rec["theta"])
        planner_i = _blend_planner_theta_with_safe(
            planner_theta=planner_i_raw,
            safe_planner=SAFE_PLANNER,
            safe_blend=STAGE3_PLANNER_SAFE_BLEND,
        )
        _push_anchor(_merge_theta(planner_i, best_costs))

    for rec in _top_records(PHASE1_RESULTS, TOP_K_COST_ANCHORS):
        costs_i = _extract_cost_theta(rec["theta"])
        _push_anchor(_merge_theta(best_planner, costs_i))

    planner_sigmas: Dict[str, float] = {}
    for k, (lo, hi) in PLANNER_BOUNDS.items():
        if k == "mpc.bounds.max_vx":
            frac = PHASE2_SAFE_PLANNER_SIGMA_FRAC_VMAX
        elif k == "mpc.cost.q_sdot":
            frac = PHASE2_SAFE_PLANNER_SIGMA_FRAC_QSDOT
        else:
            frac = PHASE2_SAFE_PLANNER_SIGMA_FRAC_MU
        planner_sigmas[k] = frac * (hi - lo)

    cost_sigmas = {k: 0.10 * math.log(hi / lo) for k, (lo, hi) in COST_BOUNDS.items()}

    while len(anchors) < n_points:
        th: Dict[str, float] = {}

        for k, (lo, hi) in PLANNER_BOUNDS.items():
            draw = best_planner[k] + rng.gauss(0.0, planner_sigmas[k])
            th[k] = float(_clamp(draw, lo, hi))

        for k, (lo, hi) in COST_BOUNDS.items():
            draw = math.log(best_costs[k]) + rng.gauss(0.0, cost_sigmas[k])
            th[k] = float(_clamp(math.exp(draw), lo, hi))

        _push_anchor(th)

    return anchors[:n_points], center_theta


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
    extra_log: Optional[Dict[str, Any]] = None,
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
    ALL_RESULTS.append(rec)

    if phase == PHASE1_NAME:
        PHASE1_RESULTS.append(rec)
    elif phase == PHASE2_NAME:
        PHASE2_RESULTS.append(rec)
    elif phase == PHASE3_NAME:
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
    if extra_log:
        log_rec.update(extra_log)

    _append_run_log(log_rec)
    _update_top10_file()


# ============================================================
# Stage 1
# ============================================================
def run_stage1_random_search(
    n_trials: int,
    startup_trials: int,
    critical_crash_multiplier: float,
) -> Tuple[Dict[str, float], float]:
    rng = random.Random(GLOBAL_SEED + 1001)

    startup = generate_lhs_trials(GLOBAL_SEED + 101, startup_trials, COST_BOUNDS)

    best_theta = dict(SAFE_PLANNER)
    best_J = float("inf")

    for i in range(min(n_trials, len(startup))):
        theta = dict(SAFE_PLANNER)
        theta.update(startup[i])

        J, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE1_NAME,
            i,
            theta,
            J,
            info,
            extra_log={
                "proposal_kind": "stage1_startup_lhs",
                "is_global": False,
            },
        )

        print(
            f"[{PHASE1_NAME}][startup {i:03d}] "
            f"obj={J:.6g} mean={info.get('mean_cost', J):.6g} std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(theta)

    for i in range(len(PHASE1_RESULTS), n_trials):
        do_global = rng.random() < STAGE1_GLOBAL_RANDOM_FRACTION

        if do_global or len(PHASE1_RESULTS) == 0:
            theta = sample_theta_global(rng, COST_BOUNDS, fixed_params=SAFE_PLANNER)
            proposal_kind = "stage1_global"
            sigma_log_map = None
        else:
            G1, _ = build_G_L_from_phase1()
            center_rec = _weighted_choice_record(rng, G1)

            sigma_log_map = _cost_sigmas_from_records(
                recs=G1,
                inflate=STAGE1_COST_SIGMA_INFLATE,
                min_frac=STAGE1_COST_LOG_SIGMA_MIN_FRAC,
                max_frac=STAGE1_COST_LOG_SIGMA_MAX_FRAC,
            )

            theta = _sample_local_from_center(
                rng=rng,
                center_theta=center_rec["theta"],
                bounds=COST_BOUNDS,
                sigma_map=sigma_log_map,
                fixed_params=SAFE_PLANNER,
            )
            proposal_kind = "stage1_local"

        J, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE1_NAME,
            i,
            theta,
            J,
            info,
            extra_log={
                "proposal_kind": proposal_kind,
                "is_global": bool(do_global),
                "global_fraction": float(STAGE1_GLOBAL_RANDOM_FRACTION),
                "sigma_log_map": sigma_log_map,
            },
        )

        print(
            f"[{PHASE1_NAME}][trial   {i:03d}] "
            f"kind={proposal_kind:<13} "
            f"obj={J:.6g} mean={info.get('mean_cost', J):.6g} std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(theta)

    return best_theta, best_J


# ============================================================
# Stage 2
# ============================================================
def run_stage2_planner_random(
    n_trials: int,
    startup_trials: int,
    critical_crash_multiplier: float,
    fixed_costs: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    rng = random.Random(GLOBAL_SEED + 2002)

    startup = build_phase2_startup_planner_only(
        fixed_costs=fixed_costs,
        n_points=startup_trials,
        seed=GLOBAL_SEED + 202,
    )

    best_theta = _merge_theta(dict(SAFE_PLANNER), fixed_costs)
    best_J = float("inf")

    for i in range(min(n_trials, len(startup))):
        theta = dict(startup[i])

        J, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE2_NAME,
            i,
            theta,
            J,
            info,
            extra_log={
                "proposal_kind": "stage2_startup_safe_cloud_lhs",
                "is_global": False,
            },
        )

        print(
            f"[{PHASE2_NAME}][startup {i:03d}] "
            f"obj={J:.6g} mean={info.get('mean_cost', J):.6g} std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(theta)

    for i in range(len(PHASE2_RESULTS), n_trials):
        do_global = rng.random() < STAGE2_GLOBAL_RANDOM_FRACTION

        if do_global or len(PHASE2_RESULTS) == 0:
            theta = sample_theta_global(rng, PLANNER_BOUNDS, fixed_params=fixed_costs)
            proposal_kind = "stage2_global"
            planner_sigma_map = None
        else:
            G2, _ = build_G_L_from_phase2()
            center_rec = _weighted_choice_record(rng, G2)
            planner_sigma_map = _planner_sigmas_from_records(
                recs=G2,
                inflate=STAGE2_PLANNER_SIGMA_INFLATE,
                min_frac=STAGE2_PLANNER_SIGMA_MIN_FRAC,
                max_frac=STAGE2_PLANNER_SIGMA_MAX_FRAC,
            )

            theta = _sample_local_from_center(
                rng=rng,
                center_theta=center_rec["theta"],
                bounds=PLANNER_BOUNDS,
                sigma_map=planner_sigma_map,
                fixed_params=fixed_costs,
            )
            proposal_kind = "stage2_local"

        J, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE2_NAME,
            i,
            theta,
            J,
            info,
            extra_log={
                "proposal_kind": proposal_kind,
                "is_global": bool(do_global),
                "global_fraction": float(STAGE2_GLOBAL_RANDOM_FRACTION),
                "planner_sigma_map": planner_sigma_map,
            },
        )

        print(
            f"[{PHASE2_NAME}][trial   {i:03d}] "
            f"kind={proposal_kind:<13} "
            f"obj={J:.6g} mean={info.get('mean_cost', J):.6g} std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(theta)

    return best_theta, best_J


# ============================================================
# Stage 3
# ============================================================
def run_stage3_joint_random(
    n_trials: int,
    startup_trials: int,
    critical_crash_multiplier: float,
    best_stage1_theta: Dict[str, float],
    best_stage2_theta: Dict[str, float],
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    rng = random.Random(GLOBAL_SEED + 3003)

    startup, center_theta = build_phase3_startup_joint_anchors(
        best_stage1_theta=best_stage1_theta,
        best_stage2_theta=best_stage2_theta,
        n_points=startup_trials,
        seed=GLOBAL_SEED + 303,
    )

    best_theta = dict(center_theta)
    best_J = float("inf")

    for i in range(min(n_trials, len(startup))):
        theta = dict(startup[i])

        J, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE3_NAME,
            i,
            theta,
            J,
            info,
            extra_log={
                "proposal_kind": "stage3_startup_anchor",
                "is_global": False,
            },
        )

        print(
            f"[{PHASE3_NAME}][startup {i:03d}] "
            f"obj={J:.6g} mean={info.get('mean_cost', J):.6g} std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(theta)

    for i in range(len(PHASE3_RESULTS), n_trials):
        do_global = rng.random() < STAGE3_GLOBAL_RANDOM_FRACTION

        if do_global or len(PHASE3_RESULTS) == 0:
            theta = sample_theta_global(rng, ALL_BOUNDS, fixed_params=None)
            proposal_kind = "stage3_global"
            planner_sigma_map = None
            cost_sigma_map = None
        else:
            G3, _ = build_G_L_from_phase3()
            center_rec = _weighted_choice_record(rng, G3)

            planner_sigma_map = _planner_sigmas_from_records(
                recs=G3,
                inflate=STAGE3_PLANNER_SIGMA_INFLATE,
                min_frac=STAGE3_PLANNER_SIGMA_MIN_FRAC,
                max_frac=STAGE3_PLANNER_SIGMA_MAX_FRAC,
            )
            cost_sigma_map = _cost_sigmas_from_records(
                recs=G3,
                inflate=STAGE3_COST_SIGMA_INFLATE,
                min_frac=STAGE3_COST_LOG_SIGMA_MIN_FRAC,
                max_frac=STAGE3_COST_LOG_SIGMA_MAX_FRAC,
            )

            sigma_map: Dict[str, float] = {}
            sigma_map.update(planner_sigma_map)
            sigma_map.update(cost_sigma_map)

            theta = _sample_local_from_center(
                rng=rng,
                center_theta=center_rec["theta"],
                bounds=ALL_BOUNDS,
                sigma_map=sigma_map,
                fixed_params=None,
            )
            proposal_kind = "stage3_local"

        J, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE3_NAME,
            i,
            theta,
            J,
            info,
            extra_log={
                "proposal_kind": proposal_kind,
                "is_global": bool(do_global),
                "global_fraction": float(STAGE3_GLOBAL_RANDOM_FRACTION),
                "planner_sigma_map": planner_sigma_map,
                "cost_sigma_map": cost_sigma_map,
            },
        )

        print(
            f"[{PHASE3_NAME}][trial   {i:03d}] "
            f"kind={proposal_kind:<13} "
            f"obj={J:.6g} mean={info.get('mean_cost', J):.6g} std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(theta)

    return best_theta, best_J, center_theta


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
    G1_size: int,
    L1_size: int,
    stage3_center_theta: Dict[str, float],
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
            "phase1_trials": PHASE1_N_TRIALS,
            "phase1_startup": PHASE1_STARTUP,
            "phase2_trials": PHASE2_N_TRIALS,
            "phase2_startup": PHASE2_STARTUP,
            "phase3_trials": PHASE3_N_TRIALS,
            "phase3_startup": PHASE3_STARTUP,
            "total_trials": TOTAL_TRIALS,
            "total_episodes": TOTAL_EPISODES,
            "pure_sim_hours": PURE_SIM_HOURS,
            "safe_planner": SAFE_PLANNER,
            "planner_bounds": PLANNER_BOUNDS,
            "cost_bounds": COST_BOUNDS,
            "good_frac": GOOD_FRAC,
            "phase1_G_size": G1_size,
            "phase1_L_size": L1_size,
            "phase1_critical_crash_multiplier": PHASE1_CRITICAL_CRASH_MULTIPLIER,
            "phase2_critical_crash_multiplier": PHASE2_CRITICAL_CRASH_MULTIPLIER,
            "phase3_critical_crash_multiplier": PHASE3_CRITICAL_CRASH_MULTIPLIER,
            "ucb_weight": UCB_WEIGHT,
            "common_stage_architecture": "phase1_cost_only_phase2_planner_qsdot_phase3_joint",
            "slack_tuning": False,
            "stage3_center_theta": {k: float(v) for k, v in stage3_center_theta.items()},
            "stage3_planner_safe_blend": STAGE3_PLANNER_SAFE_BLEND,
            "top_k_cost_anchors": TOP_K_COST_ANCHORS,
            "top_k_planner_anchors": TOP_K_PLANNER_ANCHORS,
            "phase2_safe_planner_startup_frac": PHASE2_SAFE_PLANNER_STARTUP_FRAC,
            "phase2_safe_planner_sigma_frac_vmax": PHASE2_SAFE_PLANNER_SIGMA_FRAC_VMAX,
            "phase2_safe_planner_sigma_frac_mu": PHASE2_SAFE_PLANNER_SIGMA_FRAC_MU,
            "phase2_safe_planner_sigma_frac_qsdot": PHASE2_SAFE_PLANNER_SIGMA_FRAC_QSDOT,
            "stage1_global_random_fraction": STAGE1_GLOBAL_RANDOM_FRACTION,
            "stage2_global_random_fraction": STAGE2_GLOBAL_RANDOM_FRACTION,
            "stage3_global_random_fraction": STAGE3_GLOBAL_RANDOM_FRACTION,
            "stage1_cost_sigma_inflate": STAGE1_COST_SIGMA_INFLATE,
            "stage1_cost_log_sigma_min_frac": STAGE1_COST_LOG_SIGMA_MIN_FRAC,
            "stage1_cost_log_sigma_max_frac": STAGE1_COST_LOG_SIGMA_MAX_FRAC,
            "stage2_planner_sigma_inflate": STAGE2_PLANNER_SIGMA_INFLATE,
            "stage2_planner_sigma_min_frac": STAGE2_PLANNER_SIGMA_MIN_FRAC,
            "stage2_planner_sigma_max_frac": STAGE2_PLANNER_SIGMA_MAX_FRAC,
            "stage3_planner_sigma_inflate": STAGE3_PLANNER_SIGMA_INFLATE,
            "stage3_planner_sigma_min_frac": STAGE3_PLANNER_SIGMA_MIN_FRAC,
            "stage3_planner_sigma_max_frac": STAGE3_PLANNER_SIGMA_MAX_FRAC,
            "stage3_cost_sigma_inflate": STAGE3_COST_SIGMA_INFLATE,
            "stage3_cost_log_sigma_min_frac": STAGE3_COST_LOG_SIGMA_MIN_FRAC,
            "stage3_cost_log_sigma_max_frac": STAGE3_COST_LOG_SIGMA_MAX_FRAC,
            "crash_penalty": CRASH_PENALTY,
            "crash_time_penalty": CRASH_TIME_PENALTY,
        },
        "results": {
            PHASE1_NAME: {
                "best_objective_cost": float(phase1_J),
                "best_theta": {k: float(v) for k, v in phase1_theta.items()},
            },
            PHASE2_NAME: {
                "best_objective_cost": float(phase2_J),
                "best_theta": {k: float(v) for k, v in phase2_theta.items()},
            },
            PHASE3_NAME: {
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
    global BEST_COST_SO_FAR, BEST_THETA_SO_FAR, EPISODES_DONE, TRIALS_DONE
    global BASE_CONTROL_TEMPLATE

    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(ACADOS_LIB, "ACADOS_LIB")

    BASE_CONTROL_TEMPLATE = _load_json(CONTROL_PARAM_JSON)

    BEST_COST_SO_FAR = float("inf")
    BEST_THETA_SO_FAR = {}
    EPISODES_DONE = 0
    TRIALS_DONE = 0
    ALL_RESULTS.clear()
    PHASE1_RESULTS.clear()
    PHASE2_RESULTS.clear()
    PHASE3_RESULTS.clear()

    for p in [RUN_LOG_PATH, SUMMARY_PATH]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    print("[CFG] CATKIN_WS  =", CATKIN_WS)
    print("[CFG] WS_SRC     =", WS_SRC)
    print("[CFG] ACADOS_LIB =", ACADOS_LIB)
    print("[CFG] MODE       =", PROFILE_NAME)
    print("[CFG] TRACK_SET  =", TRACK_SET)
    print("[CFG] SIM_TIME   =", DEFAULT_SIM_TIME_S, "s")
    print("[CFG] TOTAL_TRIALS   =", TOTAL_TRIALS)
    print("[CFG] TOTAL_EPISODES =", TOTAL_EPISODES)
    print(f"[CFG] PURE_SIM_HOURS ~= {PURE_SIM_HOURS:.3f}")
    print("[CFG] Phase1 =", PHASE1_N_TRIALS, "cost-only random trials")
    print("[CFG] Phase2 =", PHASE2_N_TRIALS, "planner+q_sdot random trials")
    print("[CFG] Phase3 =", PHASE3_N_TRIALS, "joint random trials")
    print(f"[CFG] Phase1 critical_crash_multiplier = {PHASE1_CRITICAL_CRASH_MULTIPLIER}")
    print(f"[CFG] Phase2 critical_crash_multiplier = {PHASE2_CRITICAL_CRASH_MULTIPLIER}")
    print(f"[CFG] Phase3 critical_crash_multiplier = {PHASE3_CRITICAL_CRASH_MULTIPLIER}")
    print(f"[CFG] SAFE_PLANNER = {SAFE_PLANNER}")
    print(f"[CFG] PLANNER_BOUNDS = {PLANNER_BOUNDS}")
    print(f"[CFG] COST_BOUNDS = {COST_BOUNDS}")
    print(f"[CFG] UCB_WEIGHT = {UCB_WEIGHT}")
    print(f"[CFG] STAGE3_PLANNER_SAFE_BLEND = {STAGE3_PLANNER_SAFE_BLEND}")
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

    stage3_center_theta: Dict[str, float] = {}

    try:
        print("\n================ PHASE 1: COST ONLY @ SAFE FULL-COUPLED PLANNER ================\n")
        print("[PHASE1] tune only cost weights")
        print("[PHASE1] fixed: max_vx, mux, muy, q_sdot")
        print("[PHASE1] slack penalties frozen from base JSON\n")

        phase1_theta, phase1_J = run_stage1_random_search(
            n_trials=PHASE1_N_TRIALS,
            startup_trials=PHASE1_STARTUP,
            critical_crash_multiplier=PHASE1_CRITICAL_CRASH_MULTIPLIER,
        )

        G1, L1 = build_G_L_from_phase1()
        print("[PHASE1] G size =", len(G1))
        print("[PHASE1] L size =", len(L1))

        print("\n================ PHASE 2: PLANNER+Q_SDOT @ FIXED BEST COSTS ================\n")
        print("[PHASE2] exact SAFE_PLANNER is in startup")
        print("[PHASE2] plus local cloud around SAFE_PLANNER")
        print("[PHASE2] plus wide planner LHS")
        print("[PHASE2] costs are frozen from best stage1")
        print("[PHASE2] free: max_vx, mux, muy, q_sdot")
        print("[PHASE2] slack penalties still frozen\n")

        fixed_costs = _extract_cost_theta(phase1_theta)
        phase2_theta, phase2_J = run_stage2_planner_random(
            n_trials=PHASE2_N_TRIALS,
            startup_trials=PHASE2_STARTUP,
            critical_crash_multiplier=PHASE2_CRITICAL_CRASH_MULTIPLIER,
            fixed_costs=fixed_costs,
        )

        print("\n================ PHASE 3: JOINT RANDOM SEARCH (ANCHOR STARTUP) ================\n")
        print("[PHASE3] center = best costs from stage1 + blended planner from stage2")
        print("[PHASE3] startup contains center + conservative anchor + planner anchors + cost anchors")
        print("[PHASE3] proper search = mix global / local around good phase-3 points")
        print("[PHASE3] no separate refine stage")
        print("[PHASE3] slack penalties still frozen\n")

        phase3_theta, phase3_J, stage3_center_theta = run_stage3_joint_random(
            n_trials=PHASE3_N_TRIALS,
            startup_trials=PHASE3_STARTUP,
            critical_crash_multiplier=PHASE3_CRITICAL_CRASH_MULTIPLIER,
            best_stage1_theta=phase1_theta,
            best_stage2_theta=phase2_theta,
        )

        save_best_json(BEST_THETA_SO_FAR)
        save_summary(
            phase1_theta=phase1_theta,
            phase1_J=phase1_J,
            phase2_theta=phase2_theta,
            phase2_J=phase2_J,
            phase3_theta=phase3_theta,
            phase3_J=phase3_J,
            G1_size=len(G1),
            L1_size=len(L1),
            stage3_center_theta=stage3_center_theta,
        )

    finally:
        if roscore_p is not None:
            _kill_process_group(roscore_p)

    print("\n================ DONE ================\n")
    print(f"Best global objective cost: {BEST_COST_SO_FAR:.6g}")
    print("Best global params:")
    for k in sorted(BEST_THETA_SO_FAR.keys()):
        lo, hi = ALL_BOUNDS[k]
        print(f"  {k}: {BEST_THETA_SO_FAR[k]:.6g}   (global bounds [{lo}, {hi}])")

    print(f"\nSaved best json: {BEST_JSON_PATH}")
    print(f"Saved top10 file: {TOP10_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved run log: {RUN_LOG_PATH}")
    print(f"\nTotal trials done: {TRIALS_DONE}")
    print(f"Total episodes done: {EPISODES_DONE}")
    print(f"Pure simulation hours target ~= {PURE_SIM_HOURS:.3f}")


if __name__ == "__main__":
    main()
