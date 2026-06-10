#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import random
import math
import signal
import socket
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List


# ============================================================
# Workspace
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_agh_racing/TUNE_RANDOM_SEARCH")
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
# ACADOS lib needed by dv_control
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

DEFAULT_SIM_TIME_S = int(os.environ.get("TUNE_SIM_TIME_S", "60"))
UCB_WEIGHT = float(os.environ.get("TUNE_UCB_WEIGHT", "1.0"))

# ============================================================
# Profile / budget
# ============================================================
PROFILE_NAME = os.environ.get("TUNE_PROFILE", "night_4track")

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

    PHASE1_N_TRIALS = 48
    PHASE1_STARTUP = 12

    PHASE2_N_TRIALS = 64
    PHASE2_STARTUP = 16

    PHASE3_N_TRIALS = 96
    PHASE3_STARTUP = 24

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


# Optional override: explicit training track list
_track_override = _parse_track_set_env("TUNE_TRACK_SET")
if _track_override is not None:
    TRACK_SET = _track_override

# Optional overrides: phase budgets/startups (keep existing defaults if unset)
PHASE1_N_TRIALS = int(os.environ.get("TUNE_PHASE1_N_TRIALS", str(PHASE1_N_TRIALS)))
PHASE1_STARTUP = int(os.environ.get("TUNE_PHASE1_STARTUP", str(PHASE1_STARTUP)))

PHASE2_N_TRIALS = int(os.environ.get("TUNE_PHASE2_N_TRIALS", str(PHASE2_N_TRIALS)))
PHASE2_STARTUP = int(os.environ.get("TUNE_PHASE2_STARTUP", str(PHASE2_STARTUP)))

PHASE3_N_TRIALS = int(os.environ.get("TUNE_PHASE3_N_TRIALS", str(PHASE3_N_TRIALS)))
PHASE3_STARTUP = int(os.environ.get("TUNE_PHASE3_STARTUP", str(PHASE3_STARTUP)))

EPISODES_PER_TRIAL = len(TRACK_SET)
TOTAL_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS
TOTAL_EPISODES = TOTAL_TRIALS * EPISODES_PER_TRIAL
PURE_SIM_HOURS = TOTAL_EPISODES * DEFAULT_SIM_TIME_S / 3600.0

# ============================================================
# Common critical crash multiplier
# ============================================================
COMMON_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("TUNE_COMMON_CRITICAL_CRASH_MULTIPLIER", "1.0")
)

# ============================================================
# Safe base planner
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

PLANNER_KEYS = list(PLANNER_BOUNDS.keys())
COST_KEYS = list(COST_BOUNDS.keys())

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(PLANNER_BOUNDS)
ALL_BOUNDS.update(COST_BOUNDS)

LOG_PARAMS = set(COST_BOUNDS.keys())

# ============================================================
# Phase names
# ============================================================
PHASE1_NAME = "phase1_cost_only_random"
PHASE2_NAME = "phase2_planner_only_random"
PHASE3_NAME = "phase3_joint_random"

# ============================================================
# Random-search logic
# ============================================================
GOOD_FRAC = 0.30
MIN_G_SIZE = 4

# ---------------- Phase 1 ----------------
PHASE1_GLOBAL_RANDOM_FRACTION = 0.35
PHASE1_DYNAMIC_G_CENTER_PROB = 0.50
PHASE1_LOCAL_COST_LOG_SIGMA = 0.10

# ---------------- Phase 2 ----------------
PHASE2_SAFE_PLANNER_STARTUP_FRAC = 0.35
PHASE2_SAFE_PLANNER_SIGMA_FRAC_VMAX = 0.30
PHASE2_SAFE_PLANNER_SIGMA_FRAC_MU = 0.30

PHASE2_GLOBAL_RANDOM_FRACTION = 0.35
PHASE2_DYNAMIC_G_CENTER_PROB = 0.50
PHASE2_LOCAL_PLANNER_SIGMA_FRAC_VMAX = 0.30
PHASE2_LOCAL_PLANNER_SIGMA_FRAC_MU = 0.30

# ---------------- Phase 3 ----------------
TOP_K_COST_ANCHORS = 3
TOP_K_PLANNER_ANCHORS = 5

PHASE3_GLOBAL_RANDOM_FRACTION = 0.35
PHASE3_DYNAMIC_G_CENTER_PROB = 0.50
PHASE3_PLANNER_SIGMA_FRAC = 0.30
PHASE3_COST_LOG_SIGMA_FRAC = 0.10

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
ROS_MASTER_PORT = int(os.environ.get("TUNE_ROS_MASTER_PORT", "11318"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_tune_random_staged_{PROFILE_NAME}_{GLOBAL_SEED}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Outputs
# ============================================================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
TOP10_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_TOP10_PATH",
        os.path.join(SCRIPT_DIR, f"top10_random_search_tv_staged_{PROFILE_NAME}.json"),
    )
)
SUMMARY_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_SUMMARY_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_summary_random_search_tv_staged_{PROFILE_NAME}.json"),
    )
)
RUN_LOG_PATH = os.path.abspath(
    os.environ.get(
        "TUNE_RUN_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_run_random_search_tv_staged_{PROFILE_NAME}.jsonl"),
    )
)
BEST_JSON_PATH = os.path.join(
    os.path.dirname(CONTROL_PARAM_JSON),
    f"control_param_best_lti_3stage_random_search_tv_{PROFILE_NAME}.json",
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

PHASE1_STARTUP_ANCHORS: List[Dict[str, float]] = []
PHASE2_STARTUP_ANCHORS: List[Dict[str, float]] = []
PHASE3_STARTUP_ANCHORS: List[Dict[str, float]] = []

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


def _theta_key(theta: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, round(float(v), 12)) for k, v in theta.items()))


def _merge_theta(planner_theta: Dict[str, float], cost_theta: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in PLANNER_KEYS:
        out[k] = float(planner_theta[k])
    for k in COST_KEYS:
        out[k] = float(cost_theta[k])
    return out


def _extract_cost_theta(theta: Dict[str, float]) -> Dict[str, float]:
    return {k: float(theta[k]) for k in COST_KEYS}


def _extract_planner_theta(theta: Dict[str, float]) -> Dict[str, float]:
    return {k: float(theta[k]) for k in PLANNER_KEYS}


def _result_objective(rec: Dict[str, Any]) -> float:
    return float(rec.get("objective_cost", rec.get("robust_cost", rec.get("mean_cost", float("inf")))))


def _top_records(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    ranked = sorted(results, key=_result_objective)
    return ranked[:max(0, min(k, len(ranked)))]


def _metric_float_first(metrics: Dict[str, Any], names: List[str], default: float = 0.0) -> float:
    for name in names:
        if name in metrics:
            try:
                return float(metrics[name])
            except Exception:
                pass
    return float(default)


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
    critical_crash_multiplier: float = COMMON_CRITICAL_CRASH_MULTIPLIER,
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
    critical_crash_multiplier: float = COMMON_CRITICAL_CRASH_MULTIPLIER,
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


def generate_lhs_trials(seed: int, n_trials: int, bounds: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
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
    ranked = sorted(recs, key=_result_objective)
    weights = [1.0 / (i + 1) for i in range(len(ranked))]
    idx = rng.choices(range(len(ranked)), weights=weights, k=1)[0]
    return ranked[idx]


def _weighted_choice_theta(rng: random.Random, thetas: List[Dict[str, float]]) -> Dict[str, float]:
    if not thetas:
        raise ValueError("Empty theta pool.")
    idx = rng.randrange(len(thetas))
    return dict(thetas[idx])


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


def _fixed_cost_sigma_map(sigma_log: float) -> Dict[str, float]:
    return {k: float(sigma_log) for k in COST_KEYS}


def _fixed_planner_sigma_map(vmax_frac: float, mu_frac: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, (lo, hi) in PLANNER_BOUNDS.items():
        frac = vmax_frac if k == "velocity_planner.v_max" else mu_frac
        out[k] = float(frac * (hi - lo))
    return out


# ============================================================
# G / L construction
# ============================================================
def build_G_L_from_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ranked = sorted(results, key=_result_objective)
    n = len(ranked)

    if n == 0:
        return [], []

    g_n = max(MIN_G_SIZE, int(math.ceil(GOOD_FRAC * n)))
    g_n = min(g_n, n)
    return ranked[:g_n], ranked[g_n:]


def build_G_L_from_phase1() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_G_L_from_results(PHASE1_RESULTS)


def build_G_L_from_phase2() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_G_L_from_results(PHASE2_RESULTS)


def build_G_L_from_phase3() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_G_L_from_results(PHASE3_RESULTS)


# ============================================================
# Stage-specific startup builders
# ============================================================
def build_phase1_startup_cost_only(
    n_points: int,
    seed: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for th_cost in generate_lhs_trials(seed, n_points, COST_BOUNDS):
        out.append(
            {
                "theta": _merge_theta(dict(SAFE_PLANNER), th_cost),
                "proposal_kind": "phase1_startup_cost_lhs",
            }
        )
    return out


def build_phase2_startup_planner_only(
    fixed_costs: Dict[str, float],
    n_points: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    out: List[Dict[str, Any]] = []
    seen = set()

    def _push(planner_theta: Dict[str, float], proposal_kind: str) -> None:
        full = _merge_theta(planner_theta, fixed_costs)
        key = _theta_key(full)
        if key not in seen:
            out.append(
                {
                    "theta": full,
                    "proposal_kind": proposal_kind,
                }
            )
            seen.add(key)

    _push(dict(SAFE_PLANNER), "phase2_startup_safe_anchor")

    target_safe_block = int(round(PHASE2_SAFE_PLANNER_STARTUP_FRAC * n_points))
    target_safe_block = max(1, min(target_safe_block, n_points))
    n_local_safe = max(0, target_safe_block - 1)

    for _ in range(n_local_safe):
        planner_theta: Dict[str, float] = {}
        for k, (lo, hi) in PLANNER_BOUNDS.items():
            frac = (
                PHASE2_SAFE_PLANNER_SIGMA_FRAC_VMAX
                if k == "velocity_planner.v_max"
                else PHASE2_SAFE_PLANNER_SIGMA_FRAC_MU
            )
            sigma = frac * (hi - lo)
            draw = SAFE_PLANNER[k] + rng.gauss(0.0, sigma)
            planner_theta[k] = float(_clamp(draw, lo, hi))
        _push(planner_theta, "phase2_startup_safe_cloud")
        if len(out) >= n_points:
            return out[:n_points]

    remaining = n_points - len(out)
    if remaining > 0:
        planner_lhs = generate_lhs_trials(seed + 100, max(remaining * 3, remaining), PLANNER_BOUNDS)
        for th in planner_lhs:
            _push(th, "phase2_startup_planner_lhs")
            if len(out) >= n_points:
                return out[:n_points]

    while len(out) < n_points:
        planner_theta = sample_theta_global(rng, PLANNER_BOUNDS)
        _push(planner_theta, "phase2_startup_planner_global")

    return out[:n_points]


def build_phase3_startup_joint_anchors(
    best_stage1_theta: Dict[str, float],
    best_stage2_theta: Dict[str, float],
    n_points: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rng = random.Random(seed)

    best_costs = _extract_cost_theta(best_stage1_theta)
    best_planner = _extract_planner_theta(best_stage2_theta)
    center_theta = _merge_theta(best_planner, best_costs)

    out: List[Dict[str, Any]] = []
    seen = set()

    def _push(theta: Dict[str, float], proposal_kind: str) -> None:
        key = _theta_key(theta)
        if key not in seen:
            out.append(
                {
                    "theta": dict(theta),
                    "proposal_kind": proposal_kind,
                }
            )
            seen.add(key)

    _push(center_theta, "phase3_startup_center")
    _push(_merge_theta(dict(SAFE_PLANNER), best_costs), "phase3_startup_safe_plus_best_costs")

    for rec in _top_records(PHASE2_RESULTS, TOP_K_PLANNER_ANCHORS):
        planner_i = _extract_planner_theta(rec["theta"])
        _push(_merge_theta(planner_i, best_costs), "phase3_startup_top_planner_anchor")

    for rec in _top_records(PHASE1_RESULTS, TOP_K_COST_ANCHORS):
        costs_i = _extract_cost_theta(rec["theta"])
        _push(_merge_theta(best_planner, costs_i), "phase3_startup_top_cost_anchor")

    planner_sigmas = _fixed_planner_sigma_map(PHASE3_PLANNER_SIGMA_FRAC, PHASE3_PLANNER_SIGMA_FRAC)
    cost_sigmas = _fixed_cost_sigma_map(PHASE3_COST_LOG_SIGMA_FRAC)

    sigma_map_joint: Dict[str, float] = {}
    sigma_map_joint.update(planner_sigmas)
    sigma_map_joint.update(cost_sigmas)

    while len(out) < n_points:
        th = _sample_local_from_center(
            rng=rng,
            center_theta=center_theta,
            bounds=ALL_BOUNDS,
            sigma_map=sigma_map_joint,
            fixed_params=None,
        )
        _push(th, "phase3_startup_joint_jitter")

    return out[:n_points], center_theta


# ============================================================
# Logs / ranking
# ============================================================
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
    extra_log: Optional[Dict[str, Any]] = None,
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
    ALL_RESULTS.append(rec)

    if phase == PHASE1_NAME:
        PHASE1_RESULTS.append(rec)
    elif phase == PHASE2_NAME:
        PHASE2_RESULTS.append(rec)
    elif phase == PHASE3_NAME:
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
    if extra_log:
        log_rec.update(extra_log)

    _append_run_log(log_rec)
    _update_top10_file()


# ============================================================
# Phase 1: cost-only random search
# ============================================================
def run_phase1_cost_only_random(
    n_trials: int,
    startup_trials: int,
    critical_crash_multiplier: float,
) -> Tuple[Dict[str, float], float]:
    global PHASE1_STARTUP_ANCHORS

    rng = random.Random(GLOBAL_SEED + 1001)

    startup = build_phase1_startup_cost_only(
        n_points=startup_trials,
        seed=GLOBAL_SEED + 101,
    )
    PHASE1_STARTUP_ANCHORS = [dict(x["theta"]) for x in startup]

    cost_sigma_map = _fixed_cost_sigma_map(PHASE1_LOCAL_COST_LOG_SIGMA)

    best_theta = _merge_theta(
        dict(SAFE_PLANNER),
        {k: float(math.sqrt(COST_BOUNDS[k][0] * COST_BOUNDS[k][1])) for k in COST_KEYS},
    )
    best_obj = float("inf")

    for i in range(min(n_trials, len(startup))):
        theta = dict(startup[i]["theta"])
        proposal_kind = str(startup[i]["proposal_kind"])

        obj, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE1_NAME,
            i,
            theta,
            obj,
            info,
            extra_log={
                "proposal_kind": proposal_kind,
                "is_global": False,
                "is_startup": True,
            },
        )

        print(
            f"[{PHASE1_NAME}][startup {i:03d}] "
            f"kind={proposal_kind:<28} "
            f"obj={obj:.6g} mean={info.get('mean_cost', float('nan')):.6g} "
            f"std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)}"
        )

        if obj < best_obj:
            best_obj = float(obj)
            best_theta = dict(theta)

    for i in range(len(PHASE1_RESULTS), n_trials):
        do_global = (rng.random() < PHASE1_GLOBAL_RANDOM_FRACTION)

        if do_global or len(PHASE1_RESULTS) == 0:
            theta = sample_theta_global(rng, COST_BOUNDS, fixed_params=dict(SAFE_PLANNER))
            proposal_kind = "phase1_global"
        else:
            use_dynamic_g = (len(PHASE1_RESULTS) > 0) and (rng.random() < PHASE1_DYNAMIC_G_CENTER_PROB)

            if use_dynamic_g:
                G, _ = build_G_L_from_phase1()
                if len(G) > 0:
                    center_theta = _weighted_choice_record(rng, G)["theta"]
                    proposal_kind = "phase1_local_g"
                else:
                    center_theta = _weighted_choice_theta(rng, PHASE1_STARTUP_ANCHORS)
                    proposal_kind = "phase1_local_anchor_fallback"
            else:
                center_theta = _weighted_choice_theta(rng, PHASE1_STARTUP_ANCHORS)
                proposal_kind = "phase1_local_anchor"

            theta = _sample_local_from_center(
                rng=rng,
                center_theta=center_theta,
                bounds=COST_BOUNDS,
                sigma_map=cost_sigma_map,
                fixed_params=dict(SAFE_PLANNER),
            )

        obj, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE1_NAME,
            i,
            theta,
            obj,
            info,
            extra_log={
                "proposal_kind": proposal_kind,
                "is_global": bool(do_global),
                "is_startup": False,
                "global_fraction": float(PHASE1_GLOBAL_RANDOM_FRACTION),
                "dynamic_g_center_prob": float(PHASE1_DYNAMIC_G_CENTER_PROB),
                "cost_sigma_log": float(PHASE1_LOCAL_COST_LOG_SIGMA),
            },
        )

        print(
            f"[{PHASE1_NAME}][trial   {i:03d}] "
            f"kind={proposal_kind:<28} "
            f"obj={obj:.6g} mean={info.get('mean_cost', float('nan')):.6g} "
            f"std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)}"
        )

        if obj < best_obj:
            best_obj = float(obj)
            best_theta = dict(theta)

    return best_theta, best_obj


# ============================================================
# Phase 2: planner-only random search
# ============================================================
def run_phase2_planner_only_random(
    n_trials: int,
    startup_trials: int,
    critical_crash_multiplier: float,
    fixed_costs: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    global PHASE2_STARTUP_ANCHORS

    rng = random.Random(GLOBAL_SEED + 2002)

    startup = build_phase2_startup_planner_only(
        fixed_costs=fixed_costs,
        n_points=startup_trials,
        seed=GLOBAL_SEED + 202,
    )
    PHASE2_STARTUP_ANCHORS = [dict(x["theta"]) for x in startup]

    planner_sigma_map = _fixed_planner_sigma_map(
        PHASE2_LOCAL_PLANNER_SIGMA_FRAC_VMAX,
        PHASE2_LOCAL_PLANNER_SIGMA_FRAC_MU,
    )

    best_theta = _merge_theta(dict(SAFE_PLANNER), fixed_costs)
    best_obj = float("inf")

    for i in range(min(n_trials, len(startup))):
        theta = dict(startup[i]["theta"])
        proposal_kind = str(startup[i]["proposal_kind"])

        obj, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE2_NAME,
            i,
            theta,
            obj,
            info,
            extra_log={
                "proposal_kind": proposal_kind,
                "is_global": False,
                "is_startup": True,
            },
        )

        print(
            f"[{PHASE2_NAME}][startup {i:03d}] "
            f"kind={proposal_kind:<28} "
            f"obj={obj:.6g} mean={info.get('mean_cost', float('nan')):.6g} "
            f"std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)}"
        )

        if obj < best_obj:
            best_obj = float(obj)
            best_theta = dict(theta)

    for i in range(len(PHASE2_RESULTS), n_trials):
        do_global = (rng.random() < PHASE2_GLOBAL_RANDOM_FRACTION)

        if do_global or len(PHASE2_RESULTS) == 0:
            theta = sample_theta_global(rng, PLANNER_BOUNDS, fixed_params=fixed_costs)
            proposal_kind = "phase2_global"
        else:
            use_dynamic_g = (len(PHASE2_RESULTS) > 0) and (rng.random() < PHASE2_DYNAMIC_G_CENTER_PROB)

            if use_dynamic_g:
                G, _ = build_G_L_from_phase2()
                if len(G) > 0:
                    center_theta = _weighted_choice_record(rng, G)["theta"]
                    proposal_kind = "phase2_local_g"
                else:
                    center_theta = _weighted_choice_theta(rng, PHASE2_STARTUP_ANCHORS)
                    proposal_kind = "phase2_local_anchor_fallback"
            else:
                center_theta = _weighted_choice_theta(rng, PHASE2_STARTUP_ANCHORS)
                proposal_kind = "phase2_local_anchor"

            theta = _sample_local_from_center(
                rng=rng,
                center_theta=center_theta,
                bounds=PLANNER_BOUNDS,
                sigma_map=planner_sigma_map,
                fixed_params=fixed_costs,
            )

        obj, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE2_NAME,
            i,
            theta,
            obj,
            info,
            extra_log={
                "proposal_kind": proposal_kind,
                "is_global": bool(do_global),
                "is_startup": False,
                "global_fraction": float(PHASE2_GLOBAL_RANDOM_FRACTION),
                "dynamic_g_center_prob": float(PHASE2_DYNAMIC_G_CENTER_PROB),
                "planner_sigma_frac_vmax": float(PHASE2_LOCAL_PLANNER_SIGMA_FRAC_VMAX),
                "planner_sigma_frac_mu": float(PHASE2_LOCAL_PLANNER_SIGMA_FRAC_MU),
            },
        )

        print(
            f"[{PHASE2_NAME}][trial   {i:03d}] "
            f"kind={proposal_kind:<28} "
            f"obj={obj:.6g} mean={info.get('mean_cost', float('nan')):.6g} "
            f"std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)}"
        )

        if obj < best_obj:
            best_obj = float(obj)
            best_theta = dict(theta)

    return best_theta, best_obj


# ============================================================
# Phase 3: joint random search
# ============================================================
def run_phase3_joint_random(
    n_trials: int,
    startup_trials: int,
    critical_crash_multiplier: float,
    best_stage1_theta: Dict[str, float],
    best_stage2_theta: Dict[str, float],
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    global PHASE3_STARTUP_ANCHORS

    rng = random.Random(GLOBAL_SEED + 3003)

    startup, center_theta = build_phase3_startup_joint_anchors(
        best_stage1_theta=best_stage1_theta,
        best_stage2_theta=best_stage2_theta,
        n_points=startup_trials,
        seed=GLOBAL_SEED + 303,
    )
    PHASE3_STARTUP_ANCHORS = [dict(x["theta"]) for x in startup]

    planner_sigma_map = _fixed_planner_sigma_map(PHASE3_PLANNER_SIGMA_FRAC, PHASE3_PLANNER_SIGMA_FRAC)
    cost_sigma_map = _fixed_cost_sigma_map(PHASE3_COST_LOG_SIGMA_FRAC)

    sigma_map_joint: Dict[str, float] = {}
    sigma_map_joint.update(planner_sigma_map)
    sigma_map_joint.update(cost_sigma_map)

    best_theta = dict(center_theta)
    best_obj = float("inf")

    for i in range(min(n_trials, len(startup))):
        theta = dict(startup[i]["theta"])
        proposal_kind = str(startup[i]["proposal_kind"])

        obj, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE3_NAME,
            i,
            theta,
            obj,
            info,
            extra_log={
                "proposal_kind": proposal_kind,
                "is_global": False,
                "is_startup": True,
            },
        )

        print(
            f"[{PHASE3_NAME}][startup {i:03d}] "
            f"kind={proposal_kind:<28} "
            f"obj={obj:.6g} mean={info.get('mean_cost', float('nan')):.6g} "
            f"std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)}"
        )

        if obj < best_obj:
            best_obj = float(obj)
            best_theta = dict(theta)

    for i in range(len(PHASE3_RESULTS), n_trials):
        do_global = (rng.random() < PHASE3_GLOBAL_RANDOM_FRACTION)

        if do_global or len(PHASE3_RESULTS) == 0:
            theta = sample_theta_global(rng, ALL_BOUNDS, fixed_params=None)
            proposal_kind = "phase3_global"
        else:
            use_dynamic_g = (len(PHASE3_RESULTS) > 0) and (rng.random() < PHASE3_DYNAMIC_G_CENTER_PROB)

            if use_dynamic_g:
                G, _ = build_G_L_from_phase3()
                if len(G) > 0:
                    center = _weighted_choice_record(rng, G)["theta"]
                    proposal_kind = "phase3_local_g"
                else:
                    center = _weighted_choice_theta(rng, PHASE3_STARTUP_ANCHORS)
                    proposal_kind = "phase3_local_anchor_fallback"
            else:
                center = _weighted_choice_theta(rng, PHASE3_STARTUP_ANCHORS)
                proposal_kind = "phase3_local_anchor"

            theta = _sample_local_from_center(
                rng=rng,
                center_theta=center,
                bounds=ALL_BOUNDS,
                sigma_map=sigma_map_joint,
                fixed_params=None,
            )

        obj, info = evaluate_theta(
            theta_x=theta,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            PHASE3_NAME,
            i,
            theta,
            obj,
            info,
            extra_log={
                "proposal_kind": proposal_kind,
                "is_global": bool(do_global),
                "is_startup": False,
                "global_fraction": float(PHASE3_GLOBAL_RANDOM_FRACTION),
                "dynamic_g_center_prob": float(PHASE3_DYNAMIC_G_CENTER_PROB),
                "planner_sigma_frac": float(PHASE3_PLANNER_SIGMA_FRAC),
                "cost_sigma_log": float(PHASE3_COST_LOG_SIGMA_FRAC),
            },
        )

        print(
            f"[{PHASE3_NAME}][trial   {i:03d}] "
            f"kind={proposal_kind:<28} "
            f"obj={obj:.6g} mean={info.get('mean_cost', float('nan')):.6g} "
            f"std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)}"
        )

        if obj < best_obj:
            best_obj = float(obj)
            best_theta = dict(theta)

    return best_theta, best_obj, center_theta


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
    G1_size: int,
    L1_size: int,
    G2_size: int,
    L2_size: int,
    G3_size: int,
    L3_size: int,
    phase3_center_theta: Dict[str, float],
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
            "phase1_startup": PHASE1_STARTUP,
            "phase2_trials": PHASE2_N_TRIALS,
            "phase2_startup": PHASE2_STARTUP,
            "phase3_trials": PHASE3_N_TRIALS,
            "phase3_startup": PHASE3_STARTUP,
            "total_trials": TOTAL_TRIALS,
            "total_episodes": TOTAL_EPISODES,
            "pure_sim_hours": PURE_SIM_HOURS,
            "ucb_weight": UCB_WEIGHT,
            "common_critical_crash_multiplier": COMMON_CRITICAL_CRASH_MULTIPLIER,
            "safe_planner": SAFE_PLANNER,
            "planner_bounds": PLANNER_BOUNDS,
            "cost_bounds": COST_BOUNDS,
            "good_frac": GOOD_FRAC,
            "phase1_G_size": G1_size,
            "phase1_L_size": L1_size,
            "phase2_G_size": G2_size,
            "phase2_L_size": L2_size,
            "phase3_G_size": G3_size,
            "phase3_L_size": L3_size,
            "phase1_global_random_fraction": PHASE1_GLOBAL_RANDOM_FRACTION,
            "phase1_dynamic_g_center_prob": PHASE1_DYNAMIC_G_CENTER_PROB,
            "phase1_local_cost_log_sigma": PHASE1_LOCAL_COST_LOG_SIGMA,
            "phase2_safe_planner_startup_frac": PHASE2_SAFE_PLANNER_STARTUP_FRAC,
            "phase2_safe_planner_sigma_frac_vmax": PHASE2_SAFE_PLANNER_SIGMA_FRAC_VMAX,
            "phase2_safe_planner_sigma_frac_mu": PHASE2_SAFE_PLANNER_SIGMA_FRAC_MU,
            "phase2_global_random_fraction": PHASE2_GLOBAL_RANDOM_FRACTION,
            "phase2_dynamic_g_center_prob": PHASE2_DYNAMIC_G_CENTER_PROB,
            "phase2_local_planner_sigma_frac_vmax": PHASE2_LOCAL_PLANNER_SIGMA_FRAC_VMAX,
            "phase2_local_planner_sigma_frac_mu": PHASE2_LOCAL_PLANNER_SIGMA_FRAC_MU,
            "top_k_cost_anchors": TOP_K_COST_ANCHORS,
            "top_k_planner_anchors": TOP_K_PLANNER_ANCHORS,
            "phase3_global_random_fraction": PHASE3_GLOBAL_RANDOM_FRACTION,
            "phase3_dynamic_g_center_prob": PHASE3_DYNAMIC_G_CENTER_PROB,
            "phase3_planner_sigma_frac": PHASE3_PLANNER_SIGMA_FRAC,
            "phase3_cost_log_sigma_frac": PHASE3_COST_LOG_SIGMA_FRAC,
            "phase3_center_theta": {k: float(v) for k, v in phase3_center_theta.items()},
            "phase1_startup_anchor_count": len(PHASE1_STARTUP_ANCHORS),
            "phase2_startup_anchor_count": len(PHASE2_STARTUP_ANCHORS),
            "phase3_startup_anchor_count": len(PHASE3_STARTUP_ANCHORS),
            "run_log_path": RUN_LOG_PATH,
        },
        "results": {
            PHASE1_NAME: {
                "best_objective": float(phase1_obj),
                "best_theta": {k: float(v) for k, v in phase1_theta.items()},
            },
            PHASE2_NAME: {
                "best_objective": float(phase2_obj),
                "best_theta": {k: float(v) for k, v in phase2_theta.items()},
            },
            PHASE3_NAME: {
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
    global BEST_OBJECTIVE_SO_FAR, BEST_THETA_SO_FAR, EPISODES_DONE, TRIALS_DONE
    global PHASE1_STARTUP_ANCHORS, PHASE2_STARTUP_ANCHORS, PHASE3_STARTUP_ANCHORS

    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(ACADOS_LIB, "ACADOS_LIB")

    BEST_OBJECTIVE_SO_FAR = float("inf")
    BEST_THETA_SO_FAR = {}
    EPISODES_DONE = 0
    TRIALS_DONE = 0

    PHASE1_STARTUP_ANCHORS = []
    PHASE2_STARTUP_ANCHORS = []
    PHASE3_STARTUP_ANCHORS = []

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
    print(f"[CFG] COMMON critical_crash_multiplier = {COMMON_CRITICAL_CRASH_MULTIPLIER}")
    print("[CFG] PHASE1 =", PHASE1_N_TRIALS, "random-search trials")
    print("[CFG] PHASE2 =", PHASE2_N_TRIALS, "random-search trials")
    print("[CFG] PHASE3 =", PHASE3_N_TRIALS, "random-search trials")
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

    phase3_center_theta: Dict[str, float] = {}

    try:
        print("\n================ PHASE 1: COST ONLY @ SAFE PLANNER ================\n")
        print("[PHASE1] startup = cost LHS przy zamrożonym SAFE_PLANNER")
        print("[PHASE1] main = global cost-random + local around G / startup anchors\n")

        phase1_theta, phase1_obj = run_phase1_cost_only_random(
            n_trials=PHASE1_N_TRIALS,
            startup_trials=PHASE1_STARTUP,
            critical_crash_multiplier=COMMON_CRITICAL_CRASH_MULTIPLIER,
        )

        G1, L1 = build_G_L_from_phase1()
        print("[PHASE1] G size =", len(G1))
        print("[PHASE1] L size =", len(L1))

        print("\n================ PHASE 2: PLANNER ONLY @ FIXED BEST COSTS ================\n")
        print("[PHASE2] startup = SAFE_PLANNER + safe cloud + planner LHS")
        print("[PHASE2] main = global planner-random + local around G / startup anchors")
        print("[PHASE2] koszty są zamrożone na best from phase1\n")

        fixed_costs = _extract_cost_theta(phase1_theta)
        phase2_theta, phase2_obj = run_phase2_planner_only_random(
            n_trials=PHASE2_N_TRIALS,
            startup_trials=PHASE2_STARTUP,
            critical_crash_multiplier=COMMON_CRITICAL_CRASH_MULTIPLIER,
            fixed_costs=fixed_costs,
        )

        G2, L2 = build_G_L_from_phase2()
        print("[PHASE2] G size =", len(G2))
        print("[PHASE2] L size =", len(L2))

        print("\n================ PHASE 3: JOINT RANDOM SEARCH (ANCHOR STARTUP) ================\n")
        print("[PHASE3] startup zawiera center z best planner+best costs")
        print("[PHASE3] oraz SAFE_PLANNER+best_costs, top planners z phase2 i top costs z phase1")
        print("[PHASE3] main = global joint-random + local around G / startup anchors\n")

        phase3_theta, phase3_obj, phase3_center_theta = run_phase3_joint_random(
            n_trials=PHASE3_N_TRIALS,
            startup_trials=PHASE3_STARTUP,
            critical_crash_multiplier=COMMON_CRITICAL_CRASH_MULTIPLIER,
            best_stage1_theta=phase1_theta,
            best_stage2_theta=phase2_theta,
        )

        G3, L3 = build_G_L_from_phase3()
        print("[PHASE3] G size =", len(G3))
        print("[PHASE3] L size =", len(L3))

        save_best_json(BEST_THETA_SO_FAR)
        save_summary(
            phase1_theta=phase1_theta,
            phase1_obj=phase1_obj,
            phase2_theta=phase2_theta,
            phase2_obj=phase2_obj,
            phase3_theta=phase3_theta,
            phase3_obj=phase3_obj,
            G1_size=len(G1),
            L1_size=len(L1),
            G2_size=len(G2),
            L2_size=len(L2),
            G3_size=len(G3),
            L3_size=len(L3),
            phase3_center_theta=phase3_center_theta,
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
    print(f"\nTotal trials done: {TRIALS_DONE}")
    print(f"Total episodes done: {EPISODES_DONE}")
    print(f"Pure simulation hours target ~= {PURE_SIM_HOURS:.3f}")


if __name__ == "__main__":
    main()
