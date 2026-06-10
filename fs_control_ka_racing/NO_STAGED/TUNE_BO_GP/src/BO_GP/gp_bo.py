#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import random
import signal
import socket
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List


# ============================================================
# Workspace
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_ka_racing/TUNE_BO_GP")
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
# Evaluation setup / profiles
# ============================================================
GLOBAL_SEED = 123
random.seed(GLOBAL_SEED)

DEFAULT_SIM_TIME_S = 60
UCB_WEIGHT = 1.0

# ------------------------------------------------------------
# PROFILE SWITCH
# ------------------------------------------------------------
PROFILE_NAME = "smoke_1track"
PROFILE_NAME = "night_4track"

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

# ============================================================
# Unstaged budget = exact same total as staged
# ============================================================
UNSTAGED_TOTAL_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS
UNSTAGED_STARTUP_TRIALS = PHASE1_STARTUP + PHASE2_STARTUP + PHASE3_STARTUP

TOTAL_TRIALS = UNSTAGED_TOTAL_TRIALS
TOTAL_EPISODES = TOTAL_TRIALS * EPISODES_PER_TRIAL
PURE_SIM_HOURS = TOTAL_EPISODES * DEFAULT_SIM_TIME_S / 3600.0

# ============================================================
# Common crash multiplier
# ============================================================
COMMON_CRITICAL_CRASH_MULTIPLIER = 1.00

# ============================================================
# Safe joint anchor
# ============================================================
SAFE_PLANNER = {
    "velocity_planner.v_max": 8.0,
    "velocity_planner.mux_acc": 0.4,
    "velocity_planner.mux_dec": 0.35,
    "velocity_planner.muy": 0.5,
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
    "mpc.cost.Q_y": (1.0, 100.0),
    "mpc.cost.Q_psi": (1.0, 100.0),
    "mpc.cost.R_ddelta": (1.0, 100.0),
}

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(PLANNER_BOUNDS)
ALL_BOUNDS.update(COST_BOUNDS)

LOG_PARAMS = set(COST_BOUNDS.keys())

# ============================================================
# GP-BO hyperparams
# ============================================================
GP_BASE_ESTIMATOR = "GP"
GP_ACQ_FUNC = "LCB"
GP_ACQ_OPTIMIZER = "sampling"
UNSTAGED_GP_KAPPA = 2.4

# ============================================================
# GP surrogate logging
# ============================================================
GP_SURROGATE_LOG_CLOUD_N = 4096
GP_SURROGATE_LOG_TOPK = 5

# ============================================================
# Cost function
# ============================================================
COST_WEIGHTS = {
    "vs_avg_mps": -2.0,
    "slip_ratio_metric": 175.0,
    "slip_angle_metric": 175.0,
}

CRASH_PENALTY = 160.0
CRASH_TIME_PENALTY = 0.50

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
ROS_MASTER_PORT = 11411
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_tune_gpbo_unstaged_{PROFILE_NAME}_{GLOBAL_SEED}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Outputs
# ============================================================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
TOP10_PATH = os.path.join(SCRIPT_DIR, f"top10_unstaged_gpbo_{PROFILE_NAME}.json")
SUMMARY_PATH = os.path.join(SCRIPT_DIR, f"tuning_summary_unstaged_gpbo_{PROFILE_NAME}.json")
RUN_LOG_PATH = os.path.join(SCRIPT_DIR, f"tuning_run_unstaged_gpbo_{PROFILE_NAME}.jsonl")
BEST_JSON_PATH = os.path.join(
    os.path.dirname(CONTROL_PARAM_JSON),
    f"control_param_best_lti_unstaged_gpbo_{PROFILE_NAME}.json",
)

# ============================================================
# Global state
# ============================================================
EPISODES_DONE = 0
TRIALS_DONE = 0
GLOBAL_CANDIDATE_ID = 0

BEST_OBJECTIVE_SO_FAR = float("inf")
BEST_THETA_SO_FAR: Dict[str, float] = {}

ALL_RESULTS: List[Dict[str, Any]] = []

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
            _set_json_path(data, path, float(max(lo, min(hi, float(val)))))
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
            0.0,
            DEFAULT_SIM_TIME_S - max(0.0, crash_time),
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _theta_key(theta: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, round(float(v), 12)) for k, v in theta.items()))


def _make_safe_joint_anchor() -> Dict[str, float]:
    theta = dict(SAFE_PLANNER)
    for k, (lo, hi) in COST_BOUNDS.items():
        theta[k] = float(math.sqrt(lo * hi))
    return theta


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
    ctrl_p = _popen_roslaunch(CTRL_LAUNCH, {})

    time.sleep(0.6)

    if sim_p.poll() is not None:
        rc = int(sim_p.returncode)
        metrics = {"crashed": 1, "crash_reason": f"sim_launch_failed_rc={rc}", "crash_time_s": -1}
        cost = compute_cost(metrics)
        _kill_process_group(sim_p)
        _kill_process_group(ctrl_p)
        return EpisodeResult(cost, metrics, track_index, True, str(metrics["crash_reason"]))

    if ctrl_p.poll() is not None:
        rc = int(ctrl_p.returncode)
        metrics = {"crashed": 1, "crash_reason": f"ctrl_launch_failed_rc={rc}", "crash_time_s": -1}
        cost = compute_cost(metrics)
        _kill_process_group(sim_p)
        _kill_process_group(ctrl_p)
        return EpisodeResult(cost, metrics, track_index, True, str(metrics["crash_reason"]))

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
        return EpisodeResult(cost, metrics, track_index, True, str(metrics["crash_reason"]))

    metrics = _parse_metrics_csv(METRICS_CSV)
    cost = compute_cost(metrics)
    crashed = int(metrics.get("crashed", 0)) == 1
    reason = str(metrics.get("crash_reason", "")) if crashed else ""

    return EpisodeResult(cost, metrics, track_index, crashed, reason)


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


# ============================================================
# skopt helpers
# ============================================================
def _make_skopt_space(bounds: Dict[str, Tuple[float, float]]):
    from skopt.space import Real

    keys = list(bounds.keys())
    dims = []
    for k in keys:
        lo, hi = bounds[k]
        if k in LOG_PARAMS:
            dims.append(Real(lo, hi, prior="log-uniform", name=k))
        else:
            dims.append(Real(lo, hi, prior="uniform", name=k))
    return keys, dims


def _theta_to_vector(keys: List[str], theta: Dict[str, float]) -> List[float]:
    return [float(theta[k]) for k in keys]


def _vector_to_theta(
    keys: List[str],
    x: List[float],
    fixed_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    theta: Dict[str, float] = {}
    if fixed_params:
        theta.update(fixed_params)
    for i, k in enumerate(keys):
        theta[k] = float(x[i])
    return theta


def _vector_signature(x: List[float], ndigits: int = 12) -> Tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in x)


def _deduplicate_vector(
    x: List[float],
    keys: List[str],
    bounds: Dict[str, Tuple[float, float]],
    seen: set,
    rng: random.Random,
) -> Tuple[List[float], bool]:
    sig = _vector_signature(x)
    if sig not in seen:
        return list(map(float, x)), False

    for _ in range(64):
        theta = sample_theta_global(rng, bounds, fixed_params=None)
        cand = _theta_to_vector(keys, theta)
        sig = _vector_signature(cand)
        if sig not in seen:
            return cand, True

    cand = list(map(float, x))
    for i, k in enumerate(keys):
        lo, hi = bounds[k]
        width = hi - lo
        if k in LOG_PARAMS:
            z = math.log(max(lo, min(hi, cand[i])))
            z += rng.uniform(-0.01, 0.01) * max(1e-9, math.log(hi / lo))
            cand[i] = float(_clamp(math.exp(z), lo, hi))
        else:
            cand[i] = float(_clamp(cand[i] + rng.uniform(-0.01, 0.01) * width, lo, hi))
    return cand, True


# ============================================================
# GP surrogate snapshot logging
# ============================================================
def _merge_fixed_params(
    theta_partial: Dict[str, float],
    fixed_params: Optional[Dict[str, float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if fixed_params:
        out.update({k: float(v) for k, v in fixed_params.items()})
    out.update({k: float(v) for k, v in theta_partial.items()})
    return out


def _candidate_snapshot_payload(
    theta_partial: Dict[str, float],
    fixed_params: Optional[Dict[str, float]],
    pred_mean: float,
    pred_std: float,
    acq_value: float,
) -> Dict[str, Any]:
    return {
        "theta": _merge_fixed_params(theta_partial, fixed_params),
        "pred_mean": float(pred_mean),
        "pred_std": float(pred_std),
        "acq_value": float(acq_value),
    }


def _build_gp_surrogate_snapshot(
    opt,
    keys: List[str],
    bounds: Dict[str, Tuple[float, float]],
    fixed_params: Optional[Dict[str, float]],
    kappa: float,
    seed: int,
) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "surrogate_snapshot_available": False,
    }

    try:
        if len(getattr(opt, "models", [])) == 0:
            snap["surrogate_snapshot_error"] = "no_models"
            return snap

        if len(getattr(opt, "Xi", [])) == 0:
            snap["surrogate_snapshot_error"] = "no_observations"
            return snap

        model = opt.models[-1]

        cloud_theta = generate_lhs_trials(
            seed=seed,
            n_trials=GP_SURROGATE_LOG_CLOUD_N,
            bounds=bounds,
        )
        X_cloud = [_theta_to_vector(keys, th) for th in cloud_theta]

        mu, std = model.predict(X_cloud, return_std=True)
        mu_list = [float(v) for v in mu]
        std_list = [float(max(0.0, v)) for v in std]
        acq_list = [m - kappa * s for m, s in zip(mu_list, std_list)]

        idx_best_mean = min(range(len(mu_list)), key=lambda i: mu_list[i])
        idx_best_acq = min(range(len(acq_list)), key=lambda i: acq_list[i])

        idx_obs_best = min(range(len(opt.yi)), key=lambda i: float(opt.yi[i]))
        obs_best_theta = _vector_to_theta(keys, list(opt.Xi[idx_obs_best]), fixed_params=fixed_params)

        top_mean_idx = sorted(range(len(mu_list)), key=lambda i: mu_list[i])[:GP_SURROGATE_LOG_TOPK]
        top_acq_idx = sorted(range(len(acq_list)), key=lambda i: acq_list[i])[:GP_SURROGATE_LOG_TOPK]

        return {
            "surrogate_snapshot_available": True,
            "surrogate_model_count": int(len(opt.models)),
            "surrogate_cloud_n": int(len(X_cloud)),
            "surrogate_observed_best": {
                "theta": obs_best_theta,
                "observed_objective_cost": float(opt.yi[idx_obs_best]),
            },
            "surrogate_best_by_pred_mean": _candidate_snapshot_payload(
                theta_partial=cloud_theta[idx_best_mean],
                fixed_params=fixed_params,
                pred_mean=mu_list[idx_best_mean],
                pred_std=std_list[idx_best_mean],
                acq_value=acq_list[idx_best_mean],
            ),
            "surrogate_best_by_acquisition": _candidate_snapshot_payload(
                theta_partial=cloud_theta[idx_best_acq],
                fixed_params=fixed_params,
                pred_mean=mu_list[idx_best_acq],
                pred_std=std_list[idx_best_acq],
                acq_value=acq_list[idx_best_acq],
            ),
            "surrogate_topk_by_pred_mean": [
                _candidate_snapshot_payload(
                    theta_partial=cloud_theta[i],
                    fixed_params=fixed_params,
                    pred_mean=mu_list[i],
                    pred_std=std_list[i],
                    acq_value=acq_list[i],
                )
                for i in top_mean_idx
            ],
            "surrogate_topk_by_acquisition": [
                _candidate_snapshot_payload(
                    theta_partial=cloud_theta[i],
                    fixed_params=fixed_params,
                    pred_mean=mu_list[i],
                    pred_std=std_list[i],
                    acq_value=acq_list[i],
                )
                for i in top_acq_idx
            ],
        }

    except Exception as e:
        return {
            "surrogate_snapshot_available": False,
            "surrogate_snapshot_error": str(e),
        }


# ============================================================
# Unstaged startup builder
# ============================================================
def build_unstaged_startup_joint(
    n_points: int,
    seed: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    out: List[Dict[str, float]] = []
    meta: List[Dict[str, Any]] = []
    seen = set()

    def _push(theta: Dict[str, float], source: str) -> None:
        key = _theta_key(theta)
        if key not in seen:
            out.append(dict(theta))
            meta.append({"startup_source": source})
            seen.add(key)

    if n_points <= 0:
        return out, meta

    _push(_make_safe_joint_anchor(), "safe_joint_anchor")

    if len(out) >= n_points:
        return out[:n_points], meta[:n_points]

    remaining = n_points - len(out)
    lhs = generate_lhs_trials(seed + 100, max(remaining * 3, remaining), ALL_BOUNDS)
    for th in lhs:
        _push(th, "joint_wide_lhs")
        if len(out) >= n_points:
            return out[:n_points], meta[:n_points]

    while len(out) < n_points:
        th = sample_theta_global(rng, ALL_BOUNDS, fixed_params=None)
        _push(th, "joint_random_fallback")

    return out[:n_points], meta[:n_points]


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
    global GLOBAL_CANDIDATE_ID, TRIALS_DONE, BEST_OBJECTIVE_SO_FAR, BEST_THETA_SO_FAR

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
        "ucb_weight": float(info.get("ucb_weight", UCB_WEIGHT)),
        "crashes": int(info.get("crashes", 0)),
        "critical_crash_multiplier": float(info.get("critical_crash_multiplier", 1.0)),
        "theta": {k: float(v) for k, v in theta_x.items()},
        "tracks": list(info.get("tracks", [])),
    }
    ALL_RESULTS.append(rec)

    if J < BEST_OBJECTIVE_SO_FAR:
        BEST_OBJECTIVE_SO_FAR = float(J)
        BEST_THETA_SO_FAR = {k: float(v) for k, v in theta_x.items()}

    log_rec = {
        "candidate_id": int(rec["candidate_id"]),
        "phase": str(phase),
        "phase_trial": int(phase_trial),
        "objective_cost": float(J),
        "mean_cost": float(info.get("mean_cost", J)),
        "std_cost": float(info.get("std_cost", 0.0)),
        "robust_cost": float(info.get("robust_cost", J)),
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
# Unstaged GP-BO
# ============================================================
def run_unstaged_gpbo(
    n_trials: int,
    startup_trials: int,
    critical_crash_multiplier: float,
) -> Tuple[Dict[str, float], float]:
    from skopt import Optimizer

    keys, dims = _make_skopt_space(ALL_BOUNDS)

    opt = Optimizer(
        dimensions=dims,
        base_estimator=GP_BASE_ESTIMATOR,
        acq_func=GP_ACQ_FUNC,
        acq_optimizer=GP_ACQ_OPTIMIZER,
        n_initial_points=0,
        initial_point_generator="random",
        random_state=GLOBAL_SEED + 111,
        acq_func_kwargs={"kappa": UNSTAGED_GP_KAPPA},
    )

    rng_dup = random.Random(GLOBAL_SEED + 3111)
    seen_x = set()

    startup, startup_meta = build_unstaged_startup_joint(
        n_points=startup_trials,
        seed=GLOBAL_SEED + 101,
    )

    best_theta = _make_safe_joint_anchor()
    best_J = float("inf")

    for i in range(min(n_trials, len(startup))):
        theta = dict(startup[i])

        x = _theta_to_vector(keys, theta)
        seen_x.add(_vector_signature(x))

        J, info = evaluate_theta(theta, TRACK_SET, DEFAULT_SIM_TIME_S, critical_crash_multiplier)
        opt.tell(x, J)

        surrogate_snap = _build_gp_surrogate_snapshot(
            opt=opt,
            keys=keys,
            bounds=ALL_BOUNDS,
            fixed_params=None,
            kappa=UNSTAGED_GP_KAPPA,
            seed=GLOBAL_SEED + 700000 + i,
        )

        _register_result(
            "unstaged_joint_gpbo",
            i,
            theta,
            J,
            info,
            extra_log={
                "proposal_kind": "unstaged_startup",
                "gp_kappa": UNSTAGED_GP_KAPPA,
                **startup_meta[i],
                **surrogate_snap,
            },
        )

        print(
            f"[unstaged_joint_gpbo][startup {i:03d}] "
            f"obj={J:.6g} mean={info['mean_cost']:.6g} std={info['std_cost']:.6g} "
            f"crashes={info['crashes']}/{info['episodes']} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(theta)

    for i in range(len(startup), n_trials):
        x = opt.ask()
        x, duplicate_fixed = _deduplicate_vector(x, keys, ALL_BOUNDS, seen_x, rng_dup)
        seen_x.add(_vector_signature(x))

        theta = _vector_to_theta(keys, x, fixed_params=None)

        J, info = evaluate_theta(theta, TRACK_SET, DEFAULT_SIM_TIME_S, critical_crash_multiplier)
        opt.tell(x, J)

        surrogate_snap = _build_gp_surrogate_snapshot(
            opt=opt,
            keys=keys,
            bounds=ALL_BOUNDS,
            fixed_params=None,
            kappa=UNSTAGED_GP_KAPPA,
            seed=GLOBAL_SEED + 710000 + i,
        )

        _register_result(
            "unstaged_joint_gpbo",
            i,
            theta,
            J,
            info,
            extra_log={
                "proposal_kind": "unstaged_gpbo",
                "duplicate_fixed": bool(duplicate_fixed),
                "gp_kappa": UNSTAGED_GP_KAPPA,
                **surrogate_snap,
            },
        )

        print(
            f"[unstaged_joint_gpbo][trial   {i:03d}] "
            f"obj={J:.6g} mean={info['mean_cost']:.6g} std={info['std_cost']:.6g} "
            f"crashes={info['crashes']}/{info['episodes']} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(theta)

    return best_theta, best_J


# ============================================================
# Save outputs
# ============================================================
def save_best_json(theta_x: Dict[str, float]) -> None:
    data = _load_json(CONTROL_PARAM_JSON)
    for path, val in theta_x.items():
        _set_json_path(data, path, float(val))
    _save_json_atomic(BEST_JSON_PATH, data)


def save_summary(
    best_theta: Dict[str, float],
    best_J: float,
    startup_anchor_theta: Dict[str, float],
) -> None:
    payload = {
        "config": {
            "mode": PROFILE_NAME,
            "search_mode": "unstaged_joint_gpbo",
            "sim_time_s": DEFAULT_SIM_TIME_S,
            "tracks": TRACK_SET,
            "episodes_per_trial": EPISODES_PER_TRIAL,
            "total_trials": TOTAL_TRIALS,
            "startup_trials": UNSTAGED_STARTUP_TRIALS,
            "total_episodes": TOTAL_EPISODES,
            "pure_sim_hours": PURE_SIM_HOURS,
            "safe_planner": SAFE_PLANNER,
            "startup_safe_joint_anchor": {k: float(v) for k, v in startup_anchor_theta.items()},
            "planner_bounds": PLANNER_BOUNDS,
            "cost_bounds": COST_BOUNDS,
            "common_critical_crash_multiplier": COMMON_CRITICAL_CRASH_MULTIPLIER,
            "gp_base_estimator": GP_BASE_ESTIMATOR,
            "gp_acq_func": GP_ACQ_FUNC,
            "gp_acq_optimizer": GP_ACQ_OPTIMIZER,
            "unstaged_gp_kappa": UNSTAGED_GP_KAPPA,
            "gp_surrogate_log_cloud_n": GP_SURROGATE_LOG_CLOUD_N,
            "gp_surrogate_log_topk": GP_SURROGATE_LOG_TOPK,
            "ucb_weight": UCB_WEIGHT,
            "crash_penalty": CRASH_PENALTY,
            "crash_time_penalty": CRASH_TIME_PENALTY,
            "staged_reference_budget": {
                "phase1_trials": PHASE1_N_TRIALS,
                "phase1_startup": PHASE1_STARTUP,
                "phase2_trials": PHASE2_N_TRIALS,
                "phase2_startup": PHASE2_STARTUP,
                "phase3_trials": PHASE3_N_TRIALS,
                "phase3_startup": PHASE3_STARTUP,
            },
        },
        "results": {
            "unstaged_joint_gpbo": {
                "best_objective_cost": float(best_J),
                "best_theta": {k: float(v) for k, v in best_theta.items()},
            },
            "global_best": {
                "best_objective_cost": float(BEST_OBJECTIVE_SO_FAR),
                "best_theta": {k: float(v) for k, v in BEST_THETA_SO_FAR.items()},
            },
        },
    }
    _save_json_atomic(SUMMARY_PATH, payload)


# ============================================================
# Main
# ============================================================
def main() -> None:
    try:
        import skopt  # noqa: F401
    except ImportError:
        print("Brak skopt. Zainstaluj: pip install scikit-optimize", file=sys.stderr)
        sys.exit(1)

    global BEST_OBJECTIVE_SO_FAR, BEST_THETA_SO_FAR, EPISODES_DONE, TRIALS_DONE, GLOBAL_CANDIDATE_ID

    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")

    BEST_OBJECTIVE_SO_FAR = float("inf")
    BEST_THETA_SO_FAR = {}
    EPISODES_DONE = 0
    TRIALS_DONE = 0
    GLOBAL_CANDIDATE_ID = 0
    ALL_RESULTS.clear()

    for p in [RUN_LOG_PATH, SUMMARY_PATH]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    startup_anchor_theta = _make_safe_joint_anchor()

    print("[CFG] CATKIN_WS =", CATKIN_WS)
    print("[CFG] WS_SRC    =", WS_SRC)
    print("[CFG] PROFILE_NAME =", PROFILE_NAME)
    print("[CFG] MODE = unstaged_joint_gpbo")
    print("[CFG] TRACK_SET =", TRACK_SET)
    print("[CFG] SIM_TIME  =", DEFAULT_SIM_TIME_S, "s")
    print("[CFG] TOTAL_TRIALS =", TOTAL_TRIALS)
    print("[CFG] STARTUP_TRIALS =", UNSTAGED_STARTUP_TRIALS)
    print("[CFG] TOTAL_EPISODES =", TOTAL_EPISODES)
    print(f"[CFG] PURE_SIM_HOURS ~= {PURE_SIM_HOURS:.3f}")
    print(f"[CFG] UCB_WEIGHT = {UCB_WEIGHT}")
    print(f"[CFG] COMMON critical_crash_multiplier = {COMMON_CRITICAL_CRASH_MULTIPLIER}")
    print(f"[CFG] GP_KAPPA = {UNSTAGED_GP_KAPPA}")
    print("[CFG] Safe startup joint anchor =", startup_anchor_theta)
    print("[CFG] PLANNER_BOUNDS =", PLANNER_BOUNDS)
    print("[CFG] COST_BOUNDS =", COST_BOUNDS)
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
        print("\n================ UNSTAGED JOINT GP-BO ================\n")
        print("[UNSTAGED] one optimizer over full ALL_BOUNDS from trial 0")
        print("[UNSTAGED] startup = safe joint anchor + wide joint LHS")
        print("[UNSTAGED] total trial count matched to staged baseline\n")

        best_theta, best_J = run_unstaged_gpbo(
            n_trials=UNSTAGED_TOTAL_TRIALS,
            startup_trials=UNSTAGED_STARTUP_TRIALS,
            critical_crash_multiplier=COMMON_CRITICAL_CRASH_MULTIPLIER,
        )

        save_best_json(BEST_THETA_SO_FAR)
        save_summary(
            best_theta=best_theta,
            best_J=best_J,
            startup_anchor_theta=startup_anchor_theta,
        )

    finally:
        if roscore_p is not None:
            _kill_process_group(roscore_p)

    print("\n================ DONE ================\n")
    print(f"Best global objective cost: {BEST_OBJECTIVE_SO_FAR:.6g}")
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
