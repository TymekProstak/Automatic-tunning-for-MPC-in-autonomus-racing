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
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List


# ============================================================
# Workspace
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_ka_racing/TUNE_BO_TPO")
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
# Evaluation setup / profile presets
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
# Unstaged budget = same total as staged
# ============================================================
UNSTAGED_TOTAL_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS
UNSTAGED_STARTUP_TRIALS = PHASE1_STARTUP + PHASE2_STARTUP + PHASE3_STARTUP

TOTAL_TRIALS = UNSTAGED_TOTAL_TRIALS
TOTAL_EPISODES = TOTAL_TRIALS * EPISODES_PER_TRIAL
PURE_SIM_HOURS = TOTAL_EPISODES * DEFAULT_SIM_TIME_S / 3600.0

# ============================================================
# Common critical crash multiplier
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
# TPE / unstaged logic
# ============================================================
GOOD_FRAC = 0.30
MIN_G_SIZE = 4

# ============================================================
# TPE candidate generation / KDE
# ============================================================
TPE_N_EI_CANDIDATES = 96
KDE_BW_MIN_FRAC = 0.03
KDE_BW_MAX_FRAC = 0.35
KDE_BW_FALLBACK_FRAC = 0.12
LOGPDF_FLOOR = -1e12

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
ROS_MASTER_PORT = 11412
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_tune_unstaged_{PROFILE_NAME}_{GLOBAL_SEED}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Outputs
# ============================================================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
TOP10_PATH = os.path.join(SCRIPT_DIR, f"top10_unstaged_tpe_{PROFILE_NAME}.json")
SUMMARY_PATH = os.path.join(SCRIPT_DIR, f"tuning_summary_unstaged_tpe_{PROFILE_NAME}.json")
RUN_LOG_PATH = os.path.join(SCRIPT_DIR, f"tuning_run_unstaged_tpe_{PROFILE_NAME}.jsonl")
BEST_JSON_PATH = os.path.join(
    os.path.dirname(CONTROL_PARAM_JSON),
    f"control_param_best_lti_unstaged_tpe_{PROFILE_NAME}.json"
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
    ctrl_p = _popen_roslaunch(CTRL_LAUNCH, {})

    time.sleep(0.6)

    if sim_p.poll() is not None:
        rc = int(sim_p.returncode)
        metrics = {"crashed": 1, "crash_reason": f"sim_launch_failed_rc={rc}", "crash_time_s": -1}
        cost = compute_cost(metrics)
        _kill_process_group(sim_p)
        _kill_process_group(ctrl_p)
        return EpisodeResult(
            cost=cost,
            metrics=metrics,
            track_index=track_index,
            crashed=True,
            crash_reason=str(metrics["crash_reason"]),
        )

    if ctrl_p.poll() is not None:
        rc = int(ctrl_p.returncode)
        metrics = {"crashed": 1, "crash_reason": f"ctrl_launch_failed_rc={rc}", "crash_time_s": -1}
        cost = compute_cost(metrics)
        _kill_process_group(sim_p)
        _kill_process_group(ctrl_p)
        return EpisodeResult(
            cost=cost,
            metrics=metrics,
            track_index=track_index,
            crashed=True,
            crash_reason=str(metrics["crash_reason"]),
        )

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
        return EpisodeResult(
            cost=cost,
            metrics=metrics,
            track_index=track_index,
            crashed=True,
            crash_reason=str(metrics["crash_reason"]),
        )

    metrics = _parse_metrics_csv(METRICS_CSV)
    cost = compute_cost(metrics)
    crashed = int(metrics.get("crashed", 0)) == 1
    reason = str(metrics.get("crash_reason", "")) if crashed else ""

    return EpisodeResult(
        cost=cost,
        metrics=metrics,
        track_index=track_index,
        crashed=crashed,
        crash_reason=reason,
    )


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


# ============================================================
# TPE internals
# ============================================================
def _x_to_z(name: str, x: float) -> float:
    if name in LOG_PARAMS:
        return math.log(float(x))
    return float(x)


def _z_to_x(name: str, z: float) -> float:
    if name in LOG_PARAMS:
        return float(math.exp(z))
    return float(z)


def _z_bounds(name: str, lo: float, hi: float) -> Tuple[float, float]:
    if name in LOG_PARAMS:
        return math.log(lo), math.log(hi)
    return float(lo), float(hi)


def _theta_to_zvec(theta: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in bounds.keys():
        out[k] = _x_to_z(k, float(theta[k]))
    return out


def _bandwidth_1d(vals: List[float], lo_z: float, hi_z: float) -> float:
    width = max(1e-12, hi_z - lo_z)
    bw_min = KDE_BW_MIN_FRAC * width
    bw_max = KDE_BW_MAX_FRAC * width

    if len(vals) <= 1:
        return max(bw_min, min(bw_max, KDE_BW_FALLBACK_FRAC * width))

    mu = sum(vals) / float(len(vals))
    var = sum((x - mu) * (x - mu) for x in vals) / float(len(vals))
    std = math.sqrt(max(0.0, var))

    bw = 0.9 * std * (len(vals) ** (-0.2))
    if not math.isfinite(bw) or bw <= 0.0:
        bw = KDE_BW_FALLBACK_FRAC * width

    bw = max(bw_min, min(bw_max, bw))
    return float(bw)


def _logsumexp(xs: List[float]) -> float:
    if len(xs) == 0:
        return LOGPDF_FLOOR
    m = max(xs)
    if not math.isfinite(m):
        return LOGPDF_FLOOR
    s = sum(math.exp(x - m) for x in xs)
    return m + math.log(max(s, 1e-300))


def _gauss_logpdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(1e-12, sigma)
    u = (x - mu) / sigma
    return -0.5 * u * u - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _mixture_logpdf_1d(x: float, mus: List[float], sigma: float) -> float:
    if len(mus) == 0:
        return LOGPDF_FLOOR
    logs = [_gauss_logpdf(x, m, sigma) for m in mus]
    return _logsumexp(logs) - math.log(float(len(mus)))


def _score_theta_log_ratio(
    theta: Dict[str, float],
    G: List[Dict[str, Any]],
    L: List[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
) -> float:
    if len(G) == 0:
        return -1e18
    if len(L) == 0:
        L = G

    z = _theta_to_zvec(theta, bounds)
    score = 0.0

    for k, (lo, hi) in bounds.items():
        lo_z, hi_z = _z_bounds(k, lo, hi)
        g_vals = [_x_to_z(k, float(r["theta"][k])) for r in G]
        l_vals = [_x_to_z(k, float(r["theta"][k])) for r in L]

        bw_g = _bandwidth_1d(g_vals, lo_z, hi_z)
        bw_l = _bandwidth_1d(l_vals, lo_z, hi_z)

        lg = _mixture_logpdf_1d(z[k], g_vals, bw_g)
        ll = _mixture_logpdf_1d(z[k], l_vals, bw_l)

        score += (lg - ll)

    return float(score)


def _sample_from_good_kde(
    rng: random.Random,
    G: List[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    if len(G) == 0:
        raise RuntimeError("_sample_from_good_kde: empty G")

    base = rng.choice(G)
    out: Dict[str, float] = {}

    for k, (lo, hi) in bounds.items():
        lo_z, hi_z = _z_bounds(k, lo, hi)
        g_vals = [_x_to_z(k, float(r["theta"][k])) for r in G]
        bw_g = _bandwidth_1d(g_vals, lo_z, hi_z)

        base_z = _x_to_z(k, float(base["theta"][k]))
        draw_z = base_z + rng.gauss(0.0, bw_g)
        draw_z = _clamp(draw_z, lo_z, hi_z)

        out[k] = float(_clamp(_z_to_x(k, draw_z), lo, hi))

    return out


def _suggest_tpe_candidate(
    rng: random.Random,
    G: List[Dict[str, Any]],
    L: List[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
    fixed_params: Dict[str, float],
    n_candidates: int = TPE_N_EI_CANDIDATES,
) -> Dict[str, float]:
    if len(G) == 0:
        raise RuntimeError("_suggest_tpe_candidate: empty G")

    best_theta: Optional[Dict[str, float]] = None
    best_score = -1e300

    for _ in range(max(1, n_candidates)):
        th = dict(fixed_params)
        th_draw = _sample_from_good_kde(rng, G, bounds)
        th.update(th_draw)

        s = _score_theta_log_ratio(th, G, L, bounds)
        if s > best_score or best_theta is None:
            best_score = float(s)
            best_theta = dict(th)

    return best_theta


# ============================================================
# G / L construction
# ============================================================
def build_G_L_from_results(
    results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ranked = sorted(results, key=lambda r: r["objective_cost"])
    n = len(ranked)

    if n == 0:
        return [], []

    g_n = max(MIN_G_SIZE, int(math.ceil(GOOD_FRAC * n)))
    g_n = min(g_n, n)

    G = ranked[:g_n]
    L = ranked[g_n:]
    return G, L


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
        th = {}
        for k, (lo, hi) in ALL_BOUNDS.items():
            if k in LOG_PARAMS:
                u = rng.random()
                th[k] = float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
            else:
                th[k] = float(lo + (hi - lo) * rng.random())
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
# Unstaged joint TPE
# ============================================================
def run_unstaged_joint_tpe(
    n_trials: int,
    startup_trials: int,
    critical_crash_multiplier: float,
) -> Tuple[Dict[str, float], float]:
    rng = random.Random(GLOBAL_SEED + 1001)

    startup, startup_meta = build_unstaged_startup_joint(
        n_points=startup_trials,
        seed=GLOBAL_SEED + 101,
    )

    best_theta = _make_safe_joint_anchor()
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
            "unstaged_joint_tpe",
            i,
            theta,
            J,
            info,
            extra_log={
                "proposal_kind": "unstaged_startup",
                **startup_meta[i],
            },
        )

        print(
            f"[unstaged_joint_tpe][startup {i:03d}] "
            f"obj={J:.6g} mean={info.get('mean_cost', 0.0):.6g} "
            f"std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(theta)

    for i in range(len(startup), n_trials):
        G, L = build_G_L_from_results(ALL_RESULTS)
        cand = _suggest_tpe_candidate(
            rng=rng,
            G=G,
            L=L,
            bounds=ALL_BOUNDS,
            fixed_params={},
            n_candidates=TPE_N_EI_CANDIDATES,
        )

        J, info = evaluate_theta(
            theta_x=cand,
            tracks=TRACK_SET,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        _register_result(
            "unstaged_joint_tpe",
            i,
            cand,
            J,
            info,
            extra_log={
                "proposal_kind": "unstaged_tpe",
                "g_size": len(G),
                "l_size": len(L),
            },
        )

        print(
            f"[unstaged_joint_tpe][trial   {i:03d}] "
            f"obj={J:.6g} mean={info.get('mean_cost', 0.0):.6g} "
            f"std={info.get('std_cost', 0.0):.6g} "
            f"crashes={info.get('crashes', 0)}/{info.get('episodes', 0)} "
            f"crit_mult={critical_crash_multiplier:.3f}"
        )

        if J < best_J:
            best_J = float(J)
            best_theta = dict(cand)

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
            "search_mode": "unstaged_joint_tpe",
            "sim_time_s": DEFAULT_SIM_TIME_S,
            "tracks": TRACK_SET,
            "episodes_per_trial": EPISODES_PER_TRIAL,
            "startup_trials": UNSTAGED_STARTUP_TRIALS,
            "total_trials": TOTAL_TRIALS,
            "total_episodes": TOTAL_EPISODES,
            "pure_sim_hours": PURE_SIM_HOURS,
            "safe_planner": SAFE_PLANNER,
            "startup_safe_joint_anchor": {k: float(v) for k, v in startup_anchor_theta.items()},
            "planner_bounds": PLANNER_BOUNDS,
            "cost_bounds": COST_BOUNDS,
            "good_frac": GOOD_FRAC,
            "common_critical_crash_multiplier": COMMON_CRITICAL_CRASH_MULTIPLIER,
            "tpe_n_ei_candidates": TPE_N_EI_CANDIDATES,
            "kde_bw_min_frac": KDE_BW_MIN_FRAC,
            "kde_bw_max_frac": KDE_BW_MAX_FRAC,
            "kde_bw_fallback_frac": KDE_BW_FALLBACK_FRAC,
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
            "unstaged_joint_tpe": {
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
    print("[CFG] MODE      =", PROFILE_NAME)
    print("[CFG] SEARCH_MODE = unstaged_joint_tpe")
    print("[CFG] TRACK_SET =", TRACK_SET)
    print("[CFG] SIM_TIME  =", DEFAULT_SIM_TIME_S, "s")
    print("[CFG] TOTAL_TRIALS =", TOTAL_TRIALS)
    print("[CFG] STARTUP_TRIALS =", UNSTAGED_STARTUP_TRIALS)
    print("[CFG] TOTAL_EPISODES =", TOTAL_EPISODES)
    print(f"[CFG] PURE_SIM_HOURS ~= {PURE_SIM_HOURS:.3f}")
    print(f"[CFG] COMMON critical_crash_multiplier = {COMMON_CRITICAL_CRASH_MULTIPLIER}")
    print(f"[CFG] UCB_WEIGHT = {UCB_WEIGHT}")
    print(f"[CFG] TPE_N_EI_CANDIDATES = {TPE_N_EI_CANDIDATES}")
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
        print("\n================ UNSTAGED JOINT TPE ================\n")
        print("[UNSTAGED] one TPE over full ALL_BOUNDS from trial 0")
        print("[UNSTAGED] startup = safe joint anchor + wide joint LHS")
        print("[UNSTAGED] total trial count matched to staged baseline\n")

        best_theta, best_J = run_unstaged_joint_tpe(
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
