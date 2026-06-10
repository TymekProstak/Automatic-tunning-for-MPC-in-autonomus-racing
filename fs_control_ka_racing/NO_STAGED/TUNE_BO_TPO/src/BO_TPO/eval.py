#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import csv
import copy
import random
import signal
import socket
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Workspace / paths
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_ka_racing/TUNE_BO_TPO")
CATKIN_WS = os.path.abspath(
    os.environ.get("EVAL_WS", os.environ.get("TUNE_WS", DEFAULT_CATKIN_WS))
)
WS_SRC = os.path.join(CATKIN_WS, "src")
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

# ============================================================
# Eval profiles
# ============================================================
DEFAULT_SIM_TIME_S = 60

# ------------------------------------------------------------
# EVAL PROFILE SWITCH
# ------------------------------------------------------------
EVAL_PROFILE_NAME = os.environ.get("EVAL_PROFILE", "smoke_3track")
EVAL_PROFILE_NAME = "night_7track"

if EVAL_PROFILE_NAME == "smoke_3track":
    TRAIN_PROFILE_NAME = "smoke_1track"
    DEFAULT_EVAL_TRACK_SET = "8,11,14"

    DEFAULT_RUN_LOG_NAME = "tuning_run_unstaged_tpe_smoke_1track.jsonl"
    DEFAULT_SUMMARY_NAME = "tuning_summary_unstaged_tpe_smoke_1track.json"

    DEFAULT_OUTPUT_CANDIDATES_NAME = "unstaged_tpe_test_candidates_smoke3tracks.json"
    DEFAULT_OUTPUT_CSV_NAME = "unstaged_tpe_test_results_smoke3tracks.csv"
    DEFAULT_PLOTS_DIR_NAME = "unstaged_tpe_eval_plots_smoke3tracks"

    UNSTAGED_N_TRIALS = 60
    UNSTAGED_STARTUP = 25

elif EVAL_PROFILE_NAME == "night_7track":
    TRAIN_PROFILE_NAME = "night_4track"
    DEFAULT_EVAL_TRACK_SET = "8,9,10,11,12,13,14"

    DEFAULT_RUN_LOG_NAME = "tuning_run_unstaged_tpe_night_4track.jsonl"
    DEFAULT_SUMMARY_NAME = "tuning_summary_unstaged_tpe_night_4track.json"

    DEFAULT_OUTPUT_CANDIDATES_NAME = "unstaged_tpe_test_candidates_night7tracks.json"
    DEFAULT_OUTPUT_CSV_NAME = "unstaged_tpe_test_results_night7tracks.csv"
    DEFAULT_PLOTS_DIR_NAME = "unstaged_tpe_eval_plots_night7tracks"

    UNSTAGED_N_TRIALS = 208
    UNSTAGED_STARTUP = 52

else:
    raise ValueError(f"Unknown EVAL_PROFILE_NAME: {EVAL_PROFILE_NAME}")

UNSTAGED_EFFECTIVE_TRIALS = UNSTAGED_N_TRIALS - UNSTAGED_STARTUP
UNSTAGED_PHASE_NAME = os.environ.get("EVAL_UNSTAGED_PHASE_NAME", "unstaged_joint_tpe")

# ============================================================
# Runtime paths
# ============================================================
RUN_LOG_PATH = os.path.abspath(
    os.environ.get("EVAL_RUN_LOG_PATH", os.path.join(SCRIPT_DIR, DEFAULT_RUN_LOG_NAME))
)
SUMMARY_PATH = os.path.abspath(
    os.environ.get("EVAL_SUMMARY_PATH", os.path.join(SCRIPT_DIR, DEFAULT_SUMMARY_NAME))
)
OUTPUT_CSV_PATH = os.path.abspath(
    os.environ.get("EVAL_OUTPUT_CSV", os.path.join(SCRIPT_DIR, DEFAULT_OUTPUT_CSV_NAME))
)
OUTPUT_CANDIDATES_JSON = os.path.abspath(
    os.environ.get(
        "EVAL_OUTPUT_CANDIDATES_JSON",
        os.path.join(SCRIPT_DIR, DEFAULT_OUTPUT_CANDIDATES_NAME),
    )
)
PLOTS_DIR = os.path.abspath(
    os.environ.get("EVAL_PLOTS_DIR", os.path.join(SCRIPT_DIR, DEFAULT_PLOTS_DIR_NAME))
)

# ============================================================
# ROS launch paths
# ============================================================
SIM_LAUNCH = os.path.join(WS_SRC, "lem_simulator", "launch", "sim.launch")
CTRL_LAUNCH = os.path.join(WS_SRC, "dv_control", "launch", "control.launch")

CONTROL_PARAM_CANDIDATES = [
    os.path.join(WS_SRC, "dv_control", "config", "Params", "control_param.json"),
    os.path.join(WS_SRC, "dv_control", "config", "control_param.json"),
]
METRICS_CSV = os.path.join(WS_SRC, "lem_simulator", "logs", "run_default_metrics.csv")


def _pick_first_existing(paths: List[str]) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    return paths[0]


CONTROL_PARAM_JSON = _pick_first_existing(CONTROL_PARAM_CANDIDATES)

# ============================================================
# Eval config
# ============================================================
GLOBAL_SEED = int(os.environ.get("EVAL_GLOBAL_SEED", "123"))
random.seed(GLOBAL_SEED)


def _parse_track_set() -> List[int]:
    raw = os.environ.get("EVAL_TRACK_SET", DEFAULT_EVAL_TRACK_SET).strip()
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("EVAL_TRACK_SET resolved to an empty list.")
    return vals


TEST_TRACKS = _parse_track_set()
TEST_SIM_TIME_S = int(os.environ.get("EVAL_SIM_TIME_S", "60"))
TEST_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("EVAL_TEST_CRITICAL_CRASH_MULTIPLIER", "1.0")
)
UCB_WEIGHT = float(os.environ.get("EVAL_UCB_WEIGHT", "1.0"))
CHECKPOINT_STEP = int(os.environ.get("EVAL_CHECKPOINT_STEP", "6"))

# ============================================================
# algorithm_thinks / local LHS
# ============================================================
LOCAL_LHS_N = int(os.environ.get("EVAL_LOCAL_LHS_N", "4096"))
LOCAL_BOX_EXPAND_FACTOR = float(os.environ.get("EVAL_LOCAL_BOX_EXPAND_FACTOR", "1.5"))
LOCAL_BOX_MIN_GLOBAL_FRAC = float(os.environ.get("EVAL_LOCAL_BOX_MIN_GLOBAL_FRAC", "0.10"))

GOOD_FRAC = 0.30
MIN_G_SIZE = 4

# ============================================================
# Tuner config mirrored (NO TV)
# ============================================================
PLANNER_BOUNDS = {
    "velocity_planner.v_max": (8.0, 18.0),
    "velocity_planner.mux_acc": (0.3, 1.7),
    "velocity_planner.mux_dec": (0.3, 1.7),
    "velocity_planner.muy": (0.3, 1.7),
}

COST_BOUNDS = {
    "mpc.cost.Q_y": (0.1, 30.0),
    "mpc.cost.Q_psi": (0.1, 30.0),
    "mpc.cost.R_ddelta": (0.1, 30.0),
}

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(PLANNER_BOUNDS)
ALL_BOUNDS.update(COST_BOUNDS)

LOG_PARAMS: set = set()

# ============================================================
# ROS runtime
# ============================================================
ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("EVAL_ROS_MASTER_PORT", "11412"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_eval_unstaged_tpe_{EVAL_PROFILE_NAME}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

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

SOFT_TRACK_DIV = 10.0
MEDIUM_TRACK_DIV = 2.0
HIGH_TRACK_MUL = 1.5

# ============================================================
# Cache / CSV schema
# ============================================================
EVAL_CACHE: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}
EPISODES_DONE = 0

CSV_FIELDS = [
    "selection_family",
    "selection_label",
    "checkpoint_step",
    "unstaged_phase_name",
    "unstaged_trials",
    "unstaged_startup",
    "unstaged_effective_trials",
    "checkpoint_trial_filtered",
    "checkpoint_trial_global",
    "source_candidate_id",
    "source_phase",
    "source_phase_trial",
    "source_trial_global",
    "source_objective_cost_train",
    "source_mean_cost_train",
    "source_std_cost_train",
    "source_robust_cost_train",
    "source_best_objective_so_far_train",
    "source_algorithm_kind",
    "source_tpe_score_train",
    "is_tail_padded",
    "from_cache",
    "test_track_ids",
    "n_test_tracks",
    "test_crash_count",
    "test_crash_percent",
    "test_mean_cost",
    "test_std_cost",
    "test_robust_cost",
    "test_avg_vs_avg_mps",
    "test_avg_soft_violations",
    "test_avg_medium_violations",
    "test_avg_hard_violations",
    "test_critical_crash_multiplier",
    "theta_json",
]

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
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=_ros_env(),
        preexec_fn=os.setsid,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )


def _popen_roslaunch(
    launch_file: str,
    launch_args: Dict[str, str],
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    cmd = ["roslaunch", launch_file] + [f"{k}:={v}" for k, v in launch_args.items()]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=_ros_env(extra_env),
        preexec_fn=os.setsid,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )


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


def _load_json_optional(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _save_json_atomic(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _theta_signature(theta: Dict[str, Any], ndigits: int = 12) -> Tuple[Tuple[str, float], ...]:
    items = []
    for k in sorted(theta.keys()):
        try:
            v = round(float(theta[k]), ndigits)
        except Exception:
            continue
        items.append((k, v))
    return tuple(items)


def _metric_float(metrics: Dict[str, Any], name: str) -> float:
    try:
        return float(metrics.get(name, 0.0))
    except Exception:
        return 0.0


def _metric_float_first(metrics: Dict[str, Any], names: List[str], default: float = 0.0) -> float:
    for name in names:
        if name in metrics:
            try:
                return float(metrics[name])
            except Exception:
                pass
    return float(default)


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


def _train_objective(rec: Dict[str, Any]) -> float:
    if "objective_cost" in rec:
        return float(rec["objective_cost"])
    if "robust_cost" in rec:
        return float(rec["robust_cost"])
    return float(rec.get("mean_cost", float("inf")))


def _best_so_far_value(rec: Dict[str, Any]) -> float:
    if "best_objective_so_far" in rec:
        return float(rec["best_objective_so_far"])
    if "best_cost_so_far" in rec:
        return float(rec["best_cost_so_far"])
    return float("nan")


# ============================================================
# LHS / density helpers for algorithm_thinks
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


def _x_to_z(name: str, x: float) -> float:
    return float(x)


def _z_to_x(name: str, z: float) -> float:
    return float(z)


def _z_bounds(name: str, lo: float, hi: float) -> Tuple[float, float]:
    return float(lo), float(hi)


def _theta_to_zvec(theta: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in bounds.keys():
        out[k] = _x_to_z(k, float(theta[k]))
    return out


def _bandwidth_1d(vals: List[float], lo_z: float, hi_z: float) -> float:
    width = max(1e-12, hi_z - lo_z)
    bw_min = 0.03 * width
    bw_max = 0.35 * width

    if len(vals) <= 1:
        return max(bw_min, min(bw_max, 0.12 * width))

    mu = sum(vals) / float(len(vals))
    var = sum((x - mu) * (x - mu) for x in vals) / float(len(vals))
    std = math.sqrt(max(0.0, var))

    bw = 0.9 * std * (len(vals) ** (-0.2))
    if not math.isfinite(bw) or bw <= 0.0:
        bw = 0.12 * width

    bw = max(bw_min, min(bw_max, bw))
    return float(bw)


def _logsumexp(xs: List[float]) -> float:
    if len(xs) == 0:
        return -1e12
    m = max(xs)
    if not math.isfinite(m):
        return -1e12
    s = sum(math.exp(x - m) for x in xs)
    return m + math.log(max(s, 1e-300))


def _gauss_logpdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(1e-12, sigma)
    u = (x - mu) / sigma
    return -0.5 * u * u - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _mixture_logpdf_1d(x: float, mus: List[float], sigma: float) -> float:
    if len(mus) == 0:
        return -1e12
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


def _build_G_L_generic(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ranked = sorted(records, key=lambda r: _train_objective(r))
    n = len(ranked)

    if n == 0:
        return [], []

    g_n = max(MIN_G_SIZE, int(math.ceil(GOOD_FRAC * n)))
    g_n = min(g_n, n)

    G = ranked[:g_n]
    L = ranked[g_n:]
    return G, L


def build_local_bounds_from_G(
    G: List[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
    expand_factor: float = LOCAL_BOX_EXPAND_FACTOR,
    min_global_frac: float = LOCAL_BOX_MIN_GLOBAL_FRAC,
) -> Dict[str, Tuple[float, float]]:
    local_bounds: Dict[str, Tuple[float, float]] = {}

    for k, (lo, hi) in bounds.items():
        glo, ghi = _z_bounds(k, lo, hi)
        global_w = max(1e-12, ghi - glo)

        vals = [_x_to_z(k, float(r["theta"][k])) for r in G if k in r["theta"]]
        if len(vals) == 0:
            center = 0.5 * (glo + ghi)
            local_w = global_w
        else:
            vmin = min(vals)
            vmax = max(vals)
            center = 0.5 * (vmin + vmax)
            g_w = max(0.0, vmax - vmin)
            local_w = max(expand_factor * g_w, min_global_frac * global_w)

        half = 0.5 * local_w
        lo_z = max(glo, center - half)
        hi_z = min(ghi, center + half)

        if hi_z - lo_z < 1e-12:
            pad = 0.5 * min_global_frac * global_w
            lo_z = max(glo, center - pad)
            hi_z = min(ghi, center + pad)

        local_bounds[k] = (_z_to_x(k, lo_z), _z_to_x(k, hi_z))

    return local_bounds


def build_tpe_local_lhs_estimate(
    G: List[Dict[str, Any]],
    L: List[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
    seed: int,
) -> Tuple[Dict[str, float], float]:
    if len(G) == 0:
        raise RuntimeError("build_tpe_local_lhs_estimate: empty G")

    local_bounds = build_local_bounds_from_G(
        G=G,
        bounds=bounds,
        expand_factor=LOCAL_BOX_EXPAND_FACTOR,
        min_global_frac=LOCAL_BOX_MIN_GLOBAL_FRAC,
    )

    cloud = generate_lhs_trials(seed=seed, n_trials=LOCAL_LHS_N, bounds=local_bounds)

    best_theta: Optional[Dict[str, float]] = None
    best_score = -1e300

    for th in cloud:
        score = _score_theta_log_ratio(
            theta=th,
            G=G,
            L=L,
            bounds=bounds,
        )

        if best_theta is None or score > best_score:
            best_score = float(score)
            best_theta = dict(th)

    return best_theta, best_score


# ============================================================
# Cost / JSON params
# ============================================================
def compute_cost(metrics: Dict[str, Any]) -> float:
    crashed = int(metrics.get("crashed", 0)) == 1
    if crashed:
        crash_time = float(metrics.get("crash_time_s", -1.0))
        return CRASH_PENALTY + CRASH_TIME_PENALTY * max(
            0.0,
            TEST_SIM_TIME_S - max(0.0, crash_time),
        )

    J = 0.0
    for k, w in COST_WEIGHTS.items():
        J += float(w) * _metric_float(metrics, k)

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


def apply_params_to_control_json(theta_x: Dict[str, float]) -> None:
    data = _load_json(CONTROL_PARAM_JSON)
    for path, val in theta_x.items():
        if path in ALL_BOUNDS:
            lo, hi = ALL_BOUNDS[path]
            _set_json_path(data, path, float(_clamp(float(val), lo, hi)))
    _save_json_atomic(CONTROL_PARAM_JSON, data)


def _build_sim_launch_args(track_index: int, sim_time_s: int, critical_crash_multiplier: float) -> Dict[str, str]:
    return {
        "sim_time": str(sim_time_s),
        "track_id": str(track_index),
        "low_level_controlers": str(SIM_LOW_LEVEL_CONTROLERS),
        "ins_mode": str(SIM_INS_MODE),
        "critical_crash_multiplier": str(float(critical_crash_multiplier)),
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            if "candidate_id" not in rec:
                rec["candidate_id"] = idx
            rec["_global_trial"] = idx
            out.append(rec)
    out.sort(key=lambda r: int(r.get("_global_trial", 0)))
    return out


def _is_startup_record(rec: Dict[str, Any]) -> bool:
    if "phase_trial" in rec:
        phase_trial_zero_based = int(rec.get("phase_trial", -1))
        phase_trial_one_based = phase_trial_zero_based + 1
        return phase_trial_one_based <= UNSTAGED_STARTUP
    return int(rec.get("candidate_id", rec.get("_global_trial", 0))) <= UNSTAGED_STARTUP


def _filtered_to_global_trial(filtered_trial: int) -> int:
    if filtered_trial <= 0:
        return 0
    return UNSTAGED_STARTUP + filtered_trial


def _build_filtered_checkpoints(step: int) -> List[int]:
    if step <= 0:
        return [UNSTAGED_EFFECTIVE_TRIALS]

    checkpoints = list(range(step, UNSTAGED_EFFECTIVE_TRIALS + 1, step))
    if not checkpoints or checkpoints[-1] != UNSTAGED_EFFECTIVE_TRIALS:
        checkpoints.append(UNSTAGED_EFFECTIVE_TRIALS)
    return checkpoints


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
    sim_time_s: int = TEST_SIM_TIME_S,
    critical_crash_multiplier: float = TEST_CRITICAL_CRASH_MULTIPLIER,
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


def evaluate_theta_on_test_tracks(
    theta_x: Dict[str, float],
    tracks: List[int],
    sim_time_s: int,
    critical_crash_multiplier: float,
) -> Dict[str, Any]:
    global EPISODES_DONE

    sig = _theta_signature(theta_x)
    if sig in EVAL_CACHE:
        out = copy.deepcopy(EVAL_CACHE[sig])
        out["from_cache"] = True
        return out

    per_track: List[EpisodeResult] = []

    for tid in tracks:
        res = run_one_episode(
            theta_x=theta_x,
            track_index=tid,
            sim_time_s=sim_time_s,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        EPISODES_DONE += 1
        per_track.append(res)

    n = max(1, len(per_track))

    costs = [float(r.cost) for r in per_track]
    mean_cost = sum(costs) / n
    var_cost = sum((c - mean_cost) * (c - mean_cost) for c in costs) / n
    std_cost = math.sqrt(max(0.0, var_cost))
    robust_cost = mean_cost + UCB_WEIGHT * std_cost

    mean_vs_avg_mps = sum(_metric_float_first(r.metrics, ["vs_avg_mps"], 0.0) for r in per_track) / n
    mean_soft = sum(
        _metric_float_first(
            r.metrics,
            ["soft_track_violations_count", "soft_track_violation_count", "soft_track_violation_count_"],
            0.0,
        )
        for r in per_track
    ) / n
    mean_medium = sum(
        _metric_float_first(
            r.metrics,
            ["medium_track_violations_count", "medium_track_violation_count", "medium_track_violation_count_"],
            0.0,
        )
        for r in per_track
    ) / n
    mean_hard = sum(
        _metric_float_first(
            r.metrics,
            ["high_track_violations_count", "high_track_violation_count", "high_track_violation_count_"],
            0.0,
        )
        for r in per_track
    ) / n
    crash_count = sum(1 for r in per_track if r.crashed)
    crash_percent = 100.0 * float(crash_count) / float(n)

    out = {
        "test_track_ids": ";".join(str(t) for t in tracks),
        "n_test_tracks": int(len(tracks)),
        "test_crash_count": int(crash_count),
        "test_crash_percent": float(crash_percent),
        "test_mean_cost": float(mean_cost),
        "test_std_cost": float(std_cost),
        "test_robust_cost": float(robust_cost),
        "test_avg_vs_avg_mps": float(mean_vs_avg_mps),
        "test_avg_soft_violations": float(mean_soft),
        "test_avg_medium_violations": float(mean_medium),
        "test_avg_hard_violations": float(mean_hard),
        "from_cache": False,
    }

    EVAL_CACHE[sig] = copy.deepcopy(out)
    return out


# ============================================================
# Candidate schedules
# ============================================================
def _select_best_so_far_schedule(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    non_startup = [r for r in records if not _is_startup_record(r)]
    non_startup.sort(key=lambda r: int(r.get("_global_trial", 0)))

    selected: List[Dict[str, Any]] = []
    checkpoints_filtered = _build_filtered_checkpoints(CHECKPOINT_STEP)

    if not non_startup:
        return selected

    available_effective = len(non_startup)

    for checkpoint_filtered in checkpoints_filtered:
        checkpoint_global = _filtered_to_global_trial(checkpoint_filtered)
        effective_ref = min(checkpoint_filtered, available_effective)
        tail_padded = checkpoint_filtered > available_effective

        history = non_startup[:effective_ref]
        if not history:
            continue

        best_rec = min(
            history,
            key=lambda r: (
                _train_objective(r),
                int(r.get("_global_trial", 10**18)),
            ),
        )

        best_obj_so_far = min(_train_objective(r) for r in history)

        selected.append(
            {
                "selection_family": "best_so_far",
                "selection_label": f"best_so_far_t{checkpoint_filtered:03d}",
                "checkpoint_trial_filtered": int(checkpoint_filtered),
                "checkpoint_trial_global": int(checkpoint_global),
                "source_candidate_id": int(best_rec.get("candidate_id", best_rec.get("_global_trial", 0))),
                "source_phase": str(best_rec.get("phase", UNSTAGED_PHASE_NAME)),
                "source_phase_trial": int(best_rec.get("phase_trial", -1)),
                "source_trial_global": int(best_rec.get("_global_trial", -1)),
                "source_objective_cost_train": float(_train_objective(best_rec)),
                "source_mean_cost_train": float(best_rec.get("mean_cost", float("nan"))),
                "source_std_cost_train": float(best_rec.get("std_cost", float("nan"))),
                "source_robust_cost_train": float(best_rec.get("robust_cost", float("nan"))),
                "source_best_objective_so_far_train": float(best_obj_so_far),
                "source_algorithm_kind": "best_so_far",
                "source_tpe_score_train": "",
                "is_tail_padded": bool(tail_padded),
                "theta": {k: float(v) for k, v in best_rec.get("theta", {}).items()},
            }
        )

    return selected


def _select_algorithm_thinks_schedule(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    non_startup = [r for r in records if not _is_startup_record(r)]
    non_startup.sort(key=lambda r: int(r.get("_global_trial", 0)))

    selected: List[Dict[str, Any]] = []
    checkpoints_filtered = _build_filtered_checkpoints(CHECKPOINT_STEP)

    if not non_startup:
        return selected

    available_effective = len(non_startup)

    for checkpoint_filtered in checkpoints_filtered:
        checkpoint_global = _filtered_to_global_trial(checkpoint_filtered)
        effective_ref = min(checkpoint_filtered, available_effective)
        tail_padded = checkpoint_filtered > available_effective

        history = non_startup[:effective_ref]
        if not history:
            continue

        G, L = _build_G_L_generic(history)
        if len(G) == 0:
            continue

        theta_est, score = build_tpe_local_lhs_estimate(
            G=G,
            L=L,
            bounds=ALL_BOUNDS,
            seed=GLOBAL_SEED + 1000000 + checkpoint_filtered,
        )

        src_rec = history[-1]
        best_obj_so_far = min(_train_objective(r) for r in history)

        selected.append(
            {
                "selection_family": "algorithm_thinks",
                "selection_label": f"algorithm_thinks_t{checkpoint_filtered:03d}",
                "checkpoint_trial_filtered": int(checkpoint_filtered),
                "checkpoint_trial_global": int(checkpoint_global),
                "source_candidate_id": int(src_rec.get("candidate_id", src_rec.get("_global_trial", 0))),
                "source_phase": str(src_rec.get("phase", UNSTAGED_PHASE_NAME)),
                "source_phase_trial": int(src_rec.get("phase_trial", -1)),
                "source_trial_global": int(src_rec.get("_global_trial", -1)),
                "source_objective_cost_train": float(_train_objective(src_rec)),
                "source_mean_cost_train": float(src_rec.get("mean_cost", float("nan"))),
                "source_std_cost_train": float(src_rec.get("std_cost", float("nan"))),
                "source_robust_cost_train": float(src_rec.get("robust_cost", float("nan"))),
                "source_best_objective_so_far_train": float(best_obj_so_far),
                "source_algorithm_kind": "unstaged_local_lhs_density_ratio",
                "source_tpe_score_train": float(score),
                "is_tail_padded": bool(tail_padded),
                "theta": {k: float(v) for k, v in theta_est.items()},
            }
        )

    return selected


# ============================================================
# CSV / plots
# ============================================================
def _make_csv_row(cand: Dict[str, Any], summary: Dict[str, Any], cache_hit: bool) -> Dict[str, Any]:
    return {
        "selection_family": str(cand["selection_family"]),
        "selection_label": str(cand["selection_label"]),
        "checkpoint_step": int(CHECKPOINT_STEP),
        "unstaged_phase_name": str(UNSTAGED_PHASE_NAME),
        "unstaged_trials": int(UNSTAGED_N_TRIALS),
        "unstaged_startup": int(UNSTAGED_STARTUP),
        "unstaged_effective_trials": int(UNSTAGED_EFFECTIVE_TRIALS),
        "checkpoint_trial_filtered": int(cand["checkpoint_trial_filtered"]),
        "checkpoint_trial_global": int(cand["checkpoint_trial_global"]),
        "source_candidate_id": int(cand["source_candidate_id"]),
        "source_phase": str(cand["source_phase"]),
        "source_phase_trial": int(cand["source_phase_trial"]),
        "source_trial_global": int(cand["source_trial_global"]),
        "source_objective_cost_train": float(cand["source_objective_cost_train"]),
        "source_mean_cost_train": float(cand["source_mean_cost_train"]),
        "source_std_cost_train": float(cand["source_std_cost_train"]),
        "source_robust_cost_train": float(cand["source_robust_cost_train"]),
        "source_best_objective_so_far_train": float(cand["source_best_objective_so_far_train"]),
        "source_algorithm_kind": str(cand["source_algorithm_kind"]),
        "source_tpe_score_train": cand["source_tpe_score_train"],
        "is_tail_padded": int(bool(cand["is_tail_padded"])),
        "from_cache": int(bool(cache_hit)),
        "test_track_ids": str(summary["test_track_ids"]),
        "n_test_tracks": int(summary["n_test_tracks"]),
        "test_crash_count": int(summary["test_crash_count"]),
        "test_crash_percent": float(summary["test_crash_percent"]),
        "test_mean_cost": float(summary["test_mean_cost"]),
        "test_std_cost": float(summary["test_std_cost"]),
        "test_robust_cost": float(summary["test_robust_cost"]),
        "test_avg_vs_avg_mps": float(summary["test_avg_vs_avg_mps"]),
        "test_avg_soft_violations": float(summary["test_avg_soft_violations"]),
        "test_avg_medium_violations": float(summary["test_avg_medium_violations"]),
        "test_avg_hard_violations": float(summary["test_avg_hard_violations"]),
        "test_critical_crash_multiplier": float(TEST_CRITICAL_CRASH_MULTIPLIER),
        "theta_json": json.dumps(cand["theta"], sort_keys=True),
    }


def _save_family_plots(rows: List[Dict[str, Any]], family: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fam_rows = [r for r in rows if str(r.get("selection_family", "")) == family]
    fam_rows.sort(key=lambda r: int(r["checkpoint_trial_filtered"]))

    if not fam_rows:
        return

    x = [int(r["checkpoint_trial_filtered"]) for r in fam_rows]

    metric_specs = [
        ("test_robust_cost", "Robust cost", f"{family}: robust_cost"),
        ("test_mean_cost", "Mean cost", f"{family}: mean_cost"),
        ("test_std_cost", "Cost standard deviation", f"{family}: std_cost"),
        ("test_crash_percent", "Crash rate [%]", f"{family}: crash_percent"),
        ("test_avg_vs_avg_mps", "Average speed [m/s]", f"{family}: avg_vs_avg_mps"),
        ("test_avg_soft_violations", "Average soft violations", f"{family}: avg_soft_violations"),
        ("test_avg_medium_violations", "Average medium violations", f"{family}: avg_medium_violations"),
        ("test_avg_hard_violations", "Average hard violations", f"{family}: avg_hard_violations"),
    ]

    for key, ylabel, title in metric_specs:
        y = [float(r[key]) for r in fam_rows]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker="o")
        plt.xlabel("Filtered trial index")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(PLOTS_DIR, f"{family}_{key}.png")
        plt.savefig(out_path, dpi=160, format="png")
        plt.close()


def _print_runtime_eval_line(
    idx: int,
    total: int,
    sel: Dict[str, Any],
    row: Dict[str, Any],
    cache_hit: bool,
) -> None:
    print(
        f"[TEST {idx:03d}/{total:03d}] "
        f"{sel['selection_label']} | family={sel['selection_family']} | "
        f"checkpoint_filtered={sel['checkpoint_trial_filtered']} | "
        f"checkpoint_global={sel['checkpoint_trial_global']} | "
        f"source_phase={sel['source_phase']} | "
        f"kind={sel['source_algorithm_kind']}"
    )

    if cache_hit:
        print("  [cache] reused evaluation")

    print(
        f"  mean={row['test_mean_cost']:.6g} | "
        f"std={row['test_std_cost']:.6g} | "
        f"robust={row['test_robust_cost']:.6g} | "
        f"avg_vs={row['test_avg_vs_avg_mps']:.6g} | "
        f"soft={row['test_avg_soft_violations']:.6g} | "
        f"medium={row['test_avg_medium_violations']:.6g} | "
        f"hard={row['test_avg_hard_violations']:.6g} | "
        f"crash%={row['test_crash_percent']:.6g} | "
        f"tail_padded={int(bool(sel['is_tail_padded']))}"
    )
    print("")


# ============================================================
# Main
# ============================================================
def main() -> None:
    global EPISODES_DONE

    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(WS_SRC, "WS_SRC")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(RUN_LOG_PATH, "RUN_LOG_PATH")

    original_control_json = _load_json(CONTROL_PARAM_JSON)
    summary_json = _load_json_optional(SUMMARY_PATH)

    try:
        if os.path.exists(OUTPUT_CSV_PATH):
            os.remove(OUTPUT_CSV_PATH)
    except Exception:
        pass

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("[CFG] CATKIN_WS =", CATKIN_WS)
    print("[CFG] TRAIN_PROFILE_NAME =", TRAIN_PROFILE_NAME)
    print("[CFG] EVAL_PROFILE_NAME =", EVAL_PROFILE_NAME)
    print("[CFG] RUN_LOG_PATH =", RUN_LOG_PATH)
    print("[CFG] SUMMARY_PATH =", SUMMARY_PATH, "(optional)")
    print("[CFG] OUTPUT_CSV_PATH =", OUTPUT_CSV_PATH)
    print("[CFG] OUTPUT_CANDIDATES_JSON =", OUTPUT_CANDIDATES_JSON)
    print("[CFG] PLOTS_DIR =", PLOTS_DIR)
    print("[CFG] TEST_TRACKS =", TEST_TRACKS)
    print("[CFG] SIM_TIME_S =", TEST_SIM_TIME_S)
    print("[CFG] UCB_WEIGHT =", UCB_WEIGHT)
    print("[CFG] CHECKPOINT_STEP =", CHECKPOINT_STEP)
    print("[CFG] UNSTAGED_PHASE_NAME =", UNSTAGED_PHASE_NAME)
    print("[CFG] UNSTAGED_N_TRIALS =", UNSTAGED_N_TRIALS)
    print("[CFG] UNSTAGED_STARTUP =", UNSTAGED_STARTUP)
    print("[CFG] UNSTAGED_EFFECTIVE_TRIALS =", UNSTAGED_EFFECTIVE_TRIALS)
    print("[CFG] TEST_CRITICAL_CRASH_MULTIPLIER =", TEST_CRITICAL_CRASH_MULTIPLIER)
    print("[CFG] LOCAL_LHS_N =", LOCAL_LHS_N)
    print("[CFG] LOCAL_BOX_EXPAND_FACTOR =", LOCAL_BOX_EXPAND_FACTOR)
    print("[CFG] LOCAL_BOX_MIN_GLOBAL_FRAC =", LOCAL_BOX_MIN_GLOBAL_FRAC)
    print("")

    records = load_jsonl(RUN_LOG_PATH)
    if len(records) == 0:
        raise RuntimeError(f"Empty run log: {RUN_LOG_PATH}")

    print(f"[INFO] Loaded {len(records)} training log rows.")
    print("[INFO] Unstaged TPE evaluator reconstructs best_so_far and algorithm_thinks from tuning_run jsonl.")
    print("[INFO] Startup filtering is single-block.")
    print("[INFO] Global-trial mapping is: checkpoint_global = UNSTAGED_STARTUP + checkpoint_filtered.")
    if summary_json is not None:
        print("[INFO] Summary json found:", SUMMARY_PATH)
    print("")

    best_schedule = _select_best_so_far_schedule(records)
    algo_schedule = _select_algorithm_thinks_schedule(records)

    selections = list(best_schedule) + list(algo_schedule)
    selections.sort(
        key=lambda r: (
            int(r["checkpoint_trial_filtered"]),
            0 if str(r["selection_family"]) == "algorithm_thinks" else 1,
            int(r["source_candidate_id"]),
        )
    )

    _save_json_atomic(OUTPUT_CANDIDATES_JSON, selections)

    print(f"[INFO] Selected {len(best_schedule)} best_so_far checkpoints.")
    print(f"[INFO] Selected {len(algo_schedule)} algorithm_thinks checkpoints.")
    print(f"[INFO] Total selections = {len(selections)}")
    print("")

    output_rows: List[Dict[str, Any]] = []

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
        for idx, sel in enumerate(selections, start=1):
            theta = {k: float(v) for k, v in sel["theta"].items()}
            sig = _theta_signature(theta)

            cache_hit = sig in EVAL_CACHE
            summary = evaluate_theta_on_test_tracks(
                theta_x=theta,
                tracks=TEST_TRACKS,
                sim_time_s=TEST_SIM_TIME_S,
                critical_crash_multiplier=TEST_CRITICAL_CRASH_MULTIPLIER,
            )

            row = _make_csv_row(sel, summary, cache_hit)
            output_rows.append(row)
            _append_csv_row(OUTPUT_CSV_PATH, CSV_FIELDS, row)

            _print_runtime_eval_line(
                idx=idx,
                total=len(selections),
                sel=sel,
                row=row,
                cache_hit=cache_hit,
            )

        _save_family_plots(output_rows, "best_so_far")
        _save_family_plots(output_rows, "algorithm_thinks")

    finally:
        try:
            _save_json_atomic(CONTROL_PARAM_JSON, original_control_json)
        except Exception:
            pass

        if roscore_p is not None:
            _kill_process_group(roscore_p)

    print("")
    print("[DONE] Saved CSV:", OUTPUT_CSV_PATH)
    print("[DONE] Saved candidates JSON:", OUTPUT_CANDIDATES_JSON)
    print("[DONE] Saved plots dir:", PLOTS_DIR)
    print("[DONE] Unique evaluated thetas:", len(EVAL_CACHE))
    print("[DONE] Total CSV rows:", len(output_rows))
    print("[DONE] Total test episodes run:", EPISODES_DONE)


if __name__ == "__main__":
    main()
