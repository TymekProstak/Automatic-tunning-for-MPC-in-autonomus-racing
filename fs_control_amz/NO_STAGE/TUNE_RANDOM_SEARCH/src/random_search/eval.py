#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import time
import math
import signal
import socket
import copy
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List


# ============================================================
# Workspace / paths
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_amz/TUNE_RANDOM_SEARCH")
CATKIN_WS = os.path.abspath(os.environ.get("EVAL_WS", os.environ.get("TUNE_WS", DEFAULT_CATKIN_WS)))
WS_SRC = os.path.join(CATKIN_WS, "src")
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

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
ACADOS_LIB = os.path.abspath(
    os.environ.get("EVAL_ACADOS_LIB", os.environ.get("TUNE_ACADOS_LIB", DEFAULT_ACADOS_LIB))
)

# ============================================================
# Eval config
# ============================================================
GLOBAL_SEED = 123
random.seed(GLOBAL_SEED)

DEFAULT_SIM_TIME_S = int(os.environ.get("EVAL_SIM_TIME_S", os.environ.get("TUNE_SIM_TIME_S", "60")))
UCB_WEIGHT = float(os.environ.get("EVAL_UCB_WEIGHT", os.environ.get("TUNE_UCB_WEIGHT", "1.0")))
PROFILE_NAME = os.environ.get("EVAL_PROFILE", os.environ.get("TUNE_PROFILE", "night_4track")).strip()
UNSTAGED_PHASE_NAME = "unstaged_joint_random"

if PROFILE_NAME == "smoke_1track":
    TRAIN_TRACK_SET = [1]
    DEFAULT_TEST_TRACK_SET = "8,11,14"
    DEFAULT_BEST_SO_FAR_EVERY = "10"
    DEFAULT_THINK_EVERY = "12"
    UNSTAGED_N_TRIALS = 60
    UNSTAGED_STARTUP = 25
elif PROFILE_NAME == "night_4track":
    TRAIN_TRACK_SET = [1, 2, 3, 4]
    DEFAULT_TEST_TRACK_SET = "8,9,10,11,12,13,14"
    DEFAULT_BEST_SO_FAR_EVERY = "10"
    DEFAULT_THINK_EVERY = "12"
    UNSTAGED_N_TRIALS = 224
    UNSTAGED_STARTUP = 64
else:
    raise ValueError(f"Unknown PROFILE_NAME: {PROFILE_NAME}")


def _parse_track_set(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("Resolved empty test track set.")
    return out


TEST_TRACK_SET = _parse_track_set(os.environ.get("EVAL_TRACK_SET", DEFAULT_TEST_TRACK_SET))
BEST_SO_FAR_EVERY = int(os.environ.get("EVAL_BEST_SO_FAR_EVERY", DEFAULT_BEST_SO_FAR_EVERY))
THINK_EVERY = int(os.environ.get("EVAL_THINK_EVERY", os.environ.get("EVAL_SURROGATE_EVERY", DEFAULT_THINK_EVERY)))
TOPK = int(os.environ.get("EVAL_TOPK", "3"))
TEST_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("EVAL_TEST_CRITICAL_CRASH_MULTIPLIER", "1.0")
)

RUN_LOG_PATH = os.path.abspath(
    os.environ.get(
        "EVAL_RUN_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_run_full_coupled_unstaged_random_search_{PROFILE_NAME}.jsonl"),
    )
)
OUTPUT_CSV = os.path.abspath(
    os.environ.get(
        "EVAL_OUTPUT_CSV",
        os.path.join(SCRIPT_DIR, f"random_search_unstaged_test_eval_full_coupled_{PROFILE_NAME}.csv"),
    )
)
CANDIDATES_JSON = os.path.abspath(
    os.environ.get(
        "EVAL_CANDIDATES_JSON",
        os.path.join(SCRIPT_DIR, f"random_search_unstaged_test_candidates_full_coupled_{PROFILE_NAME}.json"),
    )
)

# ============================================================
# Tuner config copied from full_coupled_unstaged_random_search.py
# ============================================================
SAFE_PLANNER = {
    "mpc.bounds.max_vx": 10.5,
    "mpc.model.mux": 0.4,
    "mpc.model.muy": 0.4,
}

PLANNER_BOUNDS = {
    "mpc.bounds.max_vx": (8.0, 18.0),
    "mpc.model.mux": (0.5, 1.7),
    "mpc.model.muy": (0.5, 1.7),
}

COST_BOUNDS = {
    "mpc.cost.q_ey": (0.1, 30.0),
    "mpc.cost.Q_epsi": (0.1, 30.0),
    "mpc.cost.R_dT": (0.1, 30.0),
    "mpc.cost.R_u_ddelta_cmd": (0.1, 30.0),
    "mpc.cost.Q_beta": (0.01, 5.0),
    "mpc.cost.R_Mtv": (1e-8, 1e-5),
}

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(PLANNER_BOUNDS)
ALL_BOUNDS.update(COST_BOUNDS)

LOG_PARAMS = set(COST_BOUNDS.keys())
GOOD_FRAC = 0.30
MIN_G_SIZE = 4

# ============================================================
# ROS / runtime
# ============================================================
SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("EVAL_ROS_MASTER_PORT", os.environ.get("TUNE_ROS_MASTER_PORT", "11822")))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_eval_random_full_coupled_unstaged_{PROFILE_NAME}_{GLOBAL_SEED}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Cost function copied from unstaged tuner
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
# Global state / cache
# ============================================================
BASE_CONTROL_TEMPLATE: Optional[Dict[str, Any]] = None
EVAL_CACHE: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}


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


def _append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
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
    out.sort(key=lambda r: int(r.get("candidate_id", r.get("_global_trial", 0))))
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _theta_signature(theta: Dict[str, float], ndigits: int = 12) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, round(float(v), ndigits)) for k, v in theta.items()))


def _train_objective(rec: Dict[str, Any]) -> float:
    if "objective_cost" in rec:
        return float(rec["objective_cost"])
    if "robust_cost" in rec:
        return float(rec["robust_cost"])
    return float(rec.get("mean_cost", float("inf")))


def _make_cost_mid_theta() -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, (lo, hi) in COST_BOUNDS.items():
        out[k] = float(math.sqrt(lo * hi))
    return out


def _weighted_center_from_records(
    recs: List[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if len(recs) == 0:
        for k, (lo, hi) in bounds.items():
            if k in LOG_PARAMS:
                out[k] = float(math.sqrt(lo * hi))
            else:
                out[k] = float(0.5 * (lo + hi))
        return out

    ranked = sorted(recs, key=_train_objective)
    weights = [1.0 / (i + 1) for i in range(len(ranked))]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    for k, (lo, hi) in bounds.items():
        if k in LOG_PARAMS:
            acc = 0.0
            for w, rec in zip(weights, ranked):
                acc += w * math.log(float(rec["theta"][k]))
            out[k] = float(_clamp(math.exp(acc), lo, hi))
        else:
            acc = 0.0
            for w, rec in zip(weights, ranked):
                acc += w * float(rec["theta"][k])
            out[k] = float(_clamp(acc, lo, hi))

    return out


def apply_params_to_control_json(theta_x: Dict[str, float]) -> None:
    global BASE_CONTROL_TEMPLATE

    if BASE_CONTROL_TEMPLATE is None:
        raise RuntimeError("BASE_CONTROL_TEMPLATE is not initialized")

    data = copy.deepcopy(BASE_CONTROL_TEMPLATE)
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


@dataclass
class EvalTask:
    selection_family: str
    selection_label: str
    checkpoint_trial: int
    source_candidate_id: Optional[int]
    source_phase: str
    source_objective_cost_train: float
    source_mean_cost_train: float
    source_std_cost_train: float
    source_best_objective_so_far_train: float
    theta: Dict[str, float]


def run_one_episode(
    theta_x: Dict[str, float],
    track_index: int,
    sim_time_s: int = DEFAULT_SIM_TIME_S,
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


def evaluate_theta_on_test_tracks(theta_x: Dict[str, float]) -> Dict[str, Any]:
    cache_key = _theta_signature(theta_x)
    if cache_key in EVAL_CACHE:
        out = copy.deepcopy(EVAL_CACHE[cache_key])
        out["from_cache"] = True
        return out

    per_track: List[Dict[str, Any]] = []
    costs: List[float] = []
    vs_vals: List[float] = []
    soft_vals: List[float] = []
    med_vals: List[float] = []
    hard_vals: List[float] = []
    crash_count = 0

    for tid in TEST_TRACK_SET:
        res = run_one_episode(
            theta_x=theta_x,
            track_index=tid,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=TEST_CRITICAL_CRASH_MULTIPLIER,
        )

        metrics = dict(res.metrics)
        vs_avg = _metric_float_first(metrics, ["vs_avg_mps"], 0.0)
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
        hard_cnt = _metric_float_first(
            metrics,
            ["high_track_violations_count", "high_track_violation_count", "high_track_violation_count_"],
            0.0,
        )

        costs.append(float(res.cost))
        vs_vals.append(float(vs_avg))
        soft_vals.append(float(soft_cnt))
        med_vals.append(float(med_cnt))
        hard_vals.append(float(hard_cnt))

        if res.crashed:
            crash_count += 1

        per_track.append(
            {
                "track_id": int(tid),
                "cost": float(res.cost),
                "crashed": bool(res.crashed),
                "crash_reason": str(res.crash_reason),
                "vs_avg_mps": float(vs_avg),
                "soft_track_violations": float(soft_cnt),
                "medium_track_violations": float(med_cnt),
                "hard_track_violations": float(hard_cnt),
            }
        )

    n = max(1, len(TEST_TRACK_SET))
    mean_cost = float(sum(costs) / n)
    var_cost = sum((x - mean_cost) * (x - mean_cost) for x in costs) / n
    std_cost = float(math.sqrt(max(0.0, var_cost)))
    robust_cost = float(mean_cost + UCB_WEIGHT * std_cost)

    out = {
        "test_tracks": list(TEST_TRACK_SET),
        "episodes": int(len(TEST_TRACK_SET)),
        "crash_count": int(crash_count),
        "crash_rate": float(crash_count / n),
        "mean_cost": float(mean_cost),
        "std_cost": float(std_cost),
        "robust_cost": float(robust_cost),
        "avg_vs_avg_mps": float(sum(vs_vals) / n),
        "avg_soft_violations": float(sum(soft_vals) / n),
        "avg_medium_violations": float(sum(med_vals) / n),
        "avg_hard_violations": float(sum(hard_vals) / n),
        "per_track": per_track,
        "from_cache": False,
    }

    EVAL_CACHE[cache_key] = copy.deepcopy(out)
    return out


# ============================================================
# Candidate selection
# ============================================================
def _topk_best_so_far(history: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    ranked = sorted(history, key=lambda r: (_train_objective(r), int(r.get("candidate_id", 10**18))))
    out: List[Dict[str, Any]] = []
    seen = set()
    for rec in ranked:
        theta = {k: float(v) for k, v in rec["theta"].items()}
        sig = _theta_signature(theta)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(rec)
        if len(out) >= max(1, topk):
            break
    return out


def _build_G_L(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ranked = sorted(records, key=_train_objective)
    n = len(ranked)
    if n == 0:
        return [], []

    g_n = max(MIN_G_SIZE, int(math.ceil(GOOD_FRAC * n)))
    g_n = min(g_n, n)
    return ranked[:g_n], ranked[g_n:]


def _algorithm_thinks_theta(history: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, float]], str]:
    if len(history) == 0:
        return None, "no_history"

    G, _ = _build_G_L(history)
    center = _weighted_center_from_records(G, ALL_BOUNDS)
    return center, "unstaged_G_center"


# ============================================================
# CSV row build
# ============================================================
CSV_FIELDS = [
    "checkpoint_trial",
    "checkpoint_phase",
    "selection_type",
    "selection_label",
    "source_candidate_id",
    "source_phase",
    "source_objective_cost_train",
    "source_mean_cost_train",
    "source_std_cost_train",
    "source_best_objective_so_far_train",
    "test_tracks",
    "test_episodes",
    "test_crash_count",
    "test_crash_rate",
    "test_mean_cost",
    "test_std_cost",
    "test_robust_cost",
    "test_avg_vs_avg_mps",
    "test_avg_soft_violations",
    "test_avg_medium_violations",
    "test_avg_hard_violations",
    "test_critical_crash_multiplier",
    "cache_hit",
    "theta_json",
]


def _make_csv_row(
    checkpoint_trial: int,
    checkpoint_phase: str,
    selection_type: str,
    selection_label: str,
    theta: Dict[str, float],
    eval_out: Dict[str, Any],
    source_candidate_id: Optional[int] = None,
    source_phase: Optional[str] = None,
    source_objective_cost_train: Optional[float] = None,
    source_mean_cost_train: Optional[float] = None,
    source_std_cost_train: Optional[float] = None,
    source_best_objective_so_far_train: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "checkpoint_trial": int(checkpoint_trial),
        "checkpoint_phase": str(checkpoint_phase),
        "selection_type": str(selection_type),
        "selection_label": str(selection_label),
        "source_candidate_id": "" if source_candidate_id is None else int(source_candidate_id),
        "source_phase": "" if source_phase is None else str(source_phase),
        "source_objective_cost_train": "" if source_objective_cost_train is None else float(source_objective_cost_train),
        "source_mean_cost_train": "" if source_mean_cost_train is None else float(source_mean_cost_train),
        "source_std_cost_train": "" if source_std_cost_train is None else float(source_std_cost_train),
        "source_best_objective_so_far_train": "" if source_best_objective_so_far_train is None else float(source_best_objective_so_far_train),
        "test_tracks": json.dumps(eval_out["test_tracks"]),
        "test_episodes": int(eval_out["episodes"]),
        "test_crash_count": int(eval_out["crash_count"]),
        "test_crash_rate": float(eval_out["crash_rate"]),
        "test_mean_cost": float(eval_out["mean_cost"]),
        "test_std_cost": float(eval_out["std_cost"]),
        "test_robust_cost": float(eval_out["robust_cost"]),
        "test_avg_vs_avg_mps": float(eval_out["avg_vs_avg_mps"]),
        "test_avg_soft_violations": float(eval_out["avg_soft_violations"]),
        "test_avg_medium_violations": float(eval_out["avg_medium_violations"]),
        "test_avg_hard_violations": float(eval_out["avg_hard_violations"]),
        "test_critical_crash_multiplier": float(TEST_CRITICAL_CRASH_MULTIPLIER),
        "cache_hit": int(bool(eval_out.get("from_cache", False))),
        "theta_json": json.dumps(theta, sort_keys=True),
    }


# ============================================================
# Main eval routine
# ============================================================
def main() -> None:
    global BASE_CONTROL_TEMPLATE

    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(ACADOS_LIB, "ACADOS_LIB")
    _must_exist(RUN_LOG_PATH, "RUN_LOG_PATH")

    original_control_json = _load_json(CONTROL_PARAM_JSON)
    BASE_CONTROL_TEMPLATE = copy.deepcopy(original_control_json)

    try:
        if os.path.exists(OUTPUT_CSV):
            os.remove(OUTPUT_CSV)
    except Exception:
        pass

    print("[CFG] PROFILE_NAME            =", PROFILE_NAME)
    print("[CFG] CATKIN_WS               =", CATKIN_WS)
    print("[CFG] RUN_LOG_PATH            =", RUN_LOG_PATH)
    print("[CFG] OUTPUT_CSV              =", OUTPUT_CSV)
    print("[CFG] CANDIDATES_JSON         =", CANDIDATES_JSON)
    print("[CFG] ACADOS_LIB              =", ACADOS_LIB)
    print("[CFG] TRAIN_TRACK_SET         =", TRAIN_TRACK_SET)
    print("[CFG] TEST_TRACK_SET          =", TEST_TRACK_SET)
    print("[CFG] EVAL_SIM_TIME_S         =", DEFAULT_SIM_TIME_S)
    print("[CFG] UCB_WEIGHT              =", UCB_WEIGHT)
    print("[CFG] BEST_SO_FAR_EVERY       =", BEST_SO_FAR_EVERY)
    print("[CFG] THINK_EVERY             =", THINK_EVERY)
    print("[CFG] TOPK                    =", TOPK)
    print("[CFG] TEST_CRASH_MULTIPLIER   =", TEST_CRITICAL_CRASH_MULTIPLIER)
    print("[CFG] UNSTAGED_N_TRIALS       =", UNSTAGED_N_TRIALS)
    print("[CFG] UNSTAGED_STARTUP        =", UNSTAGED_STARTUP)
    print("")

    records = _load_jsonl(RUN_LOG_PATH)
    if len(records) == 0:
        print("[FATAL] Empty tuning jsonl.", file=sys.stderr)
        sys.exit(2)

    for rec in records:
        if str(rec.get("phase", "")) != UNSTAGED_PHASE_NAME:
            print(
                f"[WARN] Found phase={rec.get('phase')} in run log; expected {UNSTAGED_PHASE_NAME}. "
                "I will still evaluate all records present."
            )
            break

    print(f"[INFO] Loaded {len(records)} tuning records.")

    candidate_payload: List[Dict[str, Any]] = []

    roscore_p = None
    rows_written = 0
    interrupted = False

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
        # ----------------------------------------------------
        # best-so-far top-k at checkpoints
        # ----------------------------------------------------
        for checkpoint in range(BEST_SO_FAR_EVERY, len(records) + 1, BEST_SO_FAR_EVERY):
            history = records[:checkpoint]
            checkpoint_phase = str(history[-1].get("phase", ""))
            best_recs = _topk_best_so_far(history, TOPK)
            best_obj_so_far = min((_train_objective(r) for r in history), default=float("nan"))

            print(
                f"\n[CHECKPOINT {checkpoint:03d}] best-so-far top-{len(best_recs)} "
                f"(phase={checkpoint_phase})"
            )

            for rank, rec in enumerate(best_recs, start=1):
                theta = {k: float(v) for k, v in rec["theta"].items()}
                eval_out = evaluate_theta_on_test_tracks(theta)

                row = _make_csv_row(
                    checkpoint_trial=checkpoint,
                    checkpoint_phase=checkpoint_phase,
                    selection_type="best_so_far",
                    selection_label=f"top{rank}",
                    theta=theta,
                    eval_out=eval_out,
                    source_candidate_id=int(rec.get("candidate_id", -1)),
                    source_phase=str(rec.get("phase", "")),
                    source_objective_cost_train=float(_train_objective(rec)),
                    source_mean_cost_train=float(rec.get("mean_cost", float("nan"))),
                    source_std_cost_train=float(rec.get("std_cost", float("nan"))),
                    source_best_objective_so_far_train=float(best_obj_so_far),
                )
                _append_csv_row(OUTPUT_CSV, CSV_FIELDS, row)
                rows_written += 1

                candidate_payload.append(
                    {
                        "checkpoint_trial": int(checkpoint),
                        "checkpoint_phase": checkpoint_phase,
                        "selection_type": "best_so_far",
                        "selection_label": f"top{rank}",
                        "source_candidate_id": int(rec.get("candidate_id", -1)),
                        "source_phase": str(rec.get("phase", "")),
                        "source_objective_cost_train": float(_train_objective(rec)),
                        "source_mean_cost_train": float(rec.get("mean_cost", float("nan"))),
                        "source_std_cost_train": float(rec.get("std_cost", float("nan"))),
                        "source_best_objective_so_far_train": float(best_obj_so_far),
                        "theta": theta,
                    }
                )

                print(
                    f"  [top{rank}] cand_id={rec.get('candidate_id')} "
                    f"train_obj={float(_train_objective(rec)):.6g} "
                    f"test_mean={eval_out['mean_cost']:.6g} "
                    f"test_std={eval_out['std_cost']:.6g} "
                    f"test_robust={eval_out['robust_cost']:.6g} "
                    f"crashes={eval_out['crash_count']}/{eval_out['episodes']} "
                    f"from_cache={bool(eval_out.get('from_cache', False))}"
                )

        # ----------------------------------------------------
        # what the unstaged algorithm currently "thinks"
        # ----------------------------------------------------
        for checkpoint in range(THINK_EVERY, len(records) + 1, THINK_EVERY):
            history = records[:checkpoint]
            checkpoint_phase = str(history[-1].get("phase", ""))
            theta, think_label = _algorithm_thinks_theta(history)
            if theta is None:
                print(f"[WARN] checkpoint={checkpoint}: cannot build think-theta.")
                continue

            best_obj_so_far = min((_train_objective(r) for r in history), default=float("nan"))

            print(
                f"\n[CHECKPOINT {checkpoint:03d}] algorithm_thinks "
                f"(phase={checkpoint_phase}, kind={think_label})"
            )

            eval_out = evaluate_theta_on_test_tracks(theta)

            row = _make_csv_row(
                checkpoint_trial=checkpoint,
                checkpoint_phase=checkpoint_phase,
                selection_type="algorithm_thinks",
                selection_label=think_label,
                theta=theta,
                eval_out=eval_out,
                source_phase=checkpoint_phase,
                source_best_objective_so_far_train=float(best_obj_so_far),
            )
            _append_csv_row(OUTPUT_CSV, CSV_FIELDS, row)
            rows_written += 1

            candidate_payload.append(
                {
                    "checkpoint_trial": int(checkpoint),
                    "checkpoint_phase": checkpoint_phase,
                    "selection_type": "algorithm_thinks",
                    "selection_label": think_label,
                    "source_candidate_id": None,
                    "source_phase": checkpoint_phase,
                    "source_objective_cost_train": float("nan"),
                    "source_mean_cost_train": float("nan"),
                    "source_std_cost_train": float("nan"),
                    "source_best_objective_so_far_train": float(best_obj_so_far),
                    "theta": {k: float(v) for k, v in theta.items()},
                }
            )

            print(
                f"  [think] kind={think_label} "
                f"test_mean={eval_out['mean_cost']:.6g} "
                f"test_std={eval_out['std_cost']:.6g} "
                f"test_robust={eval_out['robust_cost']:.6g} "
                f"crashes={eval_out['crash_count']}/{eval_out['episodes']} "
                f"from_cache={bool(eval_out.get('from_cache', False))}"
            )

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Interrupted by user. Partial CSV has been kept.")

    finally:
        try:
            _save_json_atomic(CANDIDATES_JSON, candidate_payload)
        except Exception as e:
            print(f"[WARN] Could not save candidates json: {e}", file=sys.stderr)

        try:
            _save_json_atomic(CONTROL_PARAM_JSON, original_control_json)
        except Exception:
            pass

        if roscore_p is not None:
            _kill_process_group(roscore_p)

    print("\n================ DONE ================\n")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved candidates JSON: {CANDIDATES_JSON}")
    print(f"Unique theta cached: {len(EVAL_CACHE)}")
    print(f"Rows written: {rows_written}")

    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
