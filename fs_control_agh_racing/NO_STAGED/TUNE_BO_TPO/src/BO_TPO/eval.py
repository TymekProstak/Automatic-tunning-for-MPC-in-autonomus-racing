#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import time
import math
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
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_agh_racing/TUNE_BO_TPO")
CATKIN_WS = os.path.abspath(
    os.environ.get("EVAL_WS", os.environ.get("TUNE_WS", DEFAULT_CATKIN_WS))
)
WS_SRC = os.path.join(CATKIN_WS, "src")
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

# ============================================================
# Eval profiles (UNSTAGED)
# ============================================================
EVAL_PROFILE_NAME = os.environ.get("EVAL_PROFILE_NAME", "night_7track").strip()

if EVAL_PROFILE_NAME == "smoke_3track":
    TRAIN_PROFILE_NAME = "smoke_1track"
    DEFAULT_EVAL_TRACK_SET = "8,11,14"
    DEFAULT_CHECKPOINT_STEP = "2"

    # Suma z poprzednich faz: n_trials = 12+24+24 = 60, startup = 5+12+8 = 25
    N_TRIALS = 60
    STARTUP = 25

elif EVAL_PROFILE_NAME == "night_7track":
    TRAIN_PROFILE_NAME = "night_4track"
    DEFAULT_EVAL_TRACK_SET = "8,9,10,11,12,13,14"
    DEFAULT_CHECKPOINT_STEP = "4"

    # Suma z poprzednich faz: n_trials = 48+64+96 = 208, startup = 12+16+24 = 52
    N_TRIALS = 208
    STARTUP = 52

else:
    raise ValueError(f"Unknown EVAL_PROFILE_NAME: {EVAL_PROFILE_NAME}")

# Optional overrides: must match the tuning run log
N_TRIALS = int(os.environ.get("EVAL_N_TRIALS", str(N_TRIALS)))
STARTUP = int(os.environ.get("EVAL_STARTUP", str(STARTUP)))

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

RUN_LOG_PATH = os.path.abspath(
    os.environ.get(
        "EVAL_RUN_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_run_tpe_tv_unstaged_{TRAIN_PROFILE_NAME}.jsonl"),
    )
)
OUTPUT_CSV_PATH = os.path.abspath(
    os.environ.get(
        "EVAL_OUTPUT_CSV",
        os.path.join(SCRIPT_DIR, f"tpe_tv_test_eval_{EVAL_PROFILE_NAME}_unstaged.csv"),
    )
)
OUTPUT_CANDIDATES_JSON = os.path.abspath(
    os.environ.get(
        "EVAL_OUTPUT_CANDIDATES_JSON",
        os.path.join(SCRIPT_DIR, f"tpe_tv_test_candidates_{EVAL_PROFILE_NAME}_unstaged.json"),
    )
)
PLOTS_DIR = os.path.abspath(
    os.environ.get(
        "EVAL_PLOTS_DIR",
        os.path.join(SCRIPT_DIR, f"tpe_tv_eval_plots_{EVAL_PROFILE_NAME}_unstaged"),
    )
)

# ============================================================
# ACADOS lib needed by dv_control in this workspace
# ============================================================
DEFAULT_ACADOS_LIB = os.path.join(
    WS_SRC, "dv_control", "External", "acados", "install", "lib"
)
ACADOS_LIB = os.path.abspath(
    os.environ.get(
        "EVAL_ACADOS_LIB",
        os.environ.get("TUNE_ACADOS_LIB", DEFAULT_ACADOS_LIB),
    )
)

# ============================================================
# Eval config
# ============================================================
GLOBAL_SEED = 123
random.seed(GLOBAL_SEED)

DEFAULT_SIM_TIME_S = int(os.environ.get("EVAL_SIM_TIME_S", "60"))
UCB_WEIGHT = 1.0


def _parse_track_set() -> List[int]:
    raw = os.environ.get("EVAL_TRACK_SET", DEFAULT_EVAL_TRACK_SET).strip()
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("EVAL_TRACK_SET resolved to empty list")
    return vals


TEST_TRACKS = _parse_track_set()
TEST_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("EVAL_TEST_CRITICAL_CRASH_MULTIPLIER", "1.0")
)

# ============================================================
# TPE evaluation-side "what algorithm thinks"
# ============================================================
LOCAL_LHS_N = int(os.environ.get("EVAL_LOCAL_LHS_N", "4096"))
LOCAL_BOX_EXPAND_FACTOR = float(os.environ.get("EVAL_LOCAL_BOX_EXPAND_FACTOR", "1.5"))
LOCAL_BOX_MIN_GLOBAL_FRAC = float(os.environ.get("EVAL_LOCAL_BOX_MIN_GLOBAL_FRAC", "0.10"))

# ============================================================
# Tuner config (UNSTAGED)
# ============================================================
TOTAL_EFFECTIVE_TRIALS = N_TRIALS - STARTUP
CHECKPOINT_STEP = int(os.environ.get("EVAL_CHECKPOINT_STEP", DEFAULT_CHECKPOINT_STEP))

SAFE_PLANNER = {
    "velocity_planner.v_max": 13.5,
    "velocity_planner.mux_acc": 0.7,
    "velocity_planner.mux_dec": 0.6,
    "velocity_planner.muy": 0.75,
}

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
    "mpc.cost.R_tv": (1e-8, 1e-5),
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
ROS_MASTER_PORT = int(os.environ.get("EVAL_ROS_MASTER_PORT", "11616"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_eval_tpe_tv_{EVAL_PROFILE_NAME}_{GLOBAL_SEED}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

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
# CSV schema
# ============================================================
CSV_FIELDS = [
    "selection_family",
    "selection_label",
    "checkpoint_step",
    "n_trials",
    "startup",
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
    "source_tpe_score_train",
    "is_tail_padded",
    "cache_hit",
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
    return p

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

# ============================================================
# Cost
# ============================================================
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
            rec["_global_trial"] = int(idx)
            if "candidate_id" not in rec:
                rec["candidate_id"] = int(idx)
            out.append(rec)
    out.sort(key=lambda r: int(r.get("_global_trial", 0)))
    return out

# ============================================================
# Unstaged Filtering Helpers
# ============================================================
def _is_startup_record(rec: Dict[str, Any]) -> bool:
    return int(rec.get("_global_trial", 1)) <= STARTUP

def _filtered_to_global_trial(filtered_trial: int) -> int:
    if filtered_trial <= 0:
        return 0
    return STARTUP + filtered_trial

def _build_filtered_checkpoints(step: int) -> List[int]:
    if step <= 0:
        return []
    return list(range(step, TOTAL_EFFECTIVE_TRIALS + 1, step))

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
    checkpoint_trial_filtered: int
    checkpoint_trial_global: int
    source_candidate_id: int
    source_phase: str
    source_phase_trial: int
    source_trial_global: int
    source_objective_cost_train: float
    source_mean_cost_train: float
    source_std_cost_train: float
    source_robust_cost_train: float
    source_best_objective_so_far_train: float
    source_tpe_score_train: Optional[float]
    is_tail_padded: bool
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


def evaluate_theta_on_test_tracks(
    theta_x: Dict[str, float],
    tracks: List[int],
    sim_time_s: int,
    critical_crash_multiplier: float,
    eval_cache: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]],
) -> Dict[str, Any]:
    cache_key = _theta_signature(theta_x)
    if cache_key in eval_cache:
        out = copy.deepcopy(eval_cache[cache_key])
        out["from_cache"] = True
        return out

    per_track_results: List[EpisodeResult] = []

    for tid in tracks:
        res = run_one_episode(
            theta_x=theta_x,
            track_index=tid,
            sim_time_s=sim_time_s,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        per_track_results.append(res)

    mean_cost = sum(float(r.cost) for r in per_track_results) / max(1, len(per_track_results))
    var_cost = sum((float(r.cost) - mean_cost) * (float(r.cost) - mean_cost) for r in per_track_results) / max(1, len(per_track_results))
    std_cost = math.sqrt(max(0.0, var_cost))
    robust_cost = mean_cost + UCB_WEIGHT * std_cost

    mean_vs_avg_mps = sum(
        _metric_float_first(r.metrics, ["vs_avg_mps"], 0.0) for r in per_track_results
    ) / max(1, len(per_track_results))
    mean_soft = sum(
        _metric_float_first(
            r.metrics,
            ["soft_track_violations_count", "soft_track_violation_count", "soft_track_violation_count_"],
            0.0,
        )
        for r in per_track_results
    ) / max(1, len(per_track_results))
    mean_medium = sum(
        _metric_float_first(
            r.metrics,
            ["medium_track_violations_count", "medium_track_violation_count", "medium_track_violation_count_"],
            0.0,
        )
        for r in per_track_results
    ) / max(1, len(per_track_results))
    mean_hard = sum(
        _metric_float_first(
            r.metrics,
            ["high_track_violations_count", "high_track_violation_count", "high_track_violation_count_"],
            0.0,
        )
        for r in per_track_results
    ) / max(1, len(per_track_results))

    crash_count = sum(1 for r in per_track_results if r.crashed)
    crash_percent = 100.0 * float(crash_count) / max(1.0, float(len(per_track_results)))

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

    eval_cache[cache_key] = copy.deepcopy(out)
    return out


# ============================================================
# LHS helpers
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
        g_vals = [_x_to_z(k, float(r["theta"][k])) for r in G if k in r["theta"]]
        l_vals = [_x_to_z(k, float(r["theta"][k])) for r in L if k in r["theta"]]

        # Fallbacks just in case a key is missing from history
        if not g_vals:
            continue

        bw_g = _bandwidth_1d(g_vals, lo_z, hi_z)
        bw_l = _bandwidth_1d(l_vals, lo_z, hi_z) if l_vals else bw_g

        lg = _mixture_logpdf_1d(z[k], g_vals, bw_g)
        ll = _mixture_logpdf_1d(z[k], l_vals, bw_l) if l_vals else -1e12

        score += (lg - ll)

    return float(score)

# ============================================================
# G / L builders
# ============================================================
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

# ============================================================
# Local LHS around G
# ============================================================
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
    fixed_params: Optional[Dict[str, float]],
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

    for th_part in cloud:
        th: Dict[str, float] = {}
        if fixed_params:
            th.update({k: float(v) for k, v in fixed_params.items()})
        th.update({k: float(v) for k, v in th_part.items()})

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
# Candidate selection (UNSTAGED)
# ============================================================
def _select_best_so_far_schedule(records: List[Dict[str, Any]]) -> List[EvalTask]:
    selected: List[EvalTask] = []

    non_startup = [r for r in records if not _is_startup_record(r)]
    non_startup.sort(key=lambda r: int(r.get("_global_trial", 0)))

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
            EvalTask(
                selection_family="best_so_far",
                selection_label=f"best_so_far_t{checkpoint_filtered:03d}",
                checkpoint_trial_filtered=int(checkpoint_filtered),
                checkpoint_trial_global=int(checkpoint_global),
                source_candidate_id=int(best_rec.get("candidate_id", best_rec.get("_global_trial", -1))),
                source_phase=str(best_rec.get("phase", "")),
                source_phase_trial=int(best_rec.get("phase_trial", -1)),
                source_trial_global=int(best_rec.get("_global_trial", -1)),
                source_objective_cost_train=float(_train_objective(best_rec)),
                source_mean_cost_train=float(best_rec.get("mean_cost", float("nan"))),
                source_std_cost_train=float(best_rec.get("std_cost", float("nan"))),
                source_robust_cost_train=float(best_rec.get("robust_cost", _train_objective(best_rec))),
                source_best_objective_so_far_train=float(best_obj_so_far),
                source_tpe_score_train=None,
                is_tail_padded=bool(tail_padded),
                theta={k: float(v) for k, v in best_rec["theta"].items()},
            )
        )

    return selected

def _select_algorithm_thinks_schedule(records: List[Dict[str, Any]]) -> List[EvalTask]:
    selected: List[EvalTask] = []

    non_startup = [r for r in records if not _is_startup_record(r)]
    non_startup.sort(key=lambda r: int(r.get("_global_trial", 0)))

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

        src_rec = history[-1]
        best_obj_so_far = min(_train_objective(r) for r in history)

        G, L = _build_G_L_generic(history)
        if len(G) == 0:
            continue

        # W wersji unstaged używamy pełnych granic (ALL_BOUNDS) w każdym kroku
        theta_est, score = build_tpe_local_lhs_estimate(
            G=G,
            L=L,
            bounds=ALL_BOUNDS,
            fixed_params=None,
            seed=GLOBAL_SEED + 1000000 + checkpoint_filtered,
        )

        label = f"algorithm_thinks_unstaged_t{checkpoint_filtered:03d}"

        selected.append(
            EvalTask(
                selection_family="algorithm_thinks",
                selection_label=label,
                checkpoint_trial_filtered=int(checkpoint_filtered),
                checkpoint_trial_global=int(checkpoint_global),
                source_candidate_id=int(src_rec.get("candidate_id", src_rec.get("_global_trial", -1))),
                source_phase=str(src_rec.get("phase", "")),
                source_phase_trial=int(src_rec.get("phase_trial", -1)),
                source_trial_global=int(src_rec.get("_global_trial", -1)),
                source_objective_cost_train=float(_train_objective(src_rec)),
                source_mean_cost_train=float(src_rec.get("mean_cost", float("nan"))),
                source_std_cost_train=float(src_rec.get("std_cost", float("nan"))),
                source_robust_cost_train=float(src_rec.get("robust_cost", _train_objective(src_rec))),
                source_best_objective_so_far_train=float(best_obj_so_far),
                source_tpe_score_train=float(score),
                is_tail_padded=bool(tail_padded),
                theta={k: float(v) for k, v in theta_est.items()},
            )
        )

    return selected

# ============================================================
# Candidate JSON / plotting
# ============================================================
def _task_to_jsonable(task: EvalTask) -> Dict[str, Any]:
    return {
        "selection_family": task.selection_family,
        "selection_label": task.selection_label,
        "checkpoint_trial_filtered": int(task.checkpoint_trial_filtered),
        "checkpoint_trial_global": int(task.checkpoint_trial_global),
        "source_candidate_id": int(task.source_candidate_id),
        "source_phase": str(task.source_phase),
        "source_phase_trial": int(task.source_phase_trial),
        "source_trial_global": int(task.source_trial_global),
        "source_objective_cost_train": float(task.source_objective_cost_train),
        "source_mean_cost_train": float(task.source_mean_cost_train),
        "source_std_cost_train": float(task.source_std_cost_train),
        "source_robust_cost_train": float(task.source_robust_cost_train),
        "source_best_objective_so_far_train": float(task.source_best_objective_so_far_train),
        "source_tpe_score_train": None if task.source_tpe_score_train is None else float(task.source_tpe_score_train),
        "is_tail_padded": bool(task.is_tail_padded),
        "theta": {k: float(v) for k, v in task.theta.items()},
    }

def _save_plot(rows: List[Dict[str, Any]], selection_family: str, y_key: str, y_label: str) -> None:
    xs = []
    ys = []

    family_rows = [r for r in rows if r["selection_family"] == selection_family]
    family_rows.sort(key=lambda r: int(r["checkpoint_trial_filtered"]))

    for row in family_rows:
        xs.append(int(row["checkpoint_trial_filtered"]))
        ys.append(float(row[y_key]))

    if not xs:
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker="o")
    # Z wersji unstaged usunięto plt.axvline oznaczające końce faz, bo faz już nie ma.
    plt.xlabel("Numer triala po odfiltrowaniu startupów")
    plt.ylabel(y_label)
    plt.title(f"{selection_family}: {y_key}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"{selection_family}_{y_key}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()

def _save_family_plots(rows: List[Dict[str, Any]], selection_family: str) -> None:
    metrics = [
        ("test_robust_cost", "Robust cost"),
        ("test_mean_cost", "Mean cost"),
        ("test_crash_percent", "Crash percent"),
        ("test_avg_vs_avg_mps", "Average vs_avg_mps"),
        ("test_avg_soft_violations", "Average soft violations"),
        ("test_avg_medium_violations", "Average medium violations"),
        ("test_avg_hard_violations", "Average hard violations"),
    ]

    for y_key, y_label in metrics:
        _save_plot(rows, selection_family, y_key, y_label)

# ============================================================
# Main
# ============================================================
def main() -> None:
    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(WS_SRC, "WS_SRC")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(ACADOS_LIB, "ACADOS_LIB")
    _must_exist(RUN_LOG_PATH, "RUN_LOG_PATH")

    try:
        if os.path.exists(OUTPUT_CSV_PATH):
            os.remove(OUTPUT_CSV_PATH)
    except Exception:
        pass

    original_control_json = _load_json(CONTROL_PARAM_JSON)

    records = load_jsonl(RUN_LOG_PATH)
    if len(records) == 0:
        raise RuntimeError(f"Empty run log: {RUN_LOG_PATH}")

    best_tasks = _select_best_so_far_schedule(records)
    algo_tasks = _select_algorithm_thinks_schedule(records)

    tasks = best_tasks + algo_tasks
    tasks.sort(
        key=lambda t: (
            int(t.checkpoint_trial_filtered),
            0 if t.selection_family == "algorithm_thinks" else 1,
        )
    )

    print("[CFG] CATKIN_WS               =", CATKIN_WS)
    print("[CFG] TRAIN_PROFILE_NAME      =", TRAIN_PROFILE_NAME)
    print("[CFG] EVAL_PROFILE_NAME       =", EVAL_PROFILE_NAME)
    print("[CFG] RUN_LOG_PATH            =", RUN_LOG_PATH)
    print("[CFG] OUTPUT_CSV              =", OUTPUT_CSV_PATH)
    print("[CFG] OUTPUT_CANDIDATES_JSON  =", OUTPUT_CANDIDATES_JSON)
    print("[CFG] PLOTS_DIR               =", PLOTS_DIR)
    print("[CFG] ACADOS_LIB              =", ACADOS_LIB)
    print("[CFG] TEST_TRACK_SET          =", TEST_TRACKS)
    print("[CFG] EVAL_SIM_TIME_S         =", DEFAULT_SIM_TIME_S)
    print("[CFG] TEST_CRASH_MULTIPLIER   =", TEST_CRITICAL_CRASH_MULTIPLIER)
    print("[CFG] UCB_WEIGHT              =", UCB_WEIGHT)
    print("[CFG] N_TRIALS                =", N_TRIALS)
    print("[CFG] STARTUP                 =", STARTUP)
    print("[CFG] TOTAL_EFFECTIVE_TRIALS  =", TOTAL_EFFECTIVE_TRIALS)
    print("[CFG] CHECKPOINT_STEP         =", CHECKPOINT_STEP)
    print("[CFG] LOCAL_LHS_N             =", LOCAL_LHS_N)
    print("[CFG] LOCAL_BOX_EXPAND_FACTOR =", LOCAL_BOX_EXPAND_FACTOR)
    print("[CFG] LOCAL_BOX_MIN_GLOBAL_FRAC =", LOCAL_BOX_MIN_GLOBAL_FRAC)
    print("")
    print(f"[INFO] Loaded {len(records)} tuning records from run log jsonl.")
    print("[INFO] best_so_far biorę z pełnego run loga (UNSTAGED).")
    print("[INFO] algorithm_thinks = TPE local-LHS approximation na całej przestrzeni ALL_BOUNDS.")
    print("")
    print(f"[INFO] Selected {len(best_tasks)} best_so_far checkpoints.")
    print(f"[INFO] Selected {len(algo_tasks)} algorithm_thinks checkpoints.")
    print("")

    _save_json_atomic(
        OUTPUT_CANDIDATES_JSON,
        [_task_to_jsonable(t) for t in tasks],
    )

    eval_cache: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []
    episodes_done = 0
    interrupted = False

    roscore_p = None
    try:
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

        for idx, task in enumerate(tasks, start=1):
            theta = {k: float(v) for k, v in task.theta.items()}
            sig = _theta_signature(theta)

            cache_hit = sig in eval_cache
            if not cache_hit:
                print(
                    f"[EVAL {idx:03d}/{len(tasks):03d}] "
                    f"{task.selection_label} -> running test tracks {TEST_TRACKS}"
                )
                summary = evaluate_theta_on_test_tracks(
                    theta_x=theta,
                    tracks=TEST_TRACKS,
                    sim_time_s=DEFAULT_SIM_TIME_S,
                    critical_crash_multiplier=TEST_CRITICAL_CRASH_MULTIPLIER,
                    eval_cache=eval_cache,
                )
                episodes_done += len(TEST_TRACKS)
            else:
                print(
                    f"[EVAL {idx:03d}/{len(tasks):03d}] "
                    f"{task.selection_label} -> cache hit"
                )
                summary = copy.deepcopy(eval_cache[sig])
                summary["from_cache"] = True

            row = {
                "selection_family": task.selection_family,
                "selection_label": task.selection_label,
                "checkpoint_step": int(CHECKPOINT_STEP),
                "n_trials": int(N_TRIALS),
                "startup": int(STARTUP),
                "checkpoint_trial_filtered": int(task.checkpoint_trial_filtered),
                "checkpoint_trial_global": int(task.checkpoint_trial_global),
                "source_candidate_id": int(task.source_candidate_id),
                "source_phase": str(task.source_phase),
                "source_phase_trial": int(task.source_phase_trial),
                "source_trial_global": int(task.source_trial_global),
                "source_objective_cost_train": float(task.source_objective_cost_train),
                "source_mean_cost_train": float(task.source_mean_cost_train),
                "source_std_cost_train": float(task.source_std_cost_train),
                "source_robust_cost_train": float(task.source_robust_cost_train),
                "source_best_objective_so_far_train": float(task.source_best_objective_so_far_train),
                "source_tpe_score_train": "" if task.source_tpe_score_train is None else float(task.source_tpe_score_train),
                "is_tail_padded": int(task.is_tail_padded),
                "cache_hit": int(cache_hit),
                "test_track_ids": summary["test_track_ids"],
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
                "theta_json": json.dumps(theta, sort_keys=True),
            }

            csv_rows.append(row)
            _append_csv_row(OUTPUT_CSV_PATH, CSV_FIELDS, row)

            print(
                f"    robust={row['test_robust_cost']:.6g} | "
                f"mean={row['test_mean_cost']:.6g} | "
                f"std={row['test_std_cost']:.6g} | "
                f"vs_avg={row['test_avg_vs_avg_mps']:.6g} | "
                f"soft={row['test_avg_soft_violations']:.6g} | "
                f"medium={row['test_avg_medium_violations']:.6g} | "
                f"hard={row['test_avg_hard_violations']:.6g} | "
                f"crash%={row['test_crash_percent']:.6g} | "
                f"from_cache={int(cache_hit)}"
            )

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Interrupted by user. Saving partial artifacts.")

    finally:
        try:
            if csv_rows:
                _save_family_plots(csv_rows, "best_so_far")
                _save_family_plots(csv_rows, "algorithm_thinks")
        except Exception as e:
            print(f"[WARN] Plot save failed: {e}", file=sys.stderr)

        try:
            _save_json_atomic(CONTROL_PARAM_JSON, original_control_json)
        except Exception:
            pass

        if roscore_p is not None:
            _kill_process_group(roscore_p)

    print("")
    print("================ EVAL DONE ================")
    print(f"Rows written        : {len(csv_rows)}")
    print(f"Unique thetas evald : {len(eval_cache)}")
    print(f"Episodes done       : {episodes_done}")
    print(f"Output CSV          : {OUTPUT_CSV_PATH}")
    print(f"Candidates JSON     : {OUTPUT_CANDIDATES_JSON}")
    print(f"Plots dir           : {PLOTS_DIR}")

    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
