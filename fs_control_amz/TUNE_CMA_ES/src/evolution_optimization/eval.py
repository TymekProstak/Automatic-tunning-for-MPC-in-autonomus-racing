#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import time
import math
import copy
import signal
import socket
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Workspace
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_amz/TUNE_CMA_ES")
CATKIN_WS = os.path.abspath(
    os.environ.get("EVAL_WS", os.environ.get("TUNE_WS", DEFAULT_CATKIN_WS))
)
WS_SRC = os.path.join(CATKIN_WS, "src")
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

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
ACADOS_LIB = os.path.abspath(
    os.environ.get("EVAL_ACADOS_LIB", os.environ.get("TUNE_ACADOS_LIB", DEFAULT_ACADOS_LIB))
)

# ============================================================
# Evaluation setup
# ============================================================
DEFAULT_SIM_TIME_S = int(os.environ.get("EVAL_SIM_TIME_S", "60"))
UCB_WEIGHT = float(os.environ.get("EVAL_UCB_WEIGHT", os.environ.get("TUNE_UCB_WEIGHT", "1.0")))

# ============================================================
# Profile / exact staged budget from tuner
# ============================================================
PROFILE_NAME = os.environ.get("EVAL_PROFILE", os.environ.get("TUNE_PROFILE", "night_4track")).strip()

if PROFILE_NAME == "smoke_1track":
    TRAIN_TRACK_SET = [1]
    DEFAULT_TEST_TRACK_SET = "8,11,14"
    DEFAULT_CHECKPOINT_STEP = "2"

    PHASE1_N_TRIALS = 12
    PHASE1_LAMBDA = 4
    PHASE1_MAXITER = 3

    PHASE2_N_TRIALS = 24
    PHASE2_LAMBDA = 6
    PHASE2_MAXITER = 4

    PHASE3_N_TRIALS = 24
    PHASE3_LAMBDA = 6
    PHASE3_MAXITER = 4

elif PROFILE_NAME == "night_4track":
    TRAIN_TRACK_SET = [1, 2, 3, 4]
    DEFAULT_TEST_TRACK_SET = "8,9,10,11,12,13,14"
    DEFAULT_CHECKPOINT_STEP = "4"

    PHASE1_N_TRIALS = 64
    PHASE1_LAMBDA = 8
    PHASE1_MAXITER = 8

    PHASE2_N_TRIALS = 64
    PHASE2_LAMBDA = 8
    PHASE2_MAXITER = 8

    PHASE3_N_TRIALS = 96
    PHASE3_LAMBDA = 8
    PHASE3_MAXITER = 12

else:
    raise ValueError(f"Unknown PROFILE_NAME: {PROFILE_NAME}")

assert PHASE1_LAMBDA * PHASE1_MAXITER == PHASE1_N_TRIALS
assert PHASE2_LAMBDA * PHASE2_MAXITER == PHASE2_N_TRIALS
assert PHASE3_LAMBDA * PHASE3_MAXITER == PHASE3_N_TRIALS

TOTAL_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS
STAGE1_END_TRIAL = PHASE1_N_TRIALS
STAGE2_END_TRIAL = PHASE1_N_TRIALS + PHASE2_N_TRIALS

# ============================================================
# Test setup
# ============================================================
def _parse_track_set(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("Resolved empty EVAL_TRACK_SET")
    return out


TEST_TRACK_SET = _parse_track_set(os.environ.get("EVAL_TRACK_SET", DEFAULT_TEST_TRACK_SET))
TEST_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("EVAL_TEST_CRITICAL_CRASH_MULTIPLIER", "1.0")
)

CHECKPOINT_STEP = int(os.environ.get("EVAL_CHECKPOINT_STEP", DEFAULT_CHECKPOINT_STEP))

# ============================================================
# ROS / runtime
# ============================================================
SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("EVAL_ROS_MASTER_PORT", "11721"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_test_eval_cma_full_coupled_{PROFILE_NAME}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Outputs
# ============================================================
RUN_LOG_PATH = os.path.abspath(
    os.environ.get(
        "EVAL_RUN_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_run_full_coupled_cma_{PROFILE_NAME}.jsonl"),
    )
)
DIST_LOG_PATH = os.path.abspath(
    os.environ.get(
        "EVAL_DIST_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"cma_distribution_log_full_coupled_{PROFILE_NAME}.jsonl"),
    )
)

TEST_CANDIDATES_JSON = os.path.abspath(
    os.environ.get(
        "EVAL_CANDIDATES_JSON",
        os.path.join(SCRIPT_DIR, f"cma_full_coupled_test_candidates_{PROFILE_NAME}.json"),
    )
)
TEST_RESULTS_CSV = os.path.abspath(
    os.environ.get(
        "EVAL_RESULTS_CSV",
        os.path.join(SCRIPT_DIR, f"cma_full_coupled_test_results_{PROFILE_NAME}.csv"),
    )
)
PLOTS_DIR = os.path.abspath(
    os.environ.get(
        "EVAL_PLOTS_DIR",
        os.path.join(SCRIPT_DIR, f"cma_full_coupled_eval_plots_{PROFILE_NAME}"),
    )
)

# ============================================================
# Fixed planner / bounds
# ZGODNE Z TUNEREM full_coupled_staged_cma_fixed.py
# ============================================================
SAFE_PLANNER = {
    "mpc.bounds.max_vx": 13.5,
    "mpc.model.mux": 0.6,
    "mpc.model.muy": 0.7,
    "mpc.cost.q_sdot": 0.1,
}

PLANNER_BOUNDS = {
    "mpc.bounds.max_vx": (8.0, 18.0),
    "mpc.model.mux": (0.4, 1.7),
    "mpc.model.muy": (0.4, 1.7),
    "mpc.cost.q_sdot": (0.1, 1.0),
}

COST_BOUNDS = {
    "mpc.cost.q_ey": (1.0, 100.0),
    "mpc.cost.Q_epsi": (1.0, 100.0),
    "mpc.cost.R_dT": (1.0, 100.0),
    "mpc.cost.R_u_ddelta_cmd": (1.0, 100.0),
    "mpc.cost.R_Mtv": (1e-8, 1e-5),
    "mpc.cost.Q_beta": (0.1, 10.0),
}

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(COST_BOUNDS)
ALL_BOUNDS.update(PLANNER_BOUNDS)

# ============================================================
# Cost function
# ZGODNE Z TUNEREM full_coupled_staged_cma_fixed.py
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
# Phase names from tuner
# ============================================================
PHASE1_NAME = "phase1_cost_only_cma"
PHASE2_NAME = "phase2_planner_qsdot_cma"
PHASE3_NAME = "phase3_joint_cma"

# ============================================================
# Global cache
# ============================================================
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
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "candidate_id" not in rec:
                rec["candidate_id"] = idx
            rec["_global_trial"] = idx
            out.append(rec)
    out.sort(key=lambda r: int(r.get("candidate_id", r.get("_global_trial", 0))))
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def apply_params_to_control_json(theta_x: Dict[str, float]) -> None:
    data = _load_json(CONTROL_PARAM_JSON)
    for path, val in theta_x.items():
        if path in ALL_BOUNDS:
            lo, hi = ALL_BOUNDS[path]
            _set_json_path(data, path, float(_clamp(float(val), lo, hi)))
        else:
            _set_json_path(data, path, float(val))
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


def _theta_signature(theta: Dict[str, float], ndigits: int = 12) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, round(float(v), ndigits)) for k, v in theta.items()))


def _train_objective(rec: Dict[str, Any]) -> float:
    if "objective_cost" in rec:
        return float(rec["objective_cost"])
    if "robust_cost" in rec:
        return float(rec["robust_cost"])
    return float(rec.get("mean_cost", float("inf")))


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
    source_phase: str
    source_candidate_id: Optional[int]
    source_generation: Optional[int]
    source_global_trial_end: int
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

    vs_vals: List[float] = []
    soft_vals: List[float] = []
    medium_vals: List[float] = []
    hard_vals: List[float] = []
    cost_vals: List[float] = []
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
        medium_cnt = _metric_float_first(
            metrics,
            ["medium_track_violations_count", "medium_track_violation_count", "medium_track_violation_count_"],
            0.0,
        )
        hard_cnt = _metric_float_first(
            metrics,
            ["high_track_violations_count", "high_track_violation_count", "high_track_violation_count_"],
            0.0,
        )

        cost_vals.append(float(res.cost))
        vs_vals.append(float(vs_avg))
        soft_vals.append(float(soft_cnt))
        medium_vals.append(float(medium_cnt))
        hard_vals.append(float(hard_cnt))

        if res.crashed:
            crash_count += 1

        per_track.append(
            {
                "track_id": int(tid),
                "cost": float(res.cost),
                "vs_avg_mps": float(vs_avg),
                "soft_track_violations": float(soft_cnt),
                "medium_track_violations": float(medium_cnt),
                "hard_track_violations": float(hard_cnt),
                "crashed": bool(res.crashed),
                "crash_reason": str(res.crash_reason),
            }
        )

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    mean_cost = _mean(cost_vals)
    var_cost = sum((x - mean_cost) * (x - mean_cost) for x in cost_vals) / max(1, len(cost_vals))
    std_cost = math.sqrt(max(0.0, var_cost))
    robust_cost = mean_cost + UCB_WEIGHT * std_cost

    out = {
        "test_tracks": list(TEST_TRACK_SET),
        "n_test_tracks": int(len(TEST_TRACK_SET)),
        "critical_crash_multiplier": float(TEST_CRITICAL_CRASH_MULTIPLIER),
        "avg_vs_avg_mps": _mean(vs_vals),
        "avg_soft_track_violations": _mean(soft_vals),
        "avg_medium_track_violations": _mean(medium_vals),
        "avg_hard_track_violations": _mean(hard_vals),
        "crashes": int(crash_count),
        "crash_percent": 100.0 * float(crash_count) / max(1.0, float(len(TEST_TRACK_SET))),
        "mean_cost": float(mean_cost),
        "std_cost": float(std_cost),
        "robust_cost": float(robust_cost),
        "per_track": per_track,
        "from_cache": False,
    }

    EVAL_CACHE[cache_key] = copy.deepcopy(out)
    return out


# ============================================================
# Candidate selection
# ============================================================
def _select_best_so_far_candidates(run_log: List[Dict[str, Any]]) -> List[EvalTask]:
    if not run_log:
        return []

    max_trial = max(int(r["candidate_id"]) for r in run_log)
    selected: List[EvalTask] = []

    for checkpoint in range(CHECKPOINT_STEP, max_trial + 1, CHECKPOINT_STEP):
        prefix = [r for r in run_log if int(r["candidate_id"]) <= checkpoint]
        ranked = sorted(prefix, key=lambda r: (_train_objective(r), int(r["candidate_id"])))

        best_rec = None
        used = set()

        for rec in ranked:
            theta = {k: float(v) for k, v in rec["theta"].items()}
            sig = _theta_signature(theta)
            if sig in used:
                continue
            used.add(sig)
            best_rec = rec
            break

        if best_rec is None:
            continue

        theta = {k: float(v) for k, v in best_rec["theta"].items()}
        best_obj_so_far = min(_train_objective(r) for r in prefix)

        selected.append(
            EvalTask(
                selection_family="best_so_far",
                selection_label=f"best_so_far_t{checkpoint:03d}",
                checkpoint_trial=int(checkpoint),
                source_phase=str(best_rec.get("phase", "")),
                source_candidate_id=int(best_rec["candidate_id"]),
                source_generation=None,
                source_global_trial_end=int(best_rec["candidate_id"]),
                source_objective_cost_train=float(_train_objective(best_rec)),
                source_mean_cost_train=float(best_rec.get("mean_cost", float("nan"))),
                source_std_cost_train=float(best_rec.get("std_cost", float("nan"))),
                source_best_objective_so_far_train=float(best_obj_so_far),
                theta=theta,
            )
        )

    return selected


def _distribution_full_theta(rec: Dict[str, Any]) -> Dict[str, float]:
    if "distribution_center_theta_full" in rec:
        return {k: float(v) for k, v in rec["distribution_center_theta_full"].items()}

    center = {k: float(v) for k, v in rec.get("distribution_center_theta", {}).items()}
    phase = str(rec.get("phase", ""))

    if phase == PHASE1_NAME:
        merged = dict(SAFE_PLANNER)
        merged.update(center)
        return merged

    return center


def _build_dist_records_with_global_trial_end(dist_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    phase_offsets = {
        PHASE1_NAME: 0,
        PHASE2_NAME: PHASE1_N_TRIALS,
        PHASE3_NAME: PHASE1_N_TRIALS + PHASE2_N_TRIALS,
    }

    for rec in dist_log:
        phase = str(rec.get("phase", ""))
        generation = int(rec.get("generation", 0))
        lambda_ = int(rec.get("lambda", 0))

        if phase not in phase_offsets:
            continue

        # generation 0 / lambda 0 = initial snapshot, nie używamy jako checkpoint candidate
        if generation <= 0 or lambda_ <= 0:
            continue

        global_trial_end = phase_offsets[phase] + generation * lambda_

        tmp = dict(rec)
        tmp["global_trial_end"] = int(global_trial_end)
        out.append(tmp)

    out.sort(key=lambda r: int(r["global_trial_end"]))
    return out


def _select_distribution_center_candidates(
    dist_log: List[Dict[str, Any]],
    run_log: List[Dict[str, Any]],
) -> List[EvalTask]:
    dist_ext = _build_dist_records_with_global_trial_end(dist_log)
    if not dist_ext:
        return []

    max_trial = max(int(r["candidate_id"]) for r in run_log)
    selected: List[EvalTask] = []

    for checkpoint in range(CHECKPOINT_STEP, max_trial + 1, CHECKPOINT_STEP):
        available = [r for r in dist_ext if int(r["global_trial_end"]) <= checkpoint]
        if not available:
            continue

        rec = available[-1]
        theta = _distribution_full_theta(rec)

        prefix = [r for r in run_log if int(r["candidate_id"]) <= checkpoint]
        best_obj_so_far = min((_train_objective(r) for r in prefix), default=float("nan"))

        selected.append(
            EvalTask(
                selection_family="distribution_center",
                selection_label=f"distribution_center_t{checkpoint:03d}",
                checkpoint_trial=int(checkpoint),
                source_phase=str(rec.get("phase", "")),
                source_candidate_id=None,
                source_generation=int(rec.get("generation", 0)),
                source_global_trial_end=int(rec["global_trial_end"]),
                source_objective_cost_train=float(rec.get("generation_best_objective", float("nan"))),
                source_mean_cost_train=float("nan"),
                source_std_cost_train=float("nan"),
                source_best_objective_so_far_train=float(best_obj_so_far),
                theta=theta,
            )
        )

    return selected


# ============================================================
# CSV / plots
# ============================================================
CSV_FIELDS = [
    "selection_label",
    "selection_family",
    "checkpoint_trial",
    "source_phase",
    "source_candidate_id",
    "source_generation",
    "source_global_trial_end",
    "source_objective_cost_train",
    "source_mean_cost_train",
    "source_std_cost_train",
    "source_best_objective_so_far_train",
    "n_test_tracks",
    "critical_crash_multiplier",
    "test_mean_cost",
    "test_std_cost",
    "test_robust_cost",
    "avg_vs_avg_mps",
    "avg_soft_track_violations",
    "avg_medium_track_violations",
    "avg_hard_track_violations",
    "crashes",
    "crash_percent",
    "cache_hit",
    "theta_json",
]


def _save_plot(rows: List[Dict[str, Any]], selection_family: str, y_key: str, y_label: str) -> None:
    xs = []
    ys = []

    family_rows = [r for r in rows if r["selection_family"] == selection_family]
    family_rows.sort(key=lambda r: int(r["checkpoint_trial"]))

    for row in family_rows:
        xs.append(int(row["checkpoint_trial"]))
        ys.append(float(row[y_key]))

    if not xs:
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker="o")
    plt.axvline(x=STAGE1_END_TRIAL, linestyle="--", linewidth=1.2, label="koniec stage 1")
    plt.axvline(x=STAGE2_END_TRIAL, linestyle="--", linewidth=1.2, label="koniec stage 2")
    plt.xlabel("Global trial index")
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
        ("test_std_cost", "Std cost"),
        ("crash_percent", "Crash percent"),
        ("avg_vs_avg_mps", "Average vs_avg_mps"),
        ("avg_soft_track_violations", "Average soft violations"),
        ("avg_medium_track_violations", "Average medium violations"),
        ("avg_hard_track_violations", "Average hard violations"),
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
    _must_exist(DIST_LOG_PATH, "DIST_LOG_PATH")

    original_control_json = _load_json(CONTROL_PARAM_JSON)

    try:
        if os.path.exists(TEST_RESULTS_CSV):
            os.remove(TEST_RESULTS_CSV)
    except Exception:
        pass

    print("[CFG] PROFILE_NAME =", PROFILE_NAME)
    print("[CFG] CATKIN_WS =", CATKIN_WS)
    print("[CFG] WS_SRC    =", WS_SRC)
    print("[CFG] ACADOS_LIB =", ACADOS_LIB)
    print("[CFG] TRAIN_TRACK_SET =", TRAIN_TRACK_SET)
    print("[CFG] TEST_TRACK_SET  =", TEST_TRACK_SET)
    print("[CFG] TEST critical_crash_multiplier =", TEST_CRITICAL_CRASH_MULTIPLIER)
    print("[CFG] UCB_WEIGHT =", UCB_WEIGHT)
    print("[CFG] RUN_LOG_PATH  =", RUN_LOG_PATH)
    print("[CFG] DIST_LOG_PATH =", DIST_LOG_PATH)
    print("[CFG] TEST_RESULTS_CSV =", TEST_RESULTS_CSV)
    print("[CFG] TEST_CANDIDATES_JSON =", TEST_CANDIDATES_JSON)
    print("[CFG] PLOTS_DIR =", PLOTS_DIR)
    print("[CFG] CHECKPOINT_STEP =", CHECKPOINT_STEP)
    print("")

    run_log = _load_jsonl(RUN_LOG_PATH)
    dist_log = _load_jsonl(DIST_LOG_PATH)

    if not run_log:
        print("[FATAL] Empty run log.", file=sys.stderr)
        sys.exit(2)

    best_candidates = _select_best_so_far_candidates(run_log)
    dist_candidates = _select_distribution_center_candidates(
        dist_log=dist_log,
        run_log=run_log,
    )

    all_candidates = best_candidates + dist_candidates
    all_candidates.sort(
        key=lambda c: (
            int(c.checkpoint_trial),
            0 if c.selection_family == "distribution_center" else 1,
            10**9 if c.source_global_trial_end is None else int(c.source_global_trial_end),
        )
    )

    _save_json_atomic(
        TEST_CANDIDATES_JSON,
        [
            {
                "selection_label": c.selection_label,
                "selection_family": c.selection_family,
                "checkpoint_trial": int(c.checkpoint_trial),
                "source_phase": c.source_phase,
                "source_candidate_id": c.source_candidate_id,
                "source_generation": c.source_generation,
                "source_global_trial_end": int(c.source_global_trial_end),
                "source_objective_cost_train": float(c.source_objective_cost_train),
                "source_mean_cost_train": float(c.source_mean_cost_train),
                "source_std_cost_train": float(c.source_std_cost_train),
                "source_best_objective_so_far_train": float(c.source_best_objective_so_far_train),
                "theta": {k: float(v) for k, v in c.theta.items()},
            }
            for c in all_candidates
        ],
    )

    print(f"[INFO] Wybrałem {len(best_candidates)} kandydatów typu best-so-far.")
    print(f"[INFO] Wybrałem {len(dist_candidates)} kandydatów typu distribution-center.")
    print(f"[INFO] Łącznie do testu: {len(all_candidates)} kandydatów.")
    print("")

    roscore_p = None
    rows_written = 0
    interrupted = False
    csv_rows: List[Dict[str, Any]] = []

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

        for idx, cand in enumerate(all_candidates, start=1):
            print(
                f"[TEST {idx:03d}/{len(all_candidates):03d}] "
                f"{cand.selection_label} | family={cand.selection_family} "
                f"| checkpoint={cand.checkpoint_trial}"
            )

            test_info = evaluate_theta_on_test_tracks(cand.theta)

            row = {
                "selection_label": cand.selection_label,
                "selection_family": cand.selection_family,
                "checkpoint_trial": int(cand.checkpoint_trial),
                "source_phase": cand.source_phase,
                "source_candidate_id": cand.source_candidate_id,
                "source_generation": cand.source_generation,
                "source_global_trial_end": int(cand.source_global_trial_end),
                "source_objective_cost_train": float(cand.source_objective_cost_train),
                "source_mean_cost_train": float(cand.source_mean_cost_train),
                "source_std_cost_train": float(cand.source_std_cost_train),
                "source_best_objective_so_far_train": float(cand.source_best_objective_so_far_train),
                "n_test_tracks": int(test_info["n_test_tracks"]),
                "critical_crash_multiplier": float(test_info["critical_crash_multiplier"]),
                "test_mean_cost": float(test_info["mean_cost"]),
                "test_std_cost": float(test_info["std_cost"]),
                "test_robust_cost": float(test_info["robust_cost"]),
                "avg_vs_avg_mps": float(test_info["avg_vs_avg_mps"]),
                "avg_soft_track_violations": float(test_info["avg_soft_track_violations"]),
                "avg_medium_track_violations": float(test_info["avg_medium_track_violations"]),
                "avg_hard_track_violations": float(test_info["avg_hard_track_violations"]),
                "crashes": int(test_info["crashes"]),
                "crash_percent": float(test_info["crash_percent"]),
                "cache_hit": int(bool(test_info.get("from_cache", False))),
                "theta_json": json.dumps(cand.theta, sort_keys=True),
            }

            _append_csv_row(TEST_RESULTS_CSV, CSV_FIELDS, row)
            csv_rows.append(row)
            rows_written += 1

            print(
                f"  mean={row['test_mean_cost']:.6g} | "
                f"std={row['test_std_cost']:.6g} | "
                f"robust={row['test_robust_cost']:.6g} | "
                f"avg_vs={row['avg_vs_avg_mps']:.6g} | "
                f"soft={row['avg_soft_track_violations']:.6g} | "
                f"medium={row['avg_medium_track_violations']:.6g} | "
                f"hard={row['avg_hard_track_violations']:.6g} | "
                f"crashes={row['crashes']}/{row['n_test_tracks']} | "
                f"crash%={row['crash_percent']:.6g} | "
                f"from_cache={row['cache_hit']}"
            )
            print("")

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Interrupted by user. Partial CSV has been kept.")

    finally:
        try:
            if csv_rows:
                _save_family_plots(csv_rows, "best_so_far")
                _save_family_plots(csv_rows, "distribution_center")
        except Exception as e:
            print(f"[WARN] Plot save failed: {e}", file=sys.stderr)

        try:
            _save_json_atomic(CONTROL_PARAM_JSON, original_control_json)
        except Exception:
            pass

        if roscore_p is not None:
            _kill_process_group(roscore_p)

    print("[DONE]")
    print("Saved candidates json:", TEST_CANDIDATES_JSON)
    print("Saved test csv:", TEST_RESULTS_CSV)
    print("Saved plots dir:", PLOTS_DIR)
    print("Unique theta cached:", len(EVAL_CACHE))
    print("Rows written:", rows_written)

    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
