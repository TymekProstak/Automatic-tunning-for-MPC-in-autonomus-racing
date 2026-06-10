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
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_ka_racing/TUNE_CMA_ES")
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
UCB_WEIGHT = float(os.environ.get("EVAL_UCB_WEIGHT", "1.0"))

# ------------------------------------------------------------
# EVAL PROFILE SWITCH
# ------------------------------------------------------------
EVAL_PROFILE_NAME = os.environ.get("EVAL_PROFILE", "smoke_3track")
EVAL_PROFILE_NAME = "night_7track"

if EVAL_PROFILE_NAME == "smoke_3track":
    DEFAULT_EVAL_TRACK_SET = "8,11,14"
    DEFAULT_CHECKPOINT_STEP = "6"

    # zgodne z plikami bez TV
    DEFAULT_RUN_LOG_NAME = "tuning_run_smoke_1track.jsonl"
    DEFAULT_DIST_LOG_NAME = "cma_distribution_log_smoke_1track.jsonl"
    DEFAULT_SUMMARY_NAME = "tuning_summary_smoke_1track.json"

    DEFAULT_OUTPUT_CANDIDATES_NAME = "cma_test_candidates_smoke3track.json"
    DEFAULT_OUTPUT_CSV_NAME = "cma_test_results_smoke3track.csv"
    DEFAULT_PLOTS_DIR_NAME = "cma_eval_plots"

    PHASE1_LAMBDA = 4
    PHASE1_MAXITER = 3   # 12

    PHASE2_LAMBDA = 6
    PHASE2_MAXITER = 4   # 24

    PHASE3_LAMBDA = 6
    PHASE3_MAXITER = 4   # 24

elif EVAL_PROFILE_NAME == "night_7track":
    DEFAULT_EVAL_TRACK_SET = "8,9,10,11,12,13,14"
    DEFAULT_CHECKPOINT_STEP = "6"

    # zgodne z plikami bez TV, które masz w evolution_optimization
    DEFAULT_RUN_LOG_NAME = "tuning_run_night_4track.jsonl"
    DEFAULT_DIST_LOG_NAME = "cma_distribution_log_night_4track.jsonl"
    DEFAULT_SUMMARY_NAME = "tuning_summary_night_4track.json"

    DEFAULT_OUTPUT_CANDIDATES_NAME = "cma_test_candidates_night7track.json"
    DEFAULT_OUTPUT_CSV_NAME = "cma_test_results_night7track.csv"
    DEFAULT_PLOTS_DIR_NAME = "cma_eval_plots"

    PHASE1_LAMBDA = 8
    PHASE1_MAXITER = 6   # 48

    PHASE2_LAMBDA = 8
    PHASE2_MAXITER = 8   # 64

    PHASE3_LAMBDA = 8
    PHASE3_MAXITER = 12  # 96

else:
    raise ValueError(f"Unknown EVAL_PROFILE_NAME: {EVAL_PROFILE_NAME}")


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


TEST_TRACK_SET = _parse_track_set()
TEST_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("EVAL_TEST_CRITICAL_CRASH_MULTIPLIER", "1.0")
)

# ============================================================
# Stage names / stage sizes aligned with tuner
# ============================================================
PHASE1_NAME = "phase1_cost_only_cma"
PHASE2_NAME = "phase2_planner_only_cma"
PHASE3_NAME = "phase3_joint_cma"

PHASE1_N_TRIALS = PHASE1_LAMBDA * PHASE1_MAXITER
PHASE2_N_TRIALS = PHASE2_LAMBDA * PHASE2_MAXITER
PHASE3_N_TRIALS = PHASE3_LAMBDA * PHASE3_MAXITER

TOTAL_TRIALS = PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_N_TRIALS

# CMA: brak startup offsetu
PHASE1_STARTUP_TRIALS = 0
PHASE2_STARTUP_TRIALS = 0
PHASE3_STARTUP_TRIALS = 0

PHASE1_EFFECTIVE_TRIALS = PHASE1_N_TRIALS
PHASE2_EFFECTIVE_TRIALS = PHASE2_N_TRIALS
PHASE3_EFFECTIVE_TRIALS = PHASE3_N_TRIALS
TOTAL_EFFECTIVE_TRIALS = PHASE1_EFFECTIVE_TRIALS + PHASE2_EFFECTIVE_TRIALS + PHASE3_EFFECTIVE_TRIALS

STAGE1_END_FILTERED_TRIAL = PHASE1_EFFECTIVE_TRIALS
STAGE2_END_FILTERED_TRIAL = PHASE1_EFFECTIVE_TRIALS + PHASE2_EFFECTIVE_TRIALS

RUN_LOG_PATH = os.path.abspath(
    os.environ.get("EVAL_RUN_LOG_PATH", os.path.join(SCRIPT_DIR, DEFAULT_RUN_LOG_NAME))
)
DIST_LOG_PATH = os.path.abspath(
    os.environ.get("EVAL_DIST_LOG_PATH", os.path.join(SCRIPT_DIR, DEFAULT_DIST_LOG_NAME))
)
SUMMARY_PATH = os.path.abspath(
    os.environ.get("EVAL_SUMMARY_PATH", os.path.join(SCRIPT_DIR, DEFAULT_SUMMARY_NAME))
)

TEST_CANDIDATES_JSON = os.path.abspath(
    os.environ.get("EVAL_OUTPUT_CANDIDATES_JSON", os.path.join(SCRIPT_DIR, DEFAULT_OUTPUT_CANDIDATES_NAME))
)
TEST_RESULTS_CSV = os.path.abspath(
    os.environ.get("EVAL_OUTPUT_CSV", os.path.join(SCRIPT_DIR, DEFAULT_OUTPUT_CSV_NAME))
)
PLOTS_DIR = os.path.abspath(
    os.environ.get("EVAL_PLOTS_DIR", os.path.join(SCRIPT_DIR, DEFAULT_PLOTS_DIR_NAME))
)

# ============================================================
# Selection policy
# ============================================================
CHECKPOINT_STEP = int(os.environ.get("EVAL_CHECKPOINT_STEP", DEFAULT_CHECKPOINT_STEP))

# ============================================================
# Cost function aligned with tuner
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
# Safe planner aligned with tuner
# ============================================================
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

# ============================================================
# no-TV bounds
# ============================================================
COST_BOUNDS = {
    "mpc.cost.Q_y": (0.1, 30.0),
    "mpc.cost.Q_psi": (0.1, 30.0),
    "mpc.cost.R_ddelta": (0.1, 30.0),
}

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(COST_BOUNDS)
ALL_BOUNDS.update(PLANNER_BOUNDS)

# ============================================================
# ROS / runtime
# ============================================================
SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("EVAL_ROS_MASTER_PORT", "11313"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_test_eval_cma_{EVAL_PROFILE_NAME}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# CSV fields
# ============================================================
CSV_FIELDS = [
    "selection_family",
    "selection_label",
    "checkpoint_trial_filtered",
    "checkpoint_trial_global",
    "source_phase",
    "source_candidate_id",
    "source_generation",
    "source_global_trial_end",
    "source_objective_cost_train",
    "source_mean_cost_train",
    "source_std_cost_train",
    "source_best_objective_so_far_train",
    "source_kind",
    "is_tail_padded",
    "n_test_tracks",
    "critical_crash_multiplier",
    "avg_vs_avg_mps",
    "avg_soft_track_violations",
    "avg_medium_track_violations",
    "avg_hard_track_violations",
    "crash_count",
    "crash_percent",
    "mean_cost",
    "std_cost",
    "robust_cost",
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
            out.append(rec)
    return out


def _load_json_optional(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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


def _train_objective(rec: Dict[str, Any]) -> float:
    if "objective_cost" in rec:
        return float(rec["objective_cost"])
    if "robust_cost" in rec:
        return float(rec["robust_cost"])
    return float(rec.get("mean_cost", float("inf")))


def _is_startup_record(rec: Dict[str, Any]) -> bool:
    return False


def _filtered_to_global_trial(filtered_trial: int) -> int:
    return int(filtered_trial)


def _build_filtered_checkpoints(step: int) -> List[int]:
    if step <= 0:
        return [TOTAL_EFFECTIVE_TRIALS]

    checkpoints = list(range(step, TOTAL_EFFECTIVE_TRIALS + 1, step))
    if not checkpoints or checkpoints[-1] != TOTAL_EFFECTIVE_TRIALS:
        checkpoints.append(TOTAL_EFFECTIVE_TRIALS)
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

    n_tracks = max(1, len(TEST_TRACK_SET))
    crash_percent = 100.0 * float(crash_count) / float(n_tracks)

    return {
        "test_tracks": list(TEST_TRACK_SET),
        "n_test_tracks": int(len(TEST_TRACK_SET)),
        "critical_crash_multiplier": float(TEST_CRITICAL_CRASH_MULTIPLIER),
        "avg_vs_avg_mps": _mean(vs_vals),
        "avg_soft_track_violations": _mean(soft_vals),
        "avg_medium_track_violations": _mean(medium_vals),
        "avg_hard_track_violations": _mean(hard_vals),
        "crash_count": int(crash_count),
        "crash_percent": float(crash_percent),
        "mean_cost": float(mean_cost),
        "std_cost": float(std_cost),
        "robust_cost": float(robust_cost),
        "per_track": per_track,
    }


# ============================================================
# Distribution-center helpers
# ============================================================
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


def _build_dist_records_for_phase(
    dist_log: List[Dict[str, Any]],
    phase_name: str,
    lambda_: int,
    global_offset: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for rec in dist_log:
        phase = str(rec.get("phase", ""))
        generation = int(rec.get("generation", 0))

        if phase != phase_name or generation <= 0:
            continue

        global_trial_end = global_offset + generation * lambda_
        tmp = dict(rec)
        tmp["global_trial_end"] = int(global_trial_end)
        tmp["theta_full"] = _distribution_full_theta(rec)
        out.append(tmp)

    out.sort(key=lambda r: int(r["global_trial_end"]))
    return out


def _build_initial_distribution_record(
    summary: Optional[Dict[str, Any]],
    summary_key: str,
    fallback_phase_name: str,
    global_trial_end: int,
) -> Optional[Dict[str, Any]]:
    if summary is None:
        return None

    try:
        dist = summary["stage_distributions"][summary_key]
        theta_full = dist["distribution_center_theta_full"]
    except Exception:
        return None

    return {
        "phase": fallback_phase_name,
        "generation": 0,
        "global_trial_end": int(global_trial_end),
        "theta_full": {k: float(v) for k, v in theta_full.items()},
        "source_kind": str(summary_key),
    }


# ============================================================
# Candidate selection schedules
# ============================================================
def _select_best_so_far_schedule(run_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    non_startup = [r for r in run_log if not _is_startup_record(r)]
    non_startup.sort(key=lambda r: int(r["candidate_id"]))

    checkpoints_filtered = _build_filtered_checkpoints(CHECKPOINT_STEP)
    selected: List[Dict[str, Any]] = []

    if not non_startup:
        return selected

    max_available_global = max(int(r["candidate_id"]) for r in non_startup)

    for checkpoint_filtered in checkpoints_filtered:
        checkpoint_global = _filtered_to_global_trial(checkpoint_filtered)
        effective_global = min(checkpoint_global, max_available_global)

        prefix = [r for r in non_startup if int(r["candidate_id"]) <= effective_global]
        if not prefix:
            continue

        best = min(prefix, key=lambda r: (_train_objective(r), int(r["candidate_id"])))
        best_obj_so_far = min(_train_objective(r) for r in prefix)
        theta = {k: float(v) for k, v in best["theta"].items()}

        selected.append(
            {
                "selection_family": "best_so_far",
                "selection_label": f"best_so_far_t{checkpoint_filtered:03d}",
                "checkpoint_trial_filtered": int(checkpoint_filtered),
                "checkpoint_trial_global": int(checkpoint_global),
                "source_phase": str(best.get("phase", "")),
                "source_candidate_id": int(best["candidate_id"]),
                "source_generation": "",
                "source_global_trial_end": int(best["candidate_id"]),
                "source_objective_cost_train": float(_train_objective(best)),
                "source_mean_cost_train": float(best.get("mean_cost", float("nan"))),
                "source_std_cost_train": float(best.get("std_cost", float("nan"))),
                "source_best_objective_so_far_train": float(best_obj_so_far),
                "source_kind": "best_observed_prefix",
                "is_tail_padded": bool(checkpoint_global > max_available_global),
                "theta": theta,
            }
        )

    return selected


def _select_distribution_center_schedule(
    dist_log: List[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    phase1_dist = _build_dist_records_for_phase(
        dist_log=dist_log,
        phase_name=PHASE1_NAME,
        lambda_=PHASE1_LAMBDA,
        global_offset=0,
    )
    phase2_dist = _build_dist_records_for_phase(
        dist_log=dist_log,
        phase_name=PHASE2_NAME,
        lambda_=PHASE2_LAMBDA,
        global_offset=PHASE1_N_TRIALS,
    )
    phase3_dist = _build_dist_records_for_phase(
        dist_log=dist_log,
        phase_name=PHASE3_NAME,
        lambda_=PHASE3_LAMBDA,
        global_offset=PHASE1_N_TRIALS + PHASE2_N_TRIALS,
    )

    stage2_init = _build_initial_distribution_record(
        summary=summary,
        summary_key="stage2_initial_distribution",
        fallback_phase_name=PHASE2_NAME,
        global_trial_end=PHASE1_N_TRIALS,
    )
    stage3_init = _build_initial_distribution_record(
        summary=summary,
        summary_key="stage3_initial_distribution",
        fallback_phase_name=PHASE3_NAME,
        global_trial_end=PHASE1_N_TRIALS + PHASE2_N_TRIALS,
    )

    checkpoints_filtered = _build_filtered_checkpoints(CHECKPOINT_STEP)
    selected: List[Dict[str, Any]] = []

    max_phase1_global = max([int(r["global_trial_end"]) for r in phase1_dist], default=0)
    max_phase2_global = max([int(r["global_trial_end"]) for r in phase2_dist], default=0)
    max_phase3_global = max([int(r["global_trial_end"]) for r in phase3_dist], default=0)

    for checkpoint_filtered in checkpoints_filtered:
        checkpoint_global = _filtered_to_global_trial(checkpoint_filtered)

        rec = None
        source_kind = "generation_distribution"
        is_tail_padded = False

        if checkpoint_global <= PHASE1_N_TRIALS:
            available = [r for r in phase1_dist if int(r["global_trial_end"]) <= checkpoint_global]
            if not available:
                continue
            rec = available[-1]
            is_tail_padded = checkpoint_global > max_phase1_global

        elif checkpoint_global <= PHASE1_N_TRIALS + PHASE2_N_TRIALS:
            available = [r for r in phase2_dist if int(r["global_trial_end"]) <= checkpoint_global]
            if available:
                rec = available[-1]
                is_tail_padded = checkpoint_global > max_phase2_global if max_phase2_global > 0 else True
            else:
                if stage2_init is None:
                    continue
                rec = stage2_init
                source_kind = "stage2_initial_distribution"
                is_tail_padded = True

        else:
            available = [r for r in phase3_dist if int(r["global_trial_end"]) <= checkpoint_global]
            if available:
                rec = available[-1]
                is_tail_padded = checkpoint_global > max_phase3_global if max_phase3_global > 0 else True
            else:
                if stage3_init is not None:
                    rec = stage3_init
                    source_kind = "stage3_initial_distribution"
                    is_tail_padded = True
                else:
                    if not phase2_dist:
                        continue
                    rec = phase2_dist[-1]
                    source_kind = "phase2_fallback_distribution"
                    is_tail_padded = True

        theta = {k: float(v) for k, v in rec["theta_full"].items()}

        selected.append(
            {
                "selection_family": "distribution_center",
                "selection_label": f"distribution_center_t{checkpoint_filtered:03d}",
                "checkpoint_trial_filtered": int(checkpoint_filtered),
                "checkpoint_trial_global": int(checkpoint_global),
                "source_phase": str(rec.get("phase", "")),
                "source_candidate_id": "",
                "source_generation": "" if "generation" not in rec else int(rec.get("generation", 0)),
                "source_global_trial_end": int(rec.get("global_trial_end", 0)),
                "source_objective_cost_train": float(rec.get("generation_best_objective", float("nan"))),
                "source_mean_cost_train": float("nan"),
                "source_std_cost_train": float("nan"),
                "source_best_objective_so_far_train": float(rec.get("global_best_objective_inside_phase", float("nan"))),
                "source_kind": str(source_kind),
                "is_tail_padded": bool(is_tail_padded),
                "theta": theta,
            }
        )

    return selected


# ============================================================
# Plots
# ============================================================
def _save_family_plots(rows: List[Dict[str, Any]], family: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fam_rows = [r for r in rows if str(r.get("selection_family", "")) == family]
    fam_rows.sort(key=lambda r: int(r["checkpoint_trial_filtered"]))

    if not fam_rows:
        return

    x = [int(r["checkpoint_trial_filtered"]) for r in fam_rows]

    metric_specs = [
        ("robust_cost", "Robust cost", f"{family}: robust_cost"),
        ("mean_cost", "Mean cost", f"{family}: mean_cost"),
        ("std_cost", "Cost standard deviation", f"{family}: std_cost"),
        ("crash_percent", "Crash rate [%]", f"{family}: crash_percent"),
        ("avg_vs_avg_mps", "Average speed [m/s]", f"{family}: avg_vs_avg_mps"),
        ("avg_soft_track_violations", "Average soft violations", f"{family}: avg_soft_track_violations"),
        ("avg_medium_track_violations", "Average medium violations", f"{family}: avg_medium_track_violations"),
        ("avg_hard_track_violations", "Average hard violations", f"{family}: avg_hard_track_violations"),
    ]

    for key, ylabel, title in metric_specs:
        y = [float(r[key]) for r in fam_rows]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker="o")
        plt.axvline(
            STAGE1_END_FILTERED_TRIAL,
            color="red",
            linestyle="--",
            label="End of stage 1",
        )
        plt.axvline(
            STAGE2_END_FILTERED_TRIAL,
            color="purple",
            linestyle="--",
            label="End of stage 2",
        )
        plt.xlabel("Filtered trial index")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(PLOTS_DIR, f"{family}_{key}.png")
        plt.savefig(out_path, dpi=160, format="png")
        plt.close()


# ============================================================
# Main
# ============================================================
def main() -> None:
    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(ACADOS_LIB, "ACADOS_LIB")
    _must_exist(RUN_LOG_PATH, "RUN_LOG_PATH")
    _must_exist(DIST_LOG_PATH, "DIST_LOG_PATH")

    summary = _load_json_optional(SUMMARY_PATH)
    original_control_json = _load_json(CONTROL_PARAM_JSON)

    try:
        if os.path.exists(TEST_RESULTS_CSV):
            os.remove(TEST_RESULTS_CSV)
    except Exception:
        pass

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("[CFG] CATKIN_WS =", CATKIN_WS)
    print("[CFG] EVAL_PROFILE_NAME =", EVAL_PROFILE_NAME)
    print("[CFG] WS_SRC =", WS_SRC)
    print("[CFG] ACADOS_LIB =", ACADOS_LIB)
    print("[CFG] TEST_TRACK_SET =", TEST_TRACK_SET)
    print("[CFG] TEST critical_crash_multiplier =", TEST_CRITICAL_CRASH_MULTIPLIER)
    print("[CFG] UCB_WEIGHT =", UCB_WEIGHT)
    print("[CFG] RUN_LOG_PATH =", RUN_LOG_PATH)
    print("[CFG] DIST_LOG_PATH =", DIST_LOG_PATH)
    print("[CFG] SUMMARY_PATH =", SUMMARY_PATH, "(optional)")
    print("[CFG] TEST_RESULTS_CSV =", TEST_RESULTS_CSV)
    print("[CFG] TEST_CANDIDATES_JSON =", TEST_CANDIDATES_JSON)
    print("[CFG] PLOTS_DIR =", PLOTS_DIR)
    print("[CFG] CHECKPOINT_STEP =", CHECKPOINT_STEP)
    print("[CFG] PHASE1_N_TRIALS =", PHASE1_N_TRIALS)
    print("[CFG] PHASE2_N_TRIALS =", PHASE2_N_TRIALS)
    print("[CFG] PHASE3_N_TRIALS =", PHASE3_N_TRIALS)
    print("[CFG] TOTAL_TRIALS =", TOTAL_TRIALS)
    print("[CFG] STAGE1_END_FILTERED_TRIAL =", STAGE1_END_FILTERED_TRIAL)
    print("[CFG] STAGE2_END_FILTERED_TRIAL =", STAGE2_END_FILTERED_TRIAL)
    print("")

    run_log = _load_jsonl(RUN_LOG_PATH)
    dist_log = _load_jsonl(DIST_LOG_PATH)

    if not run_log:
        print("[FATAL] Empty run log.", file=sys.stderr)
        sys.exit(2)

    best_schedule = _select_best_so_far_schedule(run_log)
    dist_schedule = _select_distribution_center_schedule(dist_log, summary)

    all_candidates = best_schedule + dist_schedule
    all_candidates.sort(
        key=lambda r: (
            int(r["checkpoint_trial_filtered"]),
            0 if str(r["selection_family"]) == "distribution_center" else 1,
        )
    )

    _save_json_atomic(TEST_CANDIDATES_JSON, all_candidates)

    print(f"[INFO] Selected {len(best_schedule)} best-so-far checkpoints.")
    print(f"[INFO] Selected {len(dist_schedule)} distribution-center checkpoints.")
    print(f"[INFO] Total candidates for evaluation: {len(all_candidates)}")
    print("")

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

    eval_cache: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []

    try:
        for idx, cand in enumerate(all_candidates):
            print(
                f"[TEST {idx + 1:03d}/{len(all_candidates):03d}] "
                f"{cand['selection_label']} | family={cand['selection_family']} "
                f"| checkpoint_filtered={cand['checkpoint_trial_filtered']} "
                f"| checkpoint_global={cand['checkpoint_trial_global']}"
            )

            sig = _theta_signature(cand["theta"])
            if sig in eval_cache:
                test_info = copy.deepcopy(eval_cache[sig])
                print("  [cache] reused evaluation")
            else:
                test_info = evaluate_theta_on_test_tracks(cand["theta"])
                eval_cache[sig] = copy.deepcopy(test_info)

            row = {
                "selection_family": cand["selection_family"],
                "selection_label": cand["selection_label"],
                "checkpoint_trial_filtered": int(cand["checkpoint_trial_filtered"]),
                "checkpoint_trial_global": int(cand["checkpoint_trial_global"]),
                "source_phase": cand["source_phase"],
                "source_candidate_id": cand["source_candidate_id"],
                "source_generation": cand.get("source_generation", ""),
                "source_global_trial_end": int(cand["source_global_trial_end"]),
                "source_objective_cost_train": float(cand.get("source_objective_cost_train", float("nan"))),
                "source_mean_cost_train": float(cand.get("source_mean_cost_train", float("nan"))),
                "source_std_cost_train": float(cand.get("source_std_cost_train", float("nan"))),
                "source_best_objective_so_far_train": float(cand.get("source_best_objective_so_far_train", float("nan"))),
                "source_kind": cand.get("source_kind", ""),
                "is_tail_padded": int(bool(cand["is_tail_padded"])),
                "n_test_tracks": int(test_info["n_test_tracks"]),
                "critical_crash_multiplier": float(test_info["critical_crash_multiplier"]),
                "avg_vs_avg_mps": float(test_info["avg_vs_avg_mps"]),
                "avg_soft_track_violations": float(test_info["avg_soft_track_violations"]),
                "avg_medium_track_violations": float(test_info["avg_medium_track_violations"]),
                "avg_hard_track_violations": float(test_info["avg_hard_track_violations"]),
                "crash_count": int(test_info["crash_count"]),
                "crash_percent": float(test_info["crash_percent"]),
                "mean_cost": float(test_info["mean_cost"]),
                "std_cost": float(test_info["std_cost"]),
                "robust_cost": float(test_info["robust_cost"]),
                "theta_json": json.dumps(cand["theta"], sort_keys=True),
            }

            csv_rows.append(row)
            _append_csv_row(TEST_RESULTS_CSV, CSV_FIELDS, row)

            print(
                f"  mean={row['mean_cost']:.6g} | "
                f"std={row['std_cost']:.6g} | "
                f"robust={row['robust_cost']:.6g} | "
                f"avg_vs={row['avg_vs_avg_mps']:.6g} | "
                f"soft={row['avg_soft_track_violations']:.6g} | "
                f"medium={row['avg_medium_track_violations']:.6g} | "
                f"hard={row['avg_hard_track_violations']:.6g} | "
                f"crash%={row['crash_percent']:.6g} | "
                f"tail_padded={row['is_tail_padded']}"
            )
            print("")

    finally:
        try:
            _save_json_atomic(CONTROL_PARAM_JSON, original_control_json)
        except Exception:
            pass

        if roscore_p is not None:
            _kill_process_group(roscore_p)

    _save_family_plots(csv_rows, "best_so_far")
    _save_family_plots(csv_rows, "distribution_center")

    print("[DONE]")
    print("Saved candidates json:", TEST_CANDIDATES_JSON)
    print("Saved test csv:", TEST_RESULTS_CSV)
    print("Saved plots dir:", PLOTS_DIR)


if __name__ == "__main__":
    main()
