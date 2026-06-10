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
# Workspace / paths
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_amz/NO_STAGE/TUNE_BO_GP")
CATKIN_WS = os.path.abspath(
    os.environ.get("EVAL_WS", os.environ.get("TUNE_WS", DEFAULT_CATKIN_WS))
)
WS_SRC = os.path.join(CATKIN_WS, "src")
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

# ============================================================
# Eval profiles
# ============================================================
EVAL_PROFILE_NAME = "smoke_3track"
EVAL_PROFILE_NAME = "night_7track"

if EVAL_PROFILE_NAME == "smoke_3track":
    TRAIN_PROFILE_NAME = "smoke_1track"
    DEFAULT_EVAL_TRACK_SET = "8,11,14"
    DEFAULT_CHECKPOINT_STEP = "2"

    UNSTAGED_N_TRIALS = 60
    UNSTAGED_STARTUP = 25

elif EVAL_PROFILE_NAME == "night_7track":
    TRAIN_PROFILE_NAME = "night_4track"
    DEFAULT_EVAL_TRACK_SET = "8,9,10,11,12,13,14"
    DEFAULT_CHECKPOINT_STEP = "4"

    UNSTAGED_N_TRIALS = 224
    UNSTAGED_STARTUP = 64

else:
    raise ValueError(f"Unknown EVAL_PROFILE_NAME: {EVAL_PROFILE_NAME}")

UNSTAGED_EFFECTIVE_TRIALS = UNSTAGED_N_TRIALS - UNSTAGED_STARTUP
UNSTAGED_PHASE_NAME = os.environ.get("TUNE_UNSTAGED_PHASE_NAME", "unstaged_joint_gpbo_full_coupled")

RUN_LOG_PATH = os.path.abspath(
    os.environ.get(
        "EVAL_RUN_LOG_PATH",
        os.path.join(SCRIPT_DIR, f"tuning_run_full_coupled_gpbo_unstaged_{TRAIN_PROFILE_NAME}.jsonl"),
    )
)
OUTPUT_CSV_PATH = os.path.abspath(
    os.environ.get(
        "EVAL_OUTPUT_CSV",
        os.path.join(SCRIPT_DIR, f"full_coupled_gpbo_unstaged_test_eval_{EVAL_PROFILE_NAME}.csv"),
    )
)
OUTPUT_CANDIDATES_JSON = os.path.abspath(
    os.environ.get(
        "EVAL_OUTPUT_CANDIDATES_JSON",
        os.path.join(SCRIPT_DIR, f"full_coupled_gpbo_unstaged_test_candidates_{EVAL_PROFILE_NAME}.json"),
    )
)
PLOTS_DIR = os.path.abspath(
    os.environ.get(
        "EVAL_PLOTS_DIR",
        os.path.join(SCRIPT_DIR, f"full_coupled_gpbo_unstaged_eval_plots_{EVAL_PROFILE_NAME}"),
    )
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
# ACADOS lib needed by dv_control
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
        raise ValueError("EVAL_TRACK_SET resolved to an empty list.")
    return vals


TEST_TRACKS = _parse_track_set()
TEST_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("EVAL_TEST_CRITICAL_CRASH_MULTIPLIER", "1.0")
)

SURROGATE_KIND = os.environ.get("EVAL_SURROGATE_KIND", "pred_mean").strip().lower()
CHECKPOINT_STEP = int(os.environ.get("EVAL_CHECKPOINT_STEP", DEFAULT_CHECKPOINT_STEP))

# ============================================================
# Names / derived counts
# ============================================================
TOTAL_TRIALS = UNSTAGED_N_TRIALS
TOTAL_EFFECTIVE_TRIALS = UNSTAGED_EFFECTIVE_TRIALS

# ============================================================
# Bounds / config retained for consistency with tuner
# ============================================================
SAFE_PLANNER = {
    "mpc.bounds.max_vx": 10.5,
    "mpc.model.mux": 0.4,
    "mpc.model.muy": 0.4,
}

PLANNER_BOUNDS = {
    "mpc.bounds.max_vx": (8.0, 18.0),
    "mpc.model.mux": (0.4, 1.7),
    "mpc.model.muy": (0.4, 1.7),
}

COST_BOUNDS = {
    "mpc.cost.q_ey": (1.0, 100.0),
    "mpc.cost.Q_epsi": (1.0, 100.0),
    "mpc.cost.R_dT": (1.0, 100.0),
    "mpc.cost.R_u_ddelta_cmd": (1.0, 100.0),
    "mpc.cost.Q_beta": (0.1, 10.0),
    "mpc.cost.q_sdot": (0.1, 1.0),
    "mpc.cost.R_Mtv": (1e-8, 1e-5),
}

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(PLANNER_BOUNDS)
ALL_BOUNDS.update(COST_BOUNDS)

# ============================================================
# ROS / runtime
# ============================================================
SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("EVAL_ROS_MASTER_PORT", "11819"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_eval_gpbo_full_coupled_unstaged_{EVAL_PROFILE_NAME}_{GLOBAL_SEED}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Cost function
# must match tuner
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
# CSV schema
# ============================================================
CSV_FIELDS = [
    "selection_family",
    "selection_label",
    "surrogate_kind",
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
    "source_surrogate_pred_mean",
    "source_surrogate_pred_std",
    "source_surrogate_acq_value",
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


def _global_trial(rec: Dict[str, Any]) -> int:
    return int(rec.get("_global_trial", 0))


def _source_phase_trial_for_unstaged(rec: Dict[str, Any]) -> int:
    """
    Dla unstaged pole phase_trial bywa mylące albo go nie ma.
    Domyślnie raportujemy więc trial zero-based oparty o global trial.
    """
    try:
        if "phase_trial" in rec and rec.get("phase_trial") is not None:
            return int(rec["phase_trial"])
    except Exception:
        pass

    gt = _global_trial(rec)
    if gt <= 0:
        return -1
    return gt - 1


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


def _is_startup_record(rec: Dict[str, Any]) -> bool:
    """
    Dla UNSTAGED startup liczymy po GLOBAL trialu:
      global_trial = 1..UNSTAGED_STARTUP  => startup
      global_trial > UNSTAGED_STARTUP     => non-startup
    """
    gt = _global_trial(rec)
    return gt <= UNSTAGED_STARTUP


def _filtered_to_global_trial(filtered_trial: int) -> int:
    if filtered_trial <= 0:
        return 0
    return UNSTAGED_STARTUP + filtered_trial


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
    source_surrogate_pred_mean: Optional[float]
    source_surrogate_pred_std: Optional[float]
    source_surrogate_acq_value: Optional[float]
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
) -> Dict[str, Any]:
    per_track: List[EpisodeResult] = []

    for tid in tracks:
        res = run_one_episode(
            theta_x=theta_x,
            track_index=tid,
            sim_time_s=sim_time_s,
            critical_crash_multiplier=critical_crash_multiplier,
        )
        per_track.append(res)

    n = max(1, len(per_track))

    mean_cost = sum(float(r.cost) for r in per_track) / n
    var_cost = sum((float(r.cost) - mean_cost) * (float(r.cost) - mean_cost) for r in per_track) / n
    std_cost = math.sqrt(max(0.0, var_cost))
    robust_cost = mean_cost + UCB_WEIGHT * std_cost

    mean_vs_avg_mps = sum(
        _metric_float_first(r.metrics, ["vs_avg_mps"], 0.0) for r in per_track
    ) / n
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

    return {
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
    }


# ============================================================
# Candidate selection
# ============================================================
def _select_best_so_far_schedule(records: List[Dict[str, Any]]) -> List[EvalTask]:
    selected: List[EvalTask] = []

    non_startup = [r for r in records if not _is_startup_record(r)]
    non_startup.sort(key=lambda r: _global_trial(r))

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
                _global_trial(r),
            ),
        )
        best_obj_so_far = min(_train_objective(r) for r in history)

        selected.append(
            EvalTask(
                selection_family="best_so_far",
                selection_label=f"best_so_far_g{checkpoint_global:03d}",
                checkpoint_trial_filtered=int(checkpoint_filtered),
                checkpoint_trial_global=int(checkpoint_global),
                source_candidate_id=int(best_rec.get("candidate_id", best_rec["_global_trial"])),
                source_phase=str(best_rec.get("phase", UNSTAGED_PHASE_NAME)),
                source_phase_trial=int(_source_phase_trial_for_unstaged(best_rec)),
                source_trial_global=int(_global_trial(best_rec)),
                source_objective_cost_train=float(_train_objective(best_rec)),
                source_mean_cost_train=float(best_rec.get("mean_cost", float("nan"))),
                source_std_cost_train=float(best_rec.get("std_cost", float("nan"))),
                source_robust_cost_train=float(best_rec.get("robust_cost", _train_objective(best_rec))),
                source_best_objective_so_far_train=float(best_obj_so_far),
                source_surrogate_pred_mean=None,
                source_surrogate_pred_std=None,
                source_surrogate_acq_value=None,
                is_tail_padded=bool(tail_padded),
                theta={k: float(v) for k, v in best_rec["theta"].items()},
            )
        )

    return selected


def _surrogate_field_name(kind: str) -> str:
    if kind == "acquisition":
        return "surrogate_best_by_acquisition"
    return "surrogate_best_by_pred_mean"


def _select_algorithm_thinks_schedule(records: List[Dict[str, Any]]) -> List[EvalTask]:
    selected: List[EvalTask] = []

    non_startup = [r for r in records if not _is_startup_record(r)]
    non_startup.sort(key=lambda r: _global_trial(r))

    checkpoints_filtered = _build_filtered_checkpoints(CHECKPOINT_STEP)
    if not non_startup:
        return selected

    field = _surrogate_field_name(SURROGATE_KIND)
    available_effective = len(non_startup)

    label_prefix = "algorithm_thinks_pred_mean"
    if SURROGATE_KIND == "acquisition":
        label_prefix = "algorithm_thinks_acquisition"

    for checkpoint_filtered in checkpoints_filtered:
        checkpoint_global = _filtered_to_global_trial(checkpoint_filtered)
        effective_ref = min(checkpoint_filtered, available_effective)
        tail_padded = checkpoint_filtered > available_effective

        history = non_startup[:effective_ref]
        if not history:
            continue

        src_rec = history[-1]

        snap = src_rec.get(field, None)
        if not bool(src_rec.get("surrogate_snapshot_available", False)):
            continue
        if not isinstance(snap, dict):
            continue
        if "theta" not in snap or not isinstance(snap["theta"], dict):
            continue

        best_obj_so_far = min(_train_objective(r) for r in history)

        pred_mean = snap.get("pred_mean", None)
        pred_std = snap.get("pred_std", None)
        acq_value = snap.get("acq_value", None)

        selected.append(
            EvalTask(
                selection_family="algorithm_thinks",
                selection_label=f"{label_prefix}_g{checkpoint_global:03d}",
                checkpoint_trial_filtered=int(checkpoint_filtered),
                checkpoint_trial_global=int(checkpoint_global),
                source_candidate_id=int(src_rec.get("candidate_id", src_rec["_global_trial"])),
                source_phase=str(src_rec.get("phase", UNSTAGED_PHASE_NAME)),
                source_phase_trial=int(_source_phase_trial_for_unstaged(src_rec)),
                source_trial_global=int(_global_trial(src_rec)),
                source_objective_cost_train=float(_train_objective(src_rec)),
                source_mean_cost_train=float(src_rec.get("mean_cost", float("nan"))),
                source_std_cost_train=float(src_rec.get("std_cost", float("nan"))),
                source_robust_cost_train=float(src_rec.get("robust_cost", _train_objective(src_rec))),
                source_best_objective_so_far_train=float(best_obj_so_far),
                source_surrogate_pred_mean=None if pred_mean is None else float(pred_mean),
                source_surrogate_pred_std=None if pred_std is None else float(pred_std),
                source_surrogate_acq_value=None if acq_value is None else float(acq_value),
                is_tail_padded=bool(tail_padded),
                theta={k: float(v) for k, v in snap["theta"].items()},
            )
        )

    return selected


# ============================================================
# Candidate JSON / plotting / runtime logging
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
        "source_surrogate_pred_mean": task.source_surrogate_pred_mean,
        "source_surrogate_pred_std": task.source_surrogate_pred_std,
        "source_surrogate_acq_value": task.source_surrogate_acq_value,
        "is_tail_padded": bool(task.is_tail_padded),
        "theta": {k: float(v) for k, v in task.theta.items()},
    }


def _save_plot(rows: List[Dict[str, Any]], selection_family: str, y_key: str, y_label: str) -> None:
    xs = []
    ys = []

    family_rows = [r for r in rows if r["selection_family"] == selection_family]
    family_rows.sort(key=lambda r: int(r["checkpoint_trial_global"]))

    for row in family_rows:
        xs.append(int(row["checkpoint_trial_global"]))
        ys.append(float(row[y_key]))

    if not xs:
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Global trial checkpoint")
    plt.ylabel(y_label)
    plt.title(f"{selection_family}: {y_key}")
    plt.grid(True, alpha=0.3)
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


def _print_runtime_row(
    idx: int,
    total: int,
    task: EvalTask,
    row: Dict[str, Any],
    cache_hit: bool,
) -> None:
    print(
        f"[TEST {idx:03d}/{total:03d}] "
        f"{task.selection_label} | family={task.selection_family} | "
        f"filtered={task.checkpoint_trial_filtered} | "
        f"global={task.checkpoint_trial_global} | "
        f"source_phase={task.source_phase} | "
        f"source_global={task.source_trial_global}"
    )

    if cache_hit:
        print("    [cache] reused evaluation")

    print(
        f"    robust={row['test_robust_cost']:.6g} | "
        f"mean={row['test_mean_cost']:.6g} | "
        f"std={row['test_std_cost']:.6g} | "
        f"vs_avg={row['test_avg_vs_avg_mps']:.6g} | "
        f"soft={row['test_avg_soft_violations']:.6g} | "
        f"medium={row['test_avg_medium_violations']:.6g} | "
        f"hard={row['test_avg_hard_violations']:.6g} | "
        f"crash%={row['test_crash_percent']:.6g} | "
        f"cache_hit={int(cache_hit)}"
    )


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

    if SURROGATE_KIND not in {"pred_mean", "acquisition"}:
        raise ValueError("EVAL_SURROGATE_KIND must be 'pred_mean' or 'acquisition'")

    try:
        if os.path.exists(OUTPUT_CSV_PATH):
            os.remove(OUTPUT_CSV_PATH)
    except Exception:
        pass

    original_control_json = _load_json(CONTROL_PARAM_JSON)

    records = load_jsonl(RUN_LOG_PATH)
    if len(records) == 0:
        raise RuntimeError(f"Empty run log: {RUN_LOG_PATH}")

    startup_count = sum(1 for r in records if _is_startup_record(r))
    nonstartup_count = len(records) - startup_count

    best_tasks = _select_best_so_far_schedule(records)
    algo_tasks = _select_algorithm_thinks_schedule(records)

    tasks = best_tasks + algo_tasks
    tasks.sort(
        key=lambda t: (
            int(t.checkpoint_trial_global),
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
    print("[CFG] UNSTAGED_PHASE_NAME     =", UNSTAGED_PHASE_NAME)
    print("[CFG] UNSTAGED_N_TRIALS       =", UNSTAGED_N_TRIALS)
    print("[CFG] UNSTAGED_STARTUP        =", UNSTAGED_STARTUP)
    print("[CFG] UNSTAGED_EFFECTIVE      =", UNSTAGED_EFFECTIVE_TRIALS)
    print("[CFG] CHECKPOINT_STEP         =", CHECKPOINT_STEP)
    print("[CFG] SURROGATE_KIND          =", SURROGATE_KIND)
    print("")
    print(f"[INFO] Loaded {len(records)} tuning records from run log jsonl.")
    print(f"[INFO] Startup filter is GLOBAL: _global_trial <= {UNSTAGED_STARTUP}")
    print(f"[INFO] Startup records        : {startup_count}")
    print(f"[INFO] Non-startup records    : {nonstartup_count}")
    print("[INFO] best_so_far biorę z pełnego run loga po odfiltrowaniu startupu.")
    print("[INFO] algorithm_thinks bierze snapshot surrogatu z ostatniego dostępnego punktu na danym checkpointcie.")
    print("[INFO] First 10 global trials :", [_global_trial(r) for r in records[:10]])
    print("")

    print(f"[INFO] Selected {len(best_tasks)} best_so_far checkpoints.")
    print(f"[INFO] Selected {len(algo_tasks)} algorithm_thinks checkpoints.")
    print("")

    _save_json_atomic(
        OUTPUT_CANDIDATES_JSON,
        [_task_to_jsonable(t) for t in tasks],
    )

    if len(tasks) == 0:
        print("[FATAL] No checkpoints selected after GLOBAL startup filtering.")
        print("[FATAL] Sprawdź czy run log naprawdę ma wystarczająco dużo rekordów > startup oraz czy snapshoty surrogatu są zapisywane.")
        print("")
        print("================ EVAL DONE ================")
        print("Rows written        : 0")
        print("Unique thetas evald : 0")
        print("Episodes done       : 0")
        print(f"Output CSV          : {OUTPUT_CSV_PATH}")
        print(f"Candidates JSON     : {OUTPUT_CANDIDATES_JSON}")
        print(f"Plots dir           : {PLOTS_DIR}")
        return

    eval_cache: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []
    episodes_done = 0

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
                summary = evaluate_theta_on_test_tracks(
                    theta_x=theta,
                    tracks=TEST_TRACKS,
                    sim_time_s=DEFAULT_SIM_TIME_S,
                    critical_crash_multiplier=TEST_CRITICAL_CRASH_MULTIPLIER,
                )
                eval_cache[sig] = copy.deepcopy(summary)
                episodes_done += len(TEST_TRACKS)
            else:
                summary = copy.deepcopy(eval_cache[sig])

            row = {
                "selection_family": task.selection_family,
                "selection_label": task.selection_label,
                "surrogate_kind": "" if task.selection_family != "algorithm_thinks" else SURROGATE_KIND,
                "checkpoint_step": int(CHECKPOINT_STEP),
                "unstaged_phase_name": str(UNSTAGED_PHASE_NAME),
                "unstaged_trials": int(UNSTAGED_N_TRIALS),
                "unstaged_startup": int(UNSTAGED_STARTUP),
                "unstaged_effective_trials": int(UNSTAGED_EFFECTIVE_TRIALS),
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
                "source_surrogate_pred_mean": "" if task.source_surrogate_pred_mean is None else float(task.source_surrogate_pred_mean),
                "source_surrogate_pred_std": "" if task.source_surrogate_pred_std is None else float(task.source_surrogate_pred_std),
                "source_surrogate_acq_value": "" if task.source_surrogate_acq_value is None else float(task.source_surrogate_acq_value),
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

            _print_runtime_row(
                idx=idx,
                total=len(tasks),
                task=task,
                row=row,
                cache_hit=cache_hit,
            )

        _save_family_plots(csv_rows, "best_so_far")
        _save_family_plots(csv_rows, "algorithm_thinks")

    finally:
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


if __name__ == "__main__":
    main()
