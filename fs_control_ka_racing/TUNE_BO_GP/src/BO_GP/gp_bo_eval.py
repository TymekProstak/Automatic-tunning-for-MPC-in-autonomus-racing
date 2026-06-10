#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import csv
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
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_ka_racing/TUNE_BO_GP")
CATKIN_WS = os.path.abspath(
    os.environ.get("EVAL_WS", os.environ.get("TUNE_WS", DEFAULT_CATKIN_WS))
)
WS_SRC = os.path.join(CATKIN_WS, "src")
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

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
# Eval profiles
# ============================================================
DEFAULT_SIM_TIME_S = 60

# ------------------------------------------------------------
# EVAL PROFILE SWITCH
# ------------------------------------------------------------
EVAL_PROFILE_NAME = os.environ.get("EVAL_PROFILE", "smoke_3track")
EVAL_PROFILE_NAME = "night_7track"

if EVAL_PROFILE_NAME == "smoke_3track":
    DEFAULT_EVAL_TRACK_SET = "8,11,14"

    # train source = smoke_1track
    DEFAULT_RUN_LOG_NAME = "tuning_run_gpbo_smoke_1track.jsonl"
    DEFAULT_OUTPUT_CSV_NAME = "gpbo_test_eval_smoke3tracks.csv"
    DEFAULT_OUTPUT_CANDIDATES_NAME = "gpbo_test_candidates_smoke3tracks.json"
    DEFAULT_PLOTS_DIR_NAME = "gpbo_eval_plots_smoke3tracks"

    PHASE1_N_TRIALS = 12
    PHASE1_STARTUP = 5

    PHASE2_N_TRIALS = 24
    PHASE2_STARTUP = 12

    PHASE3_N_TRIALS = 24
    PHASE3_STARTUP = 8

elif EVAL_PROFILE_NAME == "night_7track":
    DEFAULT_EVAL_TRACK_SET = "8,9,10,11,12,13,14"

    # train source = night_4track
    DEFAULT_RUN_LOG_NAME = "tuning_run_gpbo_night_4track.jsonl"
    DEFAULT_OUTPUT_CSV_NAME = "gpbo_test_eval_night7tracks.csv"
    DEFAULT_OUTPUT_CANDIDATES_NAME = "gpbo_test_candidates_night7tracks.json"
    DEFAULT_PLOTS_DIR_NAME = "gpbo_eval_plots_night7tracks"

    # zgodnie z tym co pisałeś: stage 1 masz 64
    PHASE1_N_TRIALS = 64
    PHASE1_STARTUP = 24

    PHASE2_N_TRIALS = 64
    PHASE2_STARTUP = 16

    PHASE3_N_TRIALS = 96
    PHASE3_STARTUP = 24

else:
    raise ValueError(f"Unknown EVAL_PROFILE_NAME: {EVAL_PROFILE_NAME}")

# ============================================================
# Derived counts
# ============================================================
PHASE1_NAME = "phase1_cost_only_gpbo"
PHASE2_NAME = "phase2_planner_only_gpbo"
PHASE3_NAME = "phase3_joint_gpbo"

PHASE1_EFFECTIVE_TRIALS = PHASE1_N_TRIALS - PHASE1_STARTUP
PHASE2_EFFECTIVE_TRIALS = PHASE2_N_TRIALS - PHASE2_STARTUP
PHASE3_EFFECTIVE_TRIALS = PHASE3_N_TRIALS - PHASE3_STARTUP

TOTAL_EFFECTIVE_TRIALS = (
    PHASE1_EFFECTIVE_TRIALS
    + PHASE2_EFFECTIVE_TRIALS
    + PHASE3_EFFECTIVE_TRIALS
)

STAGE1_END_FILTERED_TRIAL = PHASE1_EFFECTIVE_TRIALS
STAGE2_END_FILTERED_TRIAL = PHASE1_EFFECTIVE_TRIALS + PHASE2_EFFECTIVE_TRIALS

# ============================================================
# Runtime paths
# ============================================================
RUN_LOG_PATH = os.path.abspath(
    os.environ.get("EVAL_RUN_LOG_PATH", os.path.join(SCRIPT_DIR, DEFAULT_RUN_LOG_NAME))
)
OUTPUT_CSV_PATH = os.path.abspath(
    os.environ.get("EVAL_OUTPUT_CSV", os.path.join(SCRIPT_DIR, DEFAULT_OUTPUT_CSV_NAME))
)
OUTPUT_CANDIDATES_JSON = os.path.abspath(
    os.environ.get("EVAL_OUTPUT_CANDIDATES_JSON", os.path.join(SCRIPT_DIR, DEFAULT_OUTPUT_CANDIDATES_NAME))
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

# surrogate selection:
# - pred_mean   -> what the surrogate thinks is best
# - acquisition -> what the acquisition most wants to test
SURROGATE_KIND = os.environ.get("EVAL_SURROGATE_KIND", "pred_mean").strip().lower()

# fixed default: every 6 filtered trials
CHECKPOINT_STEP = int(os.environ.get("EVAL_CHECKPOINT_STEP", "6"))

# ============================================================
# Tuner config mirrored
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
}
ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(PLANNER_BOUNDS)
ALL_BOUNDS.update(COST_BOUNDS)

# ============================================================
# ROS runtime
# ============================================================
ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("EVAL_ROS_MASTER_PORT", "11311"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_eval_gpbo_{EVAL_PROFILE_NAME}")
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
# Cache / state / CSV schema
# ============================================================
EVAL_CACHE: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}
EPISODES_DONE = 0

CSV_FIELDS = [
    "selection_family",
    "selection_label",
    "surrogate_kind",
    "checkpoint_step",
    "phase1_trials",
    "phase1_startup",
    "phase2_trials",
    "phase2_startup",
    "phase3_trials",
    "phase3_startup",
    "stage1_end_filtered_trial",
    "stage2_end_filtered_trial",
    "checkpoint_trial_filtered",
    "checkpoint_trial_global",
    "source_candidate_id",
    "source_phase",
    "source_phase_trial",
    "source_objective_cost_train",
    "source_mean_cost_train",
    "source_std_cost_train",
    "source_robust_cost_train",
    "source_best_objective_so_far_train",
    "source_surrogate_pred_mean",
    "source_surrogate_pred_std",
    "source_surrogate_acq_value",
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
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    out.sort(key=lambda r: int(r.get("candidate_id", 0)))
    return out


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


def _is_startup_record(rec: Dict[str, Any]) -> bool:
    phase = str(rec.get("phase", ""))
    phase_trial_zero_based = int(rec.get("phase_trial", -1))
    phase_trial_one_based = phase_trial_zero_based + 1

    if phase == PHASE1_NAME:
        return phase_trial_one_based <= PHASE1_STARTUP
    if phase == PHASE2_NAME:
        return phase_trial_one_based <= PHASE2_STARTUP
    if phase == PHASE3_NAME:
        return phase_trial_one_based <= PHASE3_STARTUP
    return False


def _filtered_to_global_trial(filtered_trial: int) -> int:
    if filtered_trial <= 0:
        return 0

    if filtered_trial <= PHASE1_EFFECTIVE_TRIALS:
        return PHASE1_STARTUP + filtered_trial

    if filtered_trial <= PHASE1_EFFECTIVE_TRIALS + PHASE2_EFFECTIVE_TRIALS:
        rem = filtered_trial - PHASE1_EFFECTIVE_TRIALS
        return PHASE1_N_TRIALS + PHASE2_STARTUP + rem

    rem = filtered_trial - PHASE1_EFFECTIVE_TRIALS - PHASE2_EFFECTIVE_TRIALS
    return PHASE1_N_TRIALS + PHASE2_N_TRIALS + PHASE3_STARTUP + rem


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
    non_startup.sort(key=lambda r: int(r.get("candidate_id", 0)))

    selected: List[Dict[str, Any]] = []
    checkpoints_filtered = _build_filtered_checkpoints(CHECKPOINT_STEP)

    if not non_startup:
        return selected

    for checkpoint_filtered in checkpoints_filtered:
        checkpoint_global = _filtered_to_global_trial(checkpoint_filtered)

        prefix_all = [r for r in records if int(r.get("candidate_id", 0)) <= checkpoint_global]
        prefix = [r for r in non_startup if int(r.get("candidate_id", 0)) <= checkpoint_global]
        if not prefix:
            continue

        best_rec = min(
            prefix,
            key=lambda r: (
                _train_objective(r),
                int(r.get("candidate_id", 0)),
            ),
        )

        max_seen_global = max(int(r.get("candidate_id", 0)) for r in prefix_all) if prefix_all else 0
        is_tail_padded = bool(max_seen_global < checkpoint_global)

        selected.append(
            {
                "selection_family": "best_so_far",
                "selection_label": f"best_so_far_t{checkpoint_filtered:03d}",
                "checkpoint_trial_filtered": int(checkpoint_filtered),
                "checkpoint_trial_global": int(checkpoint_global),
                "source_candidate_id": int(best_rec.get("candidate_id", 0)),
                "source_phase": str(best_rec.get("phase", "")),
                "source_phase_trial": int(best_rec.get("phase_trial", -1)),
                "source_objective_cost_train": float(_train_objective(best_rec)),
                "source_mean_cost_train": float(best_rec.get("mean_cost", float("nan"))),
                "source_std_cost_train": float(best_rec.get("std_cost", float("nan"))),
                "source_robust_cost_train": float(best_rec.get("robust_cost", float("nan"))),
                "source_best_objective_so_far_train": float(_best_so_far_value(best_rec)),
                "source_surrogate_pred_mean": float("nan"),
                "source_surrogate_pred_std": float("nan"),
                "source_surrogate_acq_value": float("nan"),
                "is_tail_padded": bool(is_tail_padded),
                "theta": {k: float(v) for k, v in best_rec.get("theta", {}).items()},
            }
        )

    return selected


def _select_algorithm_thinks_schedule(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    non_startup = [r for r in records if not _is_startup_record(r)]
    non_startup.sort(key=lambda r: int(r.get("candidate_id", 0)))

    if SURROGATE_KIND == "acquisition":
        surrogate_field = "surrogate_best_by_acquisition"
        label_prefix = "algorithm_thinks_acquisition"
    else:
        surrogate_field = "surrogate_best_by_pred_mean"
        label_prefix = "algorithm_thinks_pred_mean"

    selected: List[Dict[str, Any]] = []
    checkpoints_filtered = _build_filtered_checkpoints(CHECKPOINT_STEP)

    for checkpoint_filtered in checkpoints_filtered:
        checkpoint_global = _filtered_to_global_trial(checkpoint_filtered)

        prefix_all = [r for r in records if int(r.get("candidate_id", 0)) <= checkpoint_global]
        prefix = [r for r in non_startup if int(r.get("candidate_id", 0)) <= checkpoint_global]

        if not prefix:
            continue

        phase1_prefix = [
            r for r in prefix
            if str(r.get("phase", "")) == PHASE1_NAME
            and bool(r.get("surrogate_snapshot_available", False))
            and isinstance(r.get(surrogate_field, None), dict)
            and isinstance(r.get(surrogate_field, {}).get("theta", None), dict)
        ]
        phase2_prefix = [
            r for r in prefix
            if str(r.get("phase", "")) == PHASE2_NAME
            and bool(r.get("surrogate_snapshot_available", False))
            and isinstance(r.get(surrogate_field, None), dict)
            and isinstance(r.get(surrogate_field, {}).get("theta", None), dict)
        ]
        phase3_prefix = [
            r for r in prefix
            if str(r.get("phase", "")) == PHASE3_NAME
            and bool(r.get("surrogate_snapshot_available", False))
            and isinstance(r.get(surrogate_field, None), dict)
            and isinstance(r.get(surrogate_field, {}).get("theta", None), dict)
        ]

        if checkpoint_filtered <= STAGE1_END_FILTERED_TRIAL:
            current_stage = phase1_prefix
        elif checkpoint_filtered <= STAGE2_END_FILTERED_TRIAL:
            current_stage = phase2_prefix if phase2_prefix else phase1_prefix
        else:
            current_stage = phase3_prefix if phase3_prefix else (phase2_prefix if phase2_prefix else phase1_prefix)

        if not current_stage:
            continue

        source_rec = current_stage[-1]
        snap = source_rec[surrogate_field]
        theta = {k: float(v) for k, v in snap["theta"].items()}

        max_seen_global = max(int(r.get("candidate_id", 0)) for r in prefix_all) if prefix_all else 0
        is_tail_padded = bool(max_seen_global < checkpoint_global)

        selected.append(
            {
                "selection_family": "algorithm_thinks",
                "selection_label": f"{label_prefix}_t{checkpoint_filtered:03d}",
                "checkpoint_trial_filtered": int(checkpoint_filtered),
                "checkpoint_trial_global": int(checkpoint_global),
                "source_candidate_id": int(source_rec.get("candidate_id", 0)),
                "source_phase": str(source_rec.get("phase", "")),
                "source_phase_trial": int(source_rec.get("phase_trial", -1)),
                "source_objective_cost_train": float(_train_objective(source_rec)),
                "source_mean_cost_train": float(source_rec.get("mean_cost", float("nan"))),
                "source_std_cost_train": float(source_rec.get("std_cost", float("nan"))),
                "source_robust_cost_train": float(source_rec.get("robust_cost", float("nan"))),
                "source_best_objective_so_far_train": float(_best_so_far_value(source_rec)),
                "source_surrogate_pred_mean": float(snap.get("pred_mean", float("nan"))),
                "source_surrogate_pred_std": float(snap.get("pred_std", float("nan"))),
                "source_surrogate_acq_value": float(snap.get("acq_value", float("nan"))),
                "is_tail_padded": bool(is_tail_padded),
                "theta": theta,
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
        "surrogate_kind": str(SURROGATE_KIND),
        "checkpoint_step": int(CHECKPOINT_STEP),
        "phase1_trials": int(PHASE1_N_TRIALS),
        "phase1_startup": int(PHASE1_STARTUP),
        "phase2_trials": int(PHASE2_N_TRIALS),
        "phase2_startup": int(PHASE2_STARTUP),
        "phase3_trials": int(PHASE3_N_TRIALS),
        "phase3_startup": int(PHASE3_STARTUP),
        "stage1_end_filtered_trial": int(STAGE1_END_FILTERED_TRIAL),
        "stage2_end_filtered_trial": int(STAGE2_END_FILTERED_TRIAL),
        "checkpoint_trial_filtered": int(cand["checkpoint_trial_filtered"]),
        "checkpoint_trial_global": int(cand["checkpoint_trial_global"]),
        "source_candidate_id": int(cand["source_candidate_id"]),
        "source_phase": str(cand["source_phase"]),
        "source_phase_trial": int(cand["source_phase_trial"]),
        "source_objective_cost_train": float(cand["source_objective_cost_train"]),
        "source_mean_cost_train": float(cand["source_mean_cost_train"]),
        "source_std_cost_train": float(cand["source_std_cost_train"]),
        "source_robust_cost_train": float(cand["source_robust_cost_train"]),
        "source_best_objective_so_far_train": float(cand["source_best_objective_so_far_train"]),
        "source_surrogate_pred_mean": float(cand["source_surrogate_pred_mean"]),
        "source_surrogate_pred_std": float(cand["source_surrogate_pred_std"]),
        "source_surrogate_acq_value": float(cand["source_surrogate_acq_value"]),
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
        f"source_phase={sel['source_phase']}"
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
    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(WS_SRC, "WS_SRC")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(ACADOS_LIB, "ACADOS_LIB")
    _must_exist(RUN_LOG_PATH, "RUN_LOG_PATH")

    if SURROGATE_KIND not in {"pred_mean", "acquisition"}:
        raise ValueError("EVAL_SURROGATE_KIND must be 'pred_mean' or 'acquisition'")

    original_control_json = _load_json(CONTROL_PARAM_JSON)

    try:
        if os.path.exists(OUTPUT_CSV_PATH):
            os.remove(OUTPUT_CSV_PATH)
    except Exception:
        pass

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("[CFG] CATKIN_WS =", CATKIN_WS)
    print("[CFG] EVAL_PROFILE_NAME =", EVAL_PROFILE_NAME)
    print("[CFG] RUN_LOG_PATH =", RUN_LOG_PATH)
    print("[CFG] OUTPUT_CSV_PATH =", OUTPUT_CSV_PATH)
    print("[CFG] OUTPUT_CANDIDATES_JSON =", OUTPUT_CANDIDATES_JSON)
    print("[CFG] PLOTS_DIR =", PLOTS_DIR)
    print("[CFG] TEST_TRACKS =", TEST_TRACKS)
    print("[CFG] SIM_TIME_S =", TEST_SIM_TIME_S)
    print("[CFG] UCB_WEIGHT =", UCB_WEIGHT)
    print("[CFG] CHECKPOINT_STEP =", CHECKPOINT_STEP)
    print("[CFG] PHASE1_STARTUP =", PHASE1_STARTUP)
    print("[CFG] PHASE2_STARTUP =", PHASE2_STARTUP)
    print("[CFG] PHASE3_STARTUP =", PHASE3_STARTUP)
    print("[CFG] STAGE1_END_FILTERED_TRIAL =", STAGE1_END_FILTERED_TRIAL)
    print("[CFG] STAGE2_END_FILTERED_TRIAL =", STAGE2_END_FILTERED_TRIAL)
    print("[CFG] SURROGATE_KIND =", SURROGATE_KIND)
    print("[CFG] TEST_CRITICAL_CRASH_MULTIPLIER =", TEST_CRITICAL_CRASH_MULTIPLIER)
    print("")

    records = load_jsonl(RUN_LOG_PATH)
    if len(records) == 0:
        raise RuntimeError(f"Empty run log: {RUN_LOG_PATH}")

    print(f"[INFO] Loaded {len(records)} training log rows.")
    print("[INFO] best_so_far is reconstructed from tuning_run jsonl.")
    print("[INFO] algorithm_thinks is stage-aware and uses the surrogate snapshot from the active stage.")
    print("")

    best_schedule = _select_best_so_far_schedule(records)
    thinks_schedule = _select_algorithm_thinks_schedule(records)

    selections = best_schedule + thinks_schedule
    selections.sort(
        key=lambda r: (
            int(r["checkpoint_trial_filtered"]),
            str(r["selection_family"]),
            int(r["source_candidate_id"]),
        )
    )

    _save_json_atomic(OUTPUT_CANDIDATES_JSON, selections)

    print(f"[INFO] Selected {len(best_schedule)} best_so_far checkpoints.")
    print(f"[INFO] Selected {len(thinks_schedule)} algorithm_thinks checkpoints.")
    print(f"[INFO] Total selections = {len(selections)}")
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

    try:
        output_rows: List[Dict[str, Any]] = []

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
    print("[DONE] Total CSV rows:", len(best_schedule) + len(thinks_schedule))
    print("[DONE] Total episodes executed:", EPISODES_DONE)


if __name__ == "__main__":
    main()
