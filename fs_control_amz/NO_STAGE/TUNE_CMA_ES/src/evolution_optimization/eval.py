#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import csv
import signal
import socket
import copy
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Set


# ============================================================
# Workspace
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser("~/Desktop/fs_control_amz/TUNE_CMA_ES")
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
# Evaluation setup
# ============================================================
DEFAULT_SIM_TIME_S = 60

# Ja tutaj testuję wyłącznie na zbiorze testowym.
TEST_TRACK_SET = list(range(8, 15))   # 8..14
TEST_CRITICAL_CRASH_MULTIPLIER = 1.0

# ============================================================
# ROS / runtime
# ============================================================
SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = 11813
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser("~/.ros_test_eval_cma_coupled")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

# ============================================================
# Outputs
# ============================================================
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

RUN_LOG_PATH = os.path.join(SCRIPT_DIR, "tuning_run_full_coupled_cma.jsonl")
DIST_LOG_PATH = os.path.join(SCRIPT_DIR, "cma_generations_full_coupled.jsonl")

# Zmienione pliki pod wariant coupled
TEST_CANDIDATES_JSON = os.path.join(SCRIPT_DIR, "cma_test_candidates_full_coupled.json")
TEST_RESULTS_CSV = os.path.join(SCRIPT_DIR, "cma_test_results_full_coupled.csv")

# ============================================================
# Selection policy
# ============================================================
TOPK_BEST_SO_FAR = 3
BEST_CHECKPOINT_STEP = 10
DIST_CENTER_CHECKPOINT_STEP = 12

# ============================================================
# Virtual Params config
# ============================================================
BASE_CONTROL_TEMPLATE: Optional[Dict[str, Any]] = None

VIRTUAL_SLACK_TRACK_FACTOR = "__slack_track_factor__"
VIRTUAL_SLACK_FRIC_FACTOR = "__slack_fric_factor__"
SLACK_PARAM_SET: Set[str] = {
    VIRTUAL_SLACK_TRACK_FACTOR,
    VIRTUAL_SLACK_FRIC_FACTOR,
}

SLACK_FACTOR_BOUNDS = {
    VIRTUAL_SLACK_TRACK_FACTOR: (0.70, 1.30),
    VIRTUAL_SLACK_FRIC_FACTOR: (0.70, 1.30),
}

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

SOFT_TRACK_DIV = 10.0
MEDIUM_TRACK_DIV = 2.0
HIGH_TRACK_MUL = 1.5

# ============================================================
# Phase-1 fixed planner (Zgodne z wersją Coupled!)
# ============================================================
SAFE_PHASE1_FIXED = {
    "mpc.bounds.max_vx": 13.5,
    "mpc.model.mux": 0.6,
    "mpc.model.muy": 0.7,
    VIRTUAL_SLACK_TRACK_FACTOR: 1.0,
    VIRTUAL_SLACK_FRIC_FACTOR: 1.0,
}


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


def _set_json_path(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _get_json_path(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    keys = path.split(".")
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _save_json_atomic(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _materialize_control_json(theta_x: Dict[str, float]) -> Dict[str, Any]:
    global BASE_CONTROL_TEMPLATE

    if BASE_CONTROL_TEMPLATE is None:
        raise RuntimeError("BASE_CONTROL_TEMPLATE is not initialized")

    data = copy.deepcopy(BASE_CONTROL_TEMPLATE)

    for path, val in theta_x.items():
        if path in SLACK_PARAM_SET:
            continue
        _set_json_path(data, path, float(val))

    track_factor = float(theta_x.get(VIRTUAL_SLACK_TRACK_FACTOR, 1.0))
    fric_factor = float(theta_x.get(VIRTUAL_SLACK_FRIC_FACTOR, 1.0))

    tr_lo, tr_hi = SLACK_FACTOR_BOUNDS[VIRTUAL_SLACK_TRACK_FACTOR]
    fr_lo, fr_hi = SLACK_FACTOR_BOUNDS[VIRTUAL_SLACK_FRIC_FACTOR]

    track_factor = _clamp(track_factor, tr_lo, tr_hi)
    fric_factor = _clamp(fric_factor, fr_lo, fr_hi)

    base_track_lin = float(_get_json_path(BASE_CONTROL_TEMPLATE, "mpc.cost.q_slack_track_lin", 1.0))
    base_track_quad = float(_get_json_path(BASE_CONTROL_TEMPLATE, "mpc.cost.q_slack_track_quad", 1.0))
    base_fric_lin = float(_get_json_path(BASE_CONTROL_TEMPLATE, "mpc.cost.q_slack_fric_lin", 1.0))
    base_fric_quad = float(_get_json_path(BASE_CONTROL_TEMPLATE, "mpc.cost.q_slack_fric_quad", 1.0))

    _set_json_path(data, "mpc.cost.q_slack_track_lin", track_factor * base_track_lin)
    _set_json_path(data, "mpc.cost.q_slack_track_quad", track_factor * base_track_quad)
    _set_json_path(data, "mpc.cost.q_slack_fric_lin", fric_factor * base_fric_lin)
    _set_json_path(data, "mpc.cost.q_slack_fric_quad", fric_factor * base_fric_quad)

    return data


def apply_params_to_control_json(theta_x: Dict[str, float]) -> None:
    data = _materialize_control_json(theta_x)
    _save_json_atomic(CONTROL_PARAM_JSON, data)


def _parse_metrics_csv(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
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


def _theta_signature(theta: Dict[str, float], ndigits: int = 12) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, round(float(v), ndigits)) for k, v in theta.items()))


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
    ctrl_p = _popen_roslaunch(CTRL_LAUNCH, {})

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

    return {
        "test_tracks": list(TEST_TRACK_SET),
        "n_test_tracks": int(len(TEST_TRACK_SET)),
        "critical_crash_multiplier": float(TEST_CRITICAL_CRASH_MULTIPLIER),
        "avg_vs_avg_mps": _mean(vs_vals),
        "avg_soft_track_violations": _mean(soft_vals),
        "avg_medium_track_violations": _mean(medium_vals),
        "avg_hard_track_violations": _mean(hard_vals),
        "crashes": int(crash_count),
        "avg_cost": _mean(cost_vals),
        "per_track": per_track,
    }


# ============================================================
# Candidate selection
# ============================================================
def _phase_max_candidate_id(run_log: List[Dict[str, Any]], phase_name: str) -> int:
    vals = [int(r["candidate_id"]) for r in run_log if str(r.get("phase", "")) == phase_name]
    return max(vals) if vals else 0


def _select_best_so_far_candidates(run_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not run_log:
        return []

    max_trial = max(int(r["candidate_id"]) for r in run_log)
    selected: List[Dict[str, Any]] = []

    for checkpoint in range(BEST_CHECKPOINT_STEP, max_trial + 1, BEST_CHECKPOINT_STEP):
        prefix = [r for r in run_log if int(r["candidate_id"]) <= checkpoint]
        ranked = sorted(prefix, key=lambda r: (float(r["mean_cost"]), int(r["candidate_id"])))

        used = set()
        kept = 0

        for rec in ranked:
            theta = {k: float(v) for k, v in rec["theta"].items()}
            sig = _theta_signature(theta)
            if sig in used:
                continue
            used.add(sig)

            selected.append(
                {
                    "selection_family": "best_so_far",
                    "checkpoint_trial": int(checkpoint),
                    "rank_within_checkpoint": int(kept + 1),
                    "source_phase": str(rec.get("phase", "")),
                    "source_candidate_id": int(rec["candidate_id"]),
                    "source_generation": None,
                    "source_global_trial_end": int(rec["candidate_id"]),
                    "theta": theta,
                    "selection_label": f"best_so_far_t{checkpoint:03d}_rank{kept + 1}",
                }
            )

            kept += 1
            if kept >= TOPK_BEST_SO_FAR:
                break

    return selected


def _distribution_full_theta(rec: Dict[str, Any]) -> Dict[str, float]:
    if "distribution_mean_theta" in rec:
        return {k: float(v) for k, v in rec["distribution_mean_theta"].items()}

    # Ochrona w razie starego formatu plików
    center = {k: float(v) for k, v in rec.get("distribution_center_theta", {}).items()}
    phase = str(rec.get("phase", ""))

    if phase == "phase1_cost_only_cma":
        merged = dict(SAFE_PHASE1_FIXED)
        merged.update(center)
        return merged

    return center


def _build_dist_records_with_global_trial_end(
    dist_log: List[Dict[str, Any]],
    phase1_max_trial: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for rec in dist_log:
        phase = str(rec.get("phase", ""))
        generation = int(rec.get("generation", 0))
        lambda_ = int(rec.get("lambda", 0))

        if generation <= 0 or lambda_ <= 0:
            continue

        if phase == "phase1_cost_only_cma":
            global_trial_end = generation * lambda_
        elif phase == "phase2_joint_cma":
            global_trial_end = phase1_max_trial + generation * lambda_
        else:
            continue

        tmp = dict(rec)
        tmp["global_trial_end"] = int(global_trial_end)
        out.append(tmp)

    out.sort(key=lambda r: int(r["global_trial_end"]))
    return out


def _select_distribution_center_candidates(
    dist_log: List[Dict[str, Any]],
    phase1_max_trial: int,
    phase2_max_trial: int,
) -> List[Dict[str, Any]]:
    dist_ext = _build_dist_records_with_global_trial_end(dist_log, phase1_max_trial)
    if not dist_ext:
        return []

    max_dist_trial = phase2_max_trial
    selected: List[Dict[str, Any]] = []

    for checkpoint in range(DIST_CENTER_CHECKPOINT_STEP, max_dist_trial + 1, DIST_CENTER_CHECKPOINT_STEP):
        available = [r for r in dist_ext if int(r["global_trial_end"]) <= checkpoint]
        if not available:
            continue

        rec = available[-1]
        theta = _distribution_full_theta(rec)

        selected.append(
            {
                "selection_family": "distribution_center",
                "checkpoint_trial": int(checkpoint),
                "rank_within_checkpoint": 1,
                "source_phase": str(rec.get("phase", "")),
                "source_candidate_id": None,
                "source_generation": int(rec.get("generation", 0)),
                "source_global_trial_end": int(rec["global_trial_end"]),
                "theta": theta,
                "selection_label": f"distribution_center_t{checkpoint:03d}",
            }
        )

    return selected


# ============================================================
# CSV output
# ============================================================
def _write_results_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "selection_label",
                "selection_family",
                "checkpoint_trial",
                "rank_within_checkpoint",
                "source_phase",
                "source_candidate_id",
                "source_generation",
                "source_global_trial_end",
                "n_test_tracks",
                "critical_crash_multiplier",
                "avg_vs_avg_mps",
                "avg_soft_track_violations",
                "avg_medium_track_violations",
                "avg_hard_track_violations",
                "crashes",
                "avg_cost",
                "theta_json",
            ])
        return

    fieldnames = [
        "selection_label",
        "selection_family",
        "checkpoint_trial",
        "rank_within_checkpoint",
        "source_phase",
        "source_candidate_id",
        "source_generation",
        "source_global_trial_end",
        "n_test_tracks",
        "critical_crash_multiplier",
        "avg_vs_avg_mps",
        "avg_soft_track_violations",
        "avg_medium_track_violations",
        "avg_hard_track_violations",
        "crashes",
        "avg_cost",
        "theta_json",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ============================================================
# Main
# ============================================================
def main() -> None:
    global BASE_CONTROL_TEMPLATE

    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(RUN_LOG_PATH, "RUN_LOG_PATH")
    _must_exist(DIST_LOG_PATH, "DIST_LOG_PATH")

    original_control_json = _load_json(CONTROL_PARAM_JSON)
    BASE_CONTROL_TEMPLATE = copy.deepcopy(original_control_json)

    print("[CFG] CATKIN_WS =", CATKIN_WS)
    print("[CFG] WS_SRC    =", WS_SRC)
    print("[CFG] TEST_TRACK_SET =", TEST_TRACK_SET)
    print("[CFG] TEST critical_crash_multiplier =", TEST_CRITICAL_CRASH_MULTIPLIER)
    print("[CFG] RUN_LOG_PATH  =", RUN_LOG_PATH)
    print("[CFG] DIST_LOG_PATH =", DIST_LOG_PATH)
    print("[CFG] TEST_RESULTS_CSV =", TEST_RESULTS_CSV)
    print("[CFG] TEST_CANDIDATES_JSON =", TEST_CANDIDATES_JSON)
    print("[CFG] every", BEST_CHECKPOINT_STEP, "trials -> top", TOPK_BEST_SO_FAR, "best-so-far")
    print("[CFG] every", DIST_CENTER_CHECKPOINT_STEP, "trials -> distribution center")
    print("")

    run_log = _load_jsonl(RUN_LOG_PATH)
    dist_log = _load_jsonl(DIST_LOG_PATH)

    if not run_log:
        print("[FATAL] Pusty run log.", file=sys.stderr)
        sys.exit(2)

    phase1_max_trial = _phase_max_candidate_id(run_log, "phase1_cost_only_cma")
    phase2_max_trial = _phase_max_candidate_id(run_log, "phase2_joint_cma")

    best_candidates = _select_best_so_far_candidates(run_log)
    dist_candidates = _select_distribution_center_candidates(
        dist_log=dist_log,
        phase1_max_trial=phase1_max_trial,
        phase2_max_trial=phase2_max_trial,
    )

    all_candidates = best_candidates + dist_candidates
    _save_json_atomic(TEST_CANDIDATES_JSON, all_candidates)

    print(f"[INFO] Wybrałem {len(best_candidates)} kandydatów typu best-so-far.")
    print(f"[INFO] Wybrałem {len(dist_candidates)} kandydatów typu distribution-center.")
    print(f"[INFO] Łącznie do testu: {len(all_candidates)} kandydatów.")
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

    csv_rows: List[Dict[str, Any]] = []

    try:
        for idx, cand in enumerate(all_candidates):
            print(
                f"[TEST {idx + 1:03d}/{len(all_candidates):03d}] "
                f"{cand['selection_label']} | family={cand['selection_family']} "
                f"| checkpoint={cand['checkpoint_trial']}"
            )

            test_info = evaluate_theta_on_test_tracks(cand["theta"])

            row = {
                "selection_label": cand["selection_label"],
                "selection_family": cand["selection_family"],
                "checkpoint_trial": int(cand["checkpoint_trial"]),
                "rank_within_checkpoint": int(cand["rank_within_checkpoint"]),
                "source_phase": cand["source_phase"],
                "source_candidate_id": cand["source_candidate_id"],
                "source_generation": cand["source_generation"],
                "source_global_trial_end": cand["source_global_trial_end"],
                "n_test_tracks": int(test_info["n_test_tracks"]),
                "critical_crash_multiplier": float(test_info["critical_crash_multiplier"]),
                "avg_vs_avg_mps": float(test_info["avg_vs_avg_mps"]),
                "avg_soft_track_violations": float(test_info["avg_soft_track_violations"]),
                "avg_medium_track_violations": float(test_info["avg_medium_track_violations"]),
                "avg_hard_track_violations": float(test_info["avg_hard_track_violations"]),
                "crashes": int(test_info["crashes"]),
                "avg_cost": float(test_info["avg_cost"]),
                "theta_json": json.dumps(cand["theta"], sort_keys=True),
            }
            csv_rows.append(row)

            print(
                f"  avg_vs={row['avg_vs_avg_mps']:.6g} | "
                f"soft={row['avg_soft_track_violations']:.6g} | "
                f"medium={row['avg_medium_track_violations']:.6g} | "
                f"hard={row['avg_hard_track_violations']:.6g} | "
                f"crashes={row['crashes']} | "
                f"avg_cost={row['avg_cost']:.6g}"
            )
            print("")

    finally:
        try:
            _save_json_atomic(CONTROL_PARAM_JSON, original_control_json)
        except Exception:
            pass

        if roscore_p is not None:
            _kill_process_group(roscore_p)

    _write_results_csv(csv_rows, TEST_RESULTS_CSV)

    print("[DONE]")
    print("Saved candidates json:", TEST_CANDIDATES_JSON)
    print("Saved test csv:", TEST_RESULTS_CSV)


if __name__ == "__main__":
    main()
