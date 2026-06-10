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
from typing import Dict, Any, Optional, Tuple, List, Set


# ============================================================
# Workspace / paths
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser(
    "~/Desktop/fs_control_amz/TUNE_RANDOM_SEARCH"
)
CATKIN_WS = os.path.abspath(os.environ.get("EVAL_WS", DEFAULT_CATKIN_WS))
WS_SRC = os.path.join(CATKIN_WS, "src")

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
METRICS_CSV = os.path.join(
    WS_SRC, "lem_simulator", "logs", "run_default_metrics.csv"
)

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
RUN_LOG_PATH = os.path.abspath(
    os.environ.get(
        "EVAL_RUN_LOG_PATH",
        os.path.join(SCRIPT_DIR, "tuning_run_full_coupled_random_search.jsonl"),
    )
)
OUTPUT_CSV = os.path.abspath(
    os.environ.get(
        "EVAL_OUTPUT_CSV",
        os.path.join(SCRIPT_DIR, "random_search_test_eval_full_coupled.csv"),
    )
)

# ============================================================
# Eval config
# ============================================================
GLOBAL_SEED = 123
random.seed(GLOBAL_SEED)

DEFAULT_SIM_TIME_S = int(os.environ.get("EVAL_SIM_TIME_S", "60"))
TEST_TRACK_SET = list(range(8, 15))  # 8..14

BEST_SO_FAR_EVERY = int(os.environ.get("EVAL_BEST_SO_FAR_EVERY", "10"))
THINK_EVERY = int(os.environ.get("EVAL_SURROGATE_EVERY", "12"))
TOPK = int(os.environ.get("EVAL_TOPK", "3"))

TEST_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("EVAL_TEST_CRITICAL_CRASH_MULTIPLIER", "1.0")
)

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

# ============================================================
# Tuner config copied for consistency
# ============================================================
SAFE_PLANNER = {
    "mpc.bounds.max_vx": 13.5,
    "mpc.model.mux": 0.6,
    "mpc.model.muy": 0.7,
    VIRTUAL_SLACK_TRACK_FACTOR: 1.0,
    VIRTUAL_SLACK_FRIC_FACTOR: 1.0,
}

SPEED_AGGR_BOUNDS = {
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
    "mpc.cost.q_sdot": (0.001, 1.5),
    "mpc.cost.R_Mtv": (1e-7, 1e-5),
}

SLACK_FACTOR_BOUNDS = {
    VIRTUAL_SLACK_TRACK_FACTOR: (0.70, 1.30),
    VIRTUAL_SLACK_FRIC_FACTOR: (0.70, 1.30),
}

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(SPEED_AGGR_BOUNDS)
ALL_BOUNDS.update(COST_BOUNDS)
ALL_BOUNDS.update(SLACK_FACTOR_BOUNDS)

LOG_PARAMS = set(COST_BOUNDS.keys())

GOOD_FRAC = 0.30
MIN_G_SIZE = 4

PHASE2_STARTUP = 16
REFINE_ACTIVE_ELITE_K = 5

# ============================================================
# ROS / runtime
# ============================================================
SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("EVAL_ROS_MASTER_PORT", "11734"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_eval_random_coupled_{GLOBAL_SEED}")
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

SOFT_TRACK_DIV = 10.0
MEDIUM_TRACK_DIV = 2.0
HIGH_TRACK_MUL = 1.5


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


def _save_json_atomic(path: str, data: Dict[str, Any]) -> None:
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


def _materialize_control_json(theta_x: Dict[str, float]) -> Dict[str, Any]:
    global BASE_CONTROL_TEMPLATE

    if BASE_CONTROL_TEMPLATE is None:
        raise RuntimeError("BASE_CONTROL_TEMPLATE is not initialized")

    data = copy.deepcopy(BASE_CONTROL_TEMPLATE)

    for path, val in theta_x.items():
        if path in SLACK_PARAM_SET:
            continue
        if path in ALL_BOUNDS:
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


def _metric_float_first(
    metrics: Dict[str, Any], names: List[str], default: float = 0.0
) -> float:
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
        ["soft_track_violations_count", "soft_track_violation_count"],
        0.0,
    )
    med_cnt = _metric_float_first(
        metrics,
        ["medium_track_violations_count", "medium_track_violation_count"],
        0.0,
    )
    high_cnt = _metric_float_first(
        metrics,
        ["high_track_violations_count", "high_track_violation_count"],
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

    sim_args = _build_sim_launch_args(
        track_index, sim_time_s, critical_crash_multiplier
    )
    sim_p = _popen_roslaunch(SIM_LAUNCH, sim_args)
    ctrl_p = _popen_roslaunch(CTRL_LAUNCH, {})

    time.sleep(0.6)

    if sim_p.poll() is not None:
        rc = int(sim_p.returncode)
        metrics = {
            "crashed": 1,
            "crash_reason": f"sim_launch_failed_rc={rc}",
            "crash_time_s": -1,
        }
        cost = compute_cost(metrics)
        _kill_process_group(sim_p)
        _kill_process_group(ctrl_p)
        return EpisodeResult(cost, metrics, track_index, True, str(metrics["crash_reason"]))

    if ctrl_p.poll() is not None:
        rc = int(ctrl_p.returncode)
        metrics = {
            "crashed": 1,
            "crash_reason": f"ctrl_launch_failed_rc={rc}",
            "crash_time_s": -1,
        }
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
    per_track: List[Dict[str, Any]] = []
    costs: List[float] = []
    vs_vals: List[float] = []
    soft_vals: List[float] = []
    med_vals: List[float] = []
    hard_vals: List[float] = []
    crash_count = 0

    for tid in tracks:
        res = run_one_episode(
            theta_x=theta_x,
            track_index=tid,
            sim_time_s=sim_time_s,
            critical_crash_multiplier=critical_crash_multiplier,
        )

        metrics = res.metrics
        costs.append(float(res.cost))
        vs_vals.append(_metric_float_first(metrics, ["vs_avg_mps"], 0.0))
        soft_vals.append(
            _metric_float_first(
                metrics,
                ["soft_track_violations_count", "soft_track_violation_count"],
                0.0,
            )
        )
        med_vals.append(
            _metric_float_first(
                metrics,
                ["medium_track_violations_count", "medium_track_violation_count"],
                0.0,
            )
        )
        hard_vals.append(
            _metric_float_first(
                metrics,
                ["high_track_violations_count", "high_track_violation_count"],
                0.0,
            )
        )

        if res.crashed:
            crash_count += 1

        per_track.append(
            {
                "track_id": int(tid),
                "cost": float(res.cost),
                "crashed": bool(res.crashed),
                "crash_reason": str(res.crash_reason),
                "vs_avg_mps": float(vs_vals[-1]),
                "soft_track_violations": float(soft_vals[-1]),
                "medium_track_violations": float(med_vals[-1]),
                "hard_track_violations": float(hard_vals[-1]),
            }
        )

    n = max(1, len(tracks))
    mean_cost = sum(costs) / n
    mean_vs = sum(vs_vals) / n
    mean_soft = sum(soft_vals) / n
    mean_med = sum(med_vals) / n
    mean_hard = sum(hard_vals) / n

    return {
        "test_tracks": list(tracks),
        "episodes": int(len(tracks)),
        "crash_count": int(crash_count),
        "crash_rate": float(crash_count / n),
        "avg_cost": float(mean_cost),
        "avg_vs_avg_mps": float(mean_vs),
        "avg_soft_violations": float(mean_soft),
        "avg_medium_violations": float(mean_med),
        "avg_hard_violations": float(mean_hard),
        "per_track": per_track,
    }


# ============================================================
# History / selection logic
# ============================================================
def _load_run_log_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            out.append(rec)

    for i, rec in enumerate(out):
        rec["_global_trial"] = int(i + 1)
    return out


def _phase_records(records: List[Dict[str, Any]], phase_name: str) -> List[Dict[str, Any]]:
    return [r for r in records if str(r.get("phase", "")) == phase_name]


def _build_G_L(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ranked = sorted(records, key=lambda r: float(r["mean_cost"]))
    n = len(ranked)
    if n == 0:
        return [], []

    g_n = max(MIN_G_SIZE, int(math.ceil(GOOD_FRAC * n)))
    g_n = min(g_n, n)
    return ranked[:g_n], ranked[g_n:]


def _build_G_L_stage2_mixed(
    phase1_records: List[Dict[str, Any]],
    phase2_records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    s1 = sorted(phase1_records, key=lambda r: float(r["mean_cost"]))
    s2 = sorted(phase2_records, key=lambda r: float(r["mean_cost"]))

    total_n = len(s1) + len(s2)
    if total_n == 0:
        return [], []

    g_target = max(MIN_G_SIZE, int(math.ceil(GOOD_FRAC * total_n)))

    g1_target = int(math.floor(0.25 * g_target))
    g2_target = g_target - g1_target

    g1_target = min(g1_target, len(s1))
    g2_target = min(g2_target, len(s2))

    while (g1_target + g2_target) < min(g_target, total_n):
        if g2_target < len(s2):
            g2_target += 1
        elif g1_target < len(s1):
            g1_target += 1
        else:
            break

    G = s1[:g1_target] + s2[:g2_target]
    used = set(id(r) for r in G)
    L = [r for r in (s1 + s2) if id(r) not in used]
    return G, L


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

    ranked = sorted(recs, key=lambda r: float(r["mean_cost"]))
    weights = [1.0 / (i + 1) for i in range(len(ranked))]
    sw = sum(weights)
    weights = [w / sw for w in weights]

    for k, (lo, hi) in bounds.items():
        if k in LOG_PARAMS:
            acc = 0.0
            for w, rec in zip(weights, ranked):
                acc += w * math.log(float(rec["theta"][k]))
            out[k] = float(math.exp(acc))
        else:
            acc = 0.0
            for w, rec in zip(weights, ranked):
                acc += w * float(rec["theta"][k])
            out[k] = float(_clamp(acc, lo, hi))

    return out


def _topk_best_so_far(
    history: List[Dict[str, Any]], topk: int
) -> List[Dict[str, Any]]:
    return sorted(history, key=lambda r: float(r["mean_cost"]))[:max(1, topk)]


def _algorithm_thinks_theta(
    history: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, float]], str]:
    if len(history) == 0:
        return None, "no_history"

    last_phase = str(history[-1].get("phase", ""))

    phase1_hist = _phase_records(history, "phase1_cost_only_random")
    phase2_hist = _phase_records(history, "phase2_joint_random")
    phase3_hist = _phase_records(history, "phase3_refine")

    if last_phase == "phase1_cost_only_random":
        G1, _ = _build_G_L(phase1_hist)
        tuned = _weighted_center_from_records(G1, COST_BOUNDS)
        full = dict(SAFE_PLANNER)
        full.update(tuned)
        return full, "phase1_G1_center"

    if last_phase == "phase2_joint_random":
        if len(phase2_hist) <= PHASE2_STARTUP:
            if len(phase2_hist) == 0:
                G1, _ = _build_G_L(phase1_hist)
                tuned_cost = _weighted_center_from_records(G1, COST_BOUNDS)
                full = dict(SAFE_PLANNER)
                full.update(tuned_cost)
                return full, "phase2_startup_fallback_stage1_center"

            center = _weighted_center_from_records(phase2_hist, ALL_BOUNDS)
            return center, "phase2_startup_center"

        Gmix, _ = _build_G_L_stage2_mixed(phase1_hist, phase2_hist)
        center = _weighted_center_from_records(Gmix, ALL_BOUNDS)
        return center, "phase2_Gmix_center"

    if last_phase == "phase3_refine":
        elite = sorted(phase3_hist, key=lambda r: float(r["mean_cost"]))[
            : max(1, min(REFINE_ACTIVE_ELITE_K, len(phase3_hist)))
        ]
        center = _weighted_center_from_records(elite, ALL_BOUNDS)
        return center, "phase3_active_elite_center"

    return None, "unknown_phase"


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
    "source_mean_cost_train",
    "test_tracks",
    "test_episodes",
    "test_crash_count",
    "test_crash_rate",
    "test_avg_cost",
    "test_avg_vs_avg_mps",
    "test_avg_soft_violations",
    "test_avg_medium_violations",
    "test_avg_hard_violations",
    "test_critical_crash_multiplier",
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
    source_mean_cost_train: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "checkpoint_trial": int(checkpoint_trial),
        "checkpoint_phase": str(checkpoint_phase),
        "selection_type": str(selection_type),
        "selection_label": str(selection_label),
        "source_candidate_id": "" if source_candidate_id is None else int(source_candidate_id),
        "source_phase": "" if source_phase is None else str(source_phase),
        "source_mean_cost_train": "" if source_mean_cost_train is None else float(source_mean_cost_train),
        "test_tracks": json.dumps(eval_out["test_tracks"]),
        "test_episodes": int(eval_out["episodes"]),
        "test_crash_count": int(eval_out["crash_count"]),
        "test_crash_rate": float(eval_out["crash_rate"]),
        "test_avg_cost": float(eval_out["avg_cost"]),
        "test_avg_vs_avg_mps": float(eval_out["avg_vs_avg_mps"]),
        "test_avg_soft_violations": float(eval_out["avg_soft_violations"]),
        "test_avg_medium_violations": float(eval_out["avg_medium_violations"]),
        "test_avg_hard_violations": float(eval_out["avg_hard_violations"]),
        "test_critical_crash_multiplier": float(TEST_CRITICAL_CRASH_MULTIPLIER),
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
    _must_exist(RUN_LOG_PATH, "RUN_LOG_PATH")

    original_control_json = _load_json(CONTROL_PARAM_JSON)
    BASE_CONTROL_TEMPLATE = copy.deepcopy(original_control_json)

    try:
        if os.path.exists(OUTPUT_CSV):
            os.remove(OUTPUT_CSV)
    except Exception:
        pass

    print("[CFG] CATKIN_WS               =", CATKIN_WS)
    print("[CFG] RUN_LOG_PATH            =", RUN_LOG_PATH)
    print("[CFG] OUTPUT_CSV              =", OUTPUT_CSV)
    print("[CFG] TEST_TRACK_SET          =", TEST_TRACK_SET)
    print("[CFG] EVAL_SIM_TIME_S         =", DEFAULT_SIM_TIME_S)
    print("[CFG] BEST_SO_FAR_EVERY       =", BEST_SO_FAR_EVERY)
    print("[CFG] THINK_EVERY             =", THINK_EVERY)
    print("[CFG] TOPK                    =", TOPK)
    print("[CFG] TEST_CRASH_MULTIPLIER   =", TEST_CRITICAL_CRASH_MULTIPLIER)
    print(f"[CFG] PHASE2_STARTUP          = {PHASE2_STARTUP}")
    print(f"[CFG] REFINE_ACTIVE_ELITE_K   = {REFINE_ACTIVE_ELITE_K}")
    print("")

    records = _load_run_log_jsonl(RUN_LOG_PATH)
    if len(records) == 0:
        print("[FATAL] Empty tuning jsonl.", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Loaded {len(records)} tuning records.")

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
        # ----------------------------------------------------
        # co 10 triali: top-3 best-so-far
        # ----------------------------------------------------
        for checkpoint in range(BEST_SO_FAR_EVERY, len(records) + 1, BEST_SO_FAR_EVERY):
            history = records[:checkpoint]
            checkpoint_phase = str(history[-1].get("phase", ""))

            best_recs = _topk_best_so_far(history, TOPK)
            print(
                f"\n[CHECKPOINT {checkpoint:03d}] best-so-far top-{len(best_recs)} "
                f"(phase={checkpoint_phase})"
            )

            for rank, rec in enumerate(best_recs, start=1):
                theta = {k: float(v) for k, v in rec["theta"].items()}
                eval_out = evaluate_theta_on_test_tracks(
                    theta_x=theta,
                    tracks=TEST_TRACK_SET,
                    sim_time_s=DEFAULT_SIM_TIME_S,
                    critical_crash_multiplier=TEST_CRITICAL_CRASH_MULTIPLIER,
                )

                row = _make_csv_row(
                    checkpoint_trial=checkpoint,
                    checkpoint_phase=checkpoint_phase,
                    selection_type="best_so_far",
                    selection_label=f"top{rank}",
                    theta=theta,
                    eval_out=eval_out,
                    source_candidate_id=int(rec.get("candidate_id", -1)),
                    source_phase=str(rec.get("phase", "")),
                    source_mean_cost_train=float(rec.get("mean_cost", float("nan"))),
                )
                _append_csv_row(OUTPUT_CSV, CSV_FIELDS, row)

                print(
                    f"  [top{rank}] cand_id={rec.get('candidate_id')} "
                    f"train_cost={float(rec.get('mean_cost')):.6g} "
                    f"test_avg_cost={eval_out['avg_cost']:.6g} "
                    f"crashes={eval_out['crash_count']}/{eval_out['episodes']}"
                )

        # ----------------------------------------------------
        # co 12 triali: to, co random-search "myśli"
        # ----------------------------------------------------
        for checkpoint in range(THINK_EVERY, len(records) + 1, THINK_EVERY):
            history = records[:checkpoint]
            checkpoint_phase = str(history[-1].get("phase", ""))

            theta, think_label = _algorithm_thinks_theta(history)
            if theta is None:
                print(f"[WARN] checkpoint={checkpoint}: cannot build think-theta.")
                continue

            print(
                f"\n[CHECKPOINT {checkpoint:03d}] algorithm_thinks "
                f"(phase={checkpoint_phase}, kind={think_label})"
            )

            eval_out = evaluate_theta_on_test_tracks(
                theta_x=theta,
                tracks=TEST_TRACK_SET,
                sim_time_s=DEFAULT_SIM_TIME_S,
                critical_crash_multiplier=TEST_CRITICAL_CRASH_MULTIPLIER,
            )

            row = _make_csv_row(
                checkpoint_trial=checkpoint,
                checkpoint_phase=checkpoint_phase,
                selection_type="algorithm_thinks",
                selection_label=think_label,
                theta=theta,
                eval_out=eval_out,
            )
            _append_csv_row(OUTPUT_CSV, CSV_FIELDS, row)

            print(
                f"  [think] kind={think_label} "
                f"test_avg_cost={eval_out['avg_cost']:.6g} "
                f"crashes={eval_out['crash_count']}/{eval_out['episodes']}"
            )

    finally:
        try:
            _save_json_atomic(CONTROL_PARAM_JSON, original_control_json)
        except Exception:
            pass

        if roscore_p is not None:
            _kill_process_group(roscore_p)

    print("\n================ DONE ================\n")
    print(f"Saved CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
