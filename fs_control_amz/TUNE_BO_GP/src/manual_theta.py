#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import copy
import signal
import socket
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List


# ============================================================
# Workspace / paths
# ============================================================
DEFAULT_CATKIN_WS = os.path.expanduser(
    "~/Desktop/fs_control_amz/TUNE_BO_GP"
)
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
    os.environ.get(
        "EVAL_ACADOS_LIB",
        os.environ.get("TUNE_ACADOS_LIB", DEFAULT_ACADOS_LIB)
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
METRICS_CSV = os.path.join(
    WS_SRC, "lem_simulator", "logs", "run_default_metrics.csv"
)


def _pick_first_existing(paths: List[str]) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    return paths[0]


CONTROL_PARAM_JSON = _pick_first_existing(CONTROL_PARAM_CANDIDATES)

# ============================================================
# Eval profiles
# Zgodnie z coupled baseline
# ============================================================
DEFAULT_SIM_TIME_S = int(os.environ.get("EVAL_SIM_TIME_S", "60"))
UCB_WEIGHT = float(os.environ.get("EVAL_UCB_WEIGHT", "1.0"))
TEST_CRITICAL_CRASH_MULTIPLIER = float(
    os.environ.get("EVAL_TEST_CRITICAL_CRASH_MULTIPLIER", "1.0")
)

EVAL_PROFILE_NAME = os.environ.get("EVAL_PROFILE", "night_4track")
# EVAL_PROFILE_NAME = "smoke_1track"

if EVAL_PROFILE_NAME == "smoke_1track":
    DEFAULT_EVAL_TRACK_SET = "1"
elif EVAL_PROFILE_NAME == "night_4track":
    DEFAULT_EVAL_TRACK_SET = "8,9,10,11,12,13,14"
else:
    raise ValueError(f"Unknown EVAL_PROFILE_NAME: {EVAL_PROFILE_NAME}")


def _parse_track_set() -> List[int]:
    raw = os.environ.get("EVAL_TRACK_SET", DEFAULT_EVAL_TRACK_SET).strip()
    vals: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("EVAL_TRACK_SET resolved to an empty list.")
    return vals


TEST_TRACKS = _parse_track_set()

# ============================================================
# Manual theta
# Edytuj tutaj swoje nastawy coupled.
#
# Możesz też nadpisać:
# - MANUAL_THETA_JSON='{"mpc.bounds.max_vx": 14.5, ...}'
# - MANUAL_THETA_JSON_PATH=/path/to/file.json
#
# Jeśli zostawisz pusty dict {}, skrypt odpali bieżący control_param.json
# bez żadnych ręcznych override'ów.
# ============================================================
MANUAL_THETA: Dict[str, float] = {
    # ---------------- Planner / model ----------------
    "mpc.bounds.max_vx": 18.00,
    "mpc.model.mux": 1.0,
    "mpc.model.muy": 1.35,
    "mpc.cost.q_sdot": 0.8,

    # ---------------- Costs ----------------
    "mpc.cost.q_ey": 10.0,
    "mpc.cost.Q_epsi": 30.0,
    "mpc.cost.R_dT": 10.0,
    "mpc.cost.R_u_ddelta_cmd": 10.0,
    "mpc.cost.R_Mtv": 5.00e-6,
    "mpc.cost.Q_beta": 0.5,
}

# ============================================================
# Full coupled tuned params / bounds
# Zgodnie z tunerem coupled
# ============================================================
COST_BOUNDS = {
    "mpc.cost.q_ey": (1.0, 100.0),
    "mpc.cost.Q_epsi": (1.0, 100.0),
    "mpc.cost.R_dT": (1.0, 100.0),
    "mpc.cost.R_u_ddelta_cmd": (1.0, 100.0),
    "mpc.cost.R_Mtv": (1e-8, 1e-5),
    "mpc.cost.Q_beta": (0.1, 10.0),
}

PLANNER_BOUNDS = {
    "mpc.bounds.max_vx": (8.0, 18.0),
    "mpc.model.mux": (0.4, 1.7),
    "mpc.model.muy": (0.4, 1.7),
    "mpc.cost.q_sdot": (0.1, 1.0),   # liniowy, zgodnie z tunerem
}

ALL_BOUNDS: Dict[str, Tuple[float, float]] = {}
ALL_BOUNDS.update(COST_BOUNDS)
ALL_BOUNDS.update(PLANNER_BOUNDS)

BASE_CONTROL_TEMPLATE: Optional[Dict[str, Any]] = None

# ============================================================
# ROS / runtime
# ============================================================
ROS_MASTER_HOST = "127.0.0.1"
ROS_MASTER_PORT = int(os.environ.get("EVAL_ROS_MASTER_PORT", "11319"))
START_ROSCORE_IF_NEEDED = True

ROS_HOME = os.path.expanduser(f"~/.ros_manual_coupled_eval_{EVAL_PROFILE_NAME}")
ROS_LOG_DIR = os.path.join(ROS_HOME, "log")

SIM_INS_MODE = "kalman"
SIM_LOW_LEVEL_CONTROLERS = "true"

METRICS_WAIT_TIMEOUT_S = 10.0
EPISODE_HARD_TIMEOUT_MARGIN_S = 20.0

# ============================================================
# Objective
# Zgodnie z coupled tunerem
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _metric_float(metrics: Dict[str, Any], name: str) -> float:
    try:
        return float(metrics.get(name, 0.0))
    except Exception:
        return 0.0


def _metric_float_first(
    metrics: Dict[str, Any],
    names: List[str],
    default: float = 0.0,
) -> float:
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


def _resolve_manual_theta() -> Dict[str, float]:
    theta = copy.deepcopy(MANUAL_THETA)

    raw_json = os.environ.get("MANUAL_THETA_JSON", "").strip()
    raw_path = os.environ.get("MANUAL_THETA_JSON_PATH", "").strip()

    if raw_path:
        with open(raw_path, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("MANUAL_THETA_JSON_PATH must point to a JSON object.")
        for k, v in payload.items():
            theta[k] = float(v)

    if raw_json:
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("MANUAL_THETA_JSON must be a JSON object.")
        for k, v in payload.items():
            theta[k] = float(v)

    return theta


def _sanitize_theta(theta: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    unknown: List[str] = []

    for k, v in theta.items():
        if k not in ALL_BOUNDS:
            unknown.append(k)
            continue
        lo, hi = ALL_BOUNDS[k]
        out[k] = float(_clamp(float(v), lo, hi))

    if unknown:
        print("[WARN] Ignoring unknown keys:", ", ".join(sorted(unknown)))

    return out


def _materialize_control_json(theta_x: Dict[str, float]) -> Dict[str, Any]:
    global BASE_CONTROL_TEMPLATE

    if BASE_CONTROL_TEMPLATE is None:
        raise RuntimeError("BASE_CONTROL_TEMPLATE is not initialized.")

    data = copy.deepcopy(BASE_CONTROL_TEMPLATE)
    for path, val in theta_x.items():
        if path in ALL_BOUNDS:
            lo, hi = ALL_BOUNDS[path]
            _set_json_path(data, path, float(_clamp(float(val), lo, hi)))
    return data


def apply_params_to_control_json(theta_x: Dict[str, float]) -> None:
    data = _materialize_control_json(theta_x)
    _save_json_atomic(CONTROL_PARAM_JSON, data)


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


# ============================================================
# Cost
# ============================================================
def compute_cost(metrics: Dict[str, Any], sim_time_s: int) -> float:
    crashed = int(metrics.get("crashed", 0)) == 1
    if crashed:
        crash_time = float(metrics.get("crash_time_s", -1.0))
        return CRASH_PENALTY + CRASH_TIME_PENALTY * max(
            0.0,
            sim_time_s - max(0.0, crash_time),
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
        metrics = {
            "crashed": 1,
            "crash_reason": f"sim_launch_failed_rc={rc}",
            "crash_time_s": -1,
        }
        cost = compute_cost(metrics, sim_time_s)
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
        cost = compute_cost(metrics, sim_time_s)
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
        metrics = {
            "crashed": 1,
            "crash_reason": "no_metrics_file",
            "crash_time_s": -1,
        }
        cost = compute_cost(metrics, sim_time_s)
        return EpisodeResult(cost, metrics, track_index, True, str(metrics["crash_reason"]))

    metrics = _parse_metrics_csv(METRICS_CSV)
    cost = compute_cost(metrics, sim_time_s)
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
    slip_ratio_vals: List[float] = []
    slip_angle_vals: List[float] = []
    soft_vals: List[float] = []
    medium_vals: List[float] = []
    hard_vals: List[float] = []
    crash_count = 0

    for tid in tracks:
        res = run_one_episode(
            theta_x=theta_x,
            track_index=tid,
            sim_time_s=sim_time_s,
            critical_crash_multiplier=critical_crash_multiplier,
        )

        metrics = dict(res.metrics)

        vs_avg = _metric_float_first(metrics, ["vs_avg_mps"], 0.0)
        slip_ratio = _metric_float_first(metrics, ["slip_ratio_metric"], 0.0)
        slip_angle = _metric_float_first(metrics, ["slip_angle_metric"], 0.0)
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

        costs.append(float(res.cost))
        vs_vals.append(float(vs_avg))
        slip_ratio_vals.append(float(slip_ratio))
        slip_angle_vals.append(float(slip_angle))
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
                "slip_ratio_metric": float(slip_ratio),
                "slip_angle_metric": float(slip_angle),
                "soft_track_violations": float(soft_cnt),
                "medium_track_violations": float(medium_cnt),
                "hard_track_violations": float(hard_cnt),
                "crashed": bool(res.crashed),
                "crash_reason": str(res.crash_reason),
            }
        )

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    mean_cost = _mean(costs)
    var_cost = sum((x - mean_cost) * (x - mean_cost) for x in costs) / max(1, len(costs))
    std_cost = math.sqrt(max(0.0, var_cost))
    robust_cost = mean_cost + UCB_WEIGHT * std_cost

    return {
        "test_tracks": list(tracks),
        "n_test_tracks": int(len(tracks)),
        "critical_crash_multiplier": float(critical_crash_multiplier),
        "mean_cost": float(mean_cost),
        "std_cost": float(std_cost),
        "robust_cost": float(robust_cost),
        "avg_vs_avg_mps": _mean(vs_vals),
        "avg_slip_ratio_metric": _mean(slip_ratio_vals),
        "avg_slip_angle_metric": _mean(slip_angle_vals),
        "avg_soft_track_violations": _mean(soft_vals),
        "avg_medium_track_violations": _mean(medium_vals),
        "avg_hard_track_violations": _mean(hard_vals),
        "crashes": int(crash_count),
        "crash_percent": 100.0 * float(crash_count) / max(1.0, float(len(tracks))),
        "per_track": per_track,
    }


def _print_theta(theta: Dict[str, float]) -> None:
    print("[MANUAL THETA / OVERRIDES]")
    if not theta:
        print("  <empty>  -> using current control_param.json as-is")
    else:
        for k in sorted(theta.keys()):
            if k in ALL_BOUNDS:
                lo, hi = ALL_BOUNDS[k]
                print(f"  {k}: {theta[k]:.6g}   (bounds [{lo}, {hi}])")
            else:
                print(f"  {k}: {theta[k]:.6g}")
    print("")


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\n================ SUMMARY ================\n")
    print(f"Tracks                    : {summary['test_tracks']}")
    print(f"n_test_tracks             : {summary['n_test_tracks']}")
    print(f"critical_crash_multiplier : {summary['critical_crash_multiplier']:.6g}")
    print("")
    print(f"mean_cost                 : {summary['mean_cost']:.6g}")
    print(f"std_cost                  : {summary['std_cost']:.6g}")
    print(f"robust_cost               : {summary['robust_cost']:.6g}")
    print(f"avg_vs_avg_mps            : {summary['avg_vs_avg_mps']:.6g}")
    print(f"avg_slip_ratio_metric     : {summary['avg_slip_ratio_metric']:.6g}")
    print(f"avg_slip_angle_metric     : {summary['avg_slip_angle_metric']:.6g}")
    print(f"avg_soft_violations       : {summary['avg_soft_track_violations']:.6g}")
    print(f"avg_medium_violations     : {summary['avg_medium_track_violations']:.6g}")
    print(f"avg_hard_violations       : {summary['avg_hard_track_violations']:.6g}")
    print(f"crashes                   : {summary['crashes']}/{summary['n_test_tracks']}")
    print(f"crash_percent             : {summary['crash_percent']:.6g}")
    print("")

    print("================ PER TRACK ================\n")
    for row in summary["per_track"]:
        print(
            f"[track {row['track_id']:02d}] "
            f"cost={row['cost']:.6g} | "
            f"vs_avg={row['vs_avg_mps']:.6g} | "
            f"slip_ratio={row['slip_ratio_metric']:.6g} | "
            f"slip_angle={row['slip_angle_metric']:.6g} | "
            f"soft={row['soft_track_violations']:.6g} | "
            f"medium={row['medium_track_violations']:.6g} | "
            f"hard={row['hard_track_violations']:.6g} | "
            f"crashed={int(bool(row['crashed']))} | "
            f"reason={row['crash_reason']}"
        )


# ============================================================
# Main
# ============================================================
def main() -> None:
    global BASE_CONTROL_TEMPLATE

    _must_exist(CATKIN_WS, "CATKIN_WS")
    _must_exist(WS_SRC, "WS_SRC")
    _must_exist(SIM_LAUNCH, "SIM_LAUNCH")
    _must_exist(CTRL_LAUNCH, "CTRL_LAUNCH")
    _must_exist(CONTROL_PARAM_JSON, "CONTROL_PARAM_JSON")
    _must_exist(ACADOS_LIB, "ACADOS_LIB")

    original_control_json = _load_json(CONTROL_PARAM_JSON)
    BASE_CONTROL_TEMPLATE = copy.deepcopy(original_control_json)

    manual_theta_raw = _resolve_manual_theta()
    manual_theta = _sanitize_theta(manual_theta_raw)

    print("[CFG] CATKIN_WS               =", CATKIN_WS)
    print("[CFG] WS_SRC                  =", WS_SRC)
    print("[CFG] ACADOS_LIB              =", ACADOS_LIB)
    print("[CFG] EVAL_PROFILE_NAME       =", EVAL_PROFILE_NAME)
    print("[CFG] TEST_TRACKS             =", TEST_TRACKS)
    print("[CFG] EVAL_SIM_TIME_S         =", DEFAULT_SIM_TIME_S)
    print("[CFG] EVAL_UCB_WEIGHT         =", UCB_WEIGHT)
    print("[CFG] TEST_CRASH_MULTIPLIER   =", TEST_CRITICAL_CRASH_MULTIPLIER)
    print("[CFG] ROS_MASTER_PORT         =", ROS_MASTER_PORT)
    print("")

    _print_theta(manual_theta)

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

        print("\n[RUN] Starting manual coupled single-config evaluation...\n")

        summary = evaluate_theta_on_test_tracks(
            theta_x=manual_theta,
            tracks=TEST_TRACKS,
            sim_time_s=DEFAULT_SIM_TIME_S,
            critical_crash_multiplier=TEST_CRITICAL_CRASH_MULTIPLIER,
        )

        _print_summary(summary)

    finally:
        try:
            _save_json_atomic(CONTROL_PARAM_JSON, original_control_json)
        except Exception:
            pass

        if roscore_p is not None:
            _kill_process_group(roscore_p)

    print("\n[DONE] Manual coupled single-config evaluation finished.\n")


if __name__ == "__main__":
    main()
