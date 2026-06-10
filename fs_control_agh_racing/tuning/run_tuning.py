#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TunerSpec:
    key: str
    staged_ws_rel: Path
    unstaged_ws_rel: Path
    tune_runner_rel: Path
    eval_runner_rel: Path
    supports_tuning_unstaged: bool = True


TUNERS: Dict[str, TunerSpec] = {
    "gpbo": TunerSpec(
        key="gpbo",
        staged_ws_rel=Path("TUNE_BO_GP"),
        unstaged_ws_rel=Path("NO_STAGED") / "TUNE_BO_GP",
        tune_runner_rel=Path("src") / "rune_tune.sh",
        eval_runner_rel=Path("src") / "run_test.sh",
        supports_tuning_unstaged=True,
    ),
    "tpe": TunerSpec(
        key="tpe",
        staged_ws_rel=Path("TUNE_BO_TPO"),
        unstaged_ws_rel=Path("NO_STAGED") / "TUNE_BO_TPO",
        tune_runner_rel=Path("src") / "rune_tune.sh",
        eval_runner_rel=Path("src") / "run_test.sh",
        # NOTE: current repo state suggests NO_STAGED/TUNE_BO_TPO/src/BO_TPO/tpo_fixed.py
        # does not implement TPE tuning (looks like random-search). We keep evaluation working,
        # but guard unstaged tuning until code is clarified.
        supports_tuning_unstaged=False,
    ),
    "cma": TunerSpec(
        key="cma",
        staged_ws_rel=Path("TUNE_CMA_ES"),
        unstaged_ws_rel=Path("NO_STAGED") / "TUNE_CMA_ES",
        tune_runner_rel=Path("src") / "run_tunning.sh",
        eval_runner_rel=Path("src") / "run_test.sh",
        supports_tuning_unstaged=True,
    ),
    "random": TunerSpec(
        key="random",
        staged_ws_rel=Path("TUNE_RANDOM_SEARCH"),
        unstaged_ws_rel=Path("NO_STAGED") / "TUNE_RANDOM_SEARCH",
        tune_runner_rel=Path("src") / "run_random_search.sh",
        eval_runner_rel=Path("src") / "run_test.sh",
        supports_tuning_unstaged=True,
    ),
}


def _normalize_tuner_key(raw: str) -> str:
    r = raw.strip().lower()
    aliases = {
        "bo_gp": "gpbo",
        "gp": "gpbo",
        "gp_bo": "gpbo",
        "bo": "gpbo",
        "bo_tpo": "tpe",
        "tpo": "tpe",
        "tpe": "tpe",
        "cma": "cma",
        "cma_es": "cma",
        "evolution": "cma",
        "random": "random",
        "random_search": "random",
        "rs": "random",
    }
    return aliases.get(r, r)


def _parse_csv_ints(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out if out else None


def _tracks_tag(tracks: Optional[List[int]]) -> str:
    if not tracks:
        return "default"
    return "-".join(str(t) for t in tracks)


def _run_checked(cmd: List[str], *, env: Dict[str, str]) -> None:
    subprocess.run(cmd, env=env, check=True)


def _control_param_json_dir(ws: Path) -> Optional[Path]:
    c1 = ws / "src" / "dv_control" / "config" / "Params" / "control_param.json"
    c2 = ws / "src" / "dv_control" / "config" / "control_param.json"
    if c1.is_file():
        return c1.parent
    if c2.is_file():
        return c2.parent
    return None


def _expected_best_json_name(tuner: str, profile: str, staged: bool) -> str:
    if tuner == "gpbo":
        # staged uses 3-stage naming; unstaged uses slightly different legacy naming.
        if staged:
            return f"control_param_best_lti_3stage_gp_bo_tv_{profile}.json"
        return f"control_param_best_lti_gp_bo_tv_{profile}.json"
    if tuner == "tpe":
        # only staged name is known here
        if staged:
            return f"control_param_best_lti_3stage_tpe_tv_{profile}.json"
        return f"control_param_best_lti_3stage_tpe_tv_{profile}.json"
    if tuner == "random":
        if staged:
            return f"control_param_best_lti_3stage_random_search_tv_{profile}.json"
        return f"control_param_best_lti_unstaged_random_search_tv_{profile}.json"
    if tuner == "cma":
        if staged:
            return f"control_param_best_lti_3stage_cma_tv_{profile}.json"
        return f"control_param_best_lti_unstaged_cma_tv_{profile}.json"
    raise ValueError(f"Unknown tuner: {tuner}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Unified tuning+testing runner (no algorithm changes): selects tuner, staged/unstaged, "
            "budget/phase overrides and track lists; stores final artifacts under testing/."
        )
    )

    ap.add_argument("--tuner", required=True, help="gpbo | tpe | cma | random (aliases: bo_gp, bo_tpo, cma_es, random_search)")
    ap.add_argument("--unstaged", action="store_true", help="Use NO_STAGED workspace variant")
    ap.add_argument("--profile", default="night_4track", help="Training profile name (default: night_4track)")

    ap.add_argument(
        "--train-tracks",
        default="",
        help="Comma-separated list of track IDs for training (overrides profile default)",
    )
    ap.add_argument(
        "--test-tracks",
        default="",
        help="Comma-separated list of track IDs for evaluation (overrides eval default)",
    )

    ap.add_argument("--phase-trials", default="", help="Comma-separated trials per phase, e.g. 48,64,96")
    ap.add_argument("--phase-startups", default="", help="Comma-separated startups per phase, e.g. 12,16,24")

    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="Forward SKIP_BUILD=1 to underlying runner scripts",
    )
    ap.add_argument(
        "--no-eval",
        action="store_true",
        help="Run tuning only (skip testing/eval step)",
    )

    args = ap.parse_args()

    tuner_key = _normalize_tuner_key(args.tuner)
    if tuner_key not in TUNERS:
        raise SystemExit(f"Unknown tuner: {args.tuner} (normalized: {tuner_key}). Choices: {', '.join(sorted(TUNERS.keys()))}")

    spec = TUNERS[tuner_key]
    staged = not args.unstaged

    ws_rel = spec.staged_ws_rel if staged else spec.unstaged_ws_rel
    ws = (REPO_ROOT / ws_rel).resolve()
    if not ws.is_dir():
        raise SystemExit(f"Workspace not found: {ws}")

    if (not staged) and (not spec.supports_tuning_unstaged):
        raise SystemExit(
            "Unstaged tuning for this tuner is disabled in this repo state. "
            "(NO_STAGED TPE tuning code looks inconsistent.) You can still run staged, "
            "or run testing against an existing JSONL log with testing/run_testing.py."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{tuner_key}_{'staged' if staged else 'unstaged'}"

    run_dir = (REPO_ROOT / "tuning" / "runs" / tuner_key / ("staged" if staged else "unstaged") / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    tune_out_dir = run_dir / "tuning_outputs"
    tune_out_dir.mkdir(parents=True, exist_ok=True)

    runner_log_dir = run_dir / "runner_logs"
    runner_log_dir.mkdir(parents=True, exist_ok=True)

    # Standardized tuning artifacts (we redirect tuner scripts into these paths)
    tuning_run_jsonl = tune_out_dir / "tuning_run.jsonl"
    tuning_summary_json = tune_out_dir / "tuning_summary.json"
    tuning_top10_json = tune_out_dir / "top10.json"
    tuning_dist_jsonl = tune_out_dir / "distribution_log.jsonl"

    train_tracks = _parse_csv_ints(args.train_tracks)
    test_tracks = _parse_csv_ints(args.test_tracks)

    phase_trials = _parse_csv_ints(args.phase_trials)
    phase_startups = _parse_csv_ints(args.phase_startups)

    if phase_trials is not None and len(phase_trials) != 3:
        raise SystemExit("--phase-trials must have exactly 3 integers: p1,p2,p3")
    if phase_startups is not None and len(phase_startups) != 3:
        raise SystemExit("--phase-startups must have exactly 3 integers: p1,p2,p3")

    env = os.environ.copy()
    env["TUNE_WS"] = str(ws)
    env["TUNE_PROFILE"] = str(args.profile)
    env["TUNE_RUN_LOG_DIR"] = str(runner_log_dir)
    env["SKIP_BUILD"] = "1" if args.skip_build else env.get("SKIP_BUILD", "0")

    if train_tracks is not None:
        env["TUNE_TRACK_SET"] = ",".join(str(t) for t in train_tracks)

    # Phase overrides (only used by scripts that read them)
    if phase_trials is not None:
        env["TUNE_PHASE1_N_TRIALS"] = str(phase_trials[0])
        env["TUNE_PHASE2_N_TRIALS"] = str(phase_trials[1])
        env["TUNE_PHASE3_N_TRIALS"] = str(phase_trials[2])

    if phase_startups is not None:
        env["TUNE_PHASE1_STARTUP"] = str(phase_startups[0])
        env["TUNE_PHASE2_STARTUP"] = str(phase_startups[1])
        env["TUNE_PHASE3_STARTUP"] = str(phase_startups[2])

    # Redirect tuner outputs into run_dir
    env["TUNE_RUN_LOG_PATH"] = str(tuning_run_jsonl)
    env["TUNE_SUMMARY_PATH"] = str(tuning_summary_json)
    env["TUNE_TOP10_PATH"] = str(tuning_top10_json)

    if tuner_key == "cma":
        env["TUNE_DIST_LOG_PATH"] = str(tuning_dist_jsonl)

    tune_runner = (ws / spec.tune_runner_rel).resolve()
    if not tune_runner.is_file():
        raise SystemExit(f"Tune runner script not found: {tune_runner}")

    manifest = {
        "run_id": run_id,
        "tuner": tuner_key,
        "staged": staged,
        "workspace": str(ws),
        "profile": args.profile,
        "train_tracks": train_tracks,
        "test_tracks": test_tracks,
        "phase_trials": phase_trials,
        "phase_startups": phase_startups,
        "tuning_outputs": {
            "run_jsonl": str(tuning_run_jsonl),
            "summary_json": str(tuning_summary_json),
            "top10_json": str(tuning_top10_json),
            "dist_jsonl": str(tuning_dist_jsonl) if tuner_key == "cma" else None,
        },
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[RUN] Tuning: tuner={tuner_key} mode={'staged' if staged else 'unstaged'} profile={args.profile}")
    print(f"[RUN] Run dir: {run_dir}")

    _run_checked(["bash", str(tune_runner)], env=env)

    if args.no_eval:
        print("[RUN] --no-eval set: skipping testing/eval")
        return 0

    # ================================
    # Evaluation step
    # ================================
    results_dir = (REPO_ROOT / "testing" / "results" / tuner_key / ("staged" if staged else "unstaged") / run_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = run_dir / "eval_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    eval_env = env.copy()
    eval_env["EVAL_WS"] = str(ws)
    eval_env["EVAL_RUN_LOG_PATH"] = str(tuning_run_jsonl)

    if test_tracks is not None:
        eval_env["EVAL_TRACK_SET"] = ",".join(str(t) for t in test_tracks)

    # Prefer explicit profile selection when evaluator supports it
    eval_env["EVAL_PROFILE_NAME"] = eval_env.get("EVAL_PROFILE_NAME", "night_7track")
    eval_env["EVAL_PROFILE"] = eval_env.get("EVAL_PROFILE", args.profile)

    # Keep evaluator consistent with tuning phase overrides
    if phase_trials is not None:
        eval_env["EVAL_PHASE1_N_TRIALS"] = str(phase_trials[0])
        eval_env["EVAL_PHASE2_N_TRIALS"] = str(phase_trials[1])
        eval_env["EVAL_PHASE3_N_TRIALS"] = str(phase_trials[2])

    if phase_startups is not None:
        eval_env["EVAL_PHASE1_STARTUP"] = str(phase_startups[0])
        eval_env["EVAL_PHASE2_STARTUP"] = str(phase_startups[1])
        eval_env["EVAL_PHASE3_STARTUP"] = str(phase_startups[2])

    # Method-specific output knobs used by current evaluators
    if tuner_key in {"gpbo", "tpe", "random"}:
        eval_env["EVAL_OUTPUT_CSV"] = str(results_dir / "eval.csv")
        eval_env["EVAL_OUTPUT_CANDIDATES_JSON"] = str(results_dir / "candidates.json")
        eval_env["EVAL_PLOTS_DIR"] = str(plots_dir)
    elif tuner_key == "cma":
        eval_env["EVAL_RESULTS_CSV"] = str(results_dir / "eval.csv")
        eval_env["EVAL_CANDIDATES_JSON"] = str(results_dir / "candidates.json")
        eval_env["EVAL_PLOTS_DIR"] = str(plots_dir)
        eval_env["EVAL_DIST_LOG_PATH"] = str(tuning_dist_jsonl)
    else:
        raise SystemExit(f"Unsupported tuner for evaluation: {tuner_key}")

    eval_runner = (ws / spec.eval_runner_rel).resolve()
    if not eval_runner.is_file():
        raise SystemExit(f"Eval runner script not found: {eval_runner}")

    print(f"[RUN] Testing/eval: outputs -> {results_dir}")
    _run_checked(["bash", str(eval_runner)], env=eval_env)

    # ================================
    # Final artifacts (stable location)
    # ================================
    key = f"{args.profile}__train_{_tracks_tag(train_tracks)}"
    final_dir = (REPO_ROOT / "testing" / "final" / tuner_key / ("staged" if staged else "unstaged") / key)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Copy only final artifacts to testing/final
    def _copy_if_exists(src: Path, dst_name: str) -> None:
        if src.is_file():
            shutil.copy2(src, final_dir / dst_name)

    _copy_if_exists(tuning_top10_json, "top10.json")
    _copy_if_exists(tuning_summary_json, "tuning_summary.json")

    # evaluation outputs
    _copy_if_exists(results_dir / "eval.csv", "eval.csv")
    _copy_if_exists(results_dir / "candidates.json", "candidates.json")

    # best param json (stored by tuner in dv_control config dir)
    best_dir = _control_param_json_dir(ws)
    if best_dir is not None:
        best_name = _expected_best_json_name(tuner_key, args.profile, staged=staged)
        best_path = best_dir / best_name
        if best_path.is_file():
            shutil.copy2(best_path, final_dir / "best_params.json")

    meta = {
        "run_id": run_id,
        "created": stamp,
        "tuner": tuner_key,
        "mode": "staged" if staged else "unstaged",
        "profile": args.profile,
        "train_tracks": train_tracks,
        "test_tracks": test_tracks,
        "phase_trials": phase_trials,
        "phase_startups": phase_startups,
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "workspace": str(ws),
    }
    (final_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] Final results -> {final_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode)
