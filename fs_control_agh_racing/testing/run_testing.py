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
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TunerSpec:
    key: str
    staged_ws_rel: Path
    unstaged_ws_rel: Path
    eval_runner_rel: Path


TUNERS: Dict[str, TunerSpec] = {
    "gpbo": TunerSpec(
        key="gpbo",
        staged_ws_rel=Path("TUNE_BO_GP"),
        unstaged_ws_rel=Path("NO_STAGED") / "TUNE_BO_GP",
        eval_runner_rel=Path("src") / "run_test.sh",
    ),
    "tpe": TunerSpec(
        key="tpe",
        staged_ws_rel=Path("TUNE_BO_TPO"),
        unstaged_ws_rel=Path("NO_STAGED") / "TUNE_BO_TPO",
        eval_runner_rel=Path("src") / "run_test.sh",
    ),
    "cma": TunerSpec(
        key="cma",
        staged_ws_rel=Path("TUNE_CMA_ES"),
        unstaged_ws_rel=Path("NO_STAGED") / "TUNE_CMA_ES",
        eval_runner_rel=Path("src") / "run_test.sh",
    ),
    "random": TunerSpec(
        key="random",
        staged_ws_rel=Path("TUNE_RANDOM_SEARCH"),
        unstaged_ws_rel=Path("NO_STAGED") / "TUNE_RANDOM_SEARCH",
        eval_runner_rel=Path("src") / "run_test.sh",
    ),
}


def _normalize_tuner_key(raw: str) -> str:
    r = raw.strip().lower()
    aliases = {
        "bo_gp": "gpbo",
        "gp": "gpbo",
        "gp_bo": "gpbo",
        "bo_tpo": "tpe",
        "tpo": "tpe",
        "tpe": "tpe",
        "cma": "cma",
        "cma_es": "cma",
        "random": "random",
        "random_search": "random",
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


def _find_latest_run_jsonl(tuner: str, staged: bool) -> Optional[Path]:
    base = REPO_ROOT / "tuning" / "runs" / tuner / ("staged" if staged else "unstaged")
    if not base.is_dir():
        return None
    candidates: List[Path] = []
    for run_dir in base.iterdir():
        p = run_dir / "tuning_outputs" / "tuning_run.jsonl"
        if p.is_file():
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _try_load_manifest(run_jsonl: Path) -> Optional[dict]:
    # Expected: .../tuning/runs/<tuner>/<mode>/<run_id>/tuning_outputs/tuning_run.jsonl
    run_dir = run_jsonl
    for _ in range(3):
        run_dir = run_dir.parent
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        try:
            return json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Unified testing/eval runner (re-run evaluation from a tuning JSONL log)")

    ap.add_argument("--tuner", required=True, help="gpbo | tpe | cma | random")
    ap.add_argument("--unstaged", action="store_true", help="Use NO_STAGED workspace variant")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-jsonl", default=None, help="Path to tuning_run.jsonl")
    g.add_argument("--latest", action="store_true", help="Use latest run under tuning/runs/")

    ap.add_argument("--profile", default=None, help="Override eval profile (otherwise from manifest or evaluator default)")
    ap.add_argument("--test-tracks", default="", help="Comma-separated track IDs for evaluation")

    ap.add_argument("--skip-build", action="store_true", help="Forward SKIP_BUILD=1 to underlying runner scripts")

    args = ap.parse_args()

    tuner_key = _normalize_tuner_key(args.tuner)
    if tuner_key not in TUNERS:
        raise SystemExit(f"Unknown tuner: {args.tuner} (normalized: {tuner_key})")

    staged = not args.unstaged
    spec = TUNERS[tuner_key]

    ws_rel = spec.staged_ws_rel if staged else spec.unstaged_ws_rel
    ws = (REPO_ROOT / ws_rel).resolve()
    if not ws.is_dir():
        raise SystemExit(f"Workspace not found: {ws}")

    if args.latest:
        run_jsonl = _find_latest_run_jsonl(tuner_key, staged=staged)
        if run_jsonl is None:
            raise SystemExit("No tuning runs found for this tuner/mode under tuning/runs/")
    else:
        run_jsonl = Path(args.run_jsonl).expanduser().resolve()

    if not run_jsonl.is_file():
        raise SystemExit(f"Tuning JSONL not found: {run_jsonl}")

    manifest = _try_load_manifest(run_jsonl) or {}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_id = f"{stamp}_{tuner_key}_{'staged' if staged else 'unstaged'}"

    results_dir = (REPO_ROOT / "testing" / "results" / tuner_key / ("staged" if staged else "unstaged") / eval_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_tracks = _parse_csv_ints(args.test_tracks)

    env = os.environ.copy()
    env["EVAL_WS"] = str(ws)
    env["EVAL_RUN_LOG_PATH"] = str(run_jsonl)
    env["SKIP_BUILD"] = "1" if args.skip_build else env.get("SKIP_BUILD", "0")

    if test_tracks is not None:
        env["EVAL_TRACK_SET"] = ",".join(str(t) for t in test_tracks)

    profile = args.profile or manifest.get("profile")
    if profile:
        env["EVAL_PROFILE_NAME"] = profile
        env["EVAL_PROFILE"] = profile

    # Keep evaluator consistent with tuning overrides if present
    phase_trials = manifest.get("phase_trials")
    phase_startups = manifest.get("phase_startups")
    if isinstance(phase_trials, list) and len(phase_trials) == 3:
        env["EVAL_PHASE1_N_TRIALS"] = str(phase_trials[0])
        env["EVAL_PHASE2_N_TRIALS"] = str(phase_trials[1])
        env["EVAL_PHASE3_N_TRIALS"] = str(phase_trials[2])
    if isinstance(phase_startups, list) and len(phase_startups) == 3:
        env["EVAL_PHASE1_STARTUP"] = str(phase_startups[0])
        env["EVAL_PHASE2_STARTUP"] = str(phase_startups[1])
        env["EVAL_PHASE3_STARTUP"] = str(phase_startups[2])

    # Current repo uses different env var names depending on method
    if tuner_key in {"gpbo", "tpe", "random"}:
        env["EVAL_OUTPUT_CSV"] = str(results_dir / "eval.csv")
        env["EVAL_OUTPUT_CANDIDATES_JSON"] = str(results_dir / "candidates.json")
        env["EVAL_PLOTS_DIR"] = str(plots_dir)
    elif tuner_key == "cma":
        env["EVAL_RESULTS_CSV"] = str(results_dir / "eval.csv")
        env["EVAL_CANDIDATES_JSON"] = str(results_dir / "candidates.json")
        env["EVAL_PLOTS_DIR"] = str(plots_dir)
        dist = manifest.get("tuning_outputs", {}).get("dist_jsonl")
        if dist:
            env["EVAL_DIST_LOG_PATH"] = str(dist)

    eval_runner = (ws / spec.eval_runner_rel).resolve()
    if not eval_runner.is_file():
        raise SystemExit(f"Eval runner script not found: {eval_runner}")

    print(f"[RUN] Testing/eval: tuner={tuner_key} mode={'staged' if staged else 'unstaged'}")
    print(f"[RUN] Input log: {run_jsonl}")
    print(f"[RUN] Outputs: {results_dir}")

    _run_checked(["bash", str(eval_runner)], env=env)

    # Stable final location
    train_tracks = manifest.get("train_tracks") if isinstance(manifest.get("train_tracks"), list) else None
    key = f"{profile or 'unknown_profile'}__train_{_tracks_tag(train_tracks)}"
    final_dir = (REPO_ROOT / "testing" / "final" / tuner_key / ("staged" if staged else "unstaged") / key)
    final_dir.mkdir(parents=True, exist_ok=True)

    def _copy_if_exists(src: Path, dst_name: str) -> None:
        if src.is_file():
            shutil.copy2(src, final_dir / dst_name)

    _copy_if_exists(results_dir / "eval.csv", "eval.csv")
    _copy_if_exists(results_dir / "candidates.json", "candidates.json")

    meta = {
        "eval_id": eval_id,
        "created": stamp,
        "tuner": tuner_key,
        "mode": "staged" if staged else "unstaged",
        "profile": profile,
        "input_run_jsonl": str(run_jsonl),
        "results_dir": str(results_dir),
        "workspace": str(ws),
        "test_tracks": test_tracks,
    }
    (final_dir / "meta_eval.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] Final results -> {final_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode)
