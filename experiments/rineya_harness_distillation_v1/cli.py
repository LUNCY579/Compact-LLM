"""Offline-only entrypoints. No model loader, paid API, or training CLI is exposed."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from .collection import Ledger, collect
from .contracts import (EMPTY, HARNESS_HASH, MODES, PROCEDURAL, Limits, Task,
                        canonical, digest, loads, require, validate_catalog)
from .datasets import compile_dataset
from .evaluation import evaluate, paired_report
from .fixtures import FixtureStudent, RuleReviewer, ScriptedExpert, make_tasks
from .storage import read_jsonl, verify_bundle, write_bundle

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "rhd001.json"


def plan(path: Path) -> dict:
    value = loads(path.read_text(encoding="utf-8"))
    require(value["experiment"] == "RHD-001" and value["native_training_authorized"] is False,
            "this release cannot grant training permission")
    Limits(**value["collection_limits"])
    require(value["training_parent_cap"] == 256 and value["attempts_per_task"] == 2,
            "new collection scale needs a separately reviewed plan")
    require(value["active_modes"] == ["student_only", "review_empty", "review_k"], "frozen arm subset changed")
    return {"plan": value, "plan_hash": digest(value), "harness_hash": HARNESS_HASH,
            "reference_hash": PROCEDURAL.fingerprint, "native_status": "BLOCKED_NOT_QUALIFIED",
            "next_gate": "exact-host tokenizer/forward/memory qualification; fresh authorization"}


def demo(directory: Path, count: int = 24, attempts: int = 2) -> dict:
    require(type(count) is int and 1 <= count <= 256 and attempts in {1, 2}, "bounded fixture demo")
    tasks = make_tasks(count)
    calibration = make_tasks(12, split="calibration", offset=1000)
    retention = make_tasks(12, split="retention_eval", offset=2000)
    validate_catalog(tasks + calibration + retention)
    limits = Limits()
    records, files, summaries = [], {}, {}
    for mode in ("student_only", "review_empty", "review_k"):
        ledger, arm_records = Ledger(limits), []
        for task in tasks:
            for attempt in range(attempts):
                arm_records.append(collect(task, FixtureStudent(), mode=mode, limits=limits,
                                           ledger=ledger, attempt=attempt, seed=20260927,
                                           reviewer=RuleReviewer() if mode.startswith("review_") else None,
                                           reference=PROCEDURAL if mode == "review_k" else EMPTY))
        rows, pairs, summary = compile_dataset(arm_records, tasks + calibration + retention)
        records.extend(arm_records)
        files[f"sft_{mode}.jsonl"] = rows
        files[f"pair_candidates_{mode}.jsonl"] = pairs
        summaries[mode] = summary
    eval_reports = {}
    for label, panel in (("calibration", calibration), ("retention", retention)):
        panel_hash = digest(sorted((t.key, t.fingerprint) for t in panel))
        base = evaluate(panel, FixtureStudent(), limits, panel_hash=panel_hash)
        expert = evaluate(panel, ScriptedExpert(), limits, panel_hash=panel_hash)
        files[f"{label}_base.jsonl"] = base
        files[f"{label}_scripted_expert.jsonl"] = expert
        eval_reports[label] = paired_report(base, expert)
    report = {"evidence": "SYNTHETIC_PROTOCOL_ONLY", "trained_models": 0,
              "native_model_loads": 0, "paid_calls": 0, "optimizer_steps": 0,
              "collection": summaries, "fixture_evaluation": eval_reports,
              "warning": "The scripted expert is hand-coded, not learned or distilled. No Rineya quality claim."}
    files.update({"catalog.jsonl": [asdict(t) for t in tasks + calibration + retention],
                  "trajectories.private.jsonl": records, "REPORT.json": report})
    write_bundle(directory, files, evidence="SYNTHETIC_PROTOCOL_ONLY")
    verify_bundle(directory)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p = sub.add_parser("demo")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", type=int, default=24)
    p.add_argument("--attempts", type=int, default=2)
    p = sub.add_parser("compile")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, required=True)
    p.add_argument("--mode", choices=sorted(MODES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(args.config)
        elif args.command == "demo":
            result = demo(args.out, args.tasks, args.attempts)
        elif args.command == "verify":
            result = verify_bundle(args.out)
        else:
            tasks = [Task.from_dict(row) for row in read_jsonl(args.catalog)]
            all_records = read_jsonl(args.trajectories)
            records = [r for r in all_records if r["mode"] == args.mode]
            sft, pairs, result = compile_dataset(records, tasks)
            result["unselected_records"] = len(all_records) - len(records)
            write_bundle(args.out, {"sft.jsonl": sft, "pair_candidates.jsonl": pairs,
                                    "REPORT.json": result}, evidence="REPLAY_VERIFIED_DATA_NOT_NATIVE_EFFICACY")
        print(canonical(result))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"RHD blocked: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
