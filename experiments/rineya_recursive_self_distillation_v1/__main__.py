"""Offline planning/comparison and explicitly supervised native commands."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .contracts import ARMS, Plan, arm_spec, digest, file_hash, read_json, require
from .data import demo_catalog, load_catalog, require_training_roles, write_catalog
from .evidence import write_new


def read_run(path):
    root = Path(path).resolve()
    require(not (root / "INFLIGHT.json").exists() and not (root / "RUN.lock").exists(),
            "active/quarantined run cannot be compared")
    run = read_json(root / "RUN.json")
    pointer = read_json(root / "LATEST.json")
    checkpoint = (root / pointer["directory"]).resolve()
    require(checkpoint.parent == root and checkpoint.name.startswith("checkpoint-"), "checkpoint path")
    require(file_hash(checkpoint / "MANIFEST.json") == pointer["manifest_sha256"], "manifest hash mismatch")
    manifest = read_json(checkpoint / "MANIFEST.json")
    require(manifest["binding"] == run["binding"] and set(manifest["files"]) == {"STATE.json", "ADAPTER_STATE.pt"},
            "checkpoint/run inventory mismatch")
    for name, value in manifest["files"].items():
        require(file_hash(checkpoint / name) == value, "checkpoint tamper")
    return run["binding"], read_json(checkpoint / "STATE.json")


def compare(left, right, step, plan):
    from .metrics import paired_effect, summary
    a, x = read_run(left)
    b, y = read_run(right)
    require(a["arm"] != b["arm"] and a["plan"] == plan.fingerprint, "distinct arms and matching plan required")
    require({k: v for k, v in a.items() if k != "arm"} == {k: v for k, v in b.items() if k != "arm"},
            "model/data/runtime/source/seed/budget mismatch; not a matched-arm comparison")
    require(type(step) is int and step > 0, "explicit prespecified evaluated step required")
    def panel(state):
        found = [e for e in state["evaluations"] if e["step"] == step]
        require(len(found) == 1, "requested evaluation checkpoint missing")
        return found[0]["rows"]
    before, after = panel(x), panel(y)
    cal = lambda rows: [r for r in rows if r["split"] == "calibration"]
    effect = paired_effect(cal(before), cal(after), samples=plan.bootstrap_samples, seed=plan.seed)
    low, high = effect["ci95"]
    interpretation = "inconclusive_on_adaptive_panel"
    if low > plan.minimum_effect:
        interpretation = "practical_gain_candidate_requires_untouched_confirmation"
    elif high < plan.minimum_effect:
        interpretation = "practical_gain_not_supported_on_this_panel_and_budget"
    return {"left_arm": a["arm"], "right_arm": b["arm"], "step": step, "evidence": a["evidence"],
            "calibration": effect, "left_summary": summary(cal(before)), "right_summary": summary(cal(after)),
            "retention": {source: paired_effect(
                [r for r in before if r["split"] == "retention_eval" and r["source"] == source],
                [r for r in after if r["split"] == "retention_eval" and r["source"] == source],
                samples=plan.bootstrap_samples, seed=plan.seed)
                for source in sorted({r["source"] for r in before if r["split"] == "retention_eval"})},
            "minimum_effect": plan.minimum_effect, "interpretation": interpretation,
            "native_efficacy_claim": False, "compute_matched": False,
            "note": "Optimizer-step matched mechanism comparison; generated/processed work can differ. "
                    "Adaptive calibration intervals are exploratory, not a universal falsification or final test."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fixture", help="Create a small owned mechanical fixture, never native evidence")
    p.add_argument("--output", required=True)
    p.add_argument("--per-role", type=int, default=4)
    p = sub.add_parser("plan", help="Validate inputs and emit arm matrix; no model imports")
    p.add_argument("--plan", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--exclusions")
    p.add_argument("--output", required=True)
    p = sub.add_parser("compare", help="Paired exploratory comparison at an explicit evaluated checkpoint")
    p.add_argument("--plan", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--output", required=True)
    p = sub.add_parser("bind", help="Hash explicit local input files; no qualification is granted")
    for name in ("e0-config", "e0-result", "catalog", "exclusions", "output"):
        p.add_argument("--" + name, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--parent")
    group.add_argument("--parent-stage", action="store_true")
    p = sub.add_parser("review-parent", help="Record an explicit operator review; not automated quality certification")
    p.add_argument("--candidate", required=True)
    p.add_argument("--reviewed-by", required=True)
    p.add_argument("--confirm-reviewed-parent", action="store_true")
    p.add_argument("--output", required=True)
    p = sub.add_parser("native", help="Exact Windows Titan; independent watchdog and fresh authorization")
    p.add_argument("--mode", choices=("parent", "probe", "qualify", "train"), required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--binding", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--arm", choices=ARMS, default="evolving")
    p.add_argument("--qualification")
    p.add_argument("--max-updates", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-model-load", action="store_true")
    p.add_argument("--allow-training", action="store_true")
    p = sub.add_parser("_worker", help=argparse.SUPPRESS)
    p.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fixture":
            write_catalog(args.output, demo_catalog(args.per_role))
            result = {"status": "FIXTURE_ONLY", "path": args.output}
        elif args.command == "plan":
            plan = Plan.load(args.plan)
            tasks, audit = load_catalog(args.catalog, args.exclusions)
            require_training_roles(tasks)
            require(len({t.parent for t in tasks if t.split == "train"}) >= plan.batch_tasks,
                    "too few acquisition parents")
            require(len({t.parent for t in tasks if t.split == "retention_train"}) >= plan.replay_tasks,
                    "too few replay parents")
            result = {"status": "OFFLINE_VALIDATED_NOT_NATIVE_QUALIFIED", "plan": asdict(plan),
                      "plan_digest": plan.fingerprint, "audit": audit,
                      "arms": [arm_spec(a) for a in ARMS], "pretrained_loads": 0, "optimizer_steps": 0}
            write_new(args.output, result)
        elif args.command == "compare":
            result = compare(args.left, args.right, args.step, Plan.load(args.plan))
            write_new(args.output, result)
        elif args.command == "bind":
            result = {"schema": 1}
            for name in ("e0_config", "e0_result", "catalog", "exclusions", "parent"):
                value = getattr(args, name)
                path = Path(value).resolve() if value else None
                result[name] = {"path": str(path), "sha256": file_hash(path)} if path else None
            write_new(args.output, result)
        elif args.command == "review-parent":
            require(args.confirm_reviewed_parent and args.reviewed_by.strip(), "explicit completed parent review required")
            result = read_json(args.candidate)
            require(result.get("native_sft_completed") is True and result.get("optimizer_steps", 0) > 0,
                    "not a completed native SFT candidate")
            from .native import checked_file
            for name in ("adapter", "source_run_receipt"):
                path = checked_file(result[name], Path(args.candidate).resolve().parent)
                result[name] = {"path": str(path), "sha256": file_hash(path)}
            result["parent_evaluation_reviewed"] = True
            result["reviewed_by"] = args.reviewed_by.strip()
            write_new(args.output, result)
        elif args.command == "native":
            require(args.max_updates > 0, "positive bounded update count required")
            require(args.mode != "train" or bool(args.qualification), "train requires --qualification")
            from .native import launch
            result = launch(args)
        else:
            from .native import worker
            result = worker(args.request)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc), "exception": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
