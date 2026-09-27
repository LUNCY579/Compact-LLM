"""Offline planning, an executable demo, and explicitly authorized native entrypoints."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .backends import DemoBackend, local_hf_backend
from .curriculum import unlocked
from .data import Catalog, import_mimo, load_catalog, verified_demonstrations, write_catalog
from .engine import Engine, validate_enrollment
from .evidence import Store, atomic_json, check_resources
from .schema import Gates, Limits, Plan, Stage, Task, require


def demo_inputs() -> tuple[Plan, Catalog]:
    tasks = []
    for split, count, offset in (("train", 400, 0), ("calibration", 32, 1000),
                                  ("retention_train", 32, 2000), ("retention_eval", 32, 3000)):
        source = "arithmetic" if split in {"train", "calibration"} else "preservation"
        for i in range(count):
            a, b = offset + i, i % 7
            tasks.append(Task(f"{split}-{i}", source, "fixture-v1", f"{split}:{i}", split,
                              f"Compute {a} + {b}. Return only the integer.", str(a + b)))
    stages = tuple(Stage(name, share, {"arithmetic": 1.0},
                         {"unknown": .25, "frontier": .6, "easy": .1, "hard": .05},
                         max_new_tokens=32, max_updates=16, eval_every=2)
                   for name, share in (("foundation", .10), ("expansion", .25), ("consolidation", .5)))
    plan = Plan("synthetic-controller-demo", stages, Gates(min_eval_groups=16, min_train_groups=4,
                  min_observations=8, bootstrap_samples=200, max_retention_drop=.05),
                Limits(generated_tokens=2_000_000, processed_tokens=10_000_000,
                       max_prompt_tokens=128, max_context_tokens=256), qualification_tasks=16)
    return plan, Catalog(tasks)


def read_rows(path: Path):
    require(path.stat().st_size <= 256 * 2**20, "JSONL export exceeds 256 MiB; prepare a bounded shard")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="PRLC-001 progressive RL curriculum")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "qualify", "train", "resume", "bootstrap"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--catalog", type=Path, required=True)
        if name not in {"validate", "plan"}:
            sub.add_argument("--run-dir", type=Path, required=True)
            sub.add_argument("--model-dir", type=Path, required=True)
            sub.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
            sub.add_argument("--allow-model-load", action="store_true")
            sub.add_argument("--allow-training", action="store_true")
            sub.add_argument("--max-actions", type=int, default=None)
        if name == "bootstrap":
            sub.add_argument("--demonstrations", type=Path, required=True)
    sub = commands.add_parser("demo")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--max-actions", type=int, default=None)
    sub.add_argument("--resume", action="store_true")
    sub = commands.add_parser("report")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub = commands.add_parser("import-mimo")
    for flag in ("input", "output"):
        sub.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("revision", "domain", "split", "group-field"):
        sub.add_argument(f"--{flag}", required=True)
    return result


def execute(args) -> dict:
    if args.command == "import-mimo":
        catalog = import_mimo(read_rows(args.input), revision=args.revision, split=args.split,
                              domain=args.domain, group_field=args.group_field)
        write_catalog(catalog, args.output)
        return {"imported": len(catalog.tasks), "catalog": catalog.fingerprint,
                "execution": "BLOCKED_UNQUALIFIED_ENVIRONMENT"}
    if args.command == "report":
        # Read-only inspection verifies hashes without loading torch or checkpoints.
        pointer = json.loads((args.run_dir / "LATEST.json").read_text())
        name = pointer["name"]
        require(name.startswith("checkpoint-") and name[11:].isdigit(), "invalid checkpoint name")
        identity = json.loads((args.run_dir / name / "receipt.json").read_text())["identity"]
        _, state = Store(args.run_dir, identity).inspect()
        return {"identity": identity, "state": state}
    if args.command == "demo":
        plan, catalog = demo_inputs()
        engine = Engine(plan, catalog, DemoBackend(plan), args.run_dir, allow_training=True)
        with engine.store.lock():
            if args.resume:
                engine.resume()
            else:
                require(not (args.run_dir / "LATEST.json").exists(), "existing run: use --resume")
                atomic_json(args.run_dir / "demo_plan.json", asdict(plan))
            report = engine.run(args.max_actions)
            atomic_json(args.run_dir / "report.json", report)
        return report
    plan = Plan.from_dict(json.loads(args.config.read_text()))
    catalog = load_catalog(args.catalog)
    if args.command in {"validate", "plan"}:
        blockers = []
        try:
            validate_enrollment(plan, catalog, frozenset({"text"}))
        except ValueError as error:
            blockers.append(str(error))
        compatible = catalog.subset("train", frozenset({"text"}))
        return {"status": "VALIDATED_OFFLINE", "plan": plan.fingerprint, "catalog": catalog.fingerprint,
                "census": catalog.census(frozenset({"text"})),
                "stages": [{"name": stage.name, "fraction_of_compatible_parent_groups": stage.fraction,
                            "unlocked_rows": len(unlocked(compatible, stage, plan.seed)),
                            "group_size": stage.group_size, "max_new_tokens": stage.max_new_tokens}
                           for stage in plan.stages],
                "model_loaded": False, "native_qualified": False, "enrollment_blockers": blockers}
    require(args.allow_model_load, "explicit --allow-model-load required")
    if args.command in {"train", "bootstrap"}:
        require(args.allow_training, "explicit --allow-training required")
    if args.command == "qualify":
        require(not args.allow_training, "qualification is inference-only; remove --allow-training")
    validate_enrollment(plan, catalog, frozenset({"text"}))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Admission lock precedes resource checks AND model allocation.
    with Store(args.run_dir, {}).lock():
        exists = (args.run_dir / "LATEST.json").is_file()
        require(not exists if args.command == "qualify" else exists,
                "qualification output already exists" if args.command == "qualify" else "run qualification before training")
        snapshots = check_resources(plan.limits, args.run_dir, cuda=args.device == "cuda", preflight=True)
        backend = local_hf_backend(args.model_dir, plan, device=args.device, allow_model_load=args.allow_model_load)
        engine = Engine(plan, catalog, backend, args.run_dir, allow_training=args.allow_training)
        if args.command == "qualify":
            engine.event("PREFLIGHT", resources=snapshots)
            report = engine.qualify()
        else:
            engine.resume()
            require(engine.identity["backend"]["kind"] == "native_hf", "synthetic evidence cannot authorize native training")
            if args.command == "bootstrap":
                from .evidence import file_hash
                demonstrations = verified_demonstrations(catalog, read_rows(args.demonstrations))
                engine.event("BOOTSTRAP_SOURCE", sha256=file_hash(args.demonstrations),
                             accepted_rows=len(demonstrations))
                report = engine.bootstrap(demonstrations)
            elif args.command == "resume" and engine.state["status"] in {"NEW", "REQUALIFY"}:
                report = engine.qualify()
            else:
                require(args.allow_training, "resuming optimization requires --allow-training")
                require(engine.state["status"] in {"READY", "TRAINING"}, "qualification did not authorize this state")
                report = engine.run(args.max_actions)
        atomic_json(args.run_dir / "report.json", report)
    return report


def main(argv=None) -> int:
    try:
        args = parser().parse_args(argv)
        if hasattr(args, "max_actions") and args.max_actions is not None:
            require(args.max_actions > 0, "max-actions must be positive")
        result = execute(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error),
                          "note": "No success inferred. Resume only from the last intact transaction."}), file=sys.stderr)
        return 2
