"""Offline-by-default CLI. Native work requires three explicit host permissions."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .contracts import ContractError, Plan, digest, matrix, read_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--plan", required=True)
    q.add_argument("--stage", choices=["screen", "full"], default="full")
    q = sub.add_parser("audit")
    q.add_argument("--catalog", required=True)
    q.add_argument("--exclusions")
    q = sub.add_parser("inventory")
    q.add_argument("--model-dir", required=True)
    q = sub.add_parser("register")
    q.add_argument("--plan", required=True)
    q.add_argument("--binding", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("preflight")
    q.add_argument("--registration", required=True)
    q = sub.add_parser("demo")
    q.add_argument("--output", required=True)
    q.add_argument("--stage", choices=["screen", "full"], default="full")
    q = sub.add_parser("select")
    q.add_argument("--registration", required=True)
    q.add_argument("--results", required=True, nargs="+")
    q.add_argument("--stage", choices=["screen", "full"], default="full")
    q.add_argument("--output", required=True)
    for name in ("run", "confirm"):
        q = sub.add_parser(name)
        q.add_argument("--registration", required=True)
        q.add_argument("--output", required=True)
        q.add_argument("--allow-model-load", action="store_true")
        q.add_argument("--ack-host-available", action="store_true")
        if name == "run":
            q.add_argument("--arm", required=True)
            q.add_argument("--allow-training", action="store_true")
        else:
            q.add_argument("--selection", required=True)
            q.add_argument("--confirmation", required=True)
            q.add_argument("--allow-confirmation", action="store_true")
    q = sub.add_parser("export-peft")
    q.add_argument("--registration", required=True)
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--output", required=True)
    a = p.parse_args(argv)
    try:
        result = execute(a)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ContractError, OSError, KeyError, TypeError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "type": type(exc).__name__, "reason": str(exc)}))
        return 2


def execute(a):
    if a.command == "plan":
        plan = Plan.load(a.plan)
        return {"experiment": "RDD-001", "stage": a.stage, "plan": asdict(plan),
                "arms": [asdict(x) for x in matrix(plan, a.stage)], "native_execution": False}
    if a.command == "audit":
        from .data import load_catalog
        _, provenance, audit = load_catalog(a.catalog, read_json(a.exclusions) if a.exclusions else None)
        return {"provenance": provenance, **audit}
    if a.command == "inventory":
        from .registry import inventory
        return inventory(a.model_dir)
    if a.command == "register":
        from .registry import register
        return register(a.plan, a.binding, a.output)
    if a.command == "preflight":
        from .registry import check_registration
        from .host import runtime_check
        r, _, _ = check_registration(a.registration)
        runtime_check()
        return {"status": "IDENTITY_AND_RUNTIME_CHECKED_NO_MODEL_LOADED", "identity": digest(r),
                "native_qualified": False, "resource_admission_checked": False}
    if a.command == "demo":
        return demo(Path(a.output), a.stage)
    if a.command == "select":
        from .registry import check_registration
        from .selection import select
        r, plan, _ = check_registration(a.registration)
        return select(plan, [Path(x) for x in a.results], Path(a.output), stage=a.stage, identity=digest(r))
    if a.command in {"run", "confirm"}:
        from .host import launch
        launch(a.registration, a.output, arm=getattr(a, "arm", None), selection=getattr(a, "selection", None),
               confirmation=getattr(a, "confirmation", None), allow_model_load=a.allow_model_load,
               allow_training=getattr(a, "allow_training", False), ack_host_available=a.ack_host_available,
               allow_confirmation=getattr(a, "allow_confirmation", False))
        return {"status": "SUPERVISED_COMMAND_COMPLETED", "output": a.output, "deployment_approved": False}
    from .integration import export_peft
    export_peft(a.checkpoint, a.registration, a.output)
    return {"status": "EXPORTED_FORMAT_ONLY_RUNTIME_PARITY_UNQUALIFIED", "output": a.output}


def demo(output: Path, stage: str):
    import torch
    from .contracts import sealed, write_new
    from .data import audit
    from .engine import run_arm
    from .fixtures import fixture_backend, fixture_catalog, fixture_plan, fixture_tasks
    from .selection import select
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output.mkdir(parents=True, exist_ok=False)
        plan, tasks = fixture_plan(), fixture_tasks()
        audit(tasks)
        identity = digest({"fixture": "RDD-001-v1", "plan": plan.fingerprint, "catalog": fixture_catalog(tasks)})
        write_new(output / "plan.json", asdict(plan))
        write_new(output / "catalog.json", fixture_catalog(tasks))
        paths = []
        for arm in matrix(plan, stage):
            backend = fixture_backend(plan, arm.seed)
            run_arm(backend, tasks, arm, output / arm.name, identity=identity,
                    bank_path=output / "banks" / f"seed-{arm.seed}.json")
            paths.append(output / arm.name / "RESULT.json")
        chosen = select(plan, paths, output / "selection", stage=stage, identity=identity)
        receipt = {"status": "MECHANICS_ONLY", "evidence_kind": "mechanical_fixture", "arms": len(paths),
                   "optimizer_steps_per_arm": plan.steps, "selection_status": chosen["status"],
                   "pretrained_models_loaded": False, "rineya_performance_claim": False}
        write_new(output / "DEMO_RESULT.json", sealed(receipt))
        return receipt
    finally:
        torch.set_num_threads(threads)
