"""Serial generation only; gold loading and scoring live in analyze.py.

The arm structure, call budget and per-call token caps are carried unchanged from
MRS-1 so that MRS-2 differs from it in exactly one registered respect: the output
contract. The bank, decoding policy, resource guards and native backend are reused.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import sys
import tempfile
from pathlib import Path

from .core import (RECORDS, CLASSIFICATION, digest, file_hash, load_frozen, model_assets,
                   mrs1_binding, public_bank, read_json, source_identity, validate_receipt,
                   write_new)
from experiments.multi_reasoning_scout_v1 import budget
from . import prompts
from .parsing import parse, majority

ROLE_CAP = 64      # MRS-1 value, carried unchanged
A1_CAP = 320       # MRS-1 value, carried unchanged
# Gated in smoke: every role whose output an arm actually consumes.
SMOKE_GATED = ("R0", "R1", "R2", "R3", "D1", "D2", "D3", "A2_selector", "A3_selector")
SMOKE_THRESHOLD = .75


def run_item(item, backend, out, binding):
    outputs = {}

    def call(key, role, context=(), cap=ROLE_CAP):
        prompt = prompts.messages(item, role, context)
        result = backend.generate(prompt, cap)
        record = {**result, **parse(result["raw_text"], role, result["terminated"]),
                  "case_id": item.id, "scenario_id": item.scenario_id,
                  "role": role, "key": key, "max_new_tokens": cap,
                  "prompt": prompt, "prompt_sha256": digest(prompt), "binding": binding}
        write_new(out / f"{item.id}.{key}.json", record)
        outputs[key] = record
        return record

    r0 = call("R0", "R0")
    call("A1", "R0", cap=A1_CAP)
    direct = [r0] + [call(k, k) for k in ("D1", "D2", "D3")]
    call("A2_selector", "S1", direct)
    roles = [r0] + [call(k, k) for k in ("R1", "R2", "R3")]
    call("A3_selector", "S2", roles)
    call("S1_diagnostic", "S1", roles)
    item_record = {"item": asdict(item), "binding": binding,
                   "outputs": outputs, "arms": {"A0": "R0", "A1": "A1", "A2": "A2_selector",
                                                "A3": "A3_selector", "S1": "S1_diagnostic"},
                   "S0": majority(roles), "A2_S0": majority(direct),
                   "logical_calls": {"A0": ["R0"], "A1": ["A1"],
                                     "A2": ["R0", "D1", "D2", "D3", "A2_selector"],
                                     "A3": ["R0", "R1", "R2", "R3", "A3_selector"],
                                     "S1": ["R0", "R1", "R2", "R3", "S1_diagnostic"]}}
    write_new(out / f"{item.id}.item.json", item_record)
    return item_record


def smoke_pass(rows):
    detail = {}
    for role in SMOKE_GATED:
        valid = sum(not r["outputs"][role]["format_failure"] and r["outputs"][role]["terminated"]
                    for r in rows)
        detail[role] = {"valid": valid, "n": len(rows), "pass": valid >= SMOKE_THRESHOLD * len(rows)}
    return all(d["pass"] for d in detail.values()), detail


def candidate(model_id, future=None):
    """Registered candidates come from the MRS-1 manifest; a future trained candidate
    must arrive through a hash-bound registration with passing P0/P1/P2 receipts."""
    from experiments.multi_reasoning_scout_v1.runner import candidate as mrs1_candidate
    if future is None:
        return mrs1_candidate(model_id)
    registration = read_json(future)
    if registration.get("frozen_protocol_sha256") != file_hash(RECORDS / "FREEZE.json"):
        raise ValueError("future candidate registered against a different experiment")
    return mrs1_candidate(model_id, future)


def worker(args):
    from experiments.model_quality_gpu_v1.supervisor import claim_worker
    from experiments.model_quality_gpu_v1.resources import lease
    out = Path(args.out).absolute()
    claim_worker(out)
    meter = None
    try:
        freeze = load_frozen()
        model = candidate(args.model, args.future_registration)
        observed = budget.snapshot()
        write_new(out / "preflight.json", {"observed": observed, "limits": budget.LIMITS,
                                           "estimate": estimate()})
        budget.check(observed, preload=True)
        # A blocked preload spends no model launch. A loaded attempt cannot retry silently.
        with lease(Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"):
            budget.check(budget.snapshot(), preload=True)
            write_new(RECORDS / "evidence" / f"MODEL_LAUNCH_{digest(model)[:20]}.json",
                      {"model": model["id"], "out": str(out), "model_identity": digest(model),
                       "retry_allowed": False})
            binding = {"experiment": "MRS-2", "classification": CLASSIFICATION,
                       "freeze_sha256": file_hash(RECORDS / "FREEZE.json"),
                       "prompts": prompts.manifest()["sha256"],
                       "mrs1": mrs1_binding(), "source": source_identity(), "model": model,
                       "model_identity": model_assets(model),
                       "runtime": budget.runtime_versions()}
            if args.future_registration:
                binding["future_registration_sha256"] = file_hash(args.future_registration)
                binding["future_registration"] = read_json(args.future_registration)
            write_new(out / "binding.json", binding)
            meter = budget.Meter(out)
            from experiments.multi_reasoning_scout_v1.native import Native
            backend = Native(model, meter)
            items = public_bank()
            qualification = {**backend.qualify(items), "stage": "NATIVE_QUALIFICATION",
                             "binding": binding}
            write_new(out / "qualification.json", qualification)
            validate_receipt(qualification, binding, "NATIVE_QUALIFICATION")
            raw = out / "raw"
            raw.mkdir()
            smoke = [run_item(i, backend, raw, binding) for i in items if i.split == "SMOKE"]
            good, detail = smoke_pass(smoke)
            write_new(out / "smoke.json", {"stage": "SCHEMA_SMOKE", "status": "PASS" if good else "FAIL",
                                           "binding": binding, "items": len(smoke),
                                           "per_role": detail, "threshold": SMOKE_THRESHOLD,
                                           "correctness_inspected": False})
            if not good:
                raise RuntimeError("schema smoke failed; no prompt repair or CHECK evaluation")
            # All selectors were frozen prospectively; CAL is descriptive, never used for tuning.
            for partition in ("CAL", "CHECK"):
                for item in items:
                    if item.split == partition:
                        run_item(item, backend, raw, binding)
            if backend.state_digest() != backend.state_before:
                raise ValueError("checkpoint state changed during inference")
            model_assets(model)
            load_frozen()
            receipt = {"status": "PASS", "stage": "FROZEN_SCOUT_GENERATION", "binding": binding,
                       "items": len(items), "state_unchanged": True,
                       "resources": meter.report(), "training_steps": 0, "efficacy": False,
                       "raw_files": {p.name: file_hash(p) for p in sorted(raw.iterdir())}}
            write_new(out / "generation.json", receipt)
        return 0
    except Exception as exc:
        write_new(out / "failure.json", {"status": "BLOCKED_OR_FAILED", "error": str(exc),
                                         "type": type(exc).__name__, "efficacy": False,
                                         "resources": meter.report() if meter else None,
                                         "model_imported": "torch" in sys.modules})
        return 2


def estimate():
    return {"experiment": "MRS-2", "output_contract": "C1_BARE", "models": 2, "model_loads": 2,
            "cases_per_model": 120, "scored_CHECK_cases": 96, "SMOKE": 8, "CAL": 16,
            "unique_calls_per_item": 11, "primary_A3_calls": 5, "A2_calls": 5,
            "role_output_cap": ROLE_CAP, "A1_token_allowance": A1_CAP,
            "unique_generation_token_allowance_per_item": 10 * ROLE_CAP + A1_CAP,
            "compute_matched": False,
            "accounting": "R0 shared across A0/A2/A3; S1 diagnostic adds one call outside the "
                          "five-call A3; logical per-arm and unique actual totals both reported",
            "A1_caveat": "Token-budget matching is vacuous under a single-field contract: the "
                         "model terminates after the answer regardless of allowance. A1 is run "
                         "to demonstrate this, not to claim compute matching.",
            "limits": budget.LIMITS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "_worker", "verify", "estimate"))
    p.add_argument("--model")
    p.add_argument("--future-registration", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.command == "estimate":
        print(estimate()); return 0
    if args.command == "verify":
        print({"freeze": digest(load_frozen()), "public_items": len(public_bank()),
               "prompts": prompts.manifest()["sha256"]}); return 0
    if not args.model or args.out is None:
        p.error("run requires --model and --out")
    if args.command == "_worker":
        return worker(args)
    load_frozen()
    candidate(args.model, args.future_registration)
    from experiments.model_quality_gpu_v1.supervisor import bounded_child
    command = [sys.executable, "-m", "experiments.multi_reasoning_scout_v2.runner", "_worker",
               "--model", args.model, "--out", str(args.out.absolute())]
    if args.future_registration:
        command += ["--future-registration", str(args.future_registration.absolute())]
    return bounded_child(command, args.out, budget.LIMITS["max_seconds_per_model"])


if __name__ == "__main__":
    raise SystemExit(main())
