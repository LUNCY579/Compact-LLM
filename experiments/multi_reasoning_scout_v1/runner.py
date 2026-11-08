"""Serial generation only; scoring and gold loading live in analysis.py."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import sys
import tempfile
from pathlib import Path

from .core import (RECORDS, digest, file_hash, load_frozen, model_assets, public_bank,
                   read_json, source_identity, validate_receipt, write_new)
from . import budget, prompts
from .parsing import parse, majority


def run_item(item, backend, out, binding):
    outputs = {}

    def call(key, role, context=(), cap=64):
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
    a1 = call("A1", "R0", cap=320)
    direct = [r0] + [call(k, k) for k in ("D1", "D2", "D3")]
    a2 = call("A2_selector", "S1", direct)
    roles = [r0] + [call(k, k) for k in ("R1", "R2", "R3")]
    a3 = call("A3_selector", "S2", roles)
    s1 = call("S1_diagnostic", "S1", roles)
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


def run_m2_item(item, backend):
    """Five-call optional sequential variant; disabled in the frozen cheap run.

    Kept separately to make information flow testable. Enabling requires a new
    prospective budget/experiment, never a silent extension of MRS-1.
    """
    records = []
    for role in ("R0", "M2_check", "M2_revise", "M2_constraint", "S2"):
        context = () if role == "R0" else records
        result = backend.generate(prompts.messages(item, role, context), 64)
        records.append({**result, **parse(result["raw_text"], role, result["terminated"]),
                        "case_id": item.id, "role": role})
    return records


def smoke_pass(rows):
    for role in ("R0", "R1", "R2", "R3", "A3_selector", "A2_selector"):
        valid = sum(not r["outputs"][role]["format_failure"] and r["outputs"][role]["terminated"] for r in rows)
        if valid < .75 * len(rows):
            return False
    return True


def candidate(model_id, future=None):
    registry = read_json(RECORDS / "MODEL_MANIFEST.json")
    if future is None:
        return next(m for m in registry["selected"] if m["id"] == model_id)
    registration = read_json(future)
    if registration["frozen_protocol_sha256"] != file_hash(RECORDS / "FREEZE.json"):
        raise ValueError("future candidate registered against different experiment")
    model = registration["model"]
    old = next(m for m in registry["selected"] if m["id"] == "d1-seed17-bf16-export")
    if model["id"] != model_id or model["id"] in {m["id"] for m in registry["selected"]}:
        raise ValueError("future model identity collision")
    for key in ("policy", "runtime_dtype", "storage_dtype", "attention_backend", "decoding"):
        if model[key] != old[key]:
            raise ValueError("future policy incompatible; new experiment required")
    if (model["assets"]["model.safetensors"] == old["assets"]["model.safetensors"]
            or Path(model["path"]).resolve() == Path(old["path"]).resolve()):
        raise ValueError("future checkpoint must have distinct weight bytes and path")
    if {k: v for k, v in model["assets"].items() if k != "model.safetensors"} != {
            k: v for k, v in old["assets"].items() if k != "model.safetensors"}:
        raise ValueError("future config/tokenizer differs; new experiment required")
    provenance = registration.get("training_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("structured training provenance required")
    receipts = {}
    for key, stage in (("dataset_acceptance", "DATASET_ACCEPTANCE"), ("training_contract", "TRAINING_CONTRACT"),
                       ("training_execution", "TRAINING_EXECUTION")):
        receipt = provenance.get(key, {})
        if receipt.get("status") != "PASS" or not isinstance(receipt.get("sha256"), str) or len(receipt["sha256"]) != 64 or not receipt.get("path"):
            raise ValueError("passing, hash-bound P0/P1/P2 receipt required: " + key)
        if file_hash(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError("changed future training receipt: " + key)
        receipts[key] = read_json(Path(receipt["path"]))
        if receipts[key].get("status") != "PASS" or receipts[key].get("stage") != stage:
            raise ValueError("future training receipt failed or wrong stage: " + key)
    if not provenance.get("source_checkpoint_sha256") or not provenance.get("dataset_manifest_sha256"):
        raise ValueError("exact training source lineage required")
    if any(model[k] != provenance.get(k) for k in ("source_repository", "source_branch", "source_sha")):
        raise ValueError("model and training source identities disagree")
    if any(r.get("dataset_manifest_sha256") != provenance["dataset_manifest_sha256"] for r in receipts.values()):
        raise ValueError("training receipts refer to different dataset manifests")
    for key in ("training_contract", "training_execution"):
        if receipts[key].get("source_checkpoint_sha256") != provenance["source_checkpoint_sha256"]:
            raise ValueError("training source checkpoint differs from receipt")
    if receipts["training_execution"].get("checkpoint_sha256") != model["assets"]["model.safetensors"]:
        raise ValueError("future checkpoint differs from training execution receipt")
    return model


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
        write_new(out / "preflight.json", {"observed": observed, "limits": budget.LIMITS, "estimate": budget.estimate()})
        budget.check(observed, preload=True)
        # A blocked preload spends no model launch. A loaded attempt cannot retry silently.
        with lease(Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"):
            budget.check(budget.snapshot(), preload=True)
            write_new(RECORDS / "evidence" / f"MODEL_LAUNCH_{digest(model)[:20]}.json",
                      {"model": model["id"], "out": str(out), "model_identity": digest(model), "retry_allowed": False})
            binding = {"freeze_sha256": file_hash(RECORDS / "FREEZE.json"), "source": source_identity(),
                       "model": model, "model_identity": model_assets(model), "runtime": budget.runtime_versions()}
            if args.future_registration:
                binding["future_registration_sha256"] = file_hash(args.future_registration)
                binding["future_registration"] = read_json(args.future_registration)
            write_new(out / "binding.json", binding)
            meter = budget.Meter(out)
            from .native import Native
            backend = Native(model, meter)
            items = public_bank()
            qualification = {**backend.qualify(items), "stage": "NATIVE_QUALIFICATION", "binding": binding}
            write_new(out / "qualification.json", qualification)
            validate_receipt(qualification, binding, "NATIVE_QUALIFICATION")
            raw = out / "raw"
            raw.mkdir()
            smoke = [run_item(i, backend, raw, binding) for i in items if i.split == "SMOKE"]
            good = smoke_pass(smoke)
            write_new(out / "smoke.json", {"stage": "SCHEMA_SMOKE", "status": "PASS" if good else "FAIL",
                                            "binding": binding, "items": len(smoke), "correctness_inspected": False})
            if not good:
                raise RuntimeError("schema smoke failed; no prompt repair or CHECK evaluation")
            # All selectors were frozen prospectively; CAL is descriptive, never used for tuning here.
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
                       "resources": meter.report(), "training_steps": 0,
                       "raw_files": {p.name: file_hash(p) for p in sorted(raw.iterdir())}}
            write_new(out / "generation.json", receipt)
        return 0
    except Exception as exc:
        write_new(out / "failure.json", {"status": "BLOCKED_OR_FAILED", "error": str(exc),
                                         "type": type(exc).__name__, "efficacy": False,
                                         "resources": meter.report() if meter else None,
                                         "model_imported": "torch" in sys.modules})
        return 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "_worker", "verify", "estimate"))
    p.add_argument("--model", choices=None)
    p.add_argument("--future-registration", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.command == "estimate":
        print(budget.estimate()); return 0
    if args.command == "verify":
        print({"freeze": digest(load_frozen()), "public_items": len(public_bank())}); return 0
    if not args.model or args.out is None:
        p.error("run requires --model and --out")
    if args.command == "_worker":
        return worker(args)
    load_frozen()
    candidate(args.model, args.future_registration)
    from experiments.model_quality_gpu_v1.supervisor import bounded_child
    command = [sys.executable, "-m", "experiments.multi_reasoning_scout_v1.runner", "_worker",
               "--model", args.model, "--out", str(args.out.absolute())]
    if args.future_registration:
        command += ["--future-registration", str(args.future_registration.absolute())]
    return bounded_child(command, args.out, budget.LIMITS["max_seconds_per_model"])


if __name__ == "__main__":
    raise SystemExit(main())
