"""MRS-FP-1: bounded, inference-only output-contract probe.

Why this exists: MRS-1 passed native qualification on both registered candidates and
then failed its frozen SCHEMA_SMOKE gate at 0/8 on every role, because the family
emits a bare in-vocabulary answer instead of the required JSON object. MRS-1 forbids
prompt repair, so its contract cannot be amended. This probe measures, on the
permanently-open SMOKE items only, which output contract the family can actually
satisfy, so that a successor instrument can be frozen on measured ground.

Registered before execution (see PROBE_PREREG.md):
  * Population: the 8 permanently-open MRS-1 SMOKE items. CAL and CHECK are never read.
  * Primary metric: strict format compliance = parsed AND terminated.
  * Selection rule: FORMAT COMPLIANCE ONLY, on both models. Answer correctness is
    recorded for description and is forbidden as a contract-selection input.
  * Inference only. Zero optimizer steps. Model state digest must be unchanged.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

from experiments.multi_reasoning_scout_v1 import core as mrs1
from experiments.multi_reasoning_scout_v1.core import (
    RECORDS, PublicItem, decode, digest, file_hash, model_assets, read_json, verify_files, write_new)
from experiments.multi_reasoning_scout_v1 import budget
from . import contracts

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
PROBE_RECORDS = ROOT / "research/mrs_format_probe_v1"
BRANCH = "research/mrs-format-probe-v1"
CLASSIFICATION = "OPEN / DEVELOPMENT / SELECTION-EXPOSED FOREVER"
OUTPUT_CAP = 64
GATE = 0.75           # the MRS-1 smoke-gate threshold, carried deliberately
GATED_ROLES = ("R0", "R2", "R3")


def source_identity():
    import subprocess

    def git(*a):
        return subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()

    if git("branch", "--show-current") != BRANCH:
        raise ValueError("wrong isolated branch")
    own = sorted(PACKAGE.glob("*.py"))
    return {"repository": "LUNCY579/ChatBubble", "branch": BRANCH, "sha": git("rev-parse", "HEAD"),
            "files": {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in own}}


def bind_mrs1():
    """Bind the MRS-1 frozen bank by hash and lineage, not by branch.

    MRS-1's own `load_frozen` additionally requires HEAD to be on the MRS-1 branch,
    which is correct for MRS-1 and wrong for a separate successor experiment. This
    reproduces every integrity property that matters here -- all frozen artifact
    bytes unchanged, MRS-1 executable bytes unchanged, and the frozen source commit
    an ancestor of this branch -- without pretending to be MRS-1.
    """
    import subprocess
    freeze = read_json(RECORDS / "FREEZE.json")
    verify_files(mrs1.ROOT, freeze["files"])
    if mrs1.source_files() != freeze["source_files"]:
        raise ValueError("stale frozen executable source")
    frozen_sha = freeze["source"]["sha"]
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", frozen_sha, "HEAD"]).returncode:
        raise ValueError("frozen MRS-1 source is not an ancestor of this probe branch")
    rows = [PublicItem(**decode(line)) for line
            in (RECORDS / "bank/public.jsonl").read_text(encoding="utf-8").splitlines()]
    if len({r.id for r in rows}) != len(rows) or len({r.scenario_id for r in rows}) != len(rows):
        raise ValueError("duplicate item/world")
    return freeze, rows


def generate(backend, messages, prefill, cap):
    """Greedy decode with a fixed prefilled assistant opening.

    Mirrors the MRS-1 native decoding policy exactly; the prefill bytes are identical
    for every item, role and model, so no item-specific information is injected.
    """
    torch = backend.torch
    backend.meter.reserve(cap)
    ids = backend.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    prefill_ids = backend.tokenizer.encode(prefill, add_special_tokens=False) if prefill else []
    ids = list(ids) + list(prefill_ids)
    if len(ids) > 2048:
        raise ValueError("prompt exceeds 2048-token input cap; no truncation")
    if len(ids) + cap > 4096:
        raise ValueError("context allowance exceeded")
    emitted, past, terminated = [], None, False
    current = torch.tensor([ids], device="cuda:0")
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        for _ in range(cap):
            backend.meter.guard()
            out = backend.model(input_ids=current, past_key_values=past, use_cache=True,
                                return_dict=True, logits_to_keep=1)
            past = out.past_key_values
            if not bool(torch.isfinite(out.logits).all()):
                raise FloatingPointError("nonfinite native logits")
            token = int(out.logits[0, -1].argmax())
            emitted.append(token)
            if token == 151645:
                terminated = True
                break
            current = torch.tensor([[token]], device="cuda:0")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    del out, past, current
    torch.cuda.empty_cache()
    backend.meter.guard(force=True)
    body = backend.tokenizer.decode(emitted[:-1] if terminated else emitted, skip_special_tokens=False)
    return {"completion_text": body, "raw_text": prefill + body, "prefill": prefill,
            "generated_ids": emitted, "terminated": terminated,
            "prompt_tokens": len(ids), "prefill_tokens": len(prefill_ids),
            "output_tokens": len(emitted), "total_tokens": len(ids) + len(emitted),
            "generation_calls": 1, "forward_calls": len(emitted), "wall_seconds": elapsed}


def worker(args):
    import tempfile
    from experiments.model_quality_gpu_v1.supervisor import claim_worker
    from experiments.model_quality_gpu_v1.resources import lease
    out = Path(args.out).absolute()
    claim_worker(out)
    meter = None
    try:
        freeze, bank = bind_mrs1()
        registry = read_json(RECORDS / "MODEL_MANIFEST.json")
        model = next(m for m in registry["selected"] if m["id"] == args.model)
        observed = budget.snapshot()
        write_new(out / "preflight.json", {"observed": observed, "limits": budget.LIMITS,
                                           "probe": {"output_cap": OUTPUT_CAP, "gate": GATE,
                                                     "contracts": list(contracts.CONTRACTS),
                                                     "roles": list(contracts.ROLES)}})
        budget.check(observed, preload=True)
        items = [i for i in bank if i.split == "SMOKE"]
        if not items or any(i.split != "SMOKE" for i in items):
            raise ValueError("probe population must be SMOKE only")
        with lease(Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"):
            budget.check(budget.snapshot(), preload=True)
            write_new(PROBE_RECORDS / "evidence" / ("PROBE_LAUNCH_" + args.model + ".json"),
                      {"model": args.model, "out": str(out), "retry_allowed": False})
            binding = {"probe": "MRS-FP-1", "classification": CLASSIFICATION,
                       "mrs1_freeze_sha256": file_hash(RECORDS / "FREEZE.json"),
                       "mrs1_frozen_source_sha": freeze["source"]["sha"],
                       "mrs1_bank_public_sha256": freeze["files"]["research/multi_reasoning_scout_v1/bank/public.jsonl"],
                       "source": source_identity(), "model": model,
                       "model_identity": model_assets(model),
                       "contracts_sha256": digest({k: {kk: vv for kk, vv in v.items() if kk != "parse"}
                                                   for k, v in contracts.CONTRACTS.items()}),
                       "roles_sha256": digest(contracts.ROLES),
                       "runtime": budget.runtime_versions(),
                       "population": [i.id for i in items], "split": "SMOKE"}
            write_new(out / "binding.json", binding)
            meter = budget.Meter(out)
            from experiments.multi_reasoning_scout_v1.native import Native
            backend = Native(model, meter)
            raw = out / "raw"
            raw.mkdir()
            calls = 0
            for cid in contracts.CONTRACTS:
                for role in GATED_ROLES:
                    for item in items:
                        msgs = contracts.messages(cid, role, item.problem)
                        result = generate(backend, msgs, contracts.CONTRACTS[cid]["prefill"], OUTPUT_CAP)
                        record = {**result,
                                  **contracts.parse(cid, result["raw_text"], role, result["terminated"]),
                                  "case_id": item.id, "scenario_id": item.scenario_id,
                                  "family": item.family, "split": item.split,
                                  "contract": cid, "role": role, "prompt": msgs,
                                  "prompt_sha256": digest(msgs)}
                        write_new(raw / (item.id + "." + cid + "." + role + ".json"), record)
                        calls += 1
            if backend.state_digest() != backend.state_before:
                raise ValueError("checkpoint state changed during inference")
            receipt = {"status": "PASS", "stage": "FORMAT_PROBE", "binding": binding,
                       "calls": calls, "items": len(items), "training_steps": 0,
                       "efficacy": False, "state_unchanged": True,
                       "resources": meter.report(),
                       "raw_files": {p.name: file_hash(p) for p in sorted(raw.iterdir())}}
            write_new(out / "probe.json", receipt)
        return 0
    except Exception as exc:
        write_new(out / "failure.json", {"status": "BLOCKED_OR_FAILED", "error": str(exc),
                                         "type": type(exc).__name__, "efficacy": False,
                                         "resources": meter.report() if meter else None,
                                         "model_imported": "torch" in sys.modules})
        return 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "_worker"))
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "_worker":
        return worker(args)
    from experiments.model_quality_gpu_v1.supervisor import bounded_child
    command = [sys.executable, "-m", "experiments.mrs_format_probe_v1.probe", "_worker",
               "--model", args.model, "--out", str(args.out.absolute())]
    return bounded_child(command, args.out, budget.LIMITS["max_seconds_per_model"])


if __name__ == "__main__":
    raise SystemExit(main())
