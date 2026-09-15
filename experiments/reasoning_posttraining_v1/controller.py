"""Supervised native controller: the only sanctioned path to the actual checkpoint.

Every model command runs in a watchdog-supervised foreground child that must claim
its launch before importing the native runtime. The parent reserves an all-attempt
budget entry first and settles it only from an observed terminal supervisor receipt,
so an unsettled launch keeps its whole reservation and blocks the next one.

This module authorizes nothing by itself. It refuses to run without an explicitly
APPROVED grant bound to the exact prepared data and frozen plan, a clean checkout of
the dedicated branch, the pinned runtime, and the unlowered MQ resource guards.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from experiments.model_quality_gpu_v1.config import Design
from experiments.model_quality_gpu_v1.data import group_updates
from experiments.model_quality_gpu_v1.integrity import (
    ROOT, code_identity, decode, digest, file_hash, read_json, regular, write_new)
from experiments.model_quality_gpu_v1.resources import GIB, PRELOAD_MINIMA, guard, lease, snapshot
from experiments.model_quality_gpu_v1.supervisor import bounded_child, claim_worker

from .budget import Journal, validate_grant
from .prepare import read_rows
from .sources import stable_hash
# Imported for its registration side effect: RRP-002-BASELINE must be in the registry
# before the foreground validates a grant, which happens before any phase dispatch.
# This module is torch-free, so importing it does not initialize the native runtime.
from . import instrument as _registers_rrp002_baseline  # noqa: F401

PACKAGE = Path(__file__).resolve().parent
BRANCH = "research/rineya-reasoning-posttraining-v1"
BASE = "58bf64cf93ff918ede4b1079b8a5716ee0e1e312"
PHASES = ("qualify", "train", "resume", "baseline")
# A measurement-only phase is a different registered experiment, not a wider RRP-001.
EXPERIMENT_FOR_PHASE = {"qualify": "RRP-001-MICROFIT", "train": "RRP-001-MICROFIT",
                        "resume": "RRP-001-MICROFIT", "baseline": "RRP-002-BASELINE"}
# A grant may name an amended successor for the same phase; the phase-to-experiment map
# is a default, and --experiment selects an amended lane explicitly.
# Which MEASURED resources.PRELOAD_MINIMA profile each phase belongs to. This selects
# among existing profiles; it never invents or loosens a threshold, and resources.py is
# not edited. `baseline` maps to the `qualify` profile because both load the fp32 model
# and run forwards with NO optimizer state, which is what that profile was measured on
# (3.89 GiB host peak, 2.22 GiB GPU). `resume` maps to `train` because it does allocate
# CpuAdamW's two fp32 moments. An unmapped phase falls back to the strictest profile.
RESOURCE_PROFILE = {"qualify": "qualify", "train": "train", "resume": "train",
                    "baseline": "qualify"}


# Disk headroom kept free on the volume regardless of the approved byte allowance.
DISK_FLOOR = 14 * 2 ** 30


def read_json_line(line: str):
    """One bank row, parsed with the canonical strict decoder."""
    return decode(line)


def package_identity() -> dict:
    """Semantic identity of this successor package, CRLF-normalized like MQ's."""
    files = {}
    for path in sorted(PACKAGE.rglob("*")):
        rel = path.relative_to(PACKAGE)
        if set(rel.parts) & {"__pycache__", ".pytest_cache", "outputs", "evidence"}:
            continue
        if path.is_file():
            files[rel.as_posix()] = hashlib.sha256(
                regular(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {"sha256": digest(files), "files": files}


def checkout() -> str:
    """Native execution requires this successor branch, clean, descended from BASE."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                       text=True, stderr=subprocess.STDOUT).strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise ValueError("full repository checkout required")
    if git("branch", "--show-current") != BRANCH:
        raise ValueError(f"native execution requires {BRANCH}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("native execution requires a clean checkout")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASE, "HEAD"],
                   check=True, capture_output=True)
    return git("rev-parse", "HEAD")


def design_for(plan: dict) -> Design:
    """The frozen plan's own exposure budget; every other knob stays at MQ defaults."""
    d = Design(processed_budget=plan["processed_tokens"],
               selection_seed=plan["selection_seed"], order_seed=plan["order_seed"])
    # processed_budget is the ONLY field allowed to differ from the canonical MQ design;
    # constructing from three kwargs makes every other knob a default by construction.
    if d != Design(processed_budget=d.processed_budget):
        raise ValueError("microfit may not re-seed the frozen design")
    # engine.microfit hardcodes the 1e-5 peak, so the plan must literally agree with it.
    if plan["optimizer"] != {"betas": [d.beta1, d.beta2], "epsilon": d.epsilon,
                             "gradient_clip": d.clip_norm, "name": "MQ CpuAdamW",
                             "peak_lr": 1e-5, "schedule": "WSD20/60/20_to_0.1_peak",
                             "weight_decay": d.weight_decay}:
        raise ValueError("frozen plan disagrees with the qualified optimizer design")
    if (d.max_sequence != plan["max_sequence"] or d.trainable_tail != 20
            or plan["precision"] != "fp32" or plan["trainable"] != "blocks8-27"):
        raise ValueError("frozen plan disagrees with the qualified native inventory")
    return d


def presentations(prepared: Path, plan: dict) -> tuple[list[dict], list[dict]]:
    """Return (unique fixed probe worlds, the exact repeated training presentations)."""
    rows = list(read_rows(prepared / "microfit-train.jsonl"))
    if ([r["id"] for r in rows] != plan["row_order_per_exposure"]
            or len({r["parent_id"] for r in rows}) != plan["unique_worlds"]):
        raise ValueError("prepared world/order differs from the frozen plan")
    if any(r["source_split"] != "train" or r["mode"] != "direct" or r["truncated"] for r in rows):
        raise ValueError("microfit consumes untruncated direct TRAIN presentations only")
    repeated = [r for _ in range(plan["exposures_per_world"]) for r in rows]
    if (len(repeated) != plan["presentations"]
            or sum(len(r["ids"]) for r in repeated) != plan["processed_tokens"]
            or sum(sum(t != -100 for t in r["labels"][1:]) for r in repeated) != plan["supervised_tokens"]
            or group_updates(repeated, Design().update_tokens) != plan["update_groups"]):
        raise ValueError("recompiled exposure/update inventory differs from the frozen plan")
    return rows, repeated


def binding_for(prepared: Path, plan: dict, source: Path, grant: dict,
                experiment: str = "RRP-001-MICROFIT") -> dict:
    """Identity every checkpoint and result is bound to; equality is required on resume."""
    from experiments.model_quality_gpu_v1.native import audit_assets, runtime

    return {"experiment": experiment,
            "plan_sha256": digest(plan),
            "data_contract_sha256": stable_hash(prepared / "DATA_CONTRACT.json"),
            "grant_sha256": digest(grant),
            "code_sha256": package_identity()["sha256"],
            "mq_code_sha256": code_identity()["sha256"],
            "checkout": checkout(),
            "runtime": runtime(require=True),
            "source": audit_assets(source)}


def inputs(args) -> tuple[dict, dict, Path, list[dict], list[dict]]:
    """Validate the grant against the exact prepared bytes before anything expensive."""
    prepared = Path(args.prepared)
    contract = read_json(prepared / "DATA_CONTRACT.json")
    for name, sha in contract["files"].items():
        if stable_hash(prepared / name) != sha:
            raise ValueError(f"prepared artifact changed: {name}")
    plan = read_json(Path(args.plan))
    if plan["experiment"] != "RRP-001-MICROFIT" or plan["objective"] != "assistant_response_plus_eos":
        raise ValueError("wrong or unregistered plan")
    grant = read_json(Path(args.grant))
    experiment = getattr(args, "experiment", None) or EXPERIMENT_FOR_PHASE[args.command]
    validate_grant(grant, stable_hash(prepared / "DATA_CONTRACT.json"), digest(plan),
                   experiment)
    probe, repeated = presentations(prepared, plan)
    return grant, plan, prepared, probe, repeated


def eos_set(source: Path, tokenizer) -> list[int]:
    """Source and candidate share this exact list; no tokenizer-only fallback."""
    value = read_json(source / "generation_config.json")["eos_token_id"]
    value = [value] if type(value) is int else value
    if (not isinstance(value, list) or not value or any(type(i) is not int for i in value)
            or tokenizer.eos_token_id not in value):
        raise ValueError("invalid or incomplete generation EOS set")
    return value


def worker(args) -> int:
    # Refuse direct or forged entry before importing the native runtime or writing output.
    try:
        claim_worker(Path(args.out))
    except Exception as exc:
        print(f"Private worker refused: {exc}", file=sys.stderr)
        return 2
    directory = Path(args.out)
    try:
        with lease(Path(args.resource_lock)):
            from .engine import microfit
            from experiments.model_quality_gpu_v1.native import (
                configure_runtime, load_model, load_tokenizer)
            from .native_runtime import evaluate, qualify

            grant, plan, prepared, probe, repeated = inputs(args)
            design = design_for(plan)
            configure_runtime(design.order_seed)
            before_load = guard(before_load=True,
                                phase=RESOURCE_PROFILE.get(args.command, "train"))
            journal = Journal(Path(args.journal), grant)
            if journal.breached:
                raise RuntimeError("approved budget already breached; no further model work")
            binding = binding_for(prepared, plan, Path(args.source), grant,
                                  args.experiment or EXPERIMENT_FOR_PHASE[args.command])

            if args.command in ("train", "resume"):
                # Two full checkpoints for a fresh attempt, one for a resumed attempt.
                needed = (2 if args.command == "train" else 1) * 314618880 * 12 + 2 * 2 ** 30
                free = shutil.disk_usage(directory).free
                if free < needed or free - needed < DISK_FLOOR:
                    raise RuntimeError(
                        f"disk refused; create-only checkpoints need {needed / 2**30:.1f} GiB "
                        f"and a {DISK_FLOOR / 2**30:.0f} GiB floor, have {free / 2**30:.1f} GiB")

            write_new(directory / "binding.json",
                      {"binding": binding, "resources_before_load": before_load,
                       "budget_before": journal.read(), "phase": args.command})

            tokenizer = load_tokenizer(Path(args.source))
            eos = eos_set(Path(args.source), tokenizer)
            model = load_model(Path(args.source), "cuda:0")
            guard()

            if args.command == "baseline":
                # A held-out measurement of an existing checkpoint: no optimizer, no
                # checkpoint, no model selection. Source integrity is proved by digesting
                # the parameters before and after and requiring equality.
                from experiments.model_quality_gpu_v1.native import parameter_digest
                from . import instrument as ins

                bank_sha = stable_hash(Path(args.bank))
                bank = [read_json_line(l) for l in
                        Path(args.bank).read_text(encoding="utf-8").splitlines() if l.strip()]
                window = args.max_sequence or design.max_sequence
                # A window wider than the trained 2048 must be qualified IN THIS RUN
                # before it is used. No silent widening, no RoPE stretching.
                window_qualification = None
                if window > design.max_sequence:
                    window_qualification = ins.qualify_evaluation_window(
                        model, tokenizer, window, journal.reserve, guard)
                # Three bank kinds, three different scoring contracts, and the worker must
                # not guess. `reference_text` alone is NOT a discriminator: the preservation
                # bank carries it too, so the old two-way check would have scored ordinary
                # prose with the Forge v2 JSON verifier. Preservation rows are the ones
                # carrying `sub_bank`.
                #
                #   forge_v1        reference is a plain string; text comparison is exact
                #   forge_v2        reference_text is strict JSON; MUST be scored offline by
                #                   the audited verifier, because four families accept
                #                   alternative optima of equal cost (CL-021)
                #   preservation    reference_text is ordinary prose or one of two exact
                #                   abstention sentences; scored offline by preserve.summarize
                #
                # In all three cases the worker records raw predictions and makes no claim.
                if not bank:
                    raise ValueError("empty bank")
                if "sub_bank" in bank[0]:
                    kind = "preservation"
                elif "reference_text" in bank[0]:
                    kind = "forge_v2"
                else:
                    kind = "forge_v1"
                v2 = kind == "forge_v2"
                evaluable, rejected = ins.compile_bank(
                    bank, tokenizer, max_sequence=window,
                    reference_field="reference" if kind == "forge_v1" else "reference_text")
                import torch

                before = parameter_digest(model)
                # evaluate() is itself no_grad-decorated; this mirrors engine.probe()'s
                # explicit wrapper so the intent is visible at the call site too.
                with torch.no_grad():
                    value = evaluate(model, evaluable, tokenizer, eos, "baseline",
                                     journal.reserve, guard,
                                     max_new=generation_allowance(args, plan))
                after = parameter_digest(model)
                if before != after:
                    raise RuntimeError("evaluation changed source parameters")
                write_new(directory / "baseline.json", {
                    **value, "binding": binding, "optimizer_steps": 0,
                    "source_state_before": before, "source_state_after": after,
                    "bank": str(Path(args.bank)), "bank_sha256": bank_sha,
                    "bank_kind": kind,
                    "scoring_note": {
                        "forge_v2": ("canonical_training_target_match is a LOWER BOUND on a "
                                     "forge_v2 bank: it is a text comparison, and four v2 "
                                     "families accept alternative optima of equal cost. Score "
                                     "the raw predictions with forge_rl.rewards.score_response."),
                        "forge_v1": "text comparison against the v1 string contract is exact here",
                        "preservation": ("canonical_training_target_match is a LOWER BOUND: the "
                                         "preservation contract allows shipped aliases and a "
                                         "declared tolerance, and it scores an answerable row "
                                         "and an abstention row by different rules. Score the "
                                         "raw predictions with preserve.summarize, which keeps "
                                         "wrong_abstention_kind as its own channel."),
                    }[kind],
                    "bank_rows": len(bank), "evaluated_rows": len(evaluable),
                    "rejected_overlength_rows": len(rejected), "rejected": rejected,
                    "max_sequence": window,
                    "trained_window": design.max_sequence,
                    "evaluation_window_qualification": window_qualification,
                    "length_bias_warning": (
                        "Overlength rows were REJECTED, never truncated, so the evaluated "
                        "subset is systematically shorter than the bank. Report both "
                        "denominators."),
                    "quality_claim": False})
                return 0

            if args.command == "qualify":
                result = qualify(model, tokenizer, probe, journal.reserve, guard)
                write_new(directory / "qualification.json", {**result, "binding": binding})
                return 0

            def evaluator(active, label, reserve):
                return evaluate(active, probe, tokenizer, eos, label, reserve, guard,
                                max_new=plan["max_new_tokens"])

            result = microfit(model, repeated, design, binding, directory / "microfit",
                              evaluate=evaluator, reserve=journal.reserve, resource_guard=guard,
                              resume=Path(args.resume) if args.resume else None)
            write_new(directory / "result.json", {**result, "phase": args.command})
        return 0
    except Exception as exc:
        # Preserve the attempt. Never overwrite an existing failure receipt.
        path = directory / "failure.json"
        if not path.exists():
            write_new(path, {"status": "FAILED_OR_BLOCKED", "phase": args.command,
                             "type": type(exc).__name__, "message": str(exc),
                             "quality_permission": False, "release_permission": False})
        traceback.print_exc()
        return 2


def generation_allowance(args, plan) -> int:
    """The per-row generation allowance: the plan's value, optionally LOWERED by the flag.

    The plan is the authorized ceiling, so this can only ever reduce it. Raising it would spend
    a lane the plan did not authorize, which is the whole reason the clamp is here rather than
    trusting the caller.
    """
    ceiling = plan["max_new_tokens"]
    wanted = getattr(args, "max_new_tokens", None)
    if wanted is None:
        return ceiling
    if wanted < 1:
        raise ValueError("generation allowance must be at least one token")
    if wanted > ceiling:
        raise ValueError(f"--max-new-tokens {wanted} exceeds the plan's authorized "
                         f"{ceiling}; the plan is the ceiling and this flag only lowers it")
    return wanted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("environment", *PHASES))
    for name in ("source", "prepared", "plan", "grant", "journal", "resume", "out", "bank"):
        parser.add_argument("--" + name)
    parser.add_argument("--run-id")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--experiment",
                        help="an amended successor lane for this phase, e.g. RRP-002-BASELINE-A2")
    parser.add_argument("--max-sequence", type=int,
                        help="evaluation window; above the trained 2048 it is qualified in-run")
    parser.add_argument("--max-new-tokens", type=int,
                        help="generation allowance per row. May only LOWER the plan's value, "
                             "never raise it: the plan is the authorized ceiling. Useful when "
                             "the bank's golds are short and the generated-token lane is the "
                             "binding constraint -- the preservation bank's golds are <=14 "
                             "tokens measured, so 128 per row would reserve 98,304 tokens to "
                             "emit a few thousand.")
    parser.add_argument("--confirm-model-execution", action="store_true")
    # Must stay byte-identical to MQ-001's default: a different path would silently
    # stop excluding the other native lane that shares this one GPU.
    parser.add_argument("--resource-lock",
                        default=str(Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"))
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return worker(args)
    if args.command == "environment":
        from experiments.model_quality_gpu_v1.native import runtime

        import json
        print(json.dumps({"runtime": runtime(), "resources": snapshot(),
                          "mq_code": code_identity()["sha256"],
                          "package_code": package_identity()["sha256"],
                          "native_qualification": "NOT_RUN", "optimizer_steps": 0}, indent=2))
        return 0
    for name in ("source", "prepared", "plan", "grant", "journal", "run_id", "timeout"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for {args.command}")
    if not args.confirm_model_execution:
        parser.error("model execution requires the explicit --confirm-model-execution flag")
    if args.command == "resume" and not args.resume:
        parser.error("resume verification requires an explicit --resume checkpoint")
    if args.command == "baseline" and not args.bank:
        parser.error("baseline measurement requires an explicit --bank file")

    # Validate the grant, plan and prepared bytes in the parent, before reserving anything.
    grant, plan, prepared, _, _ = inputs(args)
    design_for(plan)
    journal_dir = Path(args.journal).absolute()
    journal = Journal(journal_dir, grant, create=not journal_dir.exists())
    out = journal_dir / "runs" / args.run_id

    # The child always runs in ROOT. Resolve caller-relative paths now so a different
    # cwd cannot select another source, plan, grant, journal, output or lock.
    child = [args.command]
    for name in ("source", "prepared", "plan", "grant", "journal", "resume", "bank",
                 "resource_lock"):
        value = getattr(args, name)
        if value is not None:
            child.extend(["--" + name.replace("_", "-"), str(Path(value).absolute())])
    child.extend(["--out", str(out), "--run-id", args.run_id,
                  "--timeout", str(args.timeout), "--confirm-model-execution"])
    if args.max_sequence is not None:
        child.extend(["--max-sequence", str(args.max_sequence)])
    if args.experiment is not None:
        child.extend(["--experiment", args.experiment])
    if args.max_new_tokens is not None:
        child.extend(["--max-new-tokens", str(args.max_new_tokens)])
    command = [sys.executable, "-m", "experiments.reasoning_posttraining_v1.controller",
               *child, "--_worker"]

    # Refuse a launch the child's own pre-load guard would certainly reject. This uses
    # psutil only: importing torch in the parent would itself cost the child ~0.6 GiB of
    # the very headroom being measured. The child still re-runs the real guard, which
    # remains the authority; this only avoids spending a launch and a retry to learn it.
    import psutil

    profile = RESOURCE_PROFILE.get(args.command, "train")
    need = PRELOAD_MINIMA[profile][0]
    have = psutil.virtual_memory().available / GIB
    if have < need:
        raise SystemExit(
            f"refusing to reserve a launch: host RAM {have:.2f} GiB is below the "
            f"{need:.1f} GiB {profile}-profile pre-load guard for phase {args.command}. "
            f"Wait for memory to free; "
            f"do not lower the guard.")

    started = time.monotonic()
    journal.start(args.run_id, args.command, args.timeout)
    code = bounded_child(command, out, args.timeout)
    supervisor = read_json(out / "supervisor.json")
    journal.settle(args.run_id, supervisor, file_hash(out / "supervisor.json"),
                   time.monotonic() - started)
    state = journal.read()
    write_new(out / "run_summary.json",
              {"experiment": "RRP-001-MICROFIT", "phase": args.command, "run_id": args.run_id,
               "returncode": code, "supervisor": supervisor, "budget_after": state,
               "worker_failed": (out / "failure.json").exists(),
               "quality_claim": False, "release_permission": False})
    print({"phase": args.command, "run_id": args.run_id, "returncode": code,
           "supervisor_status": supervisor["status"], "budget_breached": state["budget_breached"],
           "remaining": state["remaining"]})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
