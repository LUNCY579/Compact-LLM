"""Explicit preparation -> independent data review -> GPU qualification -> one cell.

Native commands execute in a supervised foreground worker. No downloads, remote jobs,
automatic training sweeps, sealed evaluation, registry writes or release promotion.
"""
from __future__ import annotations

import argparse
import math
import re
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

from .config import ARMS, Design, validate_native_design
from .data import (compile_plan, encode_row, read_rows, review_template, targets,
                   validate_plan, validate_review)
from .integrity import (PACKAGE, clean_checkout, code_identity, digest, file_hash,
                        new_dir, read_json, seal, unseal, write_new)
from .resources import bounded_child, guard, lease, snapshot
from .supervisor import claim_worker

CHECKS = {"source_tensors", "tokenizer_plan_recompiled", "native_hidden_logits",
          "native_cached_logits", "masked_chunk_loss_and_gradient", "frozen_parameters"}


def native_binding(plan_value, review, source: Path) -> dict:
    from .native import audit_assets, runtime

    plan = validate_plan(plan_value)
    if plan["kind"] != "native_plan":
        raise ValueError("synthetic fixture cannot authorize native work")
    validate_native_design(Design(**plan["design"]))
    validate_review(review, plan_value["sha256"])
    return {"experiment": "MQ-001", "plan_sha256": plan_value["sha256"],
            "review_sha256": digest(review), "code_sha256": code_identity()["sha256"],
            "checkout": clean_checkout(), "runtime": runtime(require=True),
            "source": audit_assets(source)}


def verify_qualification(value: dict, binding: dict) -> dict:
    result = unseal(value)
    if (set(result) != {"schema", "phase", "status", "binding", "checks", "metrics",
                       "optimizer_steps", "quality_permission", "release_permission"}
            or result["schema"] != 1 or result["phase"] != "NATIVE_GPU_QUALIFICATION"
            or result["status"] != "PASS" or result["binding"] != binding
            or set(result["checks"]) != CHECKS
            or any(v is not True for v in result["checks"].values())
            or result["optimizer_steps"] != 0 or result["quality_permission"] is not False
            or result["release_permission"] is not False):
        raise ValueError("incomplete, synthetic or differently bound qualification")
    metrics = result["metrics"]
    if (set(metrics) != {"cache_max_abs", "probe_tokens", "trainable_parameters",
                         "trainable_tensors", "finite_gradient_tensors",
                         "source_state_before", "source_state_after"}
            or not isinstance(metrics.get("source_state_before"), str)
            or re.fullmatch(r"[0-9a-f]{64}", metrics["source_state_before"]) is None
            or type(metrics.get("probe_tokens")) is not int or not 1 <= metrics["probe_tokens"] <= 64
            or type(metrics.get("cache_max_abs")) not in (float, int)
            or not math.isfinite(metrics["cache_max_abs"]) or metrics["cache_max_abs"] < 0
            or metrics.get("trainable_parameters") != 314618880
            or metrics.get("trainable_tensors") != 220
            or metrics.get("finite_gradient_tensors") != 220
            or metrics.get("source_state_before") != metrics.get("source_state_after")):
        raise ValueError("qualification inventory/source-state mismatch")
    return result


def recompile(plan_value: dict, tokenizer) -> None:
    from .evaluation import classify

    plan = validate_plan(plan_value)
    raw = {}
    for split, info in plan["inputs"].items():
        path = Path(info["path"])
        if file_hash(path) != info["sha256"]:
            raise ValueError("raw data changed since compilation")
        raw[split] = read_rows(path)
    if set(raw) != {"train", "dev", "excluded"}:
        raise ValueError("all three raw banks must remain available")
    candidate = compile_plan(raw["train"], raw["dev"], raw["excluded"], tokenizer,
                             Design(**plan["design"]), plan["inputs"], plan["tokenizer"])
    if candidate != plan_value:
        raise ValueError("native tokenizer does not reproduce the frozen plan")
    # Only explicit contracts receive accuracy credit. Open text retains CE/raw outputs.
    for row in plan["dev"]:
        spec = row.get("evaluation")
        classify("", spec)
        references = spec["answers"] if spec is not None else [row["reference"]]
        if any(len(tokenizer.encode(a, add_special_tokens=False)) + 1 > 128 for a in references):
            raise ValueError("dev reference cannot fit the registered generation budget")


def qualify(model, tokenizer, plan_value: dict, binding: dict) -> dict:
    import torch
    from torch.nn import functional as F
    from .losses import chunked_sum
    from .native import hidden_states, parameter_digest, select_tail

    d = Design(**validate_plan(plan_value)["design"])
    named = select_tail(model, d.trainable_tail)
    if sum(p.numel() for _, p in named) != 314618880 or len(named) != 220:
        raise ValueError("not the exact Supa last-20 inventory")
    before = parameter_digest(model)
    example = {"id": "qualification-7", "template_family": "qualification-only",
               "messages": [{"role": "user", "content": "Reply with 7."},
                            {"role": "assistant", "content": "7"}]}
    row = encode_row(example, tokenizer, d)
    if len(row["ids"]) > 64:
        raise ValueError("native diagnostic exceeded its bound")
    ids = torch.tensor([row["ids"]], device="cuda:0", dtype=torch.long)
    labels = torch.tensor([targets(row, "response")], device="cuda:0", dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
        hidden = hidden_states(model, ids)
        torch.testing.assert_close(F.linear(hidden, model.lm_head.weight), logits, atol=0, rtol=0)
        total, count = chunked_sum(hidden, model.lm_head.weight, labels, 2, recompute=False)
        reference = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                    labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        torch.testing.assert_close(total, reference, atol=2e-4, rtol=2e-4)
        parts, past = [], None
        for start in range(0, ids.shape[1], 3):
            out = model(input_ids=ids[:, start:start + 3], past_key_values=past,
                        use_cache=True, return_dict=True)
            past = out.past_key_values; parts.append(out.logits)
        joined = torch.cat(parts, 1)
        torch.testing.assert_close(logits, joined, atol=2e-4, rtol=2e-4)
        cache_error = float((logits - joined).abs().max())
    del logits, hidden, joined, parts, past, out, total, reference
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(); model.zero_grad(set_to_none=True)
    loss, _ = chunked_sum(hidden_states(model, ids), model.lm_head.weight, labels, 2)
    (loss / count).backward()
    finite = sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named)
    if finite != 220 or not any(bool(p.grad.abs().sum() > 0) for _, p in named):
        raise ValueError("native trainable gradients missing/nonfinite/inactive")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError("frozen parameters received gradients")
    model.zero_grad(set_to_none=True); model.eval()
    after = parameter_digest(model)
    if before != after:
        raise ValueError("qualification modified source weights")
    return seal({"schema": 1, "phase": "NATIVE_GPU_QUALIFICATION", "status": "PASS",
                 "binding": binding, "checks": {k: True for k in sorted(CHECKS)},
                 "metrics": {"cache_max_abs": cache_error, "probe_tokens": ids.shape[1],
                             "trainable_parameters": 314618880, "trainable_tensors": 220,
                             "finite_gradient_tensors": finite, "source_state_before": before,
                             "source_state_after": after},
                 "optimizer_steps": 0, "quality_permission": False, "release_permission": False})


def worker(args) -> int:
    # Reject direct/private entry before importing the native runtime or writing output.
    try:
        claim_worker(Path(args.out))
    except Exception as exc:
        print(f"Private worker refused: {exc}", file=sys.stderr)
        return 2
    directory = Path(args.out)
    try:
        with lease(Path(args.resource_lock)):
            from .engine import train_cell
            from .native import (audit_assets, configure_runtime, load_model,
                                 load_tokenizer)

            configure_runtime(Design(**validate_plan(read_json(args.plan))["design"]).order_seed)
            resource = guard(before_load=True, phase=args.command)
            plan_value, review = read_json(args.plan), read_json(args.review)
            binding = native_binding(plan_value, review, Path(args.source))
            tokenizer = load_tokenizer(Path(args.source))
            if audit_assets(Path(args.source), tokenizer_only=True) != validate_plan(plan_value)["tokenizer"]:
                raise ValueError("tokenizer asset binding differs")
            recompile(plan_value, tokenizer)
            if args.command == "train":
                if args.arm not in ARMS or args.qualification is None or not args.confirm_training:
                    raise ValueError("train requires one frozen arm, valid qualification and explicit confirmation")
                verify_qualification(read_json(args.qualification), binding)
                design = Design(**validate_plan(plan_value)["design"])
                n = len(validate_plan(plan_value)["groups"])
                checkpoints = (n + design.checkpoint_every - 1) // design.checkpoint_every
                required = checkpoints * 314618880 * 12 + 2 * 2**30
                if shutil.disk_usage(directory).free < required:
                    raise RuntimeError("insufficient disk for create-only checkpoint history")
            write_new(directory / "binding.json", {"binding": binding, "resources_before_load": resource})
            model = load_model(Path(args.source), "cuda:0")
            guard()
            if args.command == "qualify":
                result = qualify(model, tokenizer, plan_value, binding)
                write_new(directory / "qualification.json", result)
            else:
                cell_binding = {**binding, "arm": args.arm}
                # Source and candidate share this exact EOS list; no tokenizer-only fallback.
                eos = read_json(Path(args.source) / "generation_config.json")["eos_token_id"]
                eos = [eos] if type(eos) is int else eos
                if (not isinstance(eos, list) or not eos or any(type(i) is not int for i in eos)
                        or tokenizer.eos_token_id not in eos):
                    raise ValueError("invalid complete generation EOS set")
                train_cell(model, plan_value, args.arm, directory, cell_binding,
                           resume=Path(args.resume) if args.resume else None,
                           resource_guard=guard, tokenizer=tokenizer, eos_ids=eos)
        return 0
    except Exception as exc:
        write_new(directory / "failure.json", {"status": "FAILED_OR_BLOCKED", "type": type(exc).__name__,
                                               "message": str(exc), "quality_permission": False,
                                               "release_permission": False})
        traceback.print_exc()
        return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("environment", "prepare", "qualify", "train"))
    for name in ("source", "train", "dev", "excluded", "plan", "review", "qualification", "resume", "out"):
        parser.add_argument("--" + name)
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--order-seed", type=int, default=17)
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--resource-lock", default=str(Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"))
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return worker(args)
    if args.command == "environment":
        from .native import runtime

        value = {"runtime": runtime(), "resources": snapshot(), "code": code_identity(),
                 "native_qualification": "NOT_RUN", "optimizer_steps": 0}
        import json
        print(json.dumps(value, indent=2))
        return 0
    if not args.source or not args.out:
        parser.error("--source and a new --out directory are required")
    if args.command == "prepare":
        from .native import audit_assets, load_tokenizer

        if not all((args.train, args.dev, args.excluded)):
            parser.error("prepare needs explicit --train, --dev and --excluded files")
        directory = new_dir(Path(args.out))
        try:
            design = Design(selection_seed=args.selection_seed, order_seed=args.order_seed)
            tokenizer = load_tokenizer(Path(args.source))
            inputs = {key: {"path": str(Path(getattr(args, key)).absolute()),
                            "sha256": file_hash(Path(getattr(args, key)))}
                      for key in ("train", "dev", "excluded")}
            banks = {key: read_rows(Path(value["path"])) for key, value in inputs.items()}
            value = compile_plan(**banks, tokenizer=tokenizer, design=design, inputs=inputs,
                                 tokenizer_identity=audit_assets(Path(args.source), tokenizer_only=True))
            write_new(directory / "plan.json", value)
            write_new(directory / "DATA_REVIEW_PENDING.json", review_template(value))
            return 0
        except Exception as exc:
            write_new(directory / "failure.json", {"status": "BLOCKED", "message": str(exc)})
            raise
    if not args.plan or not args.review:
        parser.error("native execution needs --plan and --review")
    design = Design(**validate_plan(read_json(Path(args.plan)))["design"])
    # The child always runs in this package's ROOT. Resolve caller-relative files now
    # so changing cwd cannot select another plan, source, output or resource lock.
    child_args = [args.command]
    for name in ("source", "train", "dev", "excluded", "plan", "review", "qualification",
                 "resume", "out", "resource_lock"):
        value = getattr(args, name)
        if value is not None:
            child_args.extend(["--" + name.replace("_", "-"), str(Path(value).absolute())])
    if args.arm is not None:
        child_args.extend(["--arm", args.arm])
    if args.confirm_training:
        child_args.append("--confirm-training")
    command = [sys.executable, "-m", "experiments.model_quality_gpu_v1", *child_args, "--_worker"]
    return bounded_child(command, Path(args.out), design.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
