"""FWT-001 foreground-only CLI: inspect -> P1 qualification -> paired 25-step smoke.

No DEV, sealed evaluation, compression, replacement, or recovery campaign command exists.
"""
from __future__ import annotations
import argparse
import gc
import json
import random
import shutil
from pathlib import Path
import torch
from . import guards as g


def tokenized(tokenizer, role):
    fixture = json.loads((g.ROOT / "fixtures.json").read_text(encoding="utf-8"))
    if role not in {"qualification", "engineering_train"}:
        raise ValueError("unknown fixture role")
    a, b = fixture["qualification"], fixture["engineering_train"]
    if not a or not b or len(set(a + b)) != len(a + b):
        raise ValueError("empty or overlapping mechanical fixture roles")
    result = [tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"] for text in fixture[role]]
    if any(x.shape[0] != 1 or not 4 <= x.shape[1] <= g.protocol()["engineering"]["max_tokens"] for x in result):
        raise ValueError("fixture length outside the registered bound; no silent truncation")
    return result


def qualify(directory, out, bound):
    from .model import FullWidthTransition, augmentation_budget
    from .source import load_source, state_digest
    from .checks import compare, active_checks, greedy
    source, native, tokenizer = load_source(directory, include_native=True)
    before = state_digest(source)
    batches = tokenized(tokenizer, "qualification")
    native_grid, ref = [], []
    with torch.inference_mode():
        for ids in batches:
            logits = source(ids).logits
            native_grid.append(compare(logits, native(input_ids=ids, use_cache=False).logits))
            ref.append(logits.clone())
        expected_generation = greedy(source, batches[0])
    del native
    gc.collect()
    report = {"native_grid": native_grid, "arms": {}, "source_greedy_ids": expected_generation}
    checks = {key: True for key in g.protocol()["required_checks"]["P1"]}
    checks["native_parity"] = all(row["pass"] for row in native_grid)
    if not checks["native_parity"]:
        checks = {key: False for key in checks}
        checks["source_assets"] = True
        return checks, report
    parameter_counts = []
    for mode in ("M", "P"):
        model = FullWidthTransition.from_source(source, split=14, mode=mode)
        parameter_counts.append(sum(p.numel() for p in model.parameters()))
        with torch.inference_mode():
            identity = all(torch.equal(model(ids).logits, value) for ids, value in zip(batches, ref))
            initial = model(batches[0]).logits.clone()
            generation = greedy(model, batches[0])
        checkpoint = out / f"{mode}28_init"
        model.save(checkpoint)
        checkpoint_hashes = {name: g.sha_file(checkpoint / name) for name in ("config.json", "model.safetensors")}
        del model
        gc.collect()
        model = FullWidthTransition.load(checkpoint)
        with torch.inference_mode():
            reload_ok = torch.equal(model(batches[0]).logits, initial)
        # Temporary, fixed, NOT trained activation. Do not save this as a recovery model.
        with torch.no_grad():
            for index, read in enumerate(model.reads):
                read.o_proj.weight.copy_(source.encoder[14 + index].o_proj.weight * g.protocol()["parity"]["nonzero_output_scale"])
        with torch.inference_mode():
            active = not torch.equal(model(batches[0]).logits, initial)
        grid = active_checks(model, batches[0])
        checks["initial_identity"] &= identity and generation == expected_generation
        checks["reload_identity"] &= reload_ok
        checks["active_cache_parity"] &= grid["cache_pass"]
        checks["active_causality"] &= grid["causal_pass"]
        checks["active_path"] &= active
        report["arms"][mode] = {"initial_identity": identity, "generation_ids": generation,
                                "reload_identity": reload_ok, "active_path": active,
                                "checkpoint_hashes": checkpoint_hashes, **grid}
        del model
        gc.collect()
    wanted = augmentation_budget(source.config, 14)["parameters"]
    checks["parameter_match"] = parameter_counts == [wanted, wanted]
    checks["source_unchanged"] = state_digest(source) == before and g.audit_source(directory) == bound["source"]
    report["parameter_counts"] = parameter_counts
    return checks, report


def engineering_pair(source, batches, *, split, recipe):
    """Shared bounded engine; returns engineering measurements, never permissions."""
    from .model import FullWidthTransition
    from .source import state_digest
    from .checks import objective, active_checks
    if (recipe != g.protocol()["engineering"] or recipe["steps"] != 25
            or recipe["max_tokens"] != 64 or recipe["optimizer"] != "AdamW"
            or recipe["train_policy"] != "added_only") :
        raise ValueError("engineering recipe differs from the frozen 25-step protocol")
    if not batches or any(x.ndim != 2 or x.shape[0] != 1 or x.dtype != torch.long
                          or x.device.type != "cpu" or not 4 <= x.shape[1] <= recipe["max_tokens"]
                          or bool((x < 0).any()) or bool((x >= source.config.vocab_size).any()) for x in batches):
        raise ValueError("engineering batches violate the registered bounds")
    source.eval()
    before = state_digest(source)
    source.requires_grad_(False)
    order = list(range(len(batches)))
    random.Random(recipe["seed"]).shuffle(order)
    order = [order[i % len(order)] for i in range(recipe["steps"])]
    records, backbone_ok = {}, True
    for mode in ("M", "P"):
        torch.manual_seed(recipe["seed"])
        model = FullWidthTransition.from_source(source, split=split, mode=mode)
        trainable = model.freeze_backbone()
        model.gradient_checkpointing = recipe["gradient_checkpointing"]
        optimizer = torch.optim.AdamW(trainable, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        watched = {"memory_k": model.memory.k_proj.weight, "memory_v": model.memory.v_proj.weight,
                   **{f"read_{i}_q": r.q_proj.weight for i, r in enumerate(model.reads)}}
        trajectory, upstream = [], {name: False for name in watched}
        for step, row in enumerate(order, 1):
            ids = batches[row]
            with torch.no_grad():
                teacher = source(ids).logits
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective(model(ids).logits, teacher, ids,
                                      temperature=recipe["temperature"], kl_weight=recipe["kl_weight"], ce_weight=recipe["ce_weight"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite engineering objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, recipe["clip_norm"], error_if_nonfinite=True)
            for name, parameter in watched.items():
                gradient = parameter.grad
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        raise ValueError("nonfinite added-branch gradient")
                    upstream[name] |= bool(torch.count_nonzero(gradient))
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in trainable):
                raise ValueError("nonfinite optimized parameter")
            trajectory.append({"step": step, "row_index": row, "target_tokens": ids.numel() - 1,
                               "loss": float(loss.detach()), "pre_clip_grad_norm": float(norm), **metrics})
        model.eval()
        unchanged = state_digest(model, backbone_only=True) == before
        backbone_ok &= unchanged
        records[mode] = {"order": order, "trajectory": trajectory,
                         "added_trainable_parameters": sum(p.numel() for p in trainable),
                         "outputs_awakened": all(bool(torch.count_nonzero(r.o_proj.weight)) for r in model.reads),
                         "upstream_gradients": upstream, "backbone_unchanged": unchanged,
                         "post_step_mechanics": active_checks(model, batches[0])}
        del model, optimizer, trainable, watched, teacher, loss, parameter, gradient
        gc.collect()
    checks = {key: True for key in g.protocol()["required_checks"]["P2"]}
    checks["both_arms"] = set(records) == {"M", "P"}
    checks["matched_data_order"] = records["M"]["order"] == records["P"]["order"] and records["M"]["added_trainable_parameters"] == records["P"]["added_trainable_parameters"]
    checks["outputs_awakened"] = all(x["outputs_awakened"] for x in records.values())
    checks["upstream_gradients"] = all(all(x["upstream_gradients"].values()) for x in records.values())
    checks["frozen_backbones"] = backbone_ok
    checks["post_step_cache_parity"] = all(x["post_step_mechanics"]["cache_pass"] and x["post_step_mechanics"]["causal_pass"] for x in records.values())
    checks["source_unchanged"] = state_digest(source) == before
    return checks, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "qualify", "smoke"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args(argv)
    g.check_dependencies()
    from .model import matrix
    if args.command == "inspect":
        print(json.dumps({"experiment": "FWT-001", "implementation": g.implementation(),
                          "runtime": g.runtime(), "matrix": matrix(), "permissions": g.protocol()["permissions"],
                          "official_execution": "requires a fresh source/runtime-bound P1; no receipt inferred"}, indent=2))
        return 0
    if args.source is None or args.out is None:
        parser.error("--source and --out are required")
    if args.command == "smoke" and args.parent is None:
        parser.error("smoke requires --parent pointing to a passing FWT-001 P1 receipt")
    if args.command == "qualify" and args.parent is not None:
        parser.error("P1 accepts no historical or other parent")
    torch.set_num_threads(g.protocol()["runtime"]["threads"])
    g.require_runtime()
    source_digest = g.audit_source(args.source)
    bound = g.binding(source_digest)
    parent = g.validate_parent(args.parent, bound) if args.command == "smoke" else None
    # P1 persists two FP32 research checkpoints. This is disk, not a RAM/VRAM-fit claim.
    if args.command == "qualify" and shutil.disk_usage(g.ROOT).free < 8 * 2**30:
        raise ValueError("P1 needs at least 8 GiB free output disk space")
    out = g.new_output(args.out, args.source)
    g.write_json(out / "attempt.json", {"command": args.command, "binding": bound, "parent_sha256": parent})
    stage = "P1" if args.command == "qualify" else "P2"
    try:
        if stage == "P1":
            checks, measurements = qualify(args.source, out, bound)
        else:
            from .source import load_source
            source, _, tokenizer = load_source(args.source)
            checks, measurements = engineering_pair(source, tokenized(tokenizer, "engineering_train"),
                                                    split=14, recipe=g.protocol()["engineering"])
            checks["source_unchanged"] &= g.audit_source(args.source) == source_digest
        if g.binding(source_digest) != bound:
            raise ValueError("implementation, fixtures or runtime changed during the attempt")
        result = g.receipt(stage, bound, checks, measurements, parent)
    except Exception as exc:
        result = g.receipt(stage, bound, {}, {"error_type": type(exc).__name__, "error": str(exc)}, parent)
        g.write_json(out / "receipt.json", result)
        raise
    g.write_json(out / "receipt.json", result)
    print(json.dumps({"receipt": str(out / "receipt.json"), "status": result["status"],
                      "permissions": result["permissions"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
