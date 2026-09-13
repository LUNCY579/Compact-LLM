"""Explicit inspect -> P1 -> separately requested P2. No recovery/DEV/TEST command."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import torch
from . import guards as g
from .checks import active_path, close, grid, train_pair
from .model import Config, DenseReference, DualStack, controls, matrix, presets
from experiments.full_width_transition_v1.source import load_source, state_digest


def verify_inheritance(model, source):
    """Check every initialized tensor, including the dormant encoder and both copies."""
    src = source.state_dict()
    for name, value in model.state_dict().items():
        donor = name
        if name.startswith("decoder."):
            donor = "encoder." + name[len("decoder."):]
        elif name.startswith("reads."):
            donor = "encoder." + name[len("reads."):]
        elif name.startswith("memory."):
            tail = name[len("memory."):]
            donor = "final_norm.weight" if tail == "norm.weight" else "encoder.0." + tail
        expected = src[donor]
        zero = name.startswith("reads.") and name.endswith("o_proj.weight")
        g.require(value.shape == expected.shape and value.dtype == expected.dtype,
                  f"donor tensor shape/dtype: {name}")
        g.require(bool((value == 0).all()) if zero else torch.equal(value, expected), f"donor value mismatch: {name}")
        g.require(value.data_ptr() != expected.data_ptr(), "source storage must not alias the constructed model")
    for encoder, decoder in zip(model.encoder, model.decoder):
        g.require(encoder.q_proj.weight.data_ptr() != decoder.q_proj.weight.data_ptr(), "two stacks share block storage")


def qualify(source_dir, out):
    bound = g.binding(source_dir)
    p = g.protocol()
    checks = dict.fromkeys(p["required_checks"]["P1"], False)
    checks["source_assets"] = True
    # Checkpoint roundtrips are sequential in memory but retained on disk.
    needed = sum(matrix()[k]["parameters"]*4 for k in ("H56", "P56", "C56")) + 1024**3
    g.require(shutil.disk_usage(out).free >= needed, "insufficient disk for three retained mechanical checkpoints")
    source, native, _ = load_source(source_dir, include_native=True)
    g.require(source.config == presets()["B28_1024_official_shape"], "official source dimensions changed")
    checks["source_shape"] = True
    before = state_digest(source)
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    native_errors = {}
    with torch.inference_mode():
        for name, kw in {
            "ordinary": {},
            "left_padding": {"attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]])},
            "large_positions": {"position_ids": torch.arange(source.config.max_position_embeddings-8,
                                                              source.config.max_position_embeddings)[None]},
        }.items():
            actual = source(x, **kw).logits
            expected = native(input_ids=x, use_cache=False, **kw).logits
            mask = kw.get("attention_mask", torch.ones_like(x)).bool()
            native_errors[name] = close(actual[mask], expected[mask], "native_"+name)
        baseline = source(x).logits.clone()
    checks["native_parity"] = True
    del native, actual, expected
    gc.collect()
    measurements, hp_inventory = {"native_max_errors": native_errors, "budgets": matrix()}, {}
    for mode in ("H", "P", "C"):
        model = DualStack.from_source(source, mode)
        verify_inheritance(model, source)
        nparams = sum(v.numel() for v in model.parameters())
        g.require(nparams == matrix()[mode+"56"]["parameters"], "full-shape parameter mismatch")
        with torch.inference_mode():
            original = model(x).logits.clone()
            if mode in {"H", "P"}:
                close(original, baseline, mode+"_identity", exact=True)
                hp_inventory[mode] = {n: (list(v.shape), str(v.dtype)) for n, v in model.named_parameters()}
        mechanics = active_path(model, source) if mode != "C" else grid(model)
        if mode == "C":
            # A pure decoder must actually use its bank, not merely its serial input.
            saved = model.memory.v_proj.weight.detach().clone()
            with torch.no_grad():
                model.memory.v_proj.weight.zero_()
            with torch.inference_mode():
                g.require(not torch.equal(model(x).logits, original), "C56 ignores shared memory")
            with torch.no_grad():
                model.memory.v_proj.weight.copy_(saved)
            del saved
        destination = out / (mode+"56_initial")
        model.save(destination)
        del model
        gc.collect()
        loaded = DualStack.load(destination)
        with torch.inference_mode():
            close(loaded(x).logits, original, mode+"_reload", exact=True)
        measurements[mode+"56"] = {"parameters": nparams, "mechanical_grid": mechanics,
                                     "checkpoint_weights_sha256": g.sha_file(destination / "model.safetensors"),
                                     "checkpoint_config_sha256": g.sha_file(destination / "config.json")}
        del loaded
        gc.collect()
    g.require(hp_inventory["H"] == hp_inventory["P"], "H/P tensor inventory mismatch")
    checks["H_P_parameter_match"] = checks["H_P_identity"] = checks["reload_identity"] = True
    for arm in ("D56", "C28"):
        model = controls(source, arm)
        g.require(sum(v.numel() for v in model.parameters()) == matrix()[arm]["parameters"], "control count mismatch")
        if arm == "D56":
            with torch.inference_mode():
                close(model(x).logits, baseline, "D56_identity", exact=True)
            checks["D56_identity"] = True
        measurements[arm] = {"mechanical_grid": grid(model)}
        del model
        gc.collect()
    checks["pure_construction"] = checks["active_grid"] = True
    g.require(state_digest(source) == before and g.binding(source_dir) == bound, "source/runtime/code changed during P1")
    checks["source_unchanged"] = True
    measurements["source_state_sha256"] = before
    return g.receipt("P1", bound, checks, measurements)


def smoke(source_dir, out, parent):
    bound = g.binding(source_dir)
    parent_hash = g.validate_parent(parent, bound)
    source, _, tokenizer = load_source(source_dir)
    p = g.protocol()
    fixtures = json.loads((g.ROOT / "fixtures.json").read_text(encoding="utf-8"))
    rows = [tokenizer.encode(text, add_special_tokens=False)[:p["engineering"]["max_tokens"]] for text in fixtures["rows"]]
    results = train_pair(source, rows)
    g.require(g.binding(source_dir) == bound, "source/runtime/code changed during P2")
    # These are the invariants checked by train_pair before it can return.
    checks = {"parent_valid": True, "both_arms": set(results) == {"H", "P"},
              "matched_data_order": results["H"]["order"] == results["P"]["order"],
              "finite_optimization": True, "outputs_awakened": True, "upstream_gradients": True,
              "frozen_backbones": True, "post_step_grid": True, "source_unchanged": True}
    return g.receipt("P2", bound, checks, results, parent_hash)


def tiny_source():
    c = Config(vocab_size=41, hidden_size=16, intermediate_size=40, num_attention_heads=4,
               num_key_value_heads=2, head_dim=8, encoder_layers=2, decoder_layers=0,
               architecture="plain", max_position_embeddings=256)
    torch.manual_seed(17)
    return DenseReference(c).eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    for command in ("qualify", "smoke", "tiny-smoke"):
        p = sub.add_parser(command)
        p.add_argument("--out", type=Path, required=True)
        if command != "tiny-smoke":
            p.add_argument("--source", type=Path, required=True)
        if command == "smoke":
            p.add_argument("--parent", type=Path, required=True)
            p.add_argument("--ack-engineering-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.command == "inspect":
        print(json.dumps({"experiment": "D28-001", "implementation": g.implementation(), "runtime": g.runtime(),
                          "budgets": matrix(), "permissions": g.protocol()["permissions"]}, indent=2))
        return 0
    out = g.new_output(args.out)
    started = time.perf_counter()
    try:
        if args.command == "qualify":
            result = qualify(args.source, out)
        elif args.command == "smoke":
            result = smoke(args.source, out, args.parent)
        else:
            result = {"experiment": "D28-001", "profile": "tiny_random_only", "status": "MECHANICAL_PASS",
                      "source_config": asdict(tiny_source().config), "implementation": g.implementation(),
                      "results": train_pair(tiny_source(), [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [2, 4, 6, 8, 10, 12]]),
                      "permissions": g.protocol()["permissions"], "licenses_engineering_smoke": False}
        g.write_json(out / "result.json", result)
        g.write_json(out / "execution.json", {"elapsed_seconds": time.perf_counter()-started,
                                              "runtime": g.runtime(), "command": args.command})
        print(json.dumps({"status": result["status"], "output": str(out)}))
        return 0 if result["status"] in {"PASS", "MECHANICAL_PASS"} else 1
    except (Exception, KeyboardInterrupt) as exc:
        g.write_json(out / "blocked.json", {"experiment": "D28-001", "status": "BLOCKED_OR_INTERRUPTED",
                     "command": args.command, "error_type": type(exc).__name__, "error": str(exc),
                     "elapsed_seconds": time.perf_counter()-started, "licenses_engineering_smoke": False,
                     "permissions": g.protocol()["permissions"]})
        print(f"D28-001 stopped: {exc}", file=sys.stderr)
        return 1
