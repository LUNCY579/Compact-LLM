#!/usr/bin/env python3
"""Read-only GPU restoration proof for one completed six-attempt smoke arm.

Run only after smoke completion, using the experiment's required Python.
This audit script never trains, validates, generates, or publishes arm results.
It holds the runner's global model lock and writes its own audit report only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

AUDIT = Path(__file__).resolve().parent
EXPERIMENT = AUDIT.parents[1]
REPO = EXPERIMENT.parents[1]
GPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_LAUNCH_BLOCKING": "1",
    "PYTORCH_ALLOC_CONF": "backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8",
    "PYTORCH_CUDA_ALLOC_CONF": "backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def equal_state(actual, expected, label="state"):
    """Exact value/shape/dtype equality, permitting restored GPU tensor devices."""
    import torch
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor), f"{label}: tensor type differs")
        require(actual.dtype == expected.dtype and actual.shape == expected.shape,
                f"{label}: tensor metadata differs")
        require(torch.equal(actual.detach().cpu(), expected.detach().cpu()),
                f"{label}: tensor values differ")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(),
                f"{label}: mapping keys differ")
        for key in expected:
            equal_state(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is type(expected) and len(actual) == len(expected),
                f"{label}: sequence differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal_state(left, right, f"{label}/{index}")
    else:
        require(type(actual) is type(expected) and actual == expected,
                f"{label}: scalar differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["phase1_smoke", "phase1_smoke_a1"])
    parser.add_argument("--arm", required=True, choices=["B", "C"])
    args = parser.parse_args()
    require(Path.cwd().resolve() == REPO, "run this verifier from repository root")
    for key, expected in GPU_ENV.items():
        require(os.environ.get(key, expected) == expected,
                f"fixed GPU environment differs: {key}")
        os.environ[key] = expected
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

    import torch
    from safetensors.torch import load_file
    import phase1_run as runner
    from slcmi import training
    from slcmi.checkpoint_resume import capture_rng, fingerprint, validate_progress
    from slcmi.provenance import (ExclusiveFileLock, checkpoint_files,
                                  frozen_parameter_hash, sha256, source_hashes,
                                  trainable_parameter_receipt, write_json)

    require(runner.GPU_EXECUTION_ENV == GPU_ENV, "verifier GPU policy differs from runner")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = AUDIT / f"gpu_resume_noop_{args.stage}_{args.arm}_{timestamp}.json"
    report = {
        "schema_version": 1, "status": "STARTED", "stage": args.stage, "arm": args.arm,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable, "torch": str(torch.__version__),
        "working_directory": str(REPO), "execution_environment": dict(GPU_ENV),
        "verifier_path": str(Path(__file__).resolve()),
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "scope": "one completed checkpoint GPU restore; zero new attempts, validation or generations",
        "model_load_attempts": 0, "model_loads": 0, "callback_events": [],
        "checks": {},
    }
    started = time.perf_counter()
    model = None
    arm_dir = EXPERIMENT / "results" / args.stage / f"arm_{args.arm}"
    latest = arm_dir / "checkpoints/latest.pt"
    core = None
    def snapshot_arm():
        return {str(p.relative_to(arm_dir)).replace("\\", "/"): sha256(p)
                for p in sorted(arm_dir.rglob("*")) if p.is_file()}

    # Global ownership prevents overlap with every stage/core of the live runner.
    with ExclusiveFileLock(EXPERIMENT / "results/.phase1_model.lock"):
        try:
            require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                    "exactly one visible CUDA device is required")
            report["gpu"] = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()
            require(latest.is_file(), "completed latest.pt is missing")
            report["source_sha256_before"] = source_hashes()
            report["arm_file_sha256_before"] = snapshot_arm()
            report["latest_sha256_before"] = sha256(latest)
            identity = read_json(arm_dir / "identity.json")
            config = dict(identity["config"])
            require(config.pop("arm") == args.arm, "arm identity differs")
            expected_core = "a11" if args.stage == "phase1_smoke" else "a1"
            fixed = {
                "stage": args.stage, "core": expected_core, "smoke": True,
                "device": "cuda", "max_steps": 6, "grad_accum": 8, "lr": 1e-4,
                "max_length": 3840, "seed": 17, "rank": 4, "alpha": 8,
                "dropout": 0.0, "trainable_layers": [17, 18, 19],
                "protected_layers": [0, 1, 2],
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "train_packs": 8, "dev_packs": 4,
                "execution_environment": GPU_ENV,
            }
            require(all(config.get(k) == v for k, v in fixed.items()),
                    "completed arm is not the fixed six-attempt smoke recipe")
            core = runner.CORES[expected_core]
            # The resume branch validates existing identity without writing it.
            equal_state(runner.prepare_arm(arm_dir, args.arm, config, core, resume=True),
                        identity, "read-only current identity")
            report["canonical_file_sha256_before"] = checkpoint_files(core["path"])
            manifest = read_json(arm_dir / "run_manifest.json")
            info = read_json(arm_dir / f"arm_{args.arm}_info.json")
            require(manifest.get("status") == "COMPLETE", "logical arm is incomplete")
            runner.validate_lifecycle_receipt(arm_dir, args.arm, manifest, info)
            report["run_id"] = identity["run_id"]
            report["config"] = identity["config"]

            current_data = {
                "curriculum": sha256(runner.LCR_SFT), "target_suite": sha256(runner.DEMO_SUITE),
                **{f"preservation_{cls}": sha256(runner.CORPUS_DIR / f"{cls}.jsonl")
                   for cls in runner.PRESERVATION_CLASSES},
            }
            require(current_data == config["data_sha256"], "current data file hashes differ")
            require(sha256(runner.DEMO_SUITE.parents[1] / "scripts/evaluate_predictions.py")
                    == config["external_scorer_sha256"], "current scorer hash differs")
            encoded = read_json(arm_dir / "encoded_data.json")
            tokenizer = runner.AutoTokenizer.from_pretrained(
                core["path"], trust_remote_code=True, local_files_only=True)
            packs = runner.load_packs(runner.LCR_SFT)
            reconstructed = {}
            for split, count, label in (("train", config["train_packs"], "train"),
                                        ("dev", config["dev_packs"], "dev")):
                rows = runner.sample_curriculum(packs, split, count)
                reconstructed[label], stats = runner.build_dataset(tokenizer, rows, config["max_length"])
                require(stats["n_dropped_too_long"] == 0, "fixed data reconstruction dropped rows")
            equal_state(reconstructed, encoded, "fresh encoded data")
            targets = runner.target_items(True)
            preservation = runner.build_preservation_set(per_class=config["preservation_per_class"])
            data_identity = {
                "train": fingerprint(encoded["train"]), "dev": fingerprint(encoded["dev"]),
                "target": runner._digest_rows(targets), "preservation": runner._digest_rows(preservation),
                "train_count": len(encoded["train"]), "dev_count": len(encoded["dev"]),
                "target_count": len(targets), "preservation_count": len(preservation),
            }
            equal_state(data_identity, identity["dataset_fingerprint"], "current dataset identity")
            del tokenizer, packs, reconstructed, targets, preservation, rows
            gc.collect()

            payload = torch.load(latest, map_location="cpu", weights_only=True)
            progress = payload["progress"]
            require(progress["completed_attempts"] == 6 and progress["pending_validation"] is None,
                    "checkpoint is not complete with no pending validation")
            require(payload["identity"]["effective_recipe"]["max_steps"] == 6,
                    "checkpoint does not preserve six TOTAL attempts")
            validate_progress(progress, max_steps=6, grad_accum=config["grad_accum"],
                              eval_due=lambda attempt: True, train_set=encoded["train"])
            equal_state(payload["identity"]["caller"], identity["checkpoint_fingerprints"],
                        "checkpoint caller")

            print(f"GPU no-op restore: {args.stage} arm {args.arm}; completed=6; loading one model", flush=True)
            report["model_load_attempts"] += 1
            load_started = time.perf_counter()
            tokenizer, model = runner.load_model(core["path"], "cuda", dtype=core["dtype"], gate=True)
            report["model_loads"] += 1
            report["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
            require(str(next(model.parameters()).dtype).removeprefix("torch.") == core["dtype"],
                    "fresh canonical base dtype differs")
            kind = "lora" if args.arm == "B" else "nora"
            model, adapter_info = runner.build_adapter(
                model, kind, config["trainable_layers"], config["rank"], config["alpha"],
                config["dropout"], config["seed"])
            # PEFT constructs a new wrapper in training mode; choose and later
            # verify the no-op verifier's incoming mode explicitly.
            model.eval()
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
            checks = runner.verify_adapter(model, kind, config["trainable_layers"])
            require(checks["all_passed"], "fresh adapter invariants failed")
            initializer = trainable_parameter_receipt(model)
            equal_state(initializer, identity["adapter_initialization"], "actual fresh initializer")
            report["actual_initialization_sha256"] = initializer["sha256"]
            fingerprints = {
                "recipe_sha256": fingerprint({"scientific_recipe": identity["scientific_recipe"],
                                               "dataset_fingerprint": data_identity}),
                "train_data_sha256": data_identity["train"], "eval_data_sha256": data_identity["dev"],
                "base_checkpoint_sha256": fingerprint(report["canonical_file_sha256_before"]),
                "adapter_initialization_sha256": initializer["sha256"],
            }
            equal_state(fingerprints, identity["checkpoint_fingerprints"], "verified five fingerprints")
            report["checkpoint_fingerprints"] = fingerprints
            report["frozen_parameter_sha256_before"] = frozen_parameter_hash(model)
            require(report["frozen_parameter_sha256_before"] == identity["frozen_parameter_sha256"],
                    "fresh frozen base differs from completed arm identity")

            restored = {}
            original_load_latest = training.load_latest
            def inspect_restore(*positional, **keyword):
                require(not restored, "checkpoint loader called more than once")
                result = original_load_latest(*positional, **keyword)
                optimizer = keyword["optimizer"]
                equal_state(optimizer.state_dict(), payload["optimizer"], "restored GPU AdamW")
                equal_state(keyword["scheduler"].state_dict(), payload["scheduler"], "restored scheduler")
                equal_state(keyword["scaler"].state_dict(), payload["scaler"], "restored GradScaler")
                moments = [value for state in optimizer.state.values()
                           for key, value in state.items() if key in ("exp_avg", "exp_avg_sq")]
                require(moments and all(value.device.type == "cuda" for value in moments),
                        "AdamW moments were not restored onto GPU")
                restored.update(optimizer_exact=True, scheduler_exact=True, scaler_exact=True,
                                gpu_moment_tensors=len(moments), scaler_enabled=keyword["scaler"].is_enabled())
                return result
            def forbidden(*unused, **ignored):
                raise RuntimeError("completed no-op resume attempted an update, forward, validation, or checkpoint save")
            def forbidden_step(self, *unused, **ignored):
                return forbidden()
            def callback(event):
                report["callback_events"].append(event)
                raise RuntimeError("completed no-op resume emitted an unexpected training/validation event")
            recipe = identity["scientific_recipe"]
            with ExitStack() as stack:
                stack.enter_context(patch.object(training, "load_latest", inspect_restore))
                stack.enter_context(patch.object(training, "save_latest", forbidden))
                stack.enter_context(patch.object(training, "_validation_with_gate", forbidden))
                stack.enter_context(patch.object(torch.optim.AdamW, "step", forbidden_step))
                stack.enter_context(patch.object(model, "forward", forbidden))
                result = training.train(
                    model, encoded["train"], device="cuda", max_steps=6,
                    grad_accum=config["grad_accum"], lr=config["lr"], seed=config["seed"],
                    warmup_ratio=recipe["warmup_ratio"], weight_decay=recipe["weight_decay"],
                    max_grad_norm=recipe["max_grad_norm"], gradient_checkpointing=recipe["gradient_checkpointing"],
                    eval_set=encoded["dev"], eval_every=recipe["eval_every"], log=lambda _: None,
                    progress_callback=callback, checkpoint_dir=latest.parent,
                    checkpoint_fingerprints=fingerprints, resume_from_checkpoint=True,
                )
            require(restored and not report["callback_events"], "restore observation is incomplete")
            equal_state(capture_rng("cuda"), payload["rng"], "restored CPU/CUDA/Python/NumPy RNG")
            equal_state(result["trajectory"], progress["trajectory"], "unchanged optimizer trajectory")
            equal_state(result["validation"], progress["evals"], "unchanged validation trajectory")
            require(result["optimizer_attempts"] == 6 and result["steps_applied"] == progress["applied_steps"],
                    "resume changed the total attempt/update count")
            report["restored_state"] = {**restored, "all_rng_streams_exact": True}

            saved_adapter_path = arm_dir / "adapter/adapter_model.safetensors"
            saved_adapter = load_file(str(saved_adapter_path), device="cpu")
            # PEFT save removes only the active adapter's '.default' component.
            actual_adapter = {}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                require(parameter.device.type == "cuda", f"trainable is not on GPU: {name}")
                require("lora_A" in name.split(".") or "lora_B" in name.split("."),
                        f"unexpected trainable tensor: {name}")
                key = name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
                require(key not in actual_adapter, f"normalized adapter key collision: {key}")
                actual_adapter[key] = parameter.detach().cpu()
                equal_state(parameter, payload["adapter"][name], f"checkpoint adapter/{name}")
            equal_state(actual_adapter, saved_adapter, "completed saved safetensors adapter")
            report["adapter_parity"] = {"exact": True, "tensor_count": len(actual_adapter),
                                        "parameter_count": sum(p.numel() for p in actual_adapter.values()),
                                        "saved_safetensors_sha256": sha256(saved_adapter_path),
                                        "normalization": "remove .default from LoRA A/B names only"}
            report["frozen_parameter_sha256_after"] = frozen_parameter_hash(model)
            require(report["frozen_parameter_sha256_after"] == report["frozen_parameter_sha256_before"],
                    "frozen base changed during restore")
            require(model.config.abstain_logit_bias and not model.training,
                    "model evaluation mode/gate was not restored")
            report["checks"].update(completed_checkpoint=True, fresh_identity_verified=True,
                                     current_data_verified=True, actual_initializer_verified=True,
                                     adapter_parity=True, frozen_base_unchanged=True,
                                     no_optimizer_updates=True, no_validation=True,
                                     no_forward_or_generation=True, no_checkpoint_save=True)
            report["status"] = "PASS"
        except Exception as exc:
            report["status"] = "FAIL"
            report["error"] = f"{type(exc).__name__}: {exc}"
            report["traceback"] = traceback.format_exc()
        finally:
            try:
                report["source_sha256_after"] = source_hashes()
                report["arm_file_sha256_after"] = snapshot_arm()
                report["latest_sha256_after"] = sha256(latest) if latest.is_file() else None
                for label, before, after in (
                    ("sources_unchanged", "source_sha256_before", "source_sha256_after"),
                    ("arm_files_unchanged", "arm_file_sha256_before", "arm_file_sha256_after"),
                    ("latest_file_unchanged", "latest_sha256_before", "latest_sha256_after"),
                ):
                    report["checks"][label] = before in report and report[before] == report[after]
                if core is not None:
                    report["canonical_file_sha256_after"] = checkpoint_files(core["path"])
                    report["checks"]["canonical_files_unchanged"] = (
                        report.get("canonical_file_sha256_before") == report["canonical_file_sha256_after"])
                if not all(report["checks"].values()):
                    report["status"] = "FAIL"
                    report.setdefault("error", "before/after file integrity verification failed")
            except Exception as exc:
                report["status"] = "FAIL"
                report["integrity_error"] = f"{type(exc).__name__}: {exc}"
            try:
                report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                report["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
            except Exception:
                pass
            report["wall_seconds"] = round(time.perf_counter() - started, 3)
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(report_path, report)
            del model
            gc.collect()
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "model_loads": report["model_loads"], "new_attempts": 0,
                      "error": report.get("error")}, separators=(",", ":")), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
