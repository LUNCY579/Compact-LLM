#!/usr/bin/env python3
# ruff: noqa: TRY301 -- failures inside the audited run must be persisted by the wrapper.
"""Execute actual one-step held-out-loss arms from byte-verified fixed plans."""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from rineya_pcsm.application import apply_update_plan
from rineya_pcsm.data import stream_selected_sft_rows
from rineya_pcsm.delta_audit import audit_parameter_delta
from rineya_pcsm.hashing import (
    canonical_json_hash,
    fingerprint_tensors,
    sha256_file,
)
from rineya_pcsm.integration import to_native_update_plan
from rineya_pcsm.manifest import (
    atomic_write_json,
    environment_manifest,
    git_state,
    utc_now,
)
from rineya_pcsm.optimizer import ExactSupportAdamW
from rineya_pcsm.persistence import load_plan
from rineya_pcsm.phase_c import encode_row, evaluate_examples, load_native_model
from rineya_pcsm.randomness import seed_everything


def load_validation_rows(
    corpus_root: Path, selected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    shards = list((corpus_root / "data" / "sft" / "validation").glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError("No validation SFT shards found")
    return {
        str(row["id"]): row for row in stream_selected_sft_rows(shards, selected_ids)
    }


def load_cached_gradient(
    cache: Path,
    section: str,
    *,
    expected_base: str,
    expected_target: str,
) -> torch.Tensor:
    manifest = json.loads((cache / f"{section}.json").read_text(encoding="utf-8"))
    declared_cache_hash = manifest.get("cache_sha256")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("cache_sha256", None)
    if declared_cache_hash != canonical_json_hash(unhashed_manifest):
        raise RuntimeError(f"{section} cache manifest identity mismatch")
    if (
        manifest.get("name") != section
        or manifest.get("target") != expected_target
        or manifest.get("model", {}).get("weight_sha256") != expected_base
    ):
        raise RuntimeError(f"{section} cache identity mismatch")
    tensor_path = cache / f"{section}.safetensors"
    if sha256_file(tensor_path) != manifest["tensor"]["sha256"]:
        raise RuntimeError(f"{section} cache hash mismatch")
    tensors = load_file(tensor_path, device="cpu")
    if set(tensors) != {"gradient"}:
        raise RuntimeError(f"{section} cache contains unexpected tensors")
    gradient = tensors["gradient"]
    if list(gradient.shape) != manifest["tensor"]["shape"]:
        raise RuntimeError(f"{section} cache shape mismatch")
    if str(gradient.dtype) != manifest["tensor"]["dtype"]:
        raise RuntimeError(f"{section} cache dtype mismatch")
    return gradient.float()


def select_learning_rate(
    *,
    target_weight: torch.Tensor,
    calibration_gradient: torch.Tensor,
    meta_gradient: torch.Tensor,
    indices: torch.Tensor,
    stabilizer: torch.Tensor,
    candidates: list[float],
    epsilon: float,
    weight_decay: float,
) -> tuple[float, dict[str, float]]:
    """Select LR on META_TRAIN first-order loss only; dev remains untouched."""

    selected_weight = target_weight.reshape(-1).index_select(0, indices).float()
    selected_cal = calibration_gradient.reshape(-1).index_select(0, indices).float()
    selected_meta = meta_gradient.reshape(-1).index_select(0, indices).float()
    direction = selected_cal / (selected_cal.abs() + epsilon)
    direction.add_(selected_weight, alpha=weight_decay)
    direction.mul_(stabilizer.float())
    predictions = {
        str(value): float(torch.sum(selected_meta * (-value * direction)))
        for value in candidates
    }
    selected = min(candidates, key=lambda value: (predictions[str(value)], value))
    return selected, predictions


def assert_delta_valid(
    *,
    valid: bool,
    before_unplanned_hash: str,
    after_unplanned_hash: str,
    arm: str,
) -> None:
    if not valid or before_unplanned_hash != after_unplanned_hash:
        raise RuntimeError(f"Native delta audit failed for {arm}")


def self_hash_valid(value: dict[str, Any], key: str) -> bool:
    declared = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    return isinstance(declared, str) and declared == canonical_json_hash(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-manifest", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--plan-result", type=Path, required=True)
    parser.add_argument("--gradient-cache", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "qualification"), default="pilot")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--maximum-tokens", type=int, default=40960)
    args = parser.parse_args()

    budget = json.loads(args.budget_manifest.read_text(encoding="utf-8"))
    plan_result = json.loads(args.plan_result.read_text(encoding="utf-8"))
    if not self_hash_valid(budget, "budget_manifest_sha256"):
        raise RuntimeError("Budget manifest self-seal is invalid")
    if not self_hash_valid(plan_result, "result_sha256"):
        raise RuntimeError("Plan result self-seal is invalid")
    if any(
        not self_hash_valid(record, "record_sha256")
        for record in plan_result.get("records", [])
    ):
        raise RuntimeError("Plan result contains an invalid record self-seal")
    stage = budget[args.stage]
    base_hash = str(budget["base_model"]["weight_sha256"])
    if sha256_file(args.model / "model.safetensors") != base_hash:
        raise RuntimeError("Base checkpoint does not match frozen budget")
    if sha256_file(args.model / "config.json") != budget["base_model"]["config_sha256"]:
        raise RuntimeError("Model config does not match frozen budget")
    sample_record = budget["sample_manifest"]
    sample_path = (
        args.sample_manifest.resolve()
        if args.sample_manifest is not None
        else Path(sample_record["path"])
    )
    if sha256_file(sample_path) != sample_record["sha256"]:
        raise RuntimeError("Frozen sample manifest hash mismatch")
    sample_manifest = json.loads(sample_path.read_text(encoding="utf-8"))
    if not self_hash_valid(sample_manifest, "sample_manifest_sha256"):
        raise RuntimeError("Sample manifest self-seal is invalid")
    if (
        sha256_file(args.model / "tokenizer.json")
        != sample_manifest["tokenizer_sha256"]
    ):
        raise RuntimeError("Tokenizer does not match frozen sample manifest")
    if plan_result["base_fingerprint"] != base_hash:
        raise RuntimeError("Plan result belongs to a different base")
    if (
        plan_result.get("status") != "complete"
        or plan_result.get("stage") != args.stage
    ):
        raise RuntimeError("Plan result is incomplete or belongs to another stage")
    target_name = str(budget["target_parameter"])
    selected_dev = stage["one_step_dev"]
    dev_ids = {str(row["row_id"]) for row in selected_dev}
    source_rows = load_validation_rows(args.corpus_root.resolve(), dev_ids)
    for frozen in selected_dev:
        row_id = str(frozen["row_id"])
        if canonical_json_hash(source_rows[row_id]) != frozen["content_sha256"]:
            raise RuntimeError(f"Frozen validation row changed: {row_id}")

    calibration_gradient = load_cached_gradient(
        args.gradient_cache,
        "mask_cal",
        expected_base=base_hash,
        expected_target=target_name,
    )
    meta_gradient = load_cached_gradient(
        args.gradient_cache,
        "meta_train",
        expected_base=base_hash,
        expected_target=target_name,
    )
    plan_lookup: dict[tuple[int, float, str], dict[str, Any]] = {}
    for record in plan_result["records"]:
        key = (
            int(record["seed"]),
            float(record["budget_fraction"]),
            str(record["arm"]),
        )
        if key in plan_lookup:
            raise RuntimeError(f"Duplicate plan-result cell: {key}")
        plan_lookup[key] = record
    expected_plan_keys = {
        (int(seed), float(fraction), str(arm))
        for seed in stage["seeds"]
        for fraction in stage["budget_fractions"]
        for arm in stage["arms"]
        if arm != "no_update_frozen"
    }
    if set(plan_lookup) != expected_plan_keys:
        raise RuntimeError("Plan-result cells do not exactly match the frozen stage")
    device = torch.device(args.device)
    optimizer_config = budget["native_optimizer"]
    seed_everything(17)
    budget_file_hash = sha256_file(args.budget_manifest)
    plan_result_file_hash = sha256_file(args.plan_result)
    if args.output.is_file():
        run = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            run.get("stage") != args.stage
            or run.get("base_weight_sha256") != base_hash
            or run.get("budget_manifest_sha256") != budget_file_hash
            or run.get("plan_result_sha256") != plan_result_file_hash
        ):
            raise RuntimeError("Existing run cannot resume under different evidence")
        if any(
            not self_hash_valid(record, "record_sha256")
            for record in run.get("records", [])
        ):
            raise RuntimeError("Existing run contains an invalid record self-seal")
        for key in (
            "run_sha256",
            "error",
            "error_type",
            "traceback",
            "failed_at_utc",
            "completed_at_utc",
        ):
            run.pop(key, None)
        run["status"] = "running"
        run["resumed_at_utc"] = utc_now()
    else:
        run = {
            "schema_version": "1.0",
            "status": "running",
            "stage": args.stage,
            "started_at_utc": utc_now(),
            "base_weight_sha256": base_hash,
            "budget_manifest_sha256": budget_file_hash,
            "plan_result_sha256": plan_result_file_hash,
            "device": str(device),
            "records": [],
            "environment": environment_manifest(),
            "git": git_state(Path(__file__).resolve().parents[2]),
            "sealed_test_read": False,
        }
    completed_cells: set[tuple[int, float, str]] = set()
    for record in run["records"]:
        key = (
            int(record["seed"]),
            float(record["budget_fraction"]),
            str(record["arm"]),
        )
        if key in completed_cells:
            raise RuntimeError(f"Existing run has a duplicate cell: {key}")
        completed_cells.add(key)
    atomic_write_json(args.output, run)
    try:
        # Tokenization is frozen once; every arm receives identical tensors.
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        examples = [
            encode_row(tokenizer, source_rows[row_id], args.maximum_tokens)
            for row_id in sorted(dev_ids)
        ]
        frozen_by_id = {str(row["row_id"]): row for row in selected_dev}
        for example in examples:
            frozen = frozen_by_id[example.row_id]
            if example.token_count != int(frozen["token_count"]):
                raise RuntimeError(f"Frozen token count changed: {example.row_id}")
            supervised = int(example.labels.ne(-100).sum().item())
            if supervised != int(frozen["assistant_token_count"]):
                raise RuntimeError(
                    f"Frozen assistant-token count changed: {example.row_id}"
                )

        for seed in stage["seeds"]:
            for fraction in stage["budget_fractions"]:
                for arm in stage["arms"]:
                    cell_key = (int(seed), float(fraction), str(arm))
                    if cell_key in completed_cells:
                        continue
                    started = time.perf_counter()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    model, _ = load_native_model(args.model, device=device)
                    record: dict[str, Any] = {
                        "seed": int(seed),
                        "budget_fraction": float(fraction),
                        "arm": arm,
                    }
                    if arm == "no_update_frozen":
                        evaluation = evaluate_examples(model, examples, device=device)
                        record.update(
                            {
                                "k": 0,
                                "learning_rate": 0.0,
                                "evaluation": evaluation,
                                "delta_audit": {
                                    "valid": True,
                                    "planned_k": 0,
                                    "changed_union_size": 0,
                                },
                                "unplanned_parameter_hash_unchanged": True,
                                "optimizer_state_bytes": 0,
                            }
                        )
                    else:
                        plan_record = plan_lookup[(int(seed), float(fraction), arm)]
                        plan_directory = Path(plan_record["plan_directory"])
                        if (
                            sha256_file(plan_directory / "plan.json")
                            != plan_record["plan_manifest_sha256"]
                            or sha256_file(plan_directory / "tensors.safetensors")
                            != plan_record["plan_tensor_sha256"]
                        ):
                            raise RuntimeError(f"Recorded plan hash mismatch: {arm}")
                        rich_plan = load_plan(
                            plan_directory,
                            expected_base_fingerprint=base_hash,
                        )
                        if len(rich_plan.entries) != 1:
                            raise RuntimeError(f"Expected one target plan for {arm}")
                        rich_entry = rich_plan.entries[0]
                        if (
                            rich_entry.name != target_name
                            or rich_entry.k != int(plan_record["k"])
                            or rich_entry.support_fraction != float(fraction)
                        ):
                            raise RuntimeError(f"Plan cell binding mismatch: {arm}")
                        metadata = dict(rich_plan.metadata)
                        expected_modes = {
                            "target_only_sparse_native": ("mode", "none"),
                            "coordinate_shuffled_peer_pcsm": (
                                "mode",
                                "shuffled",
                            ),
                            "layer_shuffled_peer_pcsm": (
                                "mode",
                                "layer_shuffled",
                            ),
                            "real_peer_pcsm_bootstrap": ("mode", "real"),
                            "real_peer_pcsm_calibrated": ("mode", "real"),
                            "random_sparse_native": ("method", "random"),
                            "gradient_topk_sparse_native": (
                                "method",
                                "gradient",
                            ),
                        }
                        metadata_key, metadata_value = expected_modes[arm]
                        if metadata.get(metadata_key) != metadata_value:
                            raise RuntimeError(f"Plan arm metadata mismatch: {arm}")
                        if int(metadata.get("seed", -1)) != int(seed):
                            raise RuntimeError(f"Plan seed metadata mismatch: {arm}")
                        expected_steps = (
                            int(budget["generator"]["bootstrap_steps"])
                            if arm == "real_peer_pcsm_bootstrap"
                            else int(budget["generator"]["calibration_steps"])
                            if arm
                            in {
                                "target_only_sparse_native",
                                "coordinate_shuffled_peer_pcsm",
                                "layer_shuffled_peer_pcsm",
                                "real_peer_pcsm_calibrated",
                            }
                            else None
                        )
                        if (
                            expected_steps is not None
                            and int(metadata.get("steps", -1)) != expected_steps
                        ):
                            raise RuntimeError(
                                f"Plan calibration compute mismatch: {arm}"
                            )
                        native_plan = to_native_update_plan(rich_plan)
                        target = dict(model.named_parameters())[target_name]
                        before_target = target.detach().cpu().clone()
                        unplanned_before = fingerprint_tensors(
                            dict(model.named_parameters()), exclude=[target_name]
                        )
                        entry = native_plan.entries[0]
                        learning_rate, lr_predictions = select_learning_rate(
                            target_weight=before_target,
                            calibration_gradient=calibration_gradient,
                            meta_gradient=meta_gradient,
                            indices=entry.flat_indices,
                            stabilizer=entry.stabilizer,
                            candidates=[
                                float(value)
                                for value in optimizer_config["learning_rate_grid"]
                            ],
                            epsilon=float(optimizer_config["epsilon"]),
                            weight_decay=float(optimizer_config["weight_decay"]),
                        )
                        applied = apply_update_plan(model, native_plan)
                        optimizer = ExactSupportAdamW(
                            applied,
                            lr=learning_rate,
                            betas=tuple(optimizer_config["betas"]),
                            eps=float(optimizer_config["epsilon"]),
                            weight_decay=float(optimizer_config["weight_decay"]),
                        )
                        target.grad = calibration_gradient.to(
                            device=target.device, dtype=target.dtype
                        )
                        optimizer.step()
                        target.grad = None
                        unplanned_after = fingerprint_tensors(
                            dict(model.named_parameters()), exclude=[target_name]
                        )
                        delta = audit_parameter_delta(
                            {target_name: before_target},
                            {target_name: target.detach().cpu()},
                            native_plan,
                            native_coordinate_count_override=sum(
                                parameter.numel() for parameter in model.parameters()
                            ),
                        )
                        assert_delta_valid(
                            valid=delta.valid,
                            before_unplanned_hash=unplanned_before,
                            after_unplanned_hash=unplanned_after,
                            arm=arm,
                        )
                        evaluation = evaluate_examples(model, examples, device=device)
                        optimizer_state_bytes = native_plan.k * (8 + 4 + 4)
                        record.update(
                            {
                                "k": native_plan.k,
                                "plan_hash": native_plan.plan_hash,
                                "support_hash": native_plan.support_hash,
                                "selected_scale_distribution": plan_record[
                                    "selected_scale_distribution"
                                ],
                                "learning_rate": learning_rate,
                                "meta_lr_predictions": lr_predictions,
                                "evaluation": evaluation,
                                "delta_audit": delta.to_dict(),
                                "unplanned_parameter_hash_before": unplanned_before,
                                "unplanned_parameter_hash_after": unplanned_after,
                                "unplanned_parameter_hash_unchanged": True,
                                "optimizer_state_bytes": optimizer_state_bytes,
                                "checkpoint_delta_bytes_estimate": native_plan.k * 10,
                            }
                        )
                        applied.restore()
                    record["wall_clock_seconds"] = time.perf_counter() - started
                    evaluated_tokens = sum(example.token_count for example in examples)
                    record["evaluation_tokens"] = evaluated_tokens
                    record["evaluation_tokens_per_second"] = (
                        evaluated_tokens / record["evaluation"]["elapsed_seconds"]
                    )
                    record["gradient_storage_bytes"] = (
                        calibration_gradient.numel()
                        * calibration_gradient.element_size()
                    )
                    record["peak_allocated_vram_bytes"] = (
                        torch.cuda.max_memory_allocated()
                        if device.type == "cuda"
                        else 0
                    )
                    record["peak_reserved_vram_bytes"] = (
                        torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
                    )
                    record["record_sha256"] = canonical_json_hash(record)
                    run["records"].append(record)
                    if arm != "no_update_frozen":
                        del applied, optimizer, rich_plan, native_plan, target
                    del model
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    atomic_write_json(args.output, run)
        run["status"] = "complete"
        run["completed_at_utc"] = utc_now()
        run["run_sha256"] = canonical_json_hash(run)
        atomic_write_json(args.output, run)
    except Exception as error:
        run.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(args.output, run)
        raise

    sparse = [record for record in run["records"] if record["k"] > 0]
    matched_k = all(
        len(
            {
                record["k"]
                for record in sparse
                if record["seed"] == seed and record["budget_fraction"] == fraction
            }
        )
        == 1
        for seed in stage["seeds"]
        for fraction in stage["budget_fractions"]
    )
    all_valid = all(record["delta_audit"]["valid"] for record in run["records"])
    decision: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "stage": args.stage,
        "pilot_can_advance": False if args.stage == "pilot" else None,
        "status": (
            "VALID_RESOURCE_PILOT"
            if args.stage == "pilot" and all_valid and matched_k
            else "QUALIFICATION_EVIDENCE_ONLY"
            if args.stage == "qualification" and all_valid and matched_k
            else "INVALID"
        ),
        "all_delta_audits_valid": all_valid,
        "matched_support_counts": matched_k,
        "advancement_gate_evaluated": False,
        "advancement_gate_status": (
            "not_applicable_to_pilot"
            if args.stage == "pilot"
            else "pending_audit_phase_c_gate.py"
        ),
        "sealed_test_read": False,
        "run_sha256": sha256_file(args.output),
    }
    decision["decision_sha256"] = canonical_json_hash(decision)
    atomic_write_json(args.decision, decision)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
