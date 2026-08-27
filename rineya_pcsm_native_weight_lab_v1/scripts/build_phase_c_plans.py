#!/usr/bin/env python3
"""Build and persist fixed PCSM/control plans from cached calibration gradients."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from rineya_pcsm.hashing import canonical_json_hash, sha256_file
from rineya_pcsm.integration import to_native_update_plan
from rineya_pcsm.manifest import atomic_write_json, git_state, utc_now
from rineya_pcsm.persistence import load_plan, save_plan
from rineya_pcsm.real_plans import (
    PlanBuildResult,
    calibrate_pcsm_plan,
    classical_tensor_plan,
    load_same_role_weights,
)
from rineya_pcsm.topk import budget_to_k


def slug_fraction(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def self_hash_valid(value: dict[str, Any], key: str) -> bool:
    declared = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    return isinstance(declared, str) and declared == canonical_json_hash(payload)


def load_gradient(
    cache: Path,
    section: str,
    expected_base: str,
    expected_config: str,
    expected_target: str,
) -> torch.Tensor:
    manifest_path = cache / f"{section}.json"
    tensor_path = cache / f"{section}.safetensors"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_cache_hash = manifest.get("cache_sha256")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("cache_sha256", None)
    if declared_cache_hash != canonical_json_hash(unhashed_manifest):
        raise RuntimeError(f"{section} gradient cache manifest identity mismatch")
    if manifest.get("name") != section or manifest.get("target") != expected_target:
        raise RuntimeError(f"{section} gradient cache identity mismatch")
    if manifest["model"]["weight_sha256"] != expected_base:
        raise RuntimeError(f"{section} gradient cache has the wrong base")
    if manifest["model"]["config_sha256"] != expected_config:
        raise RuntimeError(f"{section} gradient cache has the wrong config")
    if sha256_file(tensor_path) != manifest["tensor"]["sha256"]:
        raise RuntimeError(f"{section} gradient cache hash mismatch")
    tensors = load_file(tensor_path, device="cpu")
    if set(tensors) != {"gradient"}:
        raise RuntimeError(f"{section} gradient cache contains unexpected tensors")
    gradient = tensors["gradient"]
    if list(gradient.shape) != manifest["tensor"]["shape"]:
        raise RuntimeError(f"{section} gradient cache shape mismatch")
    if str(gradient.dtype) != manifest["tensor"]["dtype"]:
        raise RuntimeError(f"{section} gradient cache dtype mismatch")
    return gradient.float().contiguous()


def verify_existing_plan(destination: Path, plan: Any, base_fingerprint: str) -> None:
    load_plan(destination, expected_base_fingerprint=base_fingerprint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pcsm-plan-verify-", dir=destination.parent
    ) as temporary:
        candidate = Path(temporary) / "candidate"
        save_plan(plan, candidate)
        for name in ("plan.json", "tensors.safetensors"):
            if sha256_file(candidate / name) != sha256_file(destination / name):
                raise RuntimeError(
                    f"Existing plan content differs for {destination}: {name}"
                )


def build_arm(
    arm: str,
    *,
    weights: dict[str, torch.Tensor],
    target: str,
    calibration_gradient: torch.Tensor,
    meta_gradient: torch.Tensor,
    base_fingerprint: str,
    k: int,
    seed: int,
    generator_config: dict[str, Any],
    support_fraction: float,
) -> tuple[Any, tuple[dict[str, float], ...], dict[str, object]]:
    if arm == "random_sparse_native":
        plan = classical_tensor_plan(
            weights[target],
            calibration_gradient,
            target_name=target,
            base_fingerprint=base_fingerprint,
            k=k,
            seed=seed,
            method="random",
            support_fraction=support_fraction,
        )
        return plan, (), {}
    if arm == "gradient_topk_sparse_native":
        plan = classical_tensor_plan(
            weights[target],
            calibration_gradient,
            target_name=target,
            base_fingerprint=base_fingerprint,
            k=k,
            seed=seed,
            method="gradient",
            support_fraction=support_fraction,
        )
        return plan, (), {}
    mode = {
        "target_only_sparse_native": "none",
        "coordinate_shuffled_peer_pcsm": "shuffled",
        "layer_shuffled_peer_pcsm": "layer_shuffled",
        "real_peer_pcsm_bootstrap": "real",
        "real_peer_pcsm_calibrated": "real",
    }.get(arm)
    if mode is None:
        raise ValueError(f"Unsupported planned arm: {arm}")
    steps = (
        int(generator_config["bootstrap_steps"])
        if arm == "real_peer_pcsm_bootstrap"
        else int(generator_config["calibration_steps"])
    )
    result: PlanBuildResult = calibrate_pcsm_plan(
        weights,
        target_name=target,
        calibration_gradient=calibration_gradient,
        meta_gradient=meta_gradient,
        base_fingerprint=base_fingerprint,
        k=k,
        seed=seed,
        mode=mode,
        steps=steps,
        generator_learning_rate=float(generator_config["calibration_learning_rate"]),
        descriptor_dimension=int(generator_config["descriptor_dimension"]),
        structural_regularizer=float(
            generator_config["structural_regularizer_primary"]
        ),
        structural_proxy_learning_rate=float(
            generator_config["structural_proxy_learning_rate"]
        ),
        support_fraction=support_fraction,
        stabilizer_lower=float(generator_config["stabilizer_bounds"][0]),
        stabilizer_upper=float(generator_config["stabilizer_bounds"][1]),
        topk_ste_temperature=float(generator_config["topk_ste_temperature"]),
    )
    return result.plan, result.trace, result.attention


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--gradient-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "qualification"), default="pilot")
    args = parser.parse_args()

    budget = json.loads(args.budget_manifest.read_text(encoding="utf-8"))
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if not self_hash_valid(budget, "budget_manifest_sha256"):
        raise RuntimeError("Budget self-seal is invalid")
    if not self_hash_valid(preregistration, "preregistration_sha256"):
        raise RuntimeError("Preregistration self-seal is invalid")
    budget_file_sha = sha256_file(args.budget_manifest)
    preregistration_file_sha = sha256_file(args.preregistration)
    if preregistration["budget"]["sha256"] != budget_file_sha:
        raise RuntimeError("Preregistration does not bind the supplied budget")
    stage = budget[args.stage]
    planned_cell_grid = [
        {"seed": int(seed), "budget_fraction": float(fraction), "arm": str(arm)}
        for seed in stage["seeds"]
        for fraction in stage["budget_fractions"]
        for arm in stage["arms"]
        if arm != "no_update_frozen"
    ]
    if (
        args.stage == "qualification"
        and planned_cell_grid != preregistration["expected_plan_cells"]
    ):
        raise RuntimeError("Qualification cell grid differs from preregistration")
    base_fingerprint = str(budget["base_model"]["weight_sha256"])
    config_fingerprint = str(budget["base_model"]["config_sha256"])
    if sha256_file(args.model / "model.safetensors") != base_fingerprint:
        raise RuntimeError("Model differs from frozen budget base")
    if sha256_file(args.model / "config.json") != config_fingerprint:
        raise RuntimeError("Model config differs from frozen budget base")
    target = str(budget["target_parameter"])
    weights = load_same_role_weights(args.model)
    if target not in weights:
        raise KeyError(f"Target role weight not present: {target}")
    meta_gradient = load_gradient(
        args.gradient_cache,
        "meta_train",
        base_fingerprint,
        config_fingerprint,
        target,
    )
    calibration_gradient = load_gradient(
        args.gradient_cache,
        "mask_cal",
        base_fingerprint,
        config_fingerprint,
        target,
    )
    if meta_gradient.shape != weights[target].shape:
        raise RuntimeError("Gradient/target shapes differ")
    cache_files = {
        section: {
            "manifest_sha256": sha256_file(args.gradient_cache / f"{section}.json"),
            "tensor_sha256": sha256_file(
                args.gradient_cache / f"{section}.safetensors"
            ),
            "manifest_self_sha256": preregistration["gradient_cache"][section][
                "manifest_self_sha256"
            ],
            "row_ids": preregistration["gradient_cache"][section]["row_ids"],
        }
        for section in ("meta_train", "mask_cal")
    }
    for section, record in cache_files.items():
        frozen = preregistration["gradient_cache"][section]
        if (
            record["manifest_sha256"] != frozen["manifest"]["sha256"]
            or record["tensor_sha256"] != frozen["tensor"]["sha256"]
        ):
            raise RuntimeError(f"Preregistered {section} cache differs")

    provenance = {
        "preregistration_file_sha256": preregistration_file_sha,
        "preregistration_self_sha256": preregistration["preregistration_sha256"],
        "budget_file_sha256": budget_file_sha,
        "budget_self_sha256": budget["budget_manifest_sha256"],
        "sample_manifest_sha256": preregistration["sample_manifest"]["sha256"],
        "split_manifest_sha256": preregistration["split_manifest"]["sha256"],
        "model_weight_sha256": base_fingerprint,
        "model_config_sha256": config_fingerprint,
        "tokenizer_sha256": preregistration["model"]["tokenizer_sha256"],
        "gradient_cache": cache_files,
        "target": target,
        "role": "mlp.down_proj",
        "expected_plan_cells_sha256": canonical_json_hash(
            preregistration["expected_plan_cells"]
        ),
    }
    planner_access_log = [
        {
            "event": "cache_access",
            "purpose": "META_TRAIN gradient cache",
            "split": "train",
            "sealed": False,
        },
        {
            "event": "cache_access",
            "purpose": "MASK_CAL gradient cache",
            "split": "train",
            "sealed": False,
        },
    ]
    planner_access_log_sha = canonical_json_hash(planner_access_log)

    arms = [arm for arm in stage["arms"] if arm != "no_update_frozen"]
    records: list[dict[str, Any]] = []
    if args.result.is_file():
        partial = json.loads(args.result.read_text(encoding="utf-8"))
        if (
            partial.get("status") == "running"
            and partial.get("schema_version") == "2.0"
            and partial.get("stage") == args.stage
            and partial.get("base_fingerprint") == base_fingerprint
            and partial.get("provenance") == provenance
        ):
            records = list(partial.get("records", []))
    completed_keys: set[tuple[int, float, str]] = set()
    expected_plan_keys = {
        (int(cell["seed"]), float(cell["budget_fraction"]), str(cell["arm"]))
        for cell in planned_cell_grid
    }
    for record in records:
        if not self_hash_valid(record, "record_sha256"):
            raise RuntimeError("Partial plan record self-seal is invalid")
        key = (
            int(record["seed"]),
            float(record["budget_fraction"]),
            str(record["arm"]),
        )
        if key in completed_keys:
            raise RuntimeError(f"Duplicate partial plan record: {key}")
        destination = Path(record["plan_directory"])
        if (
            sha256_file(destination / "plan.json") != record["plan_manifest_sha256"]
            or sha256_file(destination / "tensors.safetensors")
            != record["plan_tensor_sha256"]
        ):
            raise RuntimeError(f"Partial plan artifact mismatch: {key}")
        completed_keys.add(key)
    if not completed_keys.issubset(expected_plan_keys):
        raise RuntimeError("Partial result contains a noncanonical plan cell")
    for seed in stage["seeds"]:
        for fraction in stage["budget_fractions"]:
            k = budget_to_k(weights[target].numel(), float(fraction))
            for arm in arms:
                cell_key = (int(seed), float(fraction), str(arm))
                if cell_key in completed_keys:
                    continue
                started = time.perf_counter()
                plan, trace, attention = build_arm(
                    arm,
                    weights=weights,
                    target=target,
                    calibration_gradient=calibration_gradient,
                    meta_gradient=meta_gradient,
                    base_fingerprint=base_fingerprint,
                    k=k,
                    seed=int(seed),
                    generator_config=budget["generator"],
                    support_fraction=float(fraction),
                )
                destination = (
                    args.plan_root
                    / args.stage
                    / f"seed-{seed}"
                    / f"budget-{slug_fraction(float(fraction))}"
                    / arm
                )
                if destination.exists():
                    verify_existing_plan(destination, plan, base_fingerprint)
                else:
                    save_plan(plan, destination)
                tensor_path = destination / "tensors.safetensors"
                manifest_path = destination / "plan.json"
                entry = plan.entries[0]
                native_plan = to_native_update_plan(plan)
                selected_scales = entry.stabilizer.reshape(-1).index_select(
                    0, entry.flat_indices
                )
                record: dict[str, Any] = {
                    "stage": args.stage,
                    "seed": int(seed),
                    "budget_fraction": float(fraction),
                    "k": k,
                    "arm": arm,
                    "plan_directory": str(destination.resolve()),
                    "plan_manifest_sha256": sha256_file(manifest_path),
                    "plan_tensor_sha256": sha256_file(tensor_path),
                    "plan_hash": native_plan.plan_hash,
                    "support_hash": native_plan.support_hash,
                    "support_stats": plan.support_stats(),
                    "selected_scale_distribution": {
                        "minimum": float(selected_scales.min()),
                        "maximum": float(selected_scales.max()),
                        "mean": float(selected_scales.mean()),
                        "standard_deviation": float(
                            selected_scales.std(unbiased=False)
                        ),
                        "q05": float(torch.quantile(selected_scales, 0.05)),
                        "q50": float(torch.quantile(selected_scales, 0.50)),
                        "q95": float(torch.quantile(selected_scales, 0.95)),
                    },
                    "metadata": dict(plan.metadata),
                    "attention": attention,
                    "calibration_trace": list(trace),
                    "generator_architecture_sha256": canonical_json_hash(
                        dict(entry.generator_config)
                    ),
                    "generator_parameter_count": int(
                        plan.metadata.get("generator_parameter_count", 0)
                    ),
                    "attention_architecture_sha256": canonical_json_hash(
                        plan.metadata.get("attention_config", {})
                    ),
                    "attention_parameter_count": int(
                        plan.metadata.get("attention_parameter_count", 0)
                    ),
                    "total_generator_capacity": int(
                        plan.metadata.get("generator_parameter_count", 0)
                    )
                    + int(plan.metadata.get("attention_parameter_count", 0)),
                    "meta_compute": {
                        "calibration_steps": int(plan.metadata.get("steps", 0)),
                        "backward_count": int(plan.metadata.get("steps", 0)),
                        "trace_length": len(trace),
                        "meta_train": cache_files["meta_train"],
                        "mask_cal": cache_files["mask_cal"],
                    },
                    "preregistration_file_sha256": preregistration_file_sha,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                if arm == "layer_shuffled_peer_pcsm":
                    expected_permutation = preregistration["layer_depth_permutations"][
                        str(seed)
                    ]
                    actual_mapping = plan.metadata.get("layer_depth_assignment")
                    if (
                        actual_mapping != expected_permutation["mapping"]
                        or plan.metadata.get("layer_depth_assignment_sha256")
                        != expected_permutation["mapping_sha256"]
                        or not plan.metadata.get("layer_depth_assignment_nonidentity")
                    ):
                        raise RuntimeError("Layer permutation evidence mismatch")
                    record["layer_depth_permutation"] = expected_permutation
                record["record_sha256"] = canonical_json_hash(record)
                records.append(record)
                atomic_write_json(
                    args.result,
                    {
                        "schema_version": "2.0",
                        "status": "running",
                        "updated_at_utc": utc_now(),
                        "base_fingerprint": base_fingerprint,
                        "stage": args.stage,
                        "provenance": provenance,
                        "git": git_state(Path(__file__).resolve().parents[2]),
                        "access_log": planner_access_log,
                        "access_log_sha256": planner_access_log_sha,
                        "records": records,
                        "sealed_test_read": False,
                    },
                )
    result = {
        "schema_version": "2.0",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "base_fingerprint": base_fingerprint,
        "stage": args.stage,
        "provenance": provenance,
        "git": git_state(Path(__file__).resolve().parents[2]),
        "access_log": planner_access_log,
        "access_log_sha256": planner_access_log_sha,
        "records": records,
        "sealed_test_read": False,
    }
    result["result_sha256"] = canonical_json_hash(result)
    atomic_write_json(args.result, result)
    print(json.dumps({"records": len(records), "result": str(args.result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
