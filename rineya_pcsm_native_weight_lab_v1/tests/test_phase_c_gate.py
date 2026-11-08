from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rineya_pcsm import gate as gate_module
from rineya_pcsm.gate import (
    CANONICAL_ARMS,
    CANONICAL_BUDGETS,
    CANONICAL_SEEDS,
    GateInputs,
    LoadedPlanEvidence,
    audit_phase_c_gate,
    authenticate_phase_d_gate,
    gate_self_hash_valid,
)
from rineya_pcsm.hashing import canonical_json_hash, sha256_file
from rineya_pcsm.topk import budget_to_k

TARGET = "model.layers.14.mlp.down_proj.weight"
TARGET_SHAPE = [1024, 3072]
TARGET_NUMEL = 1024 * 3072
BASE = "1" * 64
CONFIG = "2" * 64
UNPLANNED = "4" * 64
LR_GRID = [0.00005, 0.0001, 0.0002]


@dataclass(slots=True)
class EvidenceFixture:
    inputs: GateInputs
    loaded: dict[Path, LoadedPlanEvidence]

    def audit(self) -> dict[str, Any]:
        return audit_phase_c_gate(
            self.inputs,
            plan_evidence_loader=lambda path, _base: self.loaded[path.resolve()],
            commit_verifier=lambda _path: {
                "verified": True,
                "commit": "a" * 40,
                "repository_relative_path": "results/phase_c/PREREGISTRATION.json",
                "blob_sha256": sha256_file(self.inputs.preregistration),
            },
        )


def _seal(value: dict[str, Any], key: str) -> None:
    value.pop(key, None)
    value[key] = canonical_json_hash(value)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_record(path: Path, self_sha: str | None = None) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if self_sha is not None:
        result["self_sha256"] = self_sha
    return result


def _cells(arms: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {"seed": seed, "budget_fraction": fraction, "arm": arm}
        for seed in CANONICAL_SEEDS
        for fraction in CANONICAL_BUDGETS
        for arm in arms
    ]


def _optimizer() -> dict[str, Any]:
    return {
        "kind": "ExactSupportAdamW",
        "learning_rate_grid": LR_GRID,
        "selection_split": "META_TRAIN_only",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.01,
    }


def _optimizer_signature() -> dict[str, Any]:
    optimizer = _optimizer()
    result = {
        "kind": optimizer["kind"],
        "betas": optimizer["betas"],
        "epsilon": optimizer["epsilon"],
        "weight_decay": optimizer["weight_decay"],
        "learning_rate_grid": optimizer["learning_rate_grid"],
        "selection_split": optimizer["selection_split"],
        "step_count": 1,
    }
    result["signature_sha256"] = canonical_json_hash(result)
    return result


def _layer_permutation(seed: int) -> dict[str, Any]:
    names = [
        f"model.layers.{layer}.mlp.down_proj.weight"
        for layer in range(28)
        if layer != 14
    ]
    depths = [int(name.split(".")[2]) for name in names]
    shift = CANONICAL_SEEDS.index(seed) + 1
    rotated = depths[shift:] + depths[:shift]
    mapping = dict(zip(names, rotated, strict=True))
    return {
        "mapping": mapping,
        "mapping_sha256": canonical_json_hash(mapping),
        "nonidentity": True,
        "bijection": True,
    }


def _data_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": f"validation-{index}",
            "content_sha256": str(index + 5) * 64,
            "token_count": 2000 + index * 1000,
            "assistant_token_count": index + 2,
        }
        for index in range(4)
    ]


def _make_fixture(root: Path) -> EvidenceFixture:
    results = root / "results" / "phase_c"
    cache = root / "cache" / "phase_c" / "gradients"
    results.mkdir(parents=True)
    cache.mkdir(parents=True)
    dev_rows = _data_rows()
    model = root / "model"
    model.mkdir()
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        (model / name).write_text(f"fixture-{name}", encoding="utf-8")
    tokenizer_sha = sha256_file(model / "tokenizer.json")

    split_path = results / "split.json"
    split = {
        "schema_version": "1.0",
        "one_step_dev": {
            "components": [{"row_ids": [row["row_id"] for row in dev_rows]}],
            "source": "validation",
        },
        "sealed": {
            "source": "test",
            "read_during_split_build": False,
            "policy": "sealed",
        },
        "sealed_test_read": False,
    }
    _seal(split, "split_manifest_sha256")
    _write(split_path, split)

    sample_path = results / "sample.json"
    sample = {
        "schema_version": "1.0",
        "sections": {
            "meta_train": [{"row_id": "train-meta"}],
            "mask_cal": [{"row_id": "train-mask"}],
            "one_step_dev": dev_rows,
        },
        "split_manifest": {
            "path": str(split_path.resolve()),
            "sha256": sha256_file(split_path),
        },
        "tokenizer_sha256": tokenizer_sha,
        "maximum_tokens": 40960,
        "sealed_test_read": False,
    }
    _seal(sample, "sample_manifest_sha256")
    _write(sample_path, sample)

    budget_path = results / "budget.json"
    calibration = {
        "meta_train": [{"row_id": "train-meta"}],
        "mask_cal": [{"row_id": "train-mask"}],
    }
    budget = {
        "schema_version": "2.0",
        "base_model": {"weight_sha256": BASE, "config_sha256": CONFIG},
        "target_parameter": TARGET,
        "sample_manifest": {
            "path": str(sample_path.resolve()),
            "sha256": sha256_file(sample_path),
        },
        "calibration": calibration,
        "generator": {
            "bootstrap_steps": 0,
            "calibration_steps": 20,
            "calibration_learning_rate": 0.01,
            "descriptor_dimension": 24,
            "stabilizer_bounds": [0.25, 1.0],
            "structural_proxy_learning_rate": 0.0001,
            "structural_regularizer_primary": 0.0,
            "topk_ste_temperature": 1.0,
        },
        "native_optimizer": _optimizer(),
        "advancement_gate": {
            "layer_permutation_test_required": True,
            "paired_bootstrap_ci_must_exclude_zero": True,
            "pilot_can_advance": False,
            "real_over_shuffled_seed_wins": "at_least_4_of_5",
            "requires_complete_qualification": True,
        },
        "qualification": {
            "seeds": list(CANONICAL_SEEDS),
            "budget_fractions": list(CANONICAL_BUDGETS),
            "arms": list(CANONICAL_ARMS),
            "one_step_dev": dev_rows,
        },
        "pilot": {"one_step_dev": [dev_rows[0]]},
        "sealed_test_read": False,
    }
    _seal(budget, "budget_manifest_sha256")
    _write(budget_path, budget)

    cache_records: dict[str, dict[str, Any]] = {}
    for section in ("meta_train", "mask_cal"):
        tensor_path = cache / f"{section}.safetensors"
        tensor_path.write_bytes(f"{section}-gradient".encode())
        row_id = calibration[section][0]["row_id"]
        report = {
            "target": TARGET,
            "examples": [{"row_id": row_id, "loss": 1.0}],
            "mean_gradient_l2": 1.0,
        }
        manifest = {
            "schema_version": "1.0",
            "name": section,
            "target": TARGET,
            "model": {"weight_sha256": BASE, "config_sha256": CONFIG},
            "tensor": {
                "sha256": sha256_file(tensor_path),
                "shape": TARGET_SHAPE,
                "dtype": "torch.float32",
            },
            "report": report,
        }
        _seal(manifest, "cache_sha256")
        manifest_path = cache / f"{section}.json"
        _write(manifest_path, manifest)
        cache_records[section] = {
            "manifest_sha256": sha256_file(manifest_path),
            "tensor_sha256": sha256_file(tensor_path),
            "manifest_self_sha256": manifest["cache_sha256"],
            "row_ids": [row_id],
        }

    summary_path = results / "gradient_summary.json"
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "base_weight_sha256": BASE,
        "budget_manifest_sha256": sha256_file(budget_path),
        "target": TARGET,
        "native_weight_delta": "bitwise_zero",
        "computed_sections": ["meta_train", "mask_cal"],
        "resumed_sections": [],
        "cache": {},
        "sealed_test_read": False,
    }
    for section in ("meta_train", "mask_cal"):
        manifest = _read(cache / f"{section}.json")
        summary["cache"][section] = {
            "manifest_sha256": manifest["cache_sha256"],
            "tensor_sha256": manifest["tensor"]["sha256"],
            "report": manifest["report"],
        }
    _seal(summary, "summary_sha256")
    _write(summary_path, summary)

    prereg_path = results / "PREREGISTRATION.json"
    prereg = {
        "schema_version": "2.0",
        "analysis_class": "PRE_SPECIFIED_CONFIRMATORY",
        "root_of_trust": "committed immutable preregistration artifact",
        "model": {
            "weight_sha256": BASE,
            "config_sha256": CONFIG,
            "tokenizer_sha256": tokenizer_sha,
            "tokenizer_artifacts": {
                name: _file_record(model / name)
                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                )
            },
        },
        "budget": _file_record(budget_path, str(budget["budget_manifest_sha256"])),
        "sample_manifest": _file_record(
            sample_path, str(sample["sample_manifest_sha256"])
        ),
        "split_manifest": _file_record(split_path, str(split["split_manifest_sha256"])),
        "gradient_summary": _file_record(summary_path, str(summary["summary_sha256"])),
        "gradient_cache": {},
        "target": {
            "parameter_name": TARGET,
            "semantic_role": "mlp.down_proj",
            "shape": TARGET_SHAPE,
            "numel": TARGET_NUMEL,
        },
        "seeds": list(CANONICAL_SEEDS),
        "budget_fractions": list(CANONICAL_BUDGETS),
        "arms": list(CANONICAL_ARMS),
        "exact_k": {
            str(fraction): budget_to_k(TARGET_NUMEL, fraction)
            for fraction in CANONICAL_BUDGETS
        },
        "expected_run_cells": _cells(CANONICAL_ARMS),
        "expected_run_cell_count": 200,
        "expected_plan_cells": _cells(CANONICAL_ARMS[1:]),
        "expected_plan_cell_count": 175,
        "one_step_dev": dev_rows,
        "maximum_tokens": 40960,
        "encoded_one_step_dev": [
            {
                "row_id": row["row_id"],
                "input_ids_sha256": canonical_json_hash([index, row["token_count"]]),
                "labels_sha256": canonical_json_hash(
                    [index, row["assistant_token_count"]]
                ),
                "token_count": row["token_count"],
                "assistant_token_count": row["assistant_token_count"],
            }
            for index, row in enumerate(dev_rows)
        ],
        "layer_depth_permutations": {
            str(seed): _layer_permutation(seed) for seed in CANONICAL_SEEDS
        },
        "optimizer": _optimizer(),
        "generator": budget["generator"],
        "advancement_gate": budget["advancement_gate"],
        "access_policy": {
            "selection_reads": ["train", "validation"],
            "sealed_split": "test",
            "sealed_read_permitted": False,
            "one_step_dev_may_influence_plans_or_lr": False,
        },
        "sealed_test_read": False,
    }
    prereg["model"]["tokenizer_artifacts_sha256"] = canonical_json_hash(
        {
            name: record["sha256"]
            for name, record in prereg["model"]["tokenizer_artifacts"].items()
        }
    )
    prereg["encoded_data_signature_sha256"] = canonical_json_hash(
        prereg["encoded_one_step_dev"]
    )
    prereg["encoded_one_step_dev_by_stage"] = {
        "pilot": [prereg["encoded_one_step_dev"][0]],
        "qualification": prereg["encoded_one_step_dev"],
    }
    prereg["encoded_data_signature_sha256_by_stage"] = {
        name: canonical_json_hash(rows)
        for name, rows in prereg["encoded_one_step_dev_by_stage"].items()
    }
    prereg_access_log = [
        {
            "event": "data_access",
            "path": str((root / "corpus" / "data" / "sft" / "validation").resolve()),
            "purpose": "freeze ONE_STEP_DEV token IDs and labels before qualification",
            "sealed": False,
            "split": "validation",
        }
    ]
    prereg["access_log"] = prereg_access_log
    prereg["access_log_sha256"] = canonical_json_hash(prereg_access_log)
    for section in ("meta_train", "mask_cal"):
        manifest_path = cache / f"{section}.json"
        tensor_path = cache / f"{section}.safetensors"
        prereg["gradient_cache"][section] = {
            "manifest": _file_record(manifest_path),
            "manifest_self_sha256": _read(manifest_path)["cache_sha256"],
            "tensor": _file_record(tensor_path),
            "row_ids": cache_records[section]["row_ids"],
            "target": TARGET,
        }
    lab_root = Path(gate_module.__file__).resolve().parents[2]
    source_paths = [
        *sorted((lab_root / "src" / "rineya_pcsm").glob("*.py")),
        lab_root / "scripts" / "freeze_phase_c_preregistration.py",
        lab_root / "scripts" / "build_phase_c_plans.py",
        lab_root / "scripts" / "run_phase_c.py",
        lab_root / "scripts" / "audit_phase_c_gate.py",
        lab_root / "scripts" / "train_phase_d.py",
    ]
    prereg["source_files"] = {
        path.relative_to(lab_root).as_posix(): _file_record(path)
        for path in source_paths
    }
    prereg["source_tree_sha256"] = canonical_json_hash(
        {
            path.relative_to(lab_root).as_posix(): sha256_file(path)
            for path in source_paths
        }
    )
    _seal(prereg, "preregistration_sha256")
    _write(prereg_path, prereg)

    loaded: dict[Path, LoadedPlanEvidence] = {}
    plan_records: list[dict[str, Any]] = []
    plan_lookup: dict[tuple[int, float, str], dict[str, Any]] = {}
    peer_names = [
        f"model.layers.{layer}.mlp.down_proj.weight"
        for layer in range(28)
        if layer != 14
    ]
    generator_config = {
        "version": "pcsm-generator-v1",
        "descriptor_dimension": 24,
        "hidden_dimension": 64,
        "stabilizer_lower": 0.25,
        "stabilizer_upper": 1.0,
    }
    attention_config = {
        "version": "pcsm-peer-attention-v1",
        "descriptor_dimension": 24,
        "residual_hidden": 16,
        "residual_scale": 0.25,
        "temperature": 1.0,
        "depth_penalty": 0.0,
    }
    for seed in CANONICAL_SEEDS:
        for fraction in CANONICAL_BUDGETS:
            k = budget_to_k(TARGET_NUMEL, fraction)
            for arm in CANONICAL_ARMS[1:]:
                directory = (
                    root
                    / "plans"
                    / "qualification"
                    / f"seed-{seed}"
                    / f"budget-{fraction}"
                    / arm
                )
                directory.mkdir(parents=True)
                (directory / "plan.json").write_text(
                    json.dumps({"seed": seed, "fraction": fraction, "arm": arm}),
                    encoding="utf-8",
                )
                (directory / "tensors.safetensors").write_bytes(
                    f"{seed}:{fraction}:{arm}".encode()
                )
                classical = arm in {
                    "random_sparse_native",
                    "gradient_topk_sparse_native",
                }
                target_only = arm == "target_only_sparse_native"
                current_generator_config = (
                    {"version": "classical-control-v1", "method": arm}
                    if classical
                    else generator_config
                )
                current_attention = {} if classical or target_only else attention_config
                generator_count = 0 if classical else 1200
                attention_count = 0 if classical or target_only else 900
                steps = 0 if classical or arm == "real_peer_pcsm_bootstrap" else 20
                metadata: dict[str, Any] = {
                    "seed": seed,
                    "steps": steps,
                    "generator_learning_rate": 0.01,
                    "attention_config": current_attention,
                    "generator_parameter_count": generator_count,
                    "attention_parameter_count": attention_count,
                    "structural_regularizer": 0.0,
                    "structural_proxy_learning_rate": 0.0001,
                    "topk_ste_temperature": 1.0,
                    "support_frozen_once_after_calibration": True,
                    "target_plan_records": {
                        TARGET: {
                            "exact_k": k,
                            "shape": TARGET_SHAPE,
                            "semantic_role": "mlp.down_proj",
                            "requested_fraction": fraction,
                            "generator_config": current_generator_config,
                        }
                    },
                }
                if arm == "layer_shuffled_peer_pcsm":
                    permutation = prereg["layer_depth_permutations"][str(seed)]
                    metadata.update(
                        {
                            "layer_depth_assignment": permutation["mapping"],
                            "layer_depth_assignment_sha256": permutation[
                                "mapping_sha256"
                            ],
                            "layer_depth_assignment_nonidentity": True,
                        }
                    )
                plan_hash = canonical_json_hash(
                    {"kind": "plan", "seed": seed, "fraction": fraction, "arm": arm}
                )
                support_hash = canonical_json_hash(
                    {
                        "kind": "support",
                        "seed": seed,
                        "fraction": fraction,
                        "arm": arm,
                    }
                )
                support_stats = {
                    "per_target_k": {TARGET: k},
                    "total_numel": TARGET_NUMEL,
                    "union_fraction": k / TARGET_NUMEL,
                    "union_k": k,
                }
                attention = (
                    {
                        "peer_names": peer_names,
                        "alphas": [1.0 / len(peer_names)] * len(peer_names),
                        "self_alpha": 0.0,
                    }
                    if not classical and not target_only
                    else {"peer_names": [], "alphas": []}
                )
                record: dict[str, Any] = {
                    "stage": "qualification",
                    "seed": seed,
                    "budget_fraction": fraction,
                    "k": k,
                    "arm": arm,
                    "plan_directory": str(directory.resolve()),
                    "plan_manifest_sha256": sha256_file(directory / "plan.json"),
                    "plan_tensor_sha256": sha256_file(
                        directory / "tensors.safetensors"
                    ),
                    "plan_hash": plan_hash,
                    "support_hash": support_hash,
                    "support_stats": support_stats,
                    "selected_scale_distribution": {"mean": 1.0},
                    "metadata": metadata,
                    "attention": attention,
                    "calibration_trace": [
                        {"step": float(step), "objective": -1.0}
                        for step in range(steps)
                    ],
                    "generator_architecture_sha256": canonical_json_hash(
                        current_generator_config
                    ),
                    "generator_parameter_count": generator_count,
                    "attention_architecture_sha256": canonical_json_hash(
                        current_attention
                    ),
                    "attention_parameter_count": attention_count,
                    "total_generator_capacity": generator_count + attention_count,
                    "meta_compute": {
                        "calibration_steps": steps,
                        "backward_count": steps,
                        "trace_length": steps,
                        "meta_train": cache_records["meta_train"],
                        "mask_cal": cache_records["mask_cal"],
                    },
                    "preregistration_file_sha256": sha256_file(prereg_path),
                }
                if arm == "layer_shuffled_peer_pcsm":
                    record["layer_depth_permutation"] = prereg[
                        "layer_depth_permutations"
                    ][str(seed)]
                _seal(record, "record_sha256")
                plan_records.append(record)
                plan_lookup[(seed, fraction, arm)] = record
                loaded[directory.resolve()] = LoadedPlanEvidence(
                    base_fingerprint=BASE,
                    target_name=TARGET,
                    shape=tuple(TARGET_SHAPE),
                    k=k,
                    support_fraction=fraction,
                    plan_hash=plan_hash,
                    support_hash=support_hash,
                    metadata=metadata,
                    support_stats=support_stats,
                )

    shared_provenance = {
        "preregistration_file_sha256": sha256_file(prereg_path),
        "preregistration_self_sha256": prereg["preregistration_sha256"],
        "budget_file_sha256": sha256_file(budget_path),
        "budget_self_sha256": budget["budget_manifest_sha256"],
        "sample_manifest_sha256": sha256_file(sample_path),
        "split_manifest_sha256": sha256_file(split_path),
        "model_weight_sha256": BASE,
        "model_config_sha256": CONFIG,
        "tokenizer_sha256": tokenizer_sha,
        "gradient_cache": cache_records,
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
    plans_path = results / "qualification_plans.json"
    plans = {
        "schema_version": "2.0",
        "status": "complete",
        "stage": "qualification",
        "base_fingerprint": BASE,
        "provenance": {
            **shared_provenance,
            "target": TARGET,
            "role": "mlp.down_proj",
            "expected_plan_cells_sha256": canonical_json_hash(
                prereg["expected_plan_cells"]
            ),
        },
        "access_log": planner_access_log,
        "access_log_sha256": canonical_json_hash(planner_access_log),
        "records": plan_records,
        "sealed_test_read": False,
    }
    _seal(plans, "result_sha256")
    _write(plans_path, plans)

    optimizer_signature = _optimizer_signature()
    data_signature = {
        "examples": dev_rows,
        "signature_sha256": canonical_json_hash(dev_rows),
    }
    environment_signature = {"python": "fixture", "cuda_available": False}
    environment_signature["signature_sha256"] = canonical_json_hash(
        environment_signature
    )
    run_access_log = [
        {
            "event": "data_access",
            "path": str((root / "corpus" / "data" / "sft" / "validation").resolve()),
            "purpose": "ONE_STEP_DEV held-out loss only",
            "sealed": False,
            "split": "validation",
        },
        {
            "event": "artifact_access",
            "purpose": "fixed plan and calibration provenance",
            "split": "train",
            "paths": [
                str(plans_path.resolve()),
                str(summary_path.resolve()),
                str(cache.resolve()),
            ],
            "sealed": False,
        },
    ]
    run_records: list[dict[str, Any]] = []
    supervised_total = sum(row["assistant_token_count"] for row in dev_rows)
    input_total = sum(row["token_count"] for row in dev_rows)
    for seed in CANONICAL_SEEDS:
        for fraction in CANONICAL_BUDGETS:
            for arm in CANONICAL_ARMS:
                loss = (
                    1.0
                    if arm == "real_peer_pcsm_calibrated"
                    else 2.0
                    if arm
                    in {
                        "coordinate_shuffled_peer_pcsm",
                        "layer_shuffled_peer_pcsm",
                    }
                    else 1.5
                )
                k = (
                    0
                    if arm == "no_update_frozen"
                    else budget_to_k(TARGET_NUMEL, fraction)
                )
                record: dict[str, Any] = {
                    "seed": seed,
                    "budget_fraction": fraction,
                    "arm": arm,
                    "k": k,
                    "evaluation": {
                        "examples": [
                            {
                                "row_id": row["row_id"],
                                "tokens": row["token_count"],
                                "supervised_tokens": row["assistant_token_count"],
                                "loss": loss,
                            }
                            for row in dev_rows
                        ],
                        "mean_nll": loss,
                        "supervised_tokens": supervised_total,
                    },
                    "evaluation_tokens": input_total,
                    "dev_data_signature": data_signature["signature_sha256"],
                    "encoded_data_signature_sha256": prereg[
                        "encoded_data_signature_sha256"
                    ],
                    "maximum_tokens": 40960,
                    "preregistration_file_sha256": sha256_file(prereg_path),
                    "optimizer_config_sha256": optimizer_signature["signature_sha256"],
                    "base_hash_verified": True,
                    "access_log_sha256": canonical_json_hash(run_access_log),
                    "learning_rate_grid_sha256": canonical_json_hash(LR_GRID),
                    "unplanned_parameter_hash_before": UNPLANNED,
                    "unplanned_parameter_hash_after": UNPLANNED,
                    "unplanned_parameter_hash_unchanged": True,
                }
                if arm == "no_update_frozen":
                    record.update(
                        {
                            "learning_rate": 0.0,
                            "optimizer_state_bytes": 0,
                            "delta_audit": {
                                "valid": True,
                                "planned_k": 0,
                                "changed_union_size": 0,
                            },
                            "full_parameter_hash_before": UNPLANNED,
                            "full_parameter_hash_after": UNPLANNED,
                        }
                    )
                else:
                    plan_record = plan_lookup[(seed, fraction, arm)]
                    plan_hash = plan_record["plan_hash"]
                    support_hash = plan_record["support_hash"]
                    record.update(
                        {
                            "plan_hash": plan_hash,
                            "support_hash": support_hash,
                            "plan_record_sha256": plan_record["record_sha256"],
                            "plan_manifest_sha256": plan_record["plan_manifest_sha256"],
                            "plan_tensor_sha256": plan_record["plan_tensor_sha256"],
                            "learning_rate": 0.0002,
                            "meta_lr_predictions": {
                                "5e-05": -0.1,
                                "0.0001": -0.2,
                                "0.0002": -0.3,
                            },
                            "learning_rate_selection_split": "META_TRAIN_only",
                            "learning_rate_source_cache_sha256": cache_records[
                                "meta_train"
                            ]["manifest_sha256"],
                            "optimizer_state_bytes": k * 16,
                            "checkpoint_delta_bytes_estimate": k * 10,
                            "delta_audit": {
                                "valid": True,
                                "planned_k": k,
                                "support_union_size": k,
                                "total_native_coordinates": TARGET_NUMEL,
                                "support_union_fraction": k / TARGET_NUMEL,
                                "changed_union_size": k,
                                "selected_changed_count": k,
                                "outside_support_changed_count": 0,
                                "selected_delta_l2": 1.0,
                                "outside_support_delta_l2": 0.0,
                                "total_delta_l2": 1.0,
                                "selected_delta_max_abs": 0.1,
                                "outside_support_delta_max_abs": 0.0,
                                "plan_hash": plan_hash,
                                "support_hash": support_hash,
                                "leakage_examples": [],
                                "tensors": [
                                    {
                                        "name": TARGET,
                                        "numel": TARGET_NUMEL,
                                        "selected_k": k,
                                        "changed_union_size": k,
                                        "selected_changed_count": k,
                                        "outside_support_changed_count": 0,
                                        "selected_delta_l2": 1.0,
                                        "outside_support_delta_l2": 0.0,
                                        "total_delta_l2": 1.0,
                                        "selected_delta_max_abs": 0.1,
                                        "outside_support_delta_max_abs": 0.0,
                                    }
                                ],
                            },
                        }
                    )
                _seal(record, "record_sha256")
                run_records.append(record)

    run_path = results / "qualification_run.json"
    run = {
        "schema_version": "2.0",
        "status": "complete",
        "stage": "qualification",
        "device": "cpu",
        "base_weight_sha256": BASE,
        "budget_manifest_sha256": sha256_file(budget_path),
        "plan_result_sha256": sha256_file(plans_path),
        "provenance": {
            **shared_provenance,
            "gradient_summary_sha256": sha256_file(summary_path),
            "expected_run_cells_sha256": canonical_json_hash(
                prereg["expected_run_cells"]
            ),
            "optimizer_signature": optimizer_signature,
            "data_signature": data_signature,
            "tokenizer_artifacts": prereg["model"]["tokenizer_artifacts"],
            "tokenizer_artifacts_sha256": prereg["model"]["tokenizer_artifacts_sha256"],
            "encoded_data_signature_sha256": prereg["encoded_data_signature_sha256"],
            "device": "cpu",
            "maximum_tokens": 40960,
            "environment_signature": environment_signature,
        },
        "access_log": run_access_log,
        "access_log_sha256": canonical_json_hash(run_access_log),
        "records": run_records,
        "sealed_test_read": False,
    }
    _seal(run, "run_sha256")
    _write(run_path, run)
    return EvidenceFixture(
        GateInputs(
            run=run_path,
            plans=plans_path,
            preregistration=prereg_path,
            budget=budget_path,
            sample=sample_path,
            split=split_path,
            gradient_summary=summary_path,
            gradient_cache=cache,
        ),
        loaded,
    )


@pytest.fixture
def evidence(tmp_path: Path) -> EvidenceFixture:
    return _make_fixture(tmp_path)


def _reseal_records(path: Path, root_key: str) -> dict[str, Any]:
    value = _read(path)
    for record in value["records"]:
        _seal(record, "record_sha256")
    _seal(value, root_key)
    _write(path, value)
    return value


def test_complete_schema_v2_evidence_passes(evidence: EvidenceFixture) -> None:
    result = evidence.audit()
    assert result["verdict"] == "PASS_ADVANCE_TO_PHASE_D"
    assert result["passed"] is True
    assert gate_self_hash_valid(result)
    assert result["statistics"]["coordinate_shuffled_minus_real"]["wins"] == 5
    assert (
        result["statistics"]["small_n_caveat"][
            "minimum_attainable_two_sided_exact_sign_flip_p_value"
        ]
        == 0.0625
    )


def test_committed_preregistration_identity_survives_later_commits(
    evidence: EvidenceFixture,
) -> None:
    repository = evidence.inputs.run.parents[2]

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "pcsm-test@example.invalid")
    git("config", "user.name", "PCSM Test")
    git("add", str(evidence.inputs.preregistration.relative_to(repository)))
    git("commit", "-m", "freeze preregistration")

    def audit() -> dict[str, Any]:
        return audit_phase_c_gate(
            evidence.inputs,
            plan_evidence_loader=lambda path, _base: evidence.loaded[path.resolve()],
        )

    first = audit()
    assert first["passed"] is True
    unrelated = repository / "later.txt"
    unrelated.write_text("later evidence commit", encoding="utf-8")
    git("add", "later.txt")
    git("commit", "-m", "later evidence")
    second = audit()
    assert second["passed"] is True
    assert second["root_of_trust"] == first["root_of_trust"]


@pytest.mark.parametrize("mode", ["four", "duplicate"])
def test_noncanonical_or_duplicate_seed_cells_stop(
    evidence: EvidenceFixture, mode: str
) -> None:
    run = _read(evidence.inputs.run)
    if mode == "four":
        run["records"] = [record for record in run["records"] if record["seed"] != 31]
    else:
        for record in run["records"]:
            if record["seed"] == 31:
                record["seed"] = 11
    for record in run["records"]:
        _seal(record, "record_sha256")
    _seal(run, "run_sha256")
    _write(evidence.inputs.run, run)
    assert evidence.audit()["passed"] is False


@pytest.mark.parametrize("mode", ["four", "duplicate"])
def test_noncanonical_or_duplicate_budget_cells_stop(
    evidence: EvidenceFixture, mode: str
) -> None:
    run = _read(evidence.inputs.run)
    if mode == "four":
        run["records"] = [
            record
            for record in run["records"]
            if record["budget_fraction"] != CANONICAL_BUDGETS[-1]
        ]
    else:
        for record in run["records"]:
            if record["budget_fraction"] == CANONICAL_BUDGETS[-1]:
                record["budget_fraction"] = CANONICAL_BUDGETS[0]
    for record in run["records"]:
        _seal(record, "record_sha256")
    _seal(run, "run_sha256")
    _write(evidence.inputs.run, run)
    assert evidence.audit()["passed"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "reduced_arm"])
def test_nonexact_run_cells_stop(evidence: EvidenceFixture, mutation: str) -> None:
    run = _read(evidence.inputs.run)
    if mutation == "missing":
        run["records"].pop()
    elif mutation == "duplicate":
        run["records"][-1] = dict(run["records"][0])
    elif mutation == "extra":
        extra = dict(run["records"][0])
        extra["arm"] = "unregistered_arm"
        _seal(extra, "record_sha256")
        run["records"].append(extra)
    else:
        run["records"] = [
            record
            for record in run["records"]
            if record["arm"] != "random_sparse_native"
        ]
    for record in run["records"]:
        _seal(record, "record_sha256")
    _seal(run, "run_sha256")
    _write(evidence.inputs.run, run)
    assert evidence.audit()["passed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "wrong_k",
        "wrong_union",
        "wrong_delta",
        "leakage",
        "wrong_lr",
        "wrong_data",
        "wrong_encoded",
    ],
)
def test_per_cell_adversarial_mutations_stop(
    evidence: EvidenceFixture, field: str
) -> None:
    run = _read(evidence.inputs.run)
    record = next(
        value for value in run["records"] if value["arm"] == "real_peer_pcsm_calibrated"
    )
    if field == "wrong_k":
        record["k"] += 1
    elif field == "wrong_union":
        record["delta_audit"]["support_union_size"] += 1
    elif field == "wrong_delta":
        record["delta_audit"]["outside_support_delta_l2"] = 0.25
    elif field == "leakage":
        record["delta_audit"]["leakage_examples"] = [{"name": TARGET, "flat_index": 7}]
    elif field == "wrong_lr":
        record["meta_lr_predictions"].pop("5e-05")
    elif field == "wrong_data":
        record["evaluation"]["examples"][0]["row_id"] = "wrong-row"
    else:
        record["encoded_data_signature_sha256"] = "f" * 64
    _seal(record, "record_sha256")
    _seal(run, "run_sha256")
    _write(evidence.inputs.run, run)
    assert evidence.audit()["passed"] is False


def test_wrong_plan_file_hash_stops(evidence: EvidenceFixture) -> None:
    plans = _read(evidence.inputs.plans)
    directory = Path(plans["records"][0]["plan_directory"])
    (directory / "plan.json").write_text("tampered", encoding="utf-8")
    assert evidence.audit()["passed"] is False


def test_changed_tokenizer_artifact_stops(evidence: EvidenceFixture) -> None:
    preregistration = _read(evidence.inputs.preregistration)
    tokenizer = Path(
        preregistration["model"]["tokenizer_artifacts"]["tokenizer.json"]["path"]
    )
    tokenizer.write_text("changed-tokenizer", encoding="utf-8")
    assert evidence.audit()["passed"] is False


@pytest.mark.parametrize("mutation", ["missing_cell", "wrong_union"])
def test_plan_shape_or_support_union_mutation_stops(
    evidence: EvidenceFixture, mutation: str
) -> None:
    plans = _read(evidence.inputs.plans)
    if mutation == "missing_cell":
        plans["records"].pop()
    else:
        plans["records"][0]["support_stats"]["union_k"] += 1
    for record in plans["records"]:
        _seal(record, "record_sha256")
    _seal(plans, "result_sha256")
    _write(evidence.inputs.plans, plans)
    assert evidence.audit()["passed"] is False


@pytest.mark.parametrize("field", ["capacity", "cache", "permutation"])
def test_plan_fairness_mutations_stop(evidence: EvidenceFixture, field: str) -> None:
    plans = _read(evidence.inputs.plans)
    if field == "capacity":
        record = next(
            value
            for value in plans["records"]
            if value["arm"] == "coordinate_shuffled_peer_pcsm"
        )
        record["total_generator_capacity"] += 1
    elif field == "cache":
        plans["records"][0]["meta_compute"]["meta_train"]["manifest_sha256"] = "f" * 64
    else:
        record = next(
            value
            for value in plans["records"]
            if value["arm"] == "layer_shuffled_peer_pcsm"
        )
        record["metadata"]["layer_depth_assignment_nonidentity"] = False
    for record in plans["records"]:
        _seal(record, "record_sha256")
    _seal(plans, "result_sha256")
    _write(evidence.inputs.plans, plans)
    assert evidence.audit()["passed"] is False


def test_pilot_path_relabel_attack_stops(evidence: EvidenceFixture) -> None:
    plans = _read(evidence.inputs.plans)
    plans["records"][0]["plan_directory"] = plans["records"][0][
        "plan_directory"
    ].replace("qualification", "pilot")
    _seal(plans["records"][0], "record_sha256")
    _seal(plans, "result_sha256")
    _write(evidence.inputs.plans, plans)
    assert evidence.audit()["passed"] is False


def test_reversed_effect_and_ci_stop(evidence: EvidenceFixture) -> None:
    run = _read(evidence.inputs.run)
    for record in run["records"]:
        if record["arm"] in {
            "coordinate_shuffled_peer_pcsm",
            "layer_shuffled_peer_pcsm",
        }:
            record["evaluation"]["mean_nll"] = 0.5
            for example in record["evaluation"]["examples"]:
                example["loss"] = 0.5
            _seal(record, "record_sha256")
    _seal(run, "run_sha256")
    _write(evidence.inputs.run, run)
    result = evidence.audit()
    assert result["passed"] is False
    assert (
        result["statistics"]["coordinate_shuffled_minus_real"]["confidence_interval"][
            "high"
        ]
        < 0
    )


def test_resealed_forged_gate_cannot_authenticate(
    evidence: EvidenceFixture, tmp_path: Path
) -> None:
    gate = evidence.audit()
    gate["evidence"]["run"]["sha256"] = "f" * 64
    gate["authorization_sha256"] = canonical_json_hash(
        {
            key: gate[key]
            for key in (
                "schema_version",
                "analysis_class",
                "stage",
                "verdict",
                "passed",
                "reasons",
                "canonical_shape",
                "statistics",
                "evidence",
                "root_of_trust",
                "sealed_test_read",
                "pilot_can_advance",
            )
        }
    )
    _seal(gate, "gate_sha256")
    gate_path = tmp_path / "forged_gate.json"
    _write(gate_path, gate)
    assert gate_self_hash_valid(gate)
    with pytest.raises(RuntimeError, match=r"forged|changed"):
        authenticate_phase_d_gate(
            gate_path,
            evidence.inputs,
            plan_evidence_loader=lambda path, _base: evidence.loaded[path.resolve()],
            commit_verifier=lambda _path: {
                "verified": True,
                "commit": "a" * 40,
                "repository_relative_path": ("results/phase_c/PREREGISTRATION.json"),
                "blob_sha256": sha256_file(evidence.inputs.preregistration),
            },
        )
