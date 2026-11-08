#!/usr/bin/env python3
"""Freeze the immutable Phase-C root of trust before qualification outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from transformers import AutoTokenizer

from rineya_pcsm.data import (
    CorpusAccessPolicy,
    encode_chat_example,
    stream_selected_sft_rows,
)
from rineya_pcsm.hashing import canonical_json_hash, sha256_file
from rineya_pcsm.manifest import atomic_write_json, git_state, utc_now
from rineya_pcsm.real_plans import layer_depth_permutation
from rineya_pcsm.topk import budget_to_k


def self_hash_valid(value: dict[str, Any], key: str) -> bool:
    declared = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    return isinstance(declared, str) and declared == canonical_json_hash(payload)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--gradient-summary", type=Path, required=True)
    parser.add_argument("--gradient-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--maximum-tokens", type=int, default=40960)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    budget = json.loads(args.budget.read_text(encoding="utf-8"))
    sample = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    gradient_summary = json.loads(args.gradient_summary.read_text(encoding="utf-8"))
    for value, key, label in (
        (budget, "budget_manifest_sha256", "budget"),
        (sample, "sample_manifest_sha256", "sample"),
        (split, "split_manifest_sha256", "split"),
        (gradient_summary, "summary_sha256", "gradient summary"),
    ):
        if not self_hash_valid(value, key):
            raise RuntimeError(f"{label} self-seal is invalid")
        if label != "split" and value.get("sealed_test_read") is not False:
            raise RuntimeError(f"{label} does not prove sealed_test_read=false")
    budget_file_sha = sha256_file(args.budget)
    sample_file_sha = sha256_file(args.sample_manifest)
    split_file_sha = sha256_file(args.split_manifest)
    if budget.get("sample_manifest", {}).get("sha256") != sample_file_sha:
        raise RuntimeError("Budget is not bound to the supplied sample manifest")
    if sample.get("split_manifest", {}).get("sha256") != split_file_sha:
        raise RuntimeError("Sample manifest is not bound to the supplied split")
    if args.maximum_tokens != int(sample.get("maximum_tokens", -1)):
        raise RuntimeError("Maximum token limit differs from the sample manifest")
    if (
        gradient_summary.get("status") != "complete"
        or gradient_summary.get("budget_manifest_sha256") != budget_file_sha
        or gradient_summary.get("base_weight_sha256")
        != budget["base_model"]["weight_sha256"]
        or gradient_summary.get("target") != budget["target_parameter"]
        or gradient_summary.get("native_weight_delta") != "bitwise_zero"
    ):
        raise RuntimeError("Gradient summary is not bound to the frozen experiment")
    sealed = split.get("sealed")
    if (
        not isinstance(sealed, dict)
        or sealed.get("source") != "test"
        or sealed.get("read_during_split_build") is not False
    ):
        raise RuntimeError("Split manifest does not prove the test split stayed sealed")

    qualification = budget["qualification"]
    seeds = [int(value) for value in qualification["seeds"]]
    fractions = [float(value) for value in qualification["budget_fractions"]]
    arms = [str(value) for value in qualification["arms"]]
    if seeds != [11, 17, 23, 29, 31]:
        raise RuntimeError("Canonical five-seed grid changed")
    if fractions != [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        raise RuntimeError("Canonical five-budget grid changed")
    canonical_arms = [
        "no_update_frozen",
        "random_sparse_native",
        "gradient_topk_sparse_native",
        "target_only_sparse_native",
        "coordinate_shuffled_peer_pcsm",
        "layer_shuffled_peer_pcsm",
        "real_peer_pcsm_bootstrap",
        "real_peer_pcsm_calibrated",
    ]
    if arms != canonical_arms:
        raise RuntimeError("Canonical eight-arm grid changed")

    target_name = str(budget["target_parameter"])
    checkpoint = args.model / "model.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        checkpoint_keys = handle.keys()
        role_names = sorted(
            name for name in checkpoint_keys if name.endswith(".mlp.down_proj.weight")
        )
        if target_name not in role_names:
            raise RuntimeError("Frozen target is absent from the model checkpoint")
        shapes = {tuple(handle.get_slice(name).get_shape()) for name in role_names}
    if len(role_names) != 28 or shapes != {(1024, 3072)}:
        raise RuntimeError(
            "Actual same-role checkpoint inventory differs from Qwen3 v1"
        )
    target_shape = next(iter(shapes))
    target_numel = target_shape[0] * target_shape[1]
    peer_names = [name for name in role_names if name != target_name]
    permutations: dict[str, Any] = {}
    identity = {name: int(name.split(".")[2]) for name in peer_names}
    for seed in seeds:
        mapping = layer_depth_permutation(peer_names, seed=seed)
        depths = sorted(mapping.values())
        if depths != sorted(identity.values()) or mapping == identity:
            raise RuntimeError(f"Invalid layer-depth permutation for seed {seed}")
        permutations[str(seed)] = {
            "mapping": mapping,
            "mapping_sha256": canonical_json_hash(mapping),
            "nonidentity": True,
            "bijection": True,
        }

    cache: dict[str, Any] = {}
    expected_calibration = budget["calibration"]
    for section in ("meta_train", "mask_cal"):
        manifest_path = args.gradient_cache / f"{section}.json"
        tensor_path = args.gradient_cache / f"{section}.safetensors"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self_hash_valid(manifest, "cache_sha256"):
            raise RuntimeError(f"{section} cache self-seal is invalid")
        expected_rows = [row["row_id"] for row in expected_calibration[section]]
        recorded_rows = [row["row_id"] for row in manifest["report"]["examples"]]
        if recorded_rows != expected_rows:
            raise RuntimeError(f"{section} cache rows differ from budget")
        summary_section = gradient_summary.get("cache", {}).get(section, {})
        if (
            summary_section.get("manifest_sha256") != manifest["cache_sha256"]
            or summary_section.get("tensor_sha256") != sha256_file(tensor_path)
            or summary_section.get("report") != manifest["report"]
        ):
            raise RuntimeError(f"Gradient summary contradicts {section} cache")
        cache[section] = {
            "manifest": file_record(manifest_path),
            "manifest_self_sha256": manifest["cache_sha256"],
            "tensor": file_record(tensor_path),
            "row_ids": recorded_rows,
            "target": manifest["target"],
        }

    expected_run_cells = [
        {"seed": seed, "budget_fraction": fraction, "arm": arm}
        for seed in seeds
        for fraction in fractions
        for arm in arms
    ]
    expected_plan_cells = [
        cell for cell in expected_run_cells if cell["arm"] != "no_update_frozen"
    ]
    corpus_root = args.corpus_root.resolve()
    access_policy = CorpusAccessPolicy([corpus_root / "data" / "sft" / "test"])
    validation_root = access_policy.authorize(
        corpus_root / "data" / "sft" / "validation",
        "freeze ONE_STEP_DEV token IDs and labels before qualification",
    )
    stage_dev = {
        stage_name: budget[stage_name]["one_step_dev"]
        for stage_name in ("pilot", "qualification")
    }
    selected_ids = {str(row["row_id"]) for rows in stage_dev.values() for row in rows}
    source_rows = {
        str(row["id"]): row
        for row in stream_selected_sft_rows(
            list(validation_root.glob("*.jsonl")), selected_ids
        )
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded_by_stage: dict[str, list[dict[str, Any]]] = {}
    for stage_name, selected_dev in stage_dev.items():
        encoded_dev: list[dict[str, Any]] = []
        for frozen in selected_dev:
            row_id = str(frozen["row_id"])
            row = source_rows[row_id]
            if canonical_json_hash(row) != frozen["content_sha256"]:
                raise RuntimeError(f"Frozen ONE_STEP_DEV content changed: {row_id}")
            input_ids, labels = encode_chat_example(
                tokenizer,
                row["messages"],
                maximum_tokens=args.maximum_tokens,
                example_id=row_id,
            )
            if len(input_ids) != int(frozen["token_count"]) or sum(
                label != -100 for label in labels
            ) != int(frozen["assistant_token_count"]):
                raise RuntimeError(f"Frozen token counts changed: {row_id}")
            encoded_dev.append(
                {
                    "row_id": row_id,
                    "input_ids_sha256": canonical_json_hash(input_ids),
                    "labels_sha256": canonical_json_hash(labels),
                    "token_count": len(input_ids),
                    "assistant_token_count": sum(label != -100 for label in labels),
                }
            )
        encoded_by_stage[stage_name] = encoded_dev
    tokenizer_artifact_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    tokenizer_artifacts = {
        name: file_record(args.model / name) for name in tokenizer_artifact_names
    }
    lab_root = Path(__file__).resolve().parents[1]
    worktree_root = lab_root.parent
    source_paths = [
        *sorted((lab_root / "src" / "rineya_pcsm").glob("*.py")),
        lab_root / "scripts" / "freeze_phase_c_preregistration.py",
        lab_root / "scripts" / "build_phase_c_plans.py",
        lab_root / "scripts" / "run_phase_c.py",
        lab_root / "scripts" / "audit_phase_c_gate.py",
        lab_root / "scripts" / "train_phase_d.py",
    ]
    artifact: dict[str, Any] = {
        "schema_version": "2.0",
        "created_at_utc": utc_now(),
        "analysis_class": "PRE_SPECIFIED_CONFIRMATORY",
        "root_of_trust": "committed immutable preregistration artifact",
        "git": git_state(worktree_root),
        "model": {
            "path_provenance": str(args.model.resolve()),
            "weight_sha256": sha256_file(args.model / "model.safetensors"),
            "config_sha256": sha256_file(args.model / "config.json"),
            "tokenizer_sha256": sha256_file(args.model / "tokenizer.json"),
            "tokenizer_artifacts": tokenizer_artifacts,
            "tokenizer_artifacts_sha256": canonical_json_hash(
                {name: record["sha256"] for name, record in tokenizer_artifacts.items()}
            ),
        },
        "budget": {
            **file_record(args.budget),
            "self_sha256": budget["budget_manifest_sha256"],
        },
        "sample_manifest": {
            **file_record(args.sample_manifest),
            "self_sha256": sample["sample_manifest_sha256"],
        },
        "split_manifest": {
            **file_record(args.split_manifest),
            "self_sha256": split["split_manifest_sha256"],
        },
        "gradient_summary": {
            **file_record(args.gradient_summary),
            "self_sha256": gradient_summary["summary_sha256"],
        },
        "gradient_cache": cache,
        "target": {
            "parameter_name": target_name,
            "semantic_role": "mlp.down_proj",
            "shape": list(target_shape),
            "numel": target_numel,
        },
        "seeds": seeds,
        "budget_fractions": fractions,
        "exact_k": {
            str(fraction): budget_to_k(target_numel, fraction) for fraction in fractions
        },
        "arms": arms,
        "expected_run_cells": expected_run_cells,
        "expected_run_cell_count": 200,
        "expected_plan_cells": expected_plan_cells,
        "expected_plan_cell_count": 175,
        "one_step_dev": qualification["one_step_dev"],
        "maximum_tokens": args.maximum_tokens,
        "encoded_one_step_dev": encoded_by_stage["qualification"],
        "encoded_data_signature_sha256": canonical_json_hash(
            encoded_by_stage["qualification"]
        ),
        "encoded_one_step_dev_by_stage": encoded_by_stage,
        "encoded_data_signature_sha256_by_stage": {
            stage_name: canonical_json_hash(rows)
            for stage_name, rows in encoded_by_stage.items()
        },
        "layer_depth_permutations": permutations,
        "optimizer": budget["native_optimizer"],
        "generator": budget["generator"],
        "advancement_gate": budget["advancement_gate"],
        "access_policy": {
            "selection_reads": ["train", "validation"],
            "sealed_split": "test",
            "sealed_read_permitted": False,
            "one_step_dev_may_influence_plans_or_lr": False,
        },
        "access_log": [
            {**record, "split": "validation"} for record in access_policy.access_log
        ],
        "source_files": {
            str(path.relative_to(lab_root)).replace("\\", "/"): file_record(path)
            for path in source_paths
        },
        "source_tree_sha256": canonical_json_hash(
            {
                str(path.relative_to(lab_root)).replace("\\", "/"): sha256_file(path)
                for path in source_paths
            }
        ),
        "sealed_test_read": False,
    }
    if artifact["model"]["weight_sha256"] != budget["base_model"]["weight_sha256"]:
        raise RuntimeError("Budget/model weight identity mismatch")
    if artifact["model"]["config_sha256"] != budget["base_model"]["config_sha256"]:
        raise RuntimeError("Budget/model config identity mismatch")
    if artifact["model"]["tokenizer_sha256"] != sample["tokenizer_sha256"]:
        raise RuntimeError("Sample/model tokenizer identity mismatch")
    artifact["access_log_sha256"] = canonical_json_hash(artifact["access_log"])
    artifact["preregistration_sha256"] = canonical_json_hash(artifact)
    atomic_write_json(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "preregistration_sha256": artifact["preregistration_sha256"],
                "run_cells": len(expected_run_cells),
                "plan_cells": len(expected_plan_cells),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
