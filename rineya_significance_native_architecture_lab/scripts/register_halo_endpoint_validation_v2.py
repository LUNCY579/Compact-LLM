#!/usr/bin/env python3
"""Freeze prospective Halo endpoint-validation v2 before compression."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.manifest import atomic_write_json, sha256_file, utc_now  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def register(config_path: Path, registration_path: Path, protocol_path: Path) -> dict[str, Any]:
    import yaml

    dataset = LAB_ROOT / "results" / "benchmarks" / "halo_endpoint_validation_v2"
    capability = LAB_ROOT / "results" / "runs" / "halo_endpoint_validation_v2_capability"
    output = LAB_ROOT / "results" / "runs" / "halo_endpoint_validation_v2_registered"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("compressed endpoint output exists before registration")
    summary = json.loads((capability / "summary.json").read_text(encoding="utf-8"))
    if not bool(summary["advance_gate_passed"]):
        raise RuntimeError("cannot register endpoint validation after a failed B0 gate")
    by_family: defaultdict[str, list[str]] = defaultdict(list)
    for row in _jsonl(capability / "cases.jsonl"):
        if bool(row["correct"]):
            by_family[str(row["task_family"])].append(str(row["case_id"]))
    families = ("R1", "R3", "R4")
    per_family = 4
    selected: list[str] = []
    selected_by_family: dict[str, list[str]] = {}
    for family in families:
        values = sorted(by_family[family])
        if len(values) < per_family:
            raise RuntimeError(f"family {family} has fewer than {per_family} eligible clusters")
        selected_by_family[family] = values[:per_family]
        selected.extend(values[:per_family])
    selected = sorted(selected)
    config: dict[str, Any] = {
        "schema_version": "1.0",
        "registration": {
            "id": "halo-endpoint-validation-v2-registered",
            "status": "REGISTERED_NOT_RUN",
            "registered_at": utc_now(),
            "amendment_rule": "any_change_requires_new_registration_id_and_version",
        },
        "experiment": {
            "name": "halo_prospective_endpoint_validation_v2_registered",
            "compute_tier": "B",
            "status": "NOT_RUN",
            "classification": "PROSPECTIVE_REGISTERED_LOCAL_VALIDATION",
        },
        "protocol": {
            "execution_mode": "closed_loop_model_execution",
            "answer_key_access": "after_all_inference_predictions_are_durable",
            "endpoint_definition_access": "frozen_before_this_dataset_compression",
        },
        "paths": {
            "base": "repository_root",
            "model": "Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze",
            "inference": "rineya_significance_native_architecture_lab/results/benchmarks/halo_endpoint_validation_v2/inference.jsonl",
            "evaluation": "rineya_significance_native_architecture_lab/results/benchmarks/halo_endpoint_validation_v2/evaluation.jsonl",
            "capability_cases": "rineya_significance_native_architecture_lab/results/runs/halo_endpoint_validation_v2_capability/cases.jsonl",
            "output_root": "rineya_significance_native_architecture_lab/results/runs/halo_endpoint_validation_v2_registered",
        },
        "model": {
            "core_id": "rineya.core.halo.v1.2.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-v2-1e4",
            "expected_hashes": {
                "weights": "38fb96fe286776919032c4418eef0a2534e2c2c2865c1465cc9dd4cfe7e36544",
                "tokenizer": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
                "config": "2284cc1687a4f4f6e3a64a13dfcf7562e592b951afa2f91244cc7b212c9cfb85",
            },
        },
        "selection": {
            "rule": "lexicographically_first_four_B0_correct_case_ids_per_family",
            "families": list(families),
            "per_family": per_family,
            "selected_case_ids": selected,
            "selected_case_ids_by_family": selected_by_family,
            "independent_clusters": len(selected),
        },
        "conditions": [
            {"id": "B2", "policy": "significance", "exponent": 0.0},
            {"id": "B3", "policy": "significance", "exponent": 1.0},
            {"id": "B4", "policy": "position", "exponent": 0.0},
        ],
        "endpoint": {
            "version": "layer-fact-retention-v1",
            "primary": "layer_fact_exact_completion_rate",
            "definition_sha256": "31669a6a30fab93f37fdb27ca25f04fb3ccb68024fe7de6057086dca7279348f",
            "primary_contrast": "B3_minus_B2",
            "direction": "two_sided",
        },
        "runtime": {
            "device": "cpu",
            "dtype": "bfloat16",
            "max_new_tokens": 32,
            "seed": 41,
            "deterministic": True,
            "verify_query_rollback": True,
        },
        "query": {
            "mode": "conditioned",
            "max_tokens": 64,
            "probe_schedule": "every_nonfinal_chunk_boundary",
            "access_fairness": "identical_across_B2_B3_B4",
        },
        "cache": {
            "retained_history_budget": 128,
            "total_budget_slots": 128,
            "sink_tokens": 4,
            "recent_tokens": 32,
            "compression_chunk_tokens": 128,
        },
        "significance": {
            "enabled": True,
            "exponent": 1.0,
            "score_pooling": {"enabled": False, "width": 1},
        },
        "analysis": {
            "independence_unit": "cluster_id",
            "minimum_independent_clusters": 12,
            "primary_test": "exact_cluster_sign_flip",
            "confidence": 0.95,
            "bootstrap_samples": 10000,
            "randomization_samples": 100000,
            "secondary_outcomes": [
                "answer_correct",
                "B3_minus_B4_endpoint",
                "strict_all_token_diagnostic",
            ],
            "multiplicity": "one_declared_primary_endpoint_contrast",
        },
        "attention_prior": {"enabled": False},
        "memory_slots": {"enabled": False, "count": 0},
        "controller": {"enabled": False, "type": "none"},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    source_hashes = {
        "dataset_manifest": sha256_file(dataset / "manifest.json"),
        "inference": sha256_file(dataset / "inference.jsonl"),
        "evaluation": sha256_file(dataset / "evaluation.jsonl"),
        "capability_manifest": sha256_file(capability / "manifest.json"),
        "capability_summary": sha256_file(capability / "summary.json"),
        "capability_cases": sha256_file(capability / "cases.jsonl"),
    }
    registration = {
        "schema_version": "1.0",
        "status": "REGISTERED_NOT_RUN",
        "registration_id": config["registration"]["id"],
        "registered_at": config["registration"]["registered_at"],
        "registered_config": config_path.relative_to(LAB_ROOT).as_posix(),
        "registered_config_sha256": sha256_file(config_path),
        "source_hashes": source_hashes,
        "selected_case_ids": selected,
        "selected_case_ids_by_family": selected_by_family,
        "primary_endpoint": config["endpoint"],
        "analysis": config["analysis"],
        "compressed_output_absent_at_registration": not output.exists()
        or not any(output.iterdir()),
        "claim_limit": "Local prospective validation registration; not a publication preregistration.",
    }
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(registration_path, registration)
    protocol_path.write_text(
        "# Registered Halo endpoint validation v2\n\n"
        f"Registration ID: `{registration['registration_id']}`  \n"
        f"Registered at: `{registration['registered_at']}`  \n"
        f"Config SHA-256: `{registration['registered_config_sha256']}`\n\n"
        "Status: **REGISTERED_NOT_RUN** at creation.\n\n"
        "The primary endpoint is `layer_fact_exact_completion_rate`; the sole primary "
        "contrast is B3 minus B2 over 12 independent clusters (four each R1/R3/R4), "
        "using an exact cluster sign-flip test and cluster bootstrap interval. B4 and "
        "answer accuracy are secondary. Any amendment requires a new registration ID.\n",
        encoding="utf-8",
        newline="\n",
    )
    return registration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-output",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_endpoint_validation_v2_registered.yaml",
    )
    parser.add_argument(
        "--registration-output",
        type=Path,
        default=LAB_ROOT / "results" / "registrations" / "halo_endpoint_validation_v2.json",
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=LAB_ROOT / "REGISTERED_ENDPOINT_VALIDATION_V2.md",
    )
    args = parser.parse_args()
    result = register(args.config_output, args.registration_output, args.protocol_output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
