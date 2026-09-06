#!/usr/bin/env python3
"""Validate, join, and score answer-blind published-family Showcase-64 segments."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SCRIPTS = LAB_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_published_core_eval import (  # noqa: E402
    _atomic_write_json,
    _atomic_write_jsonl,
    _create_output_dir,
    _normalize_answer,
    _repo_path,
    _write_resolved_config,
)
from run_showcase64_eval import read_answer_key, read_showcase_rows  # noqa: E402

from rineya_significance_lab.config import load_config  # noqa: E402
from rineya_significance_lab.manifest import (  # noqa: E402
    build_manifest,
    sha256_file,
    utc_now,
)

FORBIDDEN_RAW_FIELDS = {
    "accepted_answers",
    "answer",
    "answerable",
    "correct",
    "failure_labels",
    "needle_profile",
    "reference_answer",
    "variant",
}
HASH_FILES = {
    "weights": "model.safetensors",
    "tokenizer": "tokenizer.json",
    "config": "config.json",
    "core": "CORE.json",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_sign_p(wins: int, losses: int) -> float:
    """Two-sided exact sign-test p-value, ignoring ties."""

    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_stats(
    rows: list[dict[str, Any]], *, core: str, candidate: str, reference: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["core"] == core]
    by_key = {(row["case_id"], row["condition"]): bool(row["correct"]) for row in selected}
    case_ids = sorted({str(row["case_id"]) for row in selected})
    case_wins = sum(by_key[(case_id, candidate)] and not by_key[(case_id, reference)] for case_id in case_ids)
    case_losses = sum(not by_key[(case_id, candidate)] and by_key[(case_id, reference)] for case_id in case_ids)
    scenario_ids = sorted({str(row["scenario_id"]) for row in selected})
    pack_wins = 0
    pack_losses = 0
    pack_deltas: dict[str, int] = {}
    for scenario_id in scenario_ids:
        candidate_total = sum(
            bool(row["correct"])
            for row in selected
            if row["scenario_id"] == scenario_id and row["condition"] == candidate
        )
        reference_total = sum(
            bool(row["correct"])
            for row in selected
            if row["scenario_id"] == scenario_id and row["condition"] == reference
        )
        delta = candidate_total - reference_total
        pack_deltas[scenario_id] = delta
        pack_wins += delta > 0
        pack_losses += delta < 0
    return {
        "candidate": candidate,
        "reference": reference,
        "case_unit": {
            "wins": case_wins,
            "losses": case_losses,
            "ties": len(case_ids) - case_wins - case_losses,
            "p_value_two_sided": exact_sign_p(case_wins, case_losses),
        },
        "scenario_pack_unit": {
            "wins": pack_wins,
            "losses": pack_losses,
            "ties": len(scenario_ids) - pack_wins - pack_losses,
            "p_value_two_sided": exact_sign_p(pack_wins, pack_losses),
            "correct_count_deltas": pack_deltas,
        },
    }


def _verify_models(config: dict[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for core_key, value in dict(config["cores"]).items():
        entry = dict(value)
        directory = (REPO_ROOT / str(entry["directory"])).resolve(strict=True)
        expected = dict(entry["expected_hashes"])
        observed = {name: sha256_file(directory / filename) for name, filename in HASH_FILES.items()}
        if observed != {name: str(expected[name]).lower() for name in HASH_FILES}:
            raise RuntimeError(f"{core_key}: checkpoint hash contract failed")
        core = json.loads((directory / "CORE.json").read_text(encoding="utf-8"))
        if str(core.get("core_id")) != str(entry["core_id"]):
            raise RuntimeError(f"{core_key}: CORE.json identity mismatch")
        verified[str(core_key)] = {
            "core_id": str(entry["core_id"]),
            "directory": _repo_path(directory),
            "hashes": observed,
        }
    return verified


def run(config_path: Path) -> dict[str, Any]:
    started_at = utc_now()
    config_path = config_path.expanduser().resolve(strict=True)
    config = load_config(config_path)
    final = dict(config["finalization"])
    output = (REPO_ROOT / str(final["output"])).resolve()
    _create_output_dir(output)
    resolved = dict(config)
    resolved["_meta"] = {"source_config": _repo_path(config_path), "output": _repo_path(output)}
    _write_resolved_config(output / "resolved_config.yaml", resolved)

    prompt_path = (REPO_ROOT / str(config["paths"]["prompts"])).resolve(strict=True)
    answer_key_path = (REPO_ROOT / str(config["paths"]["answer_key"])).resolve(strict=True)
    prompt_rows = read_showcase_rows(prompt_path)
    case_ids = tuple(str(row["id"]) for row in prompt_rows)
    scenario_by_case = {str(row["id"]): str(row["scenario_id"]) for row in prompt_rows}
    expected_case_ids = set(case_ids)
    verified_models = _verify_models(config)

    selected: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    source_ledger: dict[str, Any] = {}
    for source in final["sources"]:
        source = dict(source)
        path = (REPO_ROOT / str(source["path"])).resolve(strict=True)
        digest = sha256_file(path)
        expected_digest = source.get("expected_sha256")
        if expected_digest is not None and digest != str(expected_digest).lower():
            raise RuntimeError(f"{source['id']}: frozen source hash mismatch")
        rows = _read_jsonl(path)
        forbidden = sorted({key for row in rows for key in row} & FORBIDDEN_RAW_FIELDS)
        if forbidden:
            raise RuntimeError(f"{source['id']}: raw source exposes answer fields {forbidden}")
        source_ledger[str(source["id"])] = {
            "path": _repo_path(path),
            "sha256": digest,
            "rows": len(rows),
            "answer_fields_absent": True,
            "provenance_limit": source.get("provenance_limit"),
        }
        for destination, target in ((selected, source.get("select", {})), (compare_rows, source.get("compare_only", {}))):
            for core_key, conditions in dict(target).items():
                for row in rows:
                    if str(row.get("core")) == str(core_key) and str(row.get("condition")) in {
                        str(value) for value in conditions
                    }:
                        selected_row = dict(row)
                        if (
                            str(selected_row.get("condition")) == "B0"
                            and "kv_only_bytes" not in selected_row
                        ):
                            selected_row["kv_only_bytes"] = int(
                                selected_row["total_persistent_bytes"]
                            )
                        destination.append(selected_row)

    expected_cores = set(verified_models)
    expected_conditions = {str(row["id"]) for row in config["conditions"]}
    expected_keys = {
        (core, case_id, condition)
        for core in expected_cores
        for case_id in expected_case_ids
        for condition in expected_conditions
    }
    observed_keys = [
        (str(row.get("core")), str(row.get("case_id")), str(row.get("condition")))
        for row in selected
    ]
    if len(observed_keys) != len(set(observed_keys)):
        raise RuntimeError("duplicate selected family cells")
    if set(observed_keys) != expected_keys or len(selected) != int(final["expected_total_cells"]):
        raise RuntimeError("selected family coverage is not exactly four cores x 64 x three arms")
    for row in selected:
        core = str(row["core"])
        case_id = str(row["case_id"])
        if str(row.get("core_id")) != verified_models[core]["core_id"]:
            raise RuntimeError(f"{core}/{case_id}: core identity mismatch")
        if str(row.get("scenario_id")) != scenario_by_case[case_id]:
            raise RuntimeError(f"{core}/{case_id}: scenario mismatch")
        if row.get("status", "OK") != "OK":
            raise RuntimeError(f"{core}/{case_id}: non-OK inference cell")

    selected.sort(key=lambda row: (str(row["core"]), case_ids.index(str(row["case_id"])), str(row["condition"])))
    _atomic_write_jsonl(output / "predictions.unscored.jsonl", selected)

    # Only now is the answer key opened by this finalizer.
    answer_key = read_answer_key(answer_key_path, case_ids)
    scored: list[dict[str, Any]] = []
    for raw in selected:
        row = dict(raw)
        key = answer_key[str(row["case_id"])]
        accepted = {_normalize_answer(str(value)) for value in key["accepted_answers"]}
        row["correct"] = _normalize_answer(str(row["prediction"])) in accepted
        row["accepted_answers"] = list(key["accepted_answers"])
        row["reference_answer"] = key["answer"]
        row["variant"] = key.get("variant")
        row["answerable"] = key.get("answerable")
        row["needle_profile"] = key.get("needle_profile")
        scored.append(row)
    _atomic_write_jsonl(output / "cases.jsonl", scored)

    identity: dict[str, Any] = {}
    compare_index = {
        (str(row["core"]), str(row["case_id"]), str(row["condition"])): str(row["prediction"])
        for row in compare_rows
    }
    selected_index = {
        (str(row["core"]), str(row["case_id"]), str(row["condition"])): str(row["prediction"])
        for row in selected
    }
    for core in ("halo_uncensored", "lyra"):
        differences = [
            case_id
            for case_id in case_ids
            if compare_index.get((core, case_id, "B3")) != selected_index[(core, case_id, "B3")]
        ]
        identity[core] = {
            "compared_cases": len(case_ids),
            "exact_prediction_matches": len(case_ids) - len(differences),
            "different_case_ids": differences,
            "passed": not differences,
        }

    by_core: dict[str, Any] = {}
    comparison_stats: dict[str, Any] = {}
    for core in sorted(expected_cores):
        core_rows = [row for row in scored if row["core"] == core]
        b0_correct = {str(row["case_id"]) for row in core_rows if row["condition"] == "B0" and row["correct"]}
        conditions: dict[str, Any] = {}
        for condition in ("B0", "B2", "B3"):
            rows = [row for row in core_rows if row["condition"] == condition]
            correct = sum(bool(row["correct"]) for row in rows)
            by_profile = {
                profile: {
                    "cases": sum(str(row["needle_profile"]) == profile for row in rows),
                    "correct": sum(
                        str(row["needle_profile"]) == profile and bool(row["correct"])
                        for row in rows
                    ),
                }
                for profile in sorted({str(row["needle_profile"]) for row in rows})
            }
            retained = sum(
                bool(row["correct"]) and str(row["case_id"]) in b0_correct for row in rows
            )
            conditions[condition] = {
                "cases": len(rows),
                "correct": correct,
                "accuracy": correct / len(rows),
                "b0_conditioned_correct": retained if condition != "B0" else None,
                "b0_conditioned_accuracy": (
                    retained / len(b0_correct) if condition != "B0" and b0_correct else None
                ),
                "mean_resident_after_prefill": statistics.fmean(
                    int(row["resident_after_prefill"]) for row in rows
                ),
                "mean_kv_only_bytes": statistics.fmean(int(row.get("kv_only_bytes", 0)) for row in rows),
                "mean_total_persistent_bytes": statistics.fmean(
                    int(row["total_persistent_bytes"]) for row in rows
                ),
                "wall_seconds_sum": sum(float(row["inference_wall_seconds"]) for row in rows),
                "by_needle_profile": by_profile,
            }
        by_core[core] = {"conditions": conditions, "model": verified_models[core]}
        comparison_stats[core] = {
            "B3_minus_B2": paired_stats(scored, core=core, candidate="B3", reference="B2"),
            "B3_minus_B0": paired_stats(scored, core=core, candidate="B3", reference="B0"),
            "B2_minus_B0": paired_stats(scored, core=core, candidate="B2", reference="B0"),
        }

    ignored: dict[str, Any] = {}
    for value in final.get("ignored_incomplete_attempts", []):
        path = REPO_ROOT / str(value)
        if path.is_file():
            ignored[_repo_path(path)] = {"sha256": sha256_file(path), "rows": len(_read_jsonl(path))}

    statistics_value = {
        "schema_version": "1.0",
        "status": "PASS",
        "independence_unit": "scenario_pack",
        "comparisons_by_core": comparison_stats,
        "cross_process_B3_reproduction": identity,
        "cross_core_pooling_performed": False,
    }
    _atomic_write_json(output / "statistics.json", statistics_value)
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "BOUNDED_EXPLORATORY_FAMILY_SEPARATE_IDENTITIES",
        "total_cells": len(scored),
        "cores": sorted(expected_cores),
        "conditions": ["B0", "B2", "B3"],
        "by_core": by_core,
        "cross_process_B3_reproduction": identity,
        "cross_core_pooling_performed": False,
        "answer_key_access": {
            "finalizer_opened_after_aggregate_unscored_artifact": True,
            "researcher_blinding": False,
            "note": "The showcase answer key was already known from earlier Halo work; inference runners did not open it.",
        },
        "claim_limit": (
            "Per-core exploratory B0/B2/B3 evidence only; no pooled family, confirmatory, "
            "deployment, or GPU claim."
        ),
    }
    _atomic_write_json(output / "summary.json", summary)

    manifest = build_manifest(
        repo=REPO_ROOT,
        config=resolved,
        run_id=output.name,
        execution_mode="CLOSED_LOOP_MODEL_EXECUTION",
        model={
            "status": "VERIFIED",
            "id": "published-qwen3-0.6b-secondary-cores-separate",
            "path": "Rineya_Published_Cores/{Halo-Uncensored,Lyra,Nimbus,Pulse}",
            "architecture": str(config["model_geometry"]["expected_architecture"]),
            "hashes": {"weights": None, "tokenizer": None, "config": None, "chat_template": None},
        },
        benchmark={"name": str(config["benchmark"]["name"])},
        architecture_variant="per-core B0/B2/B3 published-family evaluation",
        seed=int(config["runtime"]["seed"]),
        compute_tier="B",
        status="COMPLETE",
        resolved_config_path="resolved_config.yaml",
    )
    manifest["created_at"] = started_at
    manifest["completed_at"] = utc_now()
    manifest["models"] = verified_models
    manifest["input_provenance"] = {
        "prompt_sha256": sha256_file(prompt_path),
        "answer_key_sha256_after_unscored_freeze": sha256_file(answer_key_path),
        "raw_sources": source_ledger,
        "ignored_incomplete_attempts": ignored,
    }
    manifest["data_access"] = summary["answer_key_access"]
    manifest["artifacts"] = {
        "resolved_config": "resolved_config.yaml",
        "predictions_unscored": "predictions.unscored.jsonl",
        "cases": "cases.jsonl",
        "statistics": "statistics.json",
        "summary": "summary.json",
    }
    manifest["notes"] = [
        "Each published core retains a separate identity and separate paired statistics.",
        "B2 and B3 were executed in the same process for every core.",
        "Halo Uncensored and Lyra B0 rows are reused from the immutable legacy part1 raw artifact.",
        "Rineya/R2 and Aegis are excluded.",
    ]
    _atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "cells": len(scored)}, indent=2))
    return {"output": str(output), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "published_family_showcase64_v2_finalize.yaml",
    )
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
