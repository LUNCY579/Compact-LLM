#!/usr/bin/env python3
"""Replay real Halo boundaries under matched adaptive layer/head budgets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.benchmarks import FrozenEvidenceAnnotation  # noqa: E402
from rineya_significance_lab.evaluation import (  # noqa: E402
    clustered_paired_analysis,
    score_observed_boundary_evidence,
    screen_adaptive_budget,
    selected_resident_positions,
    state_from_count_boundary,
)
from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    git_snapshot,
    sha256_file,
    utc_now,
)


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("adaptive-budget config must be a mapping")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _recorded_selection(
    arrays: dict[str, np.ndarray[Any, Any]],
    *,
    boundary_index: int,
    layer_count: int,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    rows = []
    for layer in range(layer_count):
        name = f"b{boundary_index:03d}_l{layer:03d}_selected"
        value = np.asarray(arrays[name], dtype=np.bool_)
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(f"invalid recorded selection shape: {name}={value.shape}")
        rows.append(value[0])
    return np.stack(rows, axis=0)


def run(config_path: Path) -> dict[str, Any]:
    import yaml

    started_at = utc_now()
    config_path = config_path.resolve(strict=True)
    config = _load(config_path)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    count_run = (base / str(paths["count_run"])).resolve(strict=True)
    evaluation_path = (base / str(paths["evaluation"])).resolve(strict=True)
    source_run_manifest = json.loads(
        (count_run / "manifest.json").read_text(encoding="utf-8")
    )
    output = (base / str(paths["output_root"])).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty adaptive-budget output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    source_condition = str(config["source_condition"])
    cases = [
        row
        for row in _jsonl(count_run / "cases.jsonl")
        if str(row["condition"]) == source_condition
    ]
    unscored = {
        (str(row["case_id"]), str(row["condition"])): row
        for row in _jsonl(count_run / "predictions.unscored.jsonl")
    }
    evaluation = {
        str(row["case_id"]): row for row in _jsonl(evaluation_path)
    }
    if len(cases) < int(config["analysis"]["minimum_independent_clusters"]):
        raise RuntimeError("adaptive-budget screen has too few independent clusters")

    allocation = dict(config["allocation"])
    strategies = [str(value) for value in allocation["strategies"]]
    if not strategies or strategies[0] != "uniform":
        raise ValueError("uniform must be the first adaptive-budget strategy")
    expected_global = int(allocation["global_entry_budget"])
    derived_global = (
        int(config["model"]["expected_layers"])
        * int(config["model"]["expected_kv_heads"])
        * int(allocation["average_budget_per_layer_head"])
    )
    if expected_global != derived_global:
        raise ValueError("declared and derived global entry budgets differ")

    rows: list[dict[str, Any]] = []
    allocation_arrays: dict[str, np.ndarray[Any, Any]] = {}
    uniform_parity_checks = 0
    for case in sorted(cases, key=lambda row: str(row["case_id"])):
        case_id = str(case["case_id"])
        raw = unscored[(case_id, source_condition)]
        annotation = FrozenEvidenceAnnotation.from_dict(
            evaluation[case_id]["evidence_annotation"]
        )
        diagnostic_path = count_run / str(raw["boundary_diagnostics"])
        with np.load(diagnostic_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        for summary_value in raw["boundary_summaries"]:
            summary = dict(summary_value)
            boundary = int(summary["boundary_index"])
            state = state_from_count_boundary(
                arrays,
                boundary_index=boundary,
                layer_count=int(summary["layer_count"]),
            )
            recorded = _recorded_selection(
                arrays,
                boundary_index=boundary,
                layer_count=state.layers,
            )
            for strategy_index, strategy in enumerate(strategies):
                screen = screen_adaptive_budget(
                    state,
                    strategy=strategy,  # type: ignore[arg-type]
                    average_budget_per_unit=int(allocation["average_budget_per_layer_head"]),
                    minimum_per_unit=int(allocation["minimum_per_layer_head"]),
                    maximum_per_unit=int(allocation["maximum_per_layer_head"]),
                    sink_tokens=int(allocation["sink_tokens"]),
                    recent_tokens=int(allocation["recent_tokens"]),
                    exponent=float(allocation["exponent"]),
                    seed=int(allocation["seed"]) + 1000 * strategy_index + boundary,
                )
                if int(screen.budget.allocations.sum()) != expected_global:
                    raise RuntimeError("adaptive allocation violated the global entry budget")
                uniform_matches_recorded = None
                if strategy == "uniform":
                    uniform_matches_recorded = bool(
                        np.array_equal(screen.selection.mask, recorded)
                    )
                    if not uniform_matches_recorded:
                        raise RuntimeError(
                            f"uniform replay does not reproduce runtime selection: {case_id}/b{boundary}"
                        )
                    uniform_parity_checks += 1
                boundary_score = score_observed_boundary_evidence(
                    annotation,
                    selected_resident_positions(screen),
                    true_length=int(summary["true_length"]),
                )
                endpoint = (
                    None
                    if boundary_score.evidence is None
                    else float(
                        boundary_score.evidence[
                            "layer_fact_exact_completion_rate"
                        ]
                    )
                )
                key = f"{case_id}__b{boundary:03d}__{strategy}"
                allocation_arrays[key] = screen.budget.allocations.astype(
                    np.int16, copy=True
                )
                rows.append(
                    {
                        "schema_version": "1.0",
                        "case_id": case_id,
                        "cluster_id": case["scenario_id"],
                        "task_family": case["task_family"],
                        "source_condition": source_condition,
                        "boundary_index": boundary,
                        "true_length": summary["true_length"],
                        "strategy": strategy,
                        "global_entry_budget": expected_global,
                        "allocation_minimum": int(screen.budget.allocations.min()),
                        "allocation_maximum": int(screen.budget.allocations.max()),
                        "allocation_mean": float(screen.budget.allocations.mean()),
                        "allocation_std": float(screen.budget.allocations.std()),
                        "allocation_by_layer_mean": screen.budget.allocations.mean(axis=1).tolist(),
                        "allocation_by_kv_head_mean": screen.budget.allocations.mean(axis=0).tolist(),
                        "significance_entropy_mean": float(screen.diagnostics.entropy.mean()),
                        "significance_gini_mean": float(screen.diagnostics.gini.mean()),
                        "significance_top_mass_mean": float(screen.diagnostics.top_mass.mean()),
                        "uniform_matches_recorded_runtime": uniform_matches_recorded,
                        "boundary_evidence_status": boundary_score.status,
                        "fully_observed_fact_count": len(
                            boundary_score.fully_observed_fact_ids
                        ),
                        "layer_fact_exact_completion_rate": endpoint,
                        "execution_mode": "OPEN_LOOP_TRACE_SIMULATION",
                        "support": "ON_POLICY_BOUNDARY_LOCAL",
                    }
                )

    _write_jsonl(output / "rows.jsonl", rows)
    np.savez_compressed(output / "allocations.npz", **allocation_arrays)
    primary_boundary = int(config["analysis"]["primary_boundary"])
    primary = [
        row
        for row in rows
        if int(row["boundary_index"]) == primary_boundary
        and row["layer_fact_exact_completion_rate"] is not None
    ]
    by_key = {
        (str(row["case_id"]), str(row["strategy"])): row for row in primary
    }
    case_ids = sorted({str(row["case_id"]) for row in primary})
    comparisons: dict[str, Any] = {}
    for index, strategy in enumerate(strategies[1:]):
        result = clustered_paired_analysis(
            case_ids,
            [
                float(by_key[(case_id, "uniform")]["layer_fact_exact_completion_rate"])
                for case_id in case_ids
            ],
            [
                float(by_key[(case_id, strategy)]["layer_fact_exact_completion_rate"])
                for case_id in case_ids
            ],
            minimum_clusters=int(config["analysis"]["minimum_independent_clusters"]),
            confidence=float(config["analysis"]["confidence"]),
            bootstrap_samples=int(config["analysis"]["bootstrap_samples"]),
            randomization_samples=int(config["analysis"]["randomization_samples"]),
            seed=int(allocation["seed"]) + index,
        )
        comparisons[strategy] = asdict(result)
    statistics = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "OPEN_LOOP_BOUNDARY_LOCAL_ADAPTIVE_BUDGET_SCREEN",
        "primary_boundary": primary_boundary,
        "reference_strategy": "uniform",
        "comparisons": comparisons,
        "uniform_runtime_parity_checks": uniform_parity_checks,
        "claim_limit": (
            "Boundary-local replay only. Earlier heterogeneous selection would change later "
            "hidden states, and sparse physical-storage overhead is not measured."
        ),
    }
    atomic_write_json(output / "statistics.json", statistics)
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "OPEN_LOOP_BOUNDARY_LOCAL_ADAPTIVE_BUDGET_SCREEN",
        "independent_clusters": len(case_ids),
        "row_count": len(rows),
        "strategies": strategies,
        "global_entry_budget": expected_global,
        "uniform_runtime_parity_checks": uniform_parity_checks,
        "primary_comparisons": comparisons,
        "closed_loop_model_execution": False,
        "answer_quality_measured": False,
    }
    atomic_write_json(output / "summary.json", summary)
    repo = git_snapshot(REPO_ROOT)
    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "status": "COMPLETE",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
        "created_at": started_at,
        "completed_at": utc_now(),
        "repositories": {
            "chatbubble": {
                "remote": repo["remote"],
                "commit": repo["commit"],
                "dirty": repo["dirty"],
            },
            "source": {
                "remote": "https://github.com/LUNCY579/RineyaBeta.git",
                "implementation": {
                    "ref": "refs/heads/main",
                    "commit": "9e00be74098878290dc2b0d54cbffee4652c6088",
                    "path": "scripts/kv/significance.py",
                    "blob": "30ee29724dce52de5a288f6421123fb718240541",
                },
                "paper": {
                    "ref": "refs/heads/r2-healing-harness",
                    "commit": "1664f54de0bbdfde45e311b75bbf3038dbfffe42",
                    "path": "paper/significance_matrix/main.tex",
                    "blob": "b83c7f37d871f1ddb59a00ccf521c89d84b1c3e3",
                },
            },
        },
        "experiment": {
            "name": config["experiment"]["name"],
            "compute_tier": "A",
            "architecture_variant": "B boundary-local adaptive layer/head budgets",
            "benchmark": "halo_count_control_true-C_boundaries",
            "seed": allocation["seed"],
            "query_conditioned": True,
        },
        "config": {
            "resolved_path": "configs/halo_adaptive_budget_screen.yaml",
            "sha256": sha256_file(config_path),
        },
        "model": {
            "status": "HASH_RECORDED",
            "id": config["model"]["core_id"],
            "path": "Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze",
            "architecture": "Qwen3ForCausalLM",
            "hashes": dict(source_run_manifest["model"]["hashes"]),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": None,
            "transformers": None,
            "device": "cpu",
            "precision": "numpy-float32-replay",
            "hardware": None,
            "model_loaded": False,
        },
        "budget": {
            "global_entries": expected_global,
            "matched": True,
            "physical_sparse_storage_measured": False,
        },
        "artifacts": {
            "rows": "rows.jsonl",
            "allocations": "allocations.npz",
            "statistics": "statistics.json",
            "summary": "summary.json",
            "resolved_config": "resolved_config.yaml",
        },
        "input_provenance": {
            "source_run_id": count_run.name,
            "source_run_manifest_sha256": sha256_file(count_run / "manifest.json"),
            "source_content_manifest_sha256": sha256_file(
                count_run / "content_manifest.json"
            ),
            "source_cases_sha256": sha256_file(count_run / "cases.jsonl"),
            "source_predictions_sha256": sha256_file(
                count_run / "predictions.unscored.jsonl"
            ),
            "source_boundary_index_sha256": sha256_file(
                count_run / "boundary_diagnostics.jsonl"
            ),
            "evaluation_sha256": sha256_file(evaluation_path),
        },
        "notes": list(config["claim_limits"]),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_adaptive_budget_screen.yaml",
    )
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
