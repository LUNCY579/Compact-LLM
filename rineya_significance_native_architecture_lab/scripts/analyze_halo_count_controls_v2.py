#!/usr/bin/env python3
"""Audit and analyze clean eligible-heavy-pool Halo count controls v2."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.evaluation.clustered import (  # noqa: E402
    clustered_paired_analysis,
)
from rineya_significance_lab.evaluation.count_causal import holm_adjust  # noqa: E402
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402

REFERENCE = "true_realized_count"
V2_CONDITIONS = (
    REFERENCE,
    "eligible_shuffled_count_v2",
    "eligible_circular_count_v2",
)
INTERVENTIONS = V2_CONDITIONS[1:]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_npz(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _filtered_npz_equal(
    left: Path,
    right: Path,
    *,
    suffixes: tuple[str, ...] | None = None,
) -> bool:
    left_arrays = _load_npz(left)
    right_arrays = _load_npz(right)
    names = sorted(
        name
        for name in left_arrays
        if suffixes is None or name.endswith(suffixes)
    )
    if names != sorted(
        name
        for name in right_arrays
        if suffixes is None or name.endswith(suffixes)
    ):
        return False
    return bool(names) and all(
        np.array_equal(left_arrays[name], right_arrays[name]) for name in names
    )


def cross_run_reference_audit(
    current_rows: Sequence[Mapping[str, Any]],
    *,
    current_root: Path,
    prior_root: Path,
) -> dict[str, Any]:
    """Quantify separate-process reference reproducibility without explaining its cause."""

    prior_rows = _jsonl(prior_root / "cases.jsonl")
    current = {
        str(row["case_id"]): row
        for row in current_rows
        if row.get("condition") == REFERENCE
    }
    prior = {
        str(row["case_id"]): row
        for row in prior_rows
        if row.get("condition") == REFERENCE
    }
    cases = sorted(set(current).intersection(prior))
    if not cases:
        raise RuntimeError("v1/v2 reference audit has no common cases")
    checks = {
        "prediction_exact": 0,
        "correctness_exact": 0,
        "primary_endpoint_exact": 0,
        "final_resident_positions_exact": 0,
        "boundary_selection_trajectory_exact": 0,
        "true_count_trajectory_exact": 0,
        "score_trajectory_exact": 0,
    }
    for case_id in cases:
        left = prior[case_id]
        right = current[case_id]
        checks["prediction_exact"] += int(left["prediction"] == right["prediction"])
        checks["correctness_exact"] += int(bool(left["correct"]) == bool(right["correct"]))
        checks["primary_endpoint_exact"] += int(
            float(left["evidence"]["layer_fact_exact_completion_rate"])
            == float(right["evidence"]["layer_fact_exact_completion_rate"])
        )
        checks["final_resident_positions_exact"] += int(
            _filtered_npz_equal(
                prior_root / str(left["trace_ids"][0]),
                current_root / str(right["trace_ids"][0]),
            )
        )
        for name, suffixes in (
            ("boundary_selection_trajectory_exact", ("_selected", "_retained_positions")),
            ("true_count_trajectory_exact", ("_true_counts",)),
            ("score_trajectory_exact", ("_scores",)),
        ):
            checks[name] += int(
                _filtered_npz_equal(
                    prior_root / str(left["trace_ids"][1]),
                    current_root / str(right["trace_ids"][1]),
                    suffixes=suffixes,
                )
            )
    exact = all(value == len(cases) for value in checks.values())
    return {
        "status": "EXACT" if exact else "NOT_EXACT_ACROSS_SEPARATE_PROCESSES",
        "common_cases": len(cases),
        "exact_case_counts": checks,
        "prior_run": prior_root.name,
        "current_run": current_root.name,
        "prior_cases_sha256": sha256_file(prior_root / "cases.jsonl"),
        "interpretation": (
            "Separate-process CPU-bfloat16/SDPA reference traces are not bitwise reproducible. "
            "Cause is not established; paired v2 contrasts remain within one model process."
        ),
    }


def audit_v2_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_root: Path,
    sink_tokens: int = 4,
    recent_tokens: int = 32,
) -> dict[str, Any]:
    """Prove the intervention contract from every persisted boundary tensor."""

    if sink_tokens != 4 or recent_tokens != 32:
        raise ValueError("v2 audit freezes exactly four sink and 32 recent entries")
    layer_head_cells = 0
    intervention_files = 0
    eligible_values = 0
    protected_values = 0
    for row in rows:
        condition = str(row.get("condition", ""))
        if condition not in V2_CONDITIONS:
            raise ValueError(f"unexpected v2 condition: {condition!r}")
        arrays = _load_npz(artifact_root / str(row["boundary_diagnostics"]))
        for name in sorted(arrays):
            if not name.endswith("_true_counts"):
                continue
            prefix = name.removesuffix("_true_counts")
            required = {
                "true": f"{prefix}_true_counts",
                "effective": f"{prefix}_effective_counts",
                "destination": f"{prefix}_entry_ids",
                "source": f"{prefix}_effective_count_source_entry_ids",
                "eligible": f"{prefix}_eligible_heavy_mask",
            }
            missing = sorted(set(required.values()).difference(arrays))
            if missing:
                raise RuntimeError(f"{condition}: diagnostic lacks v2 arrays: {missing}")
            true = arrays[required["true"]]
            effective = arrays[required["effective"]]
            destination = arrays[required["destination"]]
            source = arrays[required["source"]]
            eligible = arrays[required["eligible"]].astype(bool)
            if not (
                true.shape
                == effective.shape
                == destination.shape
                == source.shape
                == eligible.shape
            ):
                raise RuntimeError(f"{condition}: v2 diagnostic shapes differ at {prefix}")
            expected_mask = np.zeros(true.shape[-1], dtype=bool)
            expected_mask[sink_tokens : true.shape[-1] - recent_tokens] = True
            if not np.array_equal(
                eligible,
                np.broadcast_to(expected_mask, eligible.shape),
            ):
                raise RuntimeError(f"{condition}: eligible mask crosses protection at {prefix}")
            protected = ~eligible
            if not np.array_equal(effective[protected], true[protected]):
                raise RuntimeError(f"{condition}: protected C changed at {prefix}")
            if not np.array_equal(source[protected], destination[protected]):
                raise RuntimeError(f"{condition}: protected identity changed at {prefix}")
            for batch in range(true.shape[0]):
                for head in range(true.shape[1]):
                    pool = eligible[batch, head]
                    if not np.array_equal(
                        np.sort(effective[batch, head, pool]),
                        np.sort(true[batch, head, pool]),
                    ):
                        raise RuntimeError(
                            f"{condition}: eligible C multiset changed at {prefix}/h{head}"
                        )
                    if not np.array_equal(
                        np.sort(source[batch, head, pool]),
                        np.sort(destination[batch, head, pool]),
                    ):
                        raise RuntimeError(
                            f"{condition}: eligible identity multiset changed at {prefix}/h{head}"
                        )
                    if condition == REFERENCE:
                        if not np.array_equal(
                            source[batch, head, pool], destination[batch, head, pool]
                        ) or not np.array_equal(
                            effective[batch, head, pool], true[batch, head, pool]
                        ):
                            raise RuntimeError(f"true C path changed at {prefix}/h{head}")
                    elif np.any(source[batch, head, pool] == destination[batch, head, pool]):
                        raise RuntimeError(
                            f"{condition}: semantic fixed point at {prefix}/h{head}"
                        )
                    layer_head_cells += 1
                    eligible_values += int(pool.sum())
                    protected_values += int((~pool).sum())
            if condition != REFERENCE:
                intervention_files += 1
    return {
        "status": "PASS",
        "protected_sink_tokens": sink_tokens,
        "protected_recent_tokens": recent_tokens,
        "boundary_layer_tensors_per_intervention": intervention_files // len(INTERVENTIONS),
        "audited_layer_head_cells": layer_head_cells,
        "audited_eligible_values": eligible_values,
        "audited_protected_values": protected_values,
        "protected_denominators_immutable": True,
        "eligible_count_multiset_preserved_per_layer_head": True,
        "eligible_identity_multiset_preserved_per_layer_head": True,
        "intervention_semantic_fixed_points": 0,
        "standard_true_path_exact": True,
    }


def _validate_paired_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[tuple[str, str], Mapping[str, Any]]]:
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("case_id", "")), str(row.get("condition", "")))
        if not key[0] or key[1] not in V2_CONDITIONS or key in by_key:
            raise ValueError(f"invalid or duplicate v2 cell: {key}")
        by_key[key] = row
    cases = sorted({case_id for case_id, _ in by_key})
    for case_id in cases:
        observed = {condition for observed_case, condition in by_key if observed_case == case_id}
        if observed != set(V2_CONDITIONS):
            raise ValueError(f"{case_id}: incomplete v2 condition set {sorted(observed)}")
    if len(cases) < 8:
        raise ValueError("v2 analysis requires at least eight independent clusters")
    return cases, by_key


def analyze(run_dir: Path) -> dict[str, Any]:
    run = run_dir.resolve(strict=True)
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE":
        raise RuntimeError("v2 inference/evaluation must be complete before analysis")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment", {}).get("count_control_contract") != "eligible_heavy_v2":
        raise RuntimeError("run manifest is not an eligible-heavy v2 contract")
    cases_rows = _jsonl(run / "cases.jsonl")
    unscored = _jsonl(run / "predictions.unscored.jsonl")
    cases, by_key = _validate_paired_rows(cases_rows)
    audit = audit_v2_diagnostics(unscored, artifact_root=run)
    prior_run = run.parent / "halo_count_causal_controls_v1"
    reference_reproducibility = cross_run_reference_audit(
        cases_rows,
        current_root=run,
        prior_root=prior_run.resolve(strict=True),
    )
    identity_path = run / "within_process_true_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("status") != "PASS" or not all(identity.get("checks", {}).values()):
        raise RuntimeError("same-process real-Halo true-count identity replay did not pass")

    metrics = {
        "layer_fact_exact_completion_rate": lambda row: float(
            row["evidence"]["layer_fact_exact_completion_rate"]
        ),
        "minimum_per_layer_required_token_availability_fraction": lambda row: float(
            row["evidence"]["minimum_per_layer_required_token_availability_fraction"]
        ),
        "answer_correct": lambda row: float(bool(row["correct"])),
    }
    comparisons: dict[str, Any] = {}
    primary_p_values: dict[str, float] = {}
    for condition_index, condition in enumerate(INTERVENTIONS):
        paired: dict[str, Any] = {}
        for metric_index, (metric_name, getter) in enumerate(metrics.items()):
            baseline = [getter(by_key[(case_id, REFERENCE)]) for case_id in cases]
            candidate = [getter(by_key[(case_id, condition)]) for case_id in cases]
            result = clustered_paired_analysis(
                cases,
                baseline,
                candidate,
                minimum_clusters=8,
                bootstrap_samples=10_000,
                randomization_samples=100_000,
                seed=23 + 100 * condition_index + metric_index,
            )
            paired[metric_name] = {
                **asdict(result),
                "wins": sum(right > left for left, right in zip(baseline, candidate, strict=True)),
                "losses": sum(right < left for left, right in zip(baseline, candidate, strict=True)),
                "ties": sum(right == left for left, right in zip(baseline, candidate, strict=True)),
            }
            if metric_name == "layer_fact_exact_completion_rate":
                primary_p_values[condition] = result.p_value_two_sided
        comparisons[condition] = {"paired_metrics": paired}
    for condition, adjusted in holm_adjust(primary_p_values).items():
        comparisons[condition]["paired_metrics"]["layer_fact_exact_completion_rate"][
            "holm_adjusted_p_value_across_two_clean_controls"
        ] = adjusted

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_ELIGIBLE_POOL_COUNT_CONTROLS_V2",
        "reference_condition": REFERENCE,
        "independent_clusters": len(cases),
        "conditions": list(V2_CONDITIONS),
        "runtime_invariant_audit": audit,
        "within_process_true_count_identity": {
            "status": identity["status"],
            "case_id": identity["case_id"],
            "checks": identity["checks"],
            "artifact": identity_path.name,
            "sha256": sha256_file(identity_path),
        },
        "cross_run_reference_reproducibility": reference_reproducibility,
        "comparisons": comparisons,
        "primary_multiplicity_family": list(INTERVENTIONS),
        "prior_v1_global_controls_status": (
            "GLOBALLY_DISTRIBUTION_PRESERVING_BUT_ELIGIBLE_POOL_CONFOUNDED"
        ),
        "claim_limit": (
            "Exploratory B0-selected calibration controls; clean identity interventions, "
            "but not registered confirmatory or population evidence."
        ),
        "source": {
            "cases_sha256": sha256_file(run / "cases.jsonl"),
            "unscored_predictions_sha256": sha256_file(
                run / "predictions.unscored.jsonl"
            ),
        },
    }
    atomic_write_json(run / "statistics.json", result)
    manifest.setdefault("artifacts", {})["statistics"] = "statistics.json"
    manifest["artifacts"]["within_process_true_identity"] = identity_path.name
    audit_note = (
        "Persisted v2 diagnostics pass protected-slot immutability, eligible-pool multiset, and semantic-derangement audits."
    )
    if audit_note not in manifest.setdefault("notes", []):
        manifest["notes"].append(audit_note)
    reproducibility_note = (
        "The same-process true-count identity replay is exact, but v1/v2 reference traces from "
        "separate CPU-bfloat16/SDPA processes are not bitwise reproducible; paired v2 inference "
        "therefore remains process-local and exploratory."
    )
    if reproducibility_note not in manifest["notes"]:
        manifest["notes"].append(reproducibility_note)
    atomic_write_json(run / "manifest.json", manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=LAB_ROOT / "results" / "runs" / "halo_count_causal_controls_v2",
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
