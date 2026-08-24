#!/usr/bin/env python3
"""Run deterministic, Torch-free Cornerstone-1 Tier-A experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from rineya_significance_lab.benchmarks import (  # noqa: E402
    ExposureTraceSpec,
    build_exposure_events,
    exposure_bias_table,
    generate_suite,
    write_cases_jsonl,
    write_exposure_trace,
)
from rineya_significance_lab.core import (  # noqa: E402
    BudgetLedger,
    CacheGeometry,
    CountTransformContext,
    normalized_rank,
    stable_topk,
    transform_counts,
)
from rineya_significance_lab.evaluation import (  # noqa: E402
    selection_metrics,
    summarize_paired_binary,
)
from rineya_significance_lab.manifest import git_snapshot  # noqa: E402
from rineya_significance_lab.policies import (  # noqa: E402
    PolicyContext,
    available_baselines,
    policy_from_baseline,
)
from rineya_significance_lab.traces import replay_trace, validate_trace  # noqa: E402


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    columns = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_ints(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _parse_floats(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed or any(number < 0 or number > 1 for number in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated values in [0,1]")
    return parsed


def _load_config(path: Path) -> dict[str, Any]:
    # Dynamic lookup keeps this standalone script type-checkable without pulling
    # the separately owned config module into the scoped mypy target.
    module = importlib.import_module("rineya_significance_lab.config")
    loader = cast(Callable[[Path], dict[str, Any]], module.load_config)
    return dict(loader(path))


def run(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    """Run Tier A from a repository config and return its manifest/summary.

    This callable intentionally executes no model code. Relative configured
    output paths resolve from the standalone package root, not the process CWD.
    """

    config_path = config_path.expanduser().resolve()
    config = _load_config(config_path)
    runtime = dict(config.get("runtime", {}))
    benchmark = dict(config.get("benchmark", {}))
    paths = dict(config.get("paths", {}))
    base_seed = int(runtime.get("seed", 17))
    seeds = [int(value) for value in benchmark.get("seeds", [base_seed, base_seed + 6, base_seed + 24, base_seed + 56])]
    depths = [
        float(value)
        for value in benchmark.get("evidence_depths", [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    ]
    configured_output = output_root or Path(str(paths.get("output_root", "results/runs/tier_a")))
    if not configured_output.is_absolute():
        configured_output = PACKAGE_ROOT / configured_output
    arguments = [
        "--output",
        str(configured_output),
        "--seeds",
        ",".join(map(str, seeds)),
        "--evidence-depths",
        ",".join(map(str, depths)),
    ]
    exit_code = main(arguments)
    if exit_code != 0:
        raise RuntimeError(f"Tier-A run failed with exit code {exit_code}")
    output = configured_output.expanduser().resolve()
    resolved_path = output / "resolved_config.yaml"
    resolved_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"] = {
        "resolved_path": str(resolved_path),
        "sha256": _sha256(resolved_path),
        "source_path": str(config_path),
    }
    manifest["artifacts"]["resolved_config"] = "resolved_config.yaml"
    _write_json(manifest_path, manifest)
    return {
        "output": str(output),
        "manifest": manifest,
        "summary": json.loads((output / "summary.json").read_text(encoding="utf-8")),
    }


def _late_survival(metrics: Mapping[str, Any]) -> bool:
    return float(metrics.get("all_required_evidence_survival", 0.0)) == 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New run directory")
    parser.add_argument("--seeds", type=_parse_ints, default=[17, 23, 41, 73])
    parser.add_argument(
        "--evidence-depths",
        type=_parse_floats,
        default=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    )
    parser.add_argument("--filler-count", type=int, default=24)
    args = parser.parse_args(argv)

    if args.filler_count < 4:
        parser.error("--filler-count must be at least 4")
    run_id = datetime.now(UTC).strftime("tier_a_%Y%m%dT%H%M%SZ")
    output = (args.output or PACKAGE_ROOT / "results" / run_id).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    statistics_dir = output / "statistics"
    figure_data_dir = output / "figures" / "data"
    traces_dir = output / "traces"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    cases = generate_suite(
        args.seeds,
        args.evidence_depths,
        filler_count=args.filler_count,
    )
    benchmark_cases_path = write_cases_jsonl(output / "benchmark_cases.jsonl", cases)
    cases_path = output / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            row: dict[str, Any] = {
                "schema_version": "1.0",
                "run_id": output.name,
                "case_id": case.case_id,
                "scenario_id": None,
                "condition": "TIER_A_GENERATOR_VALIDATION",
                "task_family": case.family,
                "generator_seed": case.seed,
                "prompt_sha256": _sha256_text(case.prompt()),
                "context_tokens": None,
                "evidence_depth": case.actual_evidence_depth,
                "chunk_size": None,
                "status": "OK",
                "prediction": None,
                "correct": None,
                "full_cache_correct": None,
                "compression_conditioned_eligible": None,
                "evidence": {"required_count": len(case.evidence_segment_ids)},
                "efficiency": None,
                "failure_labels": [],
                "trace_ids": [],
                "error": None,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    family_counts = {family: sum(case.family == family for case in cases) for family in ("R1", "R2", "R3", "R4")}
    case_summary = {
        "status": "PASS",
        "cases": len(cases),
        "families": family_counts,
        "seeds": args.seeds,
        "requested_evidence_depths": args.evidence_depths,
        "solver_verified": len(cases),
    }
    _write_json(statistics_dir / "benchmark_validation.json", case_summary)

    geometry = CacheGeometry(layers=28, kv_heads=8, head_dim=128)
    ledger = BudgetLedger(geometry)
    slot_matched = {
        "without_significance": asdict(ledger.report(256, significance=False)),
        "with_significance": asdict(ledger.report(256, significance=True)),
    }
    baseline_budget, byte_matched_significance = ledger.match_total_bytes(
        256,
        baseline_significance=False,
        candidate_significance=True,
    )
    budget_summary = {
        "geometry": asdict(geometry),
        "slot_matched": slot_matched,
        "total_byte_matched": {
            "without_significance": asdict(baseline_budget),
            "with_significance": asdict(byte_matched_significance),
        },
    }
    _write_json(statistics_dir / "budget_ledger.json", budget_summary)

    round_values = (3, 6, 9)
    trace_validations: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    raw_success: list[bool] = []
    normalized_success: list[bool] = []
    primary_trace: Path | None = None
    for rounds in round_values:
        spec = ExposureTraceSpec(
            case_id=f"tier-a-exposure-bias-r{rounds}",
            rounds=rounds,
            entries_per_round=2,
            layers=2,
            kv_heads=2,
            budget=2,
            seed=17,
        )
        trace_path = traces_dir / spec.case_id
        write_exposure_trace(trace_path, spec)
        trace_validations.append(validate_trace(trace_path))
        if rounds == 6:
            primary_trace = trace_path
            _write_csv(figure_data_dir / "exposure_rank_trajectories.csv", exposure_bias_table(spec))
        final_event = build_exposure_events(spec)[-1]
        state = final_event.observed_state()
        context = PolicyContext(
            budget=spec.budget,
            current_position=final_event.boundary_position,
            chunk_size=spec.entries_per_round,
            seed=spec.seed,
            event_id=final_event.event_id,
            oracle_evidence=final_event.oracle_evidence,
            allow_oracle=True,
            source_semantics_verified=True,
        )
        raw_selection = policy_from_baseline("B2").select(state, context)
        normalized_selection = policy_from_baseline("B3").select(state, context)
        raw_metrics = selection_metrics(state, raw_selection, evidence=final_event.oracle_evidence)
        normalized_metrics = selection_metrics(state, normalized_selection, evidence=final_event.oracle_evidence)
        raw_ok = _late_survival(raw_metrics)
        normalized_ok = _late_survival(normalized_metrics)
        raw_success.append(raw_ok)
        normalized_success.append(normalized_ok)
        round_rows.extend(
            [
                {
                    "rounds": rounds,
                    "policy": "B2_raw_accumulation",
                    "late_high_utility_survives": int(raw_ok),
                    "evidence_recall": raw_metrics["evidence_recall_mean"],
                },
                {
                    "rounds": rounds,
                    "policy": "B3_observation_normalized",
                    "late_high_utility_survives": int(normalized_ok),
                    "evidence_recall": normalized_metrics["evidence_recall_mean"],
                },
            ]
        )
    if primary_trace is None:
        raise AssertionError("primary trace was not generated")
    _write_csv(figure_data_dir / "compression_round_interaction.csv", round_rows)
    _write_json(statistics_dir / "trace_validation.json", {"status": "PASS", "traces": trace_validations})

    primary_spec = ExposureTraceSpec(
        case_id="tier-a-exposure-bias-r6",
        rounds=6,
        entries_per_round=2,
        layers=2,
        kv_heads=2,
        budget=2,
        seed=17,
    )
    final_event = build_exposure_events(primary_spec)[-1]
    state = final_event.observed_state()
    base_context = PolicyContext(
        budget=primary_spec.budget,
        current_position=final_event.boundary_position,
        chunk_size=primary_spec.entries_per_round,
        seed=primary_spec.seed,
        event_id=final_event.event_id,
        oracle_evidence=final_event.oracle_evidence,
        allow_oracle=True,
        source_semantics_verified=True,
    )
    policy_rows: list[dict[str, Any]] = []
    for baseline in available_baselines():
        selection = policy_from_baseline(baseline).select(state, base_context)
        metrics = selection_metrics(state, selection, evidence=final_event.oracle_evidence)
        policy_rows.append(
            {
                "baseline": baseline,
                "policy_id": selection.policy_id,
                "selected_mean": metrics["selected_mean"],
                "evidence_recall": metrics["evidence_recall_mean"],
                "all_required_evidence_survival": metrics["all_required_evidence_survival"],
            }
        )
    _write_csv(figure_data_dir / "baseline_policy_comparison.csv", policy_rows)
    _write_json(statistics_dir / "baseline_policy_comparison.json", {"rows": policy_rows})

    exponent_rows: list[dict[str, Any]] = []
    for exponent in np.linspace(0.0, 1.0, 11):
        rank = normalized_rank(state, float(exponent))
        selection = stable_topk(state, rank, primary_spec.budget, policy_id=f"e={exponent:.1f}")
        metrics = selection_metrics(state, selection, evidence=final_event.oracle_evidence)
        exponent_rows.append(
            {
                "exponent": round(float(exponent), 2),
                "evidence_recall": metrics["evidence_recall_mean"],
                "all_required_evidence_survival": metrics["all_required_evidence_survival"],
            }
        )
    _write_csv(figure_data_dir / "exponent_sweep.csv", exponent_rows)

    count_rows: list[dict[str, Any]] = []
    for name in ("true", "constant", "age", "shuffled", "circular", "delayed", "noisy", "head_averaged"):
        count_context = CountTransformContext(
            current_position=final_event.boundary_position,
            chunk_size=primary_spec.entries_per_round,
            previous_counts=final_event.state_before.counts,
            noise_std=0.25,
            seed=17,
            event_id=final_event.event_id,
        )
        transformed = transform_counts(state, name, count_context)
        rank = normalized_rank(state, 1.0, counts=transformed)
        selection = stable_topk(state, rank, primary_spec.budget, policy_id=f"count_{name}")
        metrics = selection_metrics(state, selection, evidence=final_event.oracle_evidence)
        count_rows.append(
            {
                "count_transform": name,
                "evidence_recall": metrics["evidence_recall_mean"],
                "all_required_evidence_survival": metrics["all_required_evidence_survival"],
            }
        )
    _write_csv(figure_data_dir / "count_ablation.csv", count_rows)

    replay_b2 = replay_trace(primary_trace, policy_from_baseline("B2"), replace(base_context, allow_oracle=False))
    replay_b3 = replay_trace(primary_trace, policy_from_baseline("B3"), replace(base_context, allow_oracle=False))
    replay_summary = {
        "B2": asdict(replay_b2),
        "B3": asdict(replay_b3),
        "interpretation": (
            "Both are valid only as open-loop full-candidate simulations; neither establishes changed model output quality."
        ),
    }
    _write_json(statistics_dir / "replay_summary.json", replay_summary)

    paired = summarize_paired_binary(
        raw_success,
        normalized_success,
        seed=17,
        cluster_ids=[f"rounds-{rounds}" for rounds in round_values],
        bootstrap_samples=2_000,
    )
    paired["estimand"] = "B3 minus B2 late-high-utility survival in synthetic repeated-exposure traces"
    paired["claim_limit"] = "synthetic mechanism evidence only; not a Rineya model-quality estimate"
    _write_json(statistics_dir / "paired_b3_vs_b2.json", paired)

    raw_bias_present = any(not value for value in raw_success)
    normalization_recovers = all(normalized_success)
    summary = {
        "status": "PASS" if raw_bias_present and normalization_recovers else "FAIL",
        "tier": "A",
        "execution_mode": "OPEN_LOOP_TRACE_SIMULATION",
        "model_inference_performed": False,
        "checkpoint_loaded": False,
        "gpu_used": False,
        "benchmark_cases": len(cases),
        "trace_count": len(trace_validations),
        "raw_exposure_bias_present": raw_bias_present,
        "normalization_recovers_late_high_utility": normalization_recovers,
        "claims_supported": [
            "state/policy/trace machinery executes deterministically on synthetic inputs",
            "raw cumulative score is exposure-biased in the specified toy process",
            "S/C removes that pure exposure factor in the specified toy process",
        ],
        "claims_not_supported": [
            "Rineya reasoning or retrieval quality improves",
            "the authoritative remote implementation is reproduced by synthetic traces",
            "CPU or GPU deployment latency improves",
            "open-loop replay predicts closed-loop attention or hidden-state changes",
        ],
    }
    _write_json(output / "summary.json", summary)

    resolved_direct = {
        "entrypoint": "scripts/run_tier_a.py",
        "seeds": args.seeds,
        "evidence_depths": args.evidence_depths,
        "filler_count": args.filler_count,
        "execution_mode": "open_loop_trace_simulation",
    }
    resolved_path = output / "resolved_config.yaml"
    resolved_path.write_text(json.dumps(resolved_direct, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo = git_snapshot(PACKAGE_ROOT.parent)
    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "status": "COMPLETE" if summary["status"] == "PASS" else "FAILED",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
        "created_at": started_at,
        "completed_at": _utc_now(),
        "repositories": {
            "chatbubble": {
                "remote": "https://github.com/LUNCY579/ChatBubble.git",
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
            "name": "tier_a_state_and_policy_validation",
            "compute_tier": "A",
            "architecture_variant": "B0-B8 synthetic open-loop policy laboratory",
            "benchmark": "deterministic_synthetic_R1_R4_and_exposure_traces",
            "seed": args.seeds[0],
            "seed_set": args.seeds,
            "query_conditioned": False,
            "execution_mode": "OPEN_LOOP_TRACE_SIMULATION",
        },
        "config": {
            "resolved_path": str(resolved_path),
            "sha256": _sha256(resolved_path),
        },
        "model": {
            "status": "NOT_APPLICABLE",
            "id": None,
            "path": None,
            "architecture": None,
            "hashes": {"weights": None, "tokenizer": None, "config": None, "chat_template": None},
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": None,
            "transformers": None,
            "device": "cpu",
            "precision": "numpy-fp32-state/fp64-analysis",
            "hardware": {
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
        },
        "budget": {
            "resident_entries": 256,
            "kv_only_bytes": baseline_budget.kv_bytes,
            "total_persistent_bytes": baseline_budget.total_persistent_bytes,
            "matched": True,
        },
        "artifacts": {
            "cases": "cases.jsonl",
            "benchmark_cases": "benchmark_cases.jsonl",
            "summary": "summary.json",
            "traces": "traces/",
            "statistics": "statistics/",
            "figure_data": "figures/data/",
            "resolved_config": "resolved_config.yaml",
        },
        "artifact_hashes": {
            "cases.jsonl": _sha256(cases_path),
            "benchmark_cases.jsonl": _sha256(benchmark_cases_path),
        },
        "notes": summary["claims_not_supported"],
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), **summary}, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
