from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from consensus_lab.decoding.decoder import (
    generate_adaptive_consultation,
    generate_reference,
    generate_separate_cache,
    generate_sequence_medoid,
)
from consensus_lab.evaluation.sampling import deterministic_stratified_sample
from consensus_lab.evaluation.scoring import answer_correct, is_abstention, repetition_failure
from consensus_lab.evaluation.summary import write_summary_files
from consensus_lab.experiments.config import ExperimentConfig
from consensus_lab.io import append_jsonl_durable, read_jsonl, write_json_atomic
from consensus_lab.models.committee import ExpertCommittee
from consensus_lab.paths import LabPaths


EVALUATION_PROTOCOL_VERSION = "2.0-balanced-primary-strata-exact-reference"


def _prompt_hash(messages: list[dict[str, Any]]) -> str:
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _means(steps: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(step[key])
        for step in steps
        if step.get(key) is not None and math.isfinite(float(step[key]))
    ]
    return sum(values) / len(values) if values else None


def update_live_report(
    *,
    path: Path,
    run_name: str,
    completed: int,
    total: int,
    summaries: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    best = max(summaries, key=lambda row: float(row.get("correctness") or 0.0), default=None)
    backbone = next((row for row in summaries if row.get("committee") == "backbone_k1"), None)
    display_warnings = list(warnings)
    if best and float(best.get("false_abstention_rate") or 0.0) >= 0.9:
        display_warnings.append(
            "The aggregate-best condition is abstention-degenerate; inspect answerable correctness before selection."
        )
    lines = [
        "# Live Results",
        "",
        f"- Run: `{run_name}`",
        f"- Completed example-method conditions: **{completed}/{total}**",
        f"- Pending example-method conditions: **{max(0, total - completed)}**",
        f"- Status: **{'complete' if completed >= total else 'in progress'}**",
    ]
    if best:
        lines.extend(
            [
                f"- Current best: `{best['committee']} / {best['method']}`",
                f"- Correctness: **{float(best.get('correctness') or 0):.4f}**",
                f"- Mean tokens: **{float(best.get('mean_output_tokens') or 0):.2f}**",
                f"- Mean latency: **{float(best.get('mean_latency_seconds') or 0):.4f}s**",
                f"- Mean agreement: **{float(best.get('mean_agreement') or 0):.4f}**",
            ]
        )
        if backbone:
            difference = float(best.get("correctness") or 0.0) - float(backbone.get("correctness") or 0.0)
            lines.append(f"- Current-best minus backbone correctness: **{difference:+.4f}**")
    lines.extend(["", "## Completed summaries", "", "| Dataset | Committee | Method | K | N | Correct | Tokens | Latency | VRAM MiB | Agreement |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for summary in summaries:
        vram_mib = (
            float(summary["peak_gpu_vram_bytes"]) / (1024**2)
            if summary.get("peak_gpu_vram_bytes") is not None
            else 0.0
        )
        lines.append(
            f"| {summary['dataset']} | {summary['committee']} | {summary['method']} | {summary['committee_size']} | "
            f"{summary['case_count']} | {float(summary.get('correctness') or 0):.4f} | "
            f"{float(summary.get('mean_output_tokens') or 0):.2f} | {float(summary.get('mean_latency_seconds') or 0):.4f} | "
            f"{vram_mib:.1f} | "
            f"{float(summary.get('mean_agreement') or 0):.4f} |"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in display_warnings] or ["- None."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_direct_experiment(
    *,
    paths: LabPaths,
    config: ExperimentConfig,
    committee: ExpertCommittee,
    output_path: Path,
    summary_json: Path,
    summary_csv: Path,
    live_report: Path,
    dataset_names: list[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    output_path = paths.assert_output_path(output_path)
    summary_json = paths.assert_output_path(summary_json)
    summary_csv = paths.assert_output_path(summary_csv)
    live_report = paths.assert_output_path(live_report)
    if output_path.exists() and not resume:
        raise FileExistsError(f"Results already exist; pass resume=True: {output_path}")
    existing = read_jsonl(output_path) if output_path.is_file() else []
    for row in existing:
        if row.get("config_sha256") != config.config_sha256:
            raise RuntimeError("Existing results were produced by a different experiment config")
        if row.get("evaluation_protocol_version") != EVALUATION_PROTOCOL_VERSION:
            raise RuntimeError("Existing results were produced by a different evaluation protocol")
    done = {(row["dataset"], row["committee"], row["method"], row["example_id"]) for row in existing}
    selected_datasets = dataset_names or list(config.datasets)
    sampled: dict[str, list[dict[str, Any]]] = {}
    for name in selected_datasets:
        sampled[name] = deterministic_stratified_sample(
            read_jsonl(config.datasets[name]), limit=limit, seed=config.seed
        )
    total = sum(len(rows) for rows in sampled.values()) * len(config.committees) * len(config.methods)
    all_rows = list(existing)
    is_sealed = any("sealed" in name.lower() or "test" in name.lower() for name in selected_datasets)
    warnings: list[str] = []
    if limit:
        warnings.append(
            "This is a bounded frozen sealed sample, not the full official TEST split."
            if is_sealed
            else "This is a bounded DEV sample, not a full research-ready DEV result."
        )
    warnings.append(
        "Soft-fusion parameters were selected on DEV and were not retuned after sealed outcomes."
        if is_sealed
        else "Soft-fusion comparisons use configured temperatures/weights; calibration must be fit on DEV only."
    )
    decoders = {"reference": generate_reference, "separate_cache": generate_separate_cache}
    token_decoder = decoders[config.cache_mode]
    for dataset_name, rows in sampled.items():
        for committee_spec in config.committees:
            view = committee.subset(committee_spec.experts)
            committee_metadata = view.metadata()
            for method in config.methods:
                if method == "sequence_medoid":
                    decoder = generate_sequence_medoid
                elif method == "adaptive_hard_plurality":
                    decoder = generate_adaptive_consultation
                else:
                    decoder = token_decoder
                for row in rows:
                    example_id = str(row.get("id"))
                    key = (dataset_name, committee_spec.name, method, example_id)
                    if key in done:
                        continue
                    messages = [dict(message) for message in (row.get("messages") or [])]
                    if not messages or messages[-1].get("role") != "assistant":
                        raise ValueError(f"{example_id}: benchmark row must end with assistant gold target")
                    input_ids = view.encode_messages(messages[:-1], max_sequence_length=config.max_sequence_length)
                    decoder_kwargs: dict[str, Any] = {}
                    if method == "adaptive_hard_plurality":
                        decoder_kwargs = {
                            "core_expert_names": config.adaptive_core_experts,
                            "agreement_threshold": config.adaptive_agreement_threshold,
                        }
                    result = decoder(
                        view,
                        input_ids,
                        method=method,
                        max_new_tokens=config.max_new_tokens,
                        eos_policy=config.eos_policy,  # type: ignore[arg-type]
                        eos_confidence_threshold=config.eos_confidence_threshold,
                        minimum_tokens=config.minimum_tokens,
                        **decoder_kwargs,
                    )
                    packet = {
                        "schema_version": "1.0",
                        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
                        "config_sha256": config.config_sha256,
                        "run_name": config.name,
                        "dataset": dataset_name,
                        "benchmark_path": str(config.datasets[dataset_name]),
                        "example_id": example_id,
                        "prompt_hash": _prompt_hash(messages[:-1]),
                        "committee": committee_spec.name,
                        "committee_role": committee_spec.role,
                        "experts": committee_spec.experts,
                        "method": method,
                        "domain": row.get("domain"),
                        "reasoning_family": row.get("reasoning_family"),
                        "difficulty": row.get("difficulty"),
                        "decision": row.get("decision"),
                        "gold": row.get("approved_answer") or row.get("answer"),
                        "correct": answer_correct(result.output_text, row),
                        "abstained": is_abstention(result.output_text),
                        "repetition_failure": repetition_failure(result.output_token_ids),
                        "mean_agreement": _means(result.steps, "agreement"),
                        "mean_vote_entropy": _means(result.steps, "vote_entropy"),
                        "mean_pairwise_js": _means(result.steps, "mean_pairwise_js"),
                        "minimum_agreement": min(
                            (float(step["agreement"]) for step in result.steps if step.get("agreement") is not None),
                            default=None,
                        ),
                        "tie_count": sum(bool(step.get("tie")) for step in result.steps),
                        "early_eos_disagreement": any(
                            float(step.get("eos_vote_fraction") or 0) > 0 and not step.get("winner_is_eos")
                            for step in result.steps
                        ),
                        "late_eos_disagreement": bool(
                            result.steps
                            and result.steps[-1].get("winner_is_eos")
                            and float(result.steps[-1].get("eos_vote_fraction") or 0.0) < 1.0
                        ),
                        "adaptive_trigger_count": sum(
                            bool(step.get("adaptive_consultation", {}).get("triggered"))
                            for step in result.steps
                        ),
                        "backbone_parameter_count": committee_metadata["backbone_parameter_count"],
                        "loaded_parameter_count": committee_metadata["loaded_parameter_count"],
                        "committee_adapter_parameter_count": committee_metadata[
                            "committee_adapter_parameter_count"
                        ],
                        "active_parameter_count": committee_metadata[
                            "active_parameter_count_per_expert_forward"
                        ],
                        "adapter_storage_bytes": committee_metadata["adapter_storage_bytes"],
                        "compute_proxy_expert_token_forwards": result.adapter_evaluations,
                        **result.to_dict(),
                    }
                    append_jsonl_durable(output_path, packet)
                    all_rows.append(packet)
                    done.add(key)
                    summaries = write_summary_files(all_rows, summary_json, summary_csv)
                    write_json_atomic(
                        paths.lab_root / "state/run_state.json",
                        {
                            "schema_version": "1.0",
                            "run_name": config.name,
                            "config_sha256": config.config_sha256,
                            "completed": len(done),
                            "total": total,
                            "results": str(output_path),
                        },
                    )
                    update_live_report(
                        path=live_report,
                        run_name=config.name,
                        completed=len(done),
                        total=total,
                        summaries=summaries,
                        warnings=warnings,
                    )
    # Rebuild derived artifacts even when every raw row was already complete.
    # This repairs stale summaries after an interruption between durable JSONL
    # append and summary/report generation without rerunning model inference.
    summaries = write_summary_files(all_rows, summary_json, summary_csv)
    write_json_atomic(
        paths.lab_root / "state/run_state.json",
        {
            "schema_version": "1.0",
            "run_name": config.name,
            "config_sha256": config.config_sha256,
            "completed": len(done),
            "total": total,
            "results": str(output_path),
        },
    )
    update_live_report(
        path=live_report,
        run_name=config.name,
        completed=len(done),
        total=total,
        summaries=summaries,
        warnings=warnings,
    )
    return all_rows
