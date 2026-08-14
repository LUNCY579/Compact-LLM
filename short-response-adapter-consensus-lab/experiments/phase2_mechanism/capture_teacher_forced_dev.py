from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


LAB_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from consensus_lab.evaluation.sampling import deterministic_stratified_sample  # noqa: E402
from consensus_lab.experiments.config import load_experiment_config  # noqa: E402
from consensus_lab.io import (  # noqa: E402
    append_jsonl_durable,
    canonical_json_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
from consensus_lab.models.committee import ExpertCommittee  # noqa: E402
from consensus_lab.paths import LabPaths  # noqa: E402
from consensus_lab.run_lock import exclusive_run_lock  # noqa: E402


DEFAULT_CONFIG = LAB_ROOT / "configs/fusion_sweep.yaml"
OUTPUT = LAB_ROOT / "results/phase2_mechanism/teacher_forced_dev_cache_topk64.jsonl"
MANIFEST = LAB_ROOT / "artifacts/phase2_mechanism/teacher_forced_dev_capture_topk64_manifest.json"
STATE = LAB_ROOT / "state/phase2_teacher_forced_dev_topk64_state.json"
LIVE_REPORT = LAB_ROOT / "reports/PHASE2_LIVE_RESULTS.md"
CORNERSTONE_AUDIT = LAB_ROOT / "reports/phase2_cornerstone_verification.json"

EXPERT_ORDER = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]
TEMPERATURE_GRID = [0.25, 0.5, 0.67, 0.75, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
CAPTURE_SCHEMA_VERSION = "1.1-topk64-union-temperature-grid"


def stable_topk(values: torch.Tensor, k: int) -> tuple[list[list[int]], list[list[float]]]:
    top_values, top_ids = torch.topk(values, k=k, dim=-1)
    value_rows = top_values.detach().cpu().tolist()
    id_rows = top_ids.detach().cpu().tolist()
    sorted_ids: list[list[int]] = []
    sorted_values: list[list[float]] = []
    for ids, logits in zip(id_rows, value_rows, strict=True):
        pairs = sorted(
            ((int(token_id), float(logit)) for token_id, logit in zip(ids, logits, strict=True)),
            key=lambda item: (-item[1], item[0]),
        )
        sorted_ids.append([item[0] for item in pairs])
        sorted_values.append([item[1] for item in pairs])
    return sorted_ids, sorted_values


def tokenize_target(
    committee: ExpertCommittee,
    row: dict[str, Any],
    *,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, str, int]:
    messages = [dict(message) for message in (row.get("messages") or [])]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"{row.get('id')}: expected a final assistant gold message")
    prompt_ids = committee.encode_messages(messages[:-1], max_sequence_length=max_sequence_length)
    gold = str(row.get("approved_answer") or row.get("answer") or messages[-1].get("content") or "")
    target_values = [int(value) for value in committee.tokenizer(gold, add_special_tokens=False)["input_ids"]]
    eos = committee.tokenizer.eos_token_id
    if eos is not None:
        target_values.append(int(eos))
    if not target_values:
        raise ValueError(f"{row.get('id')}: empty teacher-forced target")
    target_ids = torch.tensor(target_values, device=committee.device, dtype=prompt_ids.dtype)
    full = torch.cat([prompt_ids, target_ids[None, :]], dim=1)
    if int(full.shape[1]) > max_sequence_length:
        raise RuntimeError(f"{row.get('id')}: prompt plus gold exceeds max_sequence_length")
    model_input = full[:, :-1]
    return model_input, target_ids, gold, int(prompt_ids.shape[1])


def capture_expert(
    committee: ExpertCommittee,
    expert: Any,
    model_input: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    top_k: int,
    temperatures: list[float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    logits = committee.teacher_forced_logits(expert, model_input, target_length=len(target_ids))
    if logits.shape[0] != len(target_ids):
        raise RuntimeError(f"{expert.name}: teacher-forced target/logit length mismatch")
    raw_lse = torch.logsumexp(logits, dim=-1)
    log_probs = logits - raw_lse[:, None]
    probabilities = torch.exp(log_probs)
    top1_ids_tensor = torch.argmax(logits, dim=-1)
    top1_log_probs = log_probs.gather(dim=-1, index=top1_ids_tensor[:, None]).squeeze(-1)
    top1_probabilities = torch.exp(top1_log_probs)
    topk_ids, topk_logits = stable_topk(logits, top_k)
    top2_logits = torch.tensor(
        [values[1] for values in topk_logits],
        dtype=logits.dtype,
        device=logits.device,
    )
    top2_probabilities = torch.exp(top2_logits - raw_lse)
    gold_logits = logits.gather(dim=-1, index=target_ids[:, None]).squeeze(-1)
    gold_log_probs = gold_logits - raw_lse
    gold_probabilities = torch.exp(gold_log_probs)
    entropy = raw_lse - torch.sum(probabilities * logits, dim=-1)
    brier = torch.sum(probabilities.square(), dim=-1) - 2 * gold_probabilities + 1.0
    lse_by_temperature = []
    for temperature in temperatures:
        lse_by_temperature.append(torch.logsumexp(logits / float(temperature), dim=-1).detach().cpu())
    temperature_by_token = torch.stack(lse_by_temperature, dim=1)
    metrics = {
        "topk_token_ids": topk_ids,
        "topk_logits": topk_logits,
        "top1_token_ids": [int(value) for value in top1_ids_tensor.detach().cpu().tolist()],
        "top1_probabilities": [float(value) for value in top1_probabilities.detach().cpu().tolist()],
        "top2_probabilities": [float(value) for value in top2_probabilities.detach().cpu().tolist()],
        "margins": [
            float(value)
            for value in (top1_probabilities - top2_probabilities).detach().cpu().tolist()
        ],
        "gold_logits": [float(value) for value in gold_logits.detach().cpu().tolist()],
        "gold_log_probabilities": [float(value) for value in gold_log_probs.detach().cpu().tolist()],
        "gold_probabilities": [float(value) for value in gold_probabilities.detach().cpu().tolist()],
        "top1_correct": [
            int(predicted) == int(gold)
            for predicted, gold in zip(top1_ids_tensor.detach().cpu().tolist(), target_ids.detach().cpu().tolist(), strict=True)
        ],
        "entropies": [float(value) for value in entropy.detach().cpu().tolist()],
        "brier_scores": [float(value) for value in brier.detach().cpu().tolist()],
        "logit_means": [float(value) for value in logits.mean(dim=-1).detach().cpu().tolist()],
        "logit_stds": [float(value) for value in logits.std(dim=-1, unbiased=False).detach().cpu().tolist()],
        "logit_l2_norms": [float(value) for value in torch.linalg.vector_norm(logits, dim=-1).detach().cpu().tolist()],
        "logsumexp_by_temperature": temperature_by_token.tolist(),
    }
    return logits.detach().cpu(), metrics


def capture_example(
    committee: ExpertCommittee,
    row: dict[str, Any],
    *,
    top_k: int,
    temperatures: list[float],
    max_sequence_length: int,
    capture_config_sha256: str,
) -> dict[str, Any]:
    model_input, target_ids, gold, prompt_tokens = tokenize_target(
        committee,
        row,
        max_sequence_length=max_sequence_length,
    )
    started = time.perf_counter()
    logits_by_expert: list[torch.Tensor] = []
    metrics_by_expert: list[dict[str, Any]] = []
    for expert in committee.experts:
        logits, metrics = capture_expert(
            committee,
            expert,
            model_input,
            target_ids,
            top_k=top_k,
            temperatures=temperatures,
        )
        logits_by_expert.append(logits)
        metrics_by_expert.append(metrics)
    if committee.device.type == "cuda":
        torch.cuda.synchronize(committee.device)
    elapsed = time.perf_counter() - started
    target_values = [int(value) for value in target_ids.detach().cpu().tolist()]
    token_packets = []
    for target_index, target_token in enumerate(target_values):
        candidates = sorted(
            {
                token_id
                for metrics in metrics_by_expert
                for token_id in metrics["topk_token_ids"][target_index]
            }
        )
        candidate_tensor = torch.tensor(candidates, dtype=torch.long)
        candidate_logits = [
            [float(value) for value in expert_logits[target_index].gather(0, candidate_tensor).tolist()]
            for expert_logits in logits_by_expert
        ]
        expert_metrics = []
        for metrics in metrics_by_expert:
            expert_metrics.append(
                {
                    "topk_token_ids": metrics["topk_token_ids"][target_index],
                    "topk_logits": metrics["topk_logits"][target_index],
                    "top1_token_id": metrics["top1_token_ids"][target_index],
                    "top1_probability": metrics["top1_probabilities"][target_index],
                    "top2_probability": metrics["top2_probabilities"][target_index],
                    "margin": metrics["margins"][target_index],
                    "gold_logit": metrics["gold_logits"][target_index],
                    "gold_log_probability": metrics["gold_log_probabilities"][target_index],
                    "gold_probability": metrics["gold_probabilities"][target_index],
                    "top1_correct": metrics["top1_correct"][target_index],
                    "entropy": metrics["entropies"][target_index],
                    "brier_score": metrics["brier_scores"][target_index],
                    "logit_mean": metrics["logit_means"][target_index],
                    "logit_std": metrics["logit_stds"][target_index],
                    "logit_l2_norm": metrics["logit_l2_norms"][target_index],
                    "logsumexp_by_temperature": metrics["logsumexp_by_temperature"][target_index],
                }
            )
        token_packets.append(
            {
                "t": target_index,
                "gold_token_id": target_token,
                "gold_token_text": committee.tokenizer.decode([target_token], skip_special_tokens=False),
                "candidate_token_ids": candidates,
                "candidate_logits_by_expert": candidate_logits,
                "expert_metrics": expert_metrics,
            }
        )
    messages = [dict(message) for message in row["messages"]]
    prompt_payload = json.dumps(messages[:-1], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "capture_config_sha256": capture_config_sha256,
        "dataset": "v22_dev",
        "dataset_role": "DEV MODEL-SELECTION / CALIBRATION",
        "example_id": str(row["id"]),
        "prompt_sha256": hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest(),
        "domain": row.get("domain"),
        "reasoning_family": row.get("reasoning_family"),
        "difficulty": row.get("difficulty"),
        "decision": row.get("decision"),
        "gold": gold,
        "prompt_tokens": prompt_tokens,
        "target_token_ids": target_values,
        "target_tokens": len(target_values),
        "expert_order": [expert.name for expert in committee.experts],
        "temperature_grid": temperatures,
        "top_k": top_k,
        "tokens": token_packets,
        "latency_seconds": elapsed,
        "expert_forwards": len(committee.experts),
        "device": str(committee.device),
        "dtype": str(committee.dtype),
    }


def make_capture_config(
    *,
    config: Any,
    dataset_sha256: str,
    top_k: int,
    temperatures: list[float],
    dtype: str,
    precision_argument: str,
) -> dict[str, Any]:
    cornerstone = read_json(CORNERSTONE_AUDIT)
    if cornerstone.get("status") != "PASS":
        raise RuntimeError("Cornerstone verification must pass before DEV mechanism capture")
    adapters = cornerstone["frozen_config"]["adapter_packets"]
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "analysis_role": "DEV MODEL-SELECTION / CALIBRATION",
        "source_config": str(DEFAULT_CONFIG),
        "source_config_sha256": config.config_sha256,
        "dataset": "v22_dev",
        "dataset_path": str(config.datasets["v22_dev"]),
        "dataset_sha256": dataset_sha256,
        "expert_order": [expert.name for expert in config.experts],
        "adapter_hashes": {
            name: {
                "config": adapters[name]["adapter_config_sha256"],
                "weights": adapters[name]["adapter_weights_sha256"],
            }
            for name in EXPERT_ORDER
        },
        "top_k": top_k,
        "temperature_grid": temperatures,
        "target_encoding": "gold_text_tokenized_without_specials_plus_eos",
        "teacher_forced_alignment": "last_target_length_logits_from_prompt_plus_target_without_final_token",
        "precision_argument": precision_argument,
        "resolved_dtype": dtype,
        "seed": config.seed + 202,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }


def write_live_progress(
    *,
    completed: int,
    total: int,
    elapsed: float,
    mean_latency: float | None,
    output_bytes: int,
) -> None:
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else None
    existing_headline = (
        "Hard K=7 scores 17.259% on 4,050 v2.2 TEST examples, versus 14.593% for Track A and "
        "14.099% for probability mean."
    )
    lines = [
        "# Phase 2 Live Results",
        "",
        f"- Updated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- Cornerstone verification: **PASS (30/30 checks)**",
        "- Existing full-TEST rescue/harm analysis: **complete**",
        f"- DEV teacher-forced cache: **{completed}/{total} examples**",
        f"- Capture throughput: **{rate:.3f} examples/s**",
        f"- Mean captured-example inference latency: **{mean_latency:.3f}s**" if mean_latency is not None else "- Mean captured-example inference latency: pending",
        f"- Cache bytes: **{output_bytes:,}**",
        f"- Estimated remaining capture time: **{remaining / 60:.1f} min**" if remaining is not None else "- Estimated remaining capture time: pending",
        "- All-127 cached committee analysis: **pending cache completion**",
        "- Selected TEST mechanism confirmation: **pending; must be frozen first**",
        "",
        "## Current verified headline",
        "",
        existing_headline,
        "",
        "## Interpretation discipline",
        "",
        "The DEV cache is model-selection/calibration evidence. It is not a new TEST result.",
    ]
    LIVE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a resumable compressed DEV teacher-forced expert cache.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=8, help="Use 4050 for full DEV; default is an 8-example smoke run.")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.limit <= 0 or args.top_k < 2 or args.progress_every <= 0:
        raise ValueError("limit/progress-every must be positive and top-k must be at least 2")
    if args.top_k != 64:
        raise ValueError("The production Phase 2 cache is frozen at top-k=64")
    paths = LabPaths.discover(LAB_ROOT)
    config = load_experiment_config(args.config, paths)
    if [expert.name for expert in config.experts] != EXPERT_ORDER:
        raise RuntimeError("Unexpected expert order for Phase 2 capture")
    if "v22_dev" not in config.datasets:
        raise RuntimeError("Config does not define v22_dev")
    committee = ExpertCommittee.load(
        paths=paths,
        backbone=config.backbone,
        experts=config.experts,
        device=args.device,
        precision=args.precision,
    )
    dataset_sha = sha256_file(config.datasets["v22_dev"])
    capture_config = make_capture_config(
        config=config,
        dataset_sha256=dataset_sha,
        top_k=args.top_k,
        temperatures=TEMPERATURE_GRID,
        dtype=str(committee.dtype),
        precision_argument=args.precision,
    )
    capture_sha = canonical_json_sha256(capture_config)
    manifest_packet = {
        **capture_config,
        "capture_config_sha256": capture_sha,
        "output": str(OUTPUT),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DEV_CAPTURE_FROZEN",
    }
    if MANIFEST.exists():
        existing_manifest = read_json(MANIFEST)
        if existing_manifest.get("capture_config_sha256") != capture_sha:
            raise RuntimeError("Existing DEV capture manifest uses a different frozen capture config")
    else:
        write_json_atomic(MANIFEST, manifest_packet)
    existing_rows = read_jsonl(OUTPUT) if OUTPUT.exists() else []
    for row in existing_rows:
        if row.get("capture_config_sha256") != capture_sha:
            raise RuntimeError("Existing DEV cache row uses a different frozen capture config")
    done_ids = {str(row["example_id"]) for row in existing_rows}
    if len(done_ids) != len(existing_rows):
        raise RuntimeError("Existing DEV cache contains duplicate example IDs")
    dataset_rows = read_jsonl(config.datasets["v22_dev"])
    selected = deterministic_stratified_sample(
        dataset_rows,
        limit=min(args.limit, len(dataset_rows)),
        seed=config.seed + 202,
    )
    selected_ids = {str(row["id"]) for row in selected}
    pending = [row for row in selected if str(row["id"]) not in done_ids]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    captured_latencies = [float(row.get("latency_seconds") or 0.0) for row in existing_rows if str(row["example_id"]) in selected_ids]
    lock_path = OUTPUT.with_suffix(OUTPUT.suffix + ".lock")
    with exclusive_run_lock(lock_path):
        for index, row in enumerate(pending, start=1):
            packet = capture_example(
                committee,
                row,
                top_k=args.top_k,
                temperatures=TEMPERATURE_GRID,
                max_sequence_length=config.max_sequence_length,
                capture_config_sha256=capture_sha,
            )
            append_jsonl_durable(OUTPUT, packet)
            captured_latencies.append(float(packet["latency_seconds"]))
            done_ids.add(str(packet["example_id"]))
            completed_selected = len(done_ids.intersection(selected_ids))
            state_packet = {
                "schema_version": "1.0",
                "status": "complete" if completed_selected == len(selected) else "in_progress",
                "capture_config_sha256": capture_sha,
                "completed_selected": completed_selected,
                "selected_total": len(selected),
                "cache_unique_examples": len(done_ids),
                "dataset_total": len(dataset_rows),
                "output": str(OUTPUT),
                "output_bytes": OUTPUT.stat().st_size,
                "mean_latency_seconds": statistics_mean(captured_latencies),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json_atomic(STATE, state_packet)
            if index % args.progress_every == 0 or completed_selected == len(selected):
                write_live_progress(
                    completed=completed_selected,
                    total=len(selected),
                    elapsed=time.perf_counter() - started,
                    mean_latency=statistics_mean(captured_latencies),
                    output_bytes=OUTPUT.stat().st_size,
                )
                print(
                    json.dumps(
                        {
                            "completed": completed_selected,
                            "selected": len(selected),
                            "cache_unique_examples": len(done_ids),
                            "last_example": packet["example_id"],
                            "last_latency_seconds": packet["latency_seconds"],
                            "output_bytes": OUTPUT.stat().st_size,
                        }
                    ),
                    flush=True,
                )
    completed_selected = len(done_ids.intersection(selected_ids))
    if not pending:
        write_live_progress(
            completed=completed_selected,
            total=len(selected),
            elapsed=max(time.perf_counter() - started, 1e-9),
            mean_latency=statistics_mean(captured_latencies),
            output_bytes=OUTPUT.stat().st_size if OUTPUT.exists() else 0,
        )
    print(
        json.dumps(
            {
                "status": "complete" if completed_selected == len(selected) else "in_progress",
                "capture_config_sha256": capture_sha,
                "selected_examples": len(selected),
                "completed_selected": completed_selected,
                "cache_unique_examples": len(done_ids),
                "dataset_total": len(dataset_rows),
                "mean_latency_seconds": statistics_mean(captured_latencies),
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


def statistics_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    main()
