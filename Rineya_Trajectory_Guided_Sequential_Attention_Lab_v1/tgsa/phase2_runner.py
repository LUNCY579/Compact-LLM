"""Executable TGSA x MTP factorial training and evidence pipeline.

This runner validates the entire workflow on deterministic recurrence data:
materialization -> paired initialization -> fixed-step training -> exact verified
MTP decoding -> factorial analysis -> competence report -> closed evidence
bundle.  It is deliberately non-evidentiary unless a configuration explicitly
marks itself confirmatory and supplies a frozen development competence ``GO``.

The tiny study is a pipeline/mechanism test.  The Rineya pretrained retrofit is
implemented separately and must carry the real-model claim burden.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from .mtp import (
    MTPConfig,
    build_offset_targets,
    causal_next_token_loss,
    compute_mtp_loss,
    joint_causal_mtp_loss,
)
from .phase1_competence import (
    ArmCompetence,
    CompetenceThresholds,
    evaluate_competence,
    require_go,
)
from .phase1_model import TinyLMConfig, state_dict_sha256
from .phase2_analysis import paired_factorial_effects
from .phase2_data import (
    Phase2Batch,
    Phase2SequenceExample,
    batches_from_schedule,
    canonical_json_bytes,
    collate_phase2_examples,
    deterministic_batch_indices,
    load_phase2_examples,
    materialize_phase2_dataset,
    sha256_bytes,
    sha256_file,
    validate_phase2_dataset,
)
from .phase2_decode import greedy_generate, verified_mtp_generate
from .phase2_model import FactorialArmSpec, TinyTGSAForCausalLMWithMTP, build_factorial_models

PHASE2_STUDY_SCHEMA = "tgsa.mtp-factorial-study.v1"
PHASE2_RUN_SCHEMA = "tgsa.mtp-factorial-run.v1"
PHASE2_MATRIX_SCHEMA = "tgsa.mtp-factorial-matrix.v1"
PRIMARY_ARMS = ("parallel_ar", "parallel_mtp", "tgsa_ar", "tgsa_mtp")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
    )


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_phase2_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Phase-2 config must be a JSON object")
    if value.get("schema_version") != PHASE2_STUDY_SCHEMA:
        raise ValueError("unsupported Phase-2 study schema")
    for key in (
        "study_id",
        "confirmatory",
        "expected_base_commit",
        "seeds",
        "dataset",
        "model",
        "mtp",
        "training",
        "evaluation",
        "competence",
        "analysis",
        "runtime",
    ):
        if key not in value:
            raise ValueError(f"Phase-2 config is missing {key!r}")
    if not isinstance(value["confirmatory"], bool):
        raise TypeError("confirmatory must be an explicit boolean")
    seeds = [int(seed) for seed in value["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    training = cast(Mapping[str, object], value["training"])
    if int(training["steps"]) <= 0 or int(training["batch_size"]) <= 0:
        raise ValueError("training steps and batch_size must be positive")
    model = cast(Mapping[str, object], value["model"])
    if float(model.get("dropout", 0.0)) != 0.0:
        raise ValueError("paired Phase-2 studies require dropout=0")
    mtp = cast(Mapping[str, object], value["mtp"])
    if int(mtp.get("max_offset", 4)) < 2:
        raise ValueError("mtp.max_offset must be at least 2")
    if value["confirmatory"]:
        competence_path = value.get("competence_report")
        if not competence_path:
            raise ValueError("confirmatory configs require competence_report")
    return value


def resolve_device(runtime: Mapping[str, object]) -> torch.device:
    requested = str(runtime.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def configure_runtime(runtime: Mapping[str, object], seed: int) -> None:
    threads = int(runtime.get("num_threads", 0))
    if threads > 0:
        torch.set_num_threads(threads)
    deterministic = bool(runtime.get("deterministic_algorithms", True))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_snapshot(device: torch.device) -> dict[str, object]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(),
    }


def _model_config(
    study: Mapping[str, object],
    *,
    vocab_size: int,
    max_sequence_length: int,
) -> TinyLMConfig:
    value = cast(Mapping[str, object], study["model"])
    configured_max = int(value.get("max_sequence_length", max_sequence_length))
    if configured_max < max_sequence_length:
        raise ValueError(
            f"model.max_sequence_length={configured_max} is below dataset maximum "
            f"{max_sequence_length}"
        )
    return TinyLMConfig(
        vocab_size=vocab_size,
        max_sequence_length=configured_max,
        embed_dim=int(value.get("embed_dim", 64)),
        num_heads=int(value.get("num_heads", 4)),
        num_layers=int(value.get("num_layers", 2)),
        ffn_multiplier=float(value.get("ffn_multiplier", 4.0)),
        dropout=float(value.get("dropout", 0.0)),
        bias=bool(value.get("bias", False)),
        trajectory_rank=int(value.get("trajectory_rank", 16)),
        arm="full",
        tie_embeddings=bool(value.get("tie_embeddings", True)),
    )


def _mtp_config(
    study: Mapping[str, object],
    *,
    hidden_size: int,
    vocab_size: int,
) -> MTPConfig:
    value = cast(Mapping[str, object], study["mtp"])
    return MTPConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_offset=int(value.get("max_offset", 4)),
        rank=int(value.get("rank", 32)),
        loss_weight=float(value.get("loss_weight", 0.30)),
        offset_decay=float(value.get("offset_decay", 0.75)),
        target_mode="future",
        detach_backbone=bool(value.get("detach_backbone", False)),
        normalize_hidden=bool(value.get("normalize_hidden", True)),
        tie_output_embeddings=bool(value.get("tie_output_embeddings", True)),
        min_proposal_confidence=float(value.get("min_proposal_confidence", 0.0)),
        ignore_index=int(value.get("ignore_index", -100)),
    )


def _schedule_hash(schedule: Sequence[Sequence[int]]) -> str:
    return sha256_bytes(canonical_json_bytes([list(batch) for batch in schedule]))


def _optimizer_for(
    model: TinyTGSAForCausalLMWithMTP,
    training: Mapping[str, object],
) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(
        trainable,
        lr=float(training.get("learning_rate", 1.0e-3)),
        betas=(float(training.get("beta1", 0.9)), float(training.get("beta2", 0.95))),
        eps=float(training.get("epsilon", 1.0e-8)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def _learning_rate(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    remaining = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm().item())
        squared += norm * norm
    return math.sqrt(squared)


def train_one_cell(
    *,
    model: TinyTGSAForCausalLMWithMTP,
    spec: FactorialArmSpec,
    examples: Sequence[Phase2SequenceExample],
    schedule: Sequence[Sequence[int]],
    training: Mapping[str, object],
    mtp_config: MTPConfig,
    device: torch.device,
    run_dir: Path,
) -> dict[str, object]:
    model.to(device)
    model.train()
    optimizer = _optimizer_for(model, training)
    steps = int(training["steps"])
    base_lr = float(training.get("learning_rate", 1.0e-3))
    warmup_steps = int(training.get("warmup_steps", max(1, steps // 20)))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    log_every = max(1, int(training.get("log_every", max(1, steps // 20))))
    cell_mtp_config = replace(mtp_config, target_mode=spec.mtp_target_mode)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for step, batch in enumerate(batches_from_schedule(examples, schedule)):
        batch = batch.to(device)
        lr = _learning_rate(step, steps, base_lr, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
        if spec.mtp_enabled:
            loss = joint_causal_mtp_loss(
                output.logits,
                output.mtp_logits,
                batch.input_ids,
                cell_mtp_config,
                attention_mask=batch.attention_mask,
                target_mask=batch.target_mask,
            )
            objective = loss.total
            loss_record = loss.detached_dict()
        else:
            objective = causal_next_token_loss(
                output.logits,
                batch.input_ids,
                attention_mask=batch.attention_mask,
                target_mask=batch.target_mask,
                ignore_index=mtp_config.ignore_index,
            )
            loss_record = {
                "main": float(objective.detach().float()),
                "auxiliary": 0.0,
                "total": float(objective.detach().float()),
            }
        objective.backward()
        pre_clip = _gradient_norm(model)
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                gradient_clip,
            )
        optimizer.step()
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            rows.append(
                {
                    "step": step + 1,
                    "learning_rate": lr,
                    "gradient_norm_pre_clip": pre_clip,
                    "loss": loss_record,
                    "batch_example_ids_sha256": sha256_bytes(
                        canonical_json_bytes(list(batch.example_ids))
                    ),
                }
            )
    elapsed = time.perf_counter() - started
    write_jsonl(run_dir / "training_trace.jsonl", rows)
    checkpoint = run_dir / "final_checkpoint.pt"
    atomic_torch_save(
        checkpoint,
        {
            "schema_version": PHASE2_RUN_SCHEMA,
            "arm": spec.name,
            "model_state_dict": model.state_dict(),
        },
    )
    return {
        "fixed_final_checkpoint": True,
        "early_stopping": False,
        "steps": steps,
        "batch_size": int(training["batch_size"]),
        "elapsed_seconds": elapsed,
        "final_checkpoint": checkpoint.name,
        "final_checkpoint_sha256": sha256_file(checkpoint),
        "final_state_sha256": state_dict_sha256(model),
        "trace": "training_trace.jsonl",
        "trace_sha256": sha256_file(run_dir / "training_trace.jsonl"),
    }


def _evaluation_batches(
    examples: Sequence[Phase2SequenceExample],
    batch_size: int,
) -> Iterable[Phase2Batch]:
    for start in range(0, len(examples), batch_size):
        yield collate_phase2_examples(examples[start : start + batch_size])


def evaluate_token_metrics(
    model: TinyTGSAForCausalLMWithMTP,
    examples: Sequence[Phase2SequenceExample],
    *,
    batch_size: int,
    mtp_config: MTPConfig,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    predictions: set[int] = set()
    family_correct: dict[str, int] = {}
    family_total: dict[str, int] = {}
    mtp_loss_sum: dict[int, float] = {offset: 0.0 for offset in mtp_config.offsets}
    mtp_tokens: dict[int, int] = {offset: 0 for offset in mtp_config.offsets}
    mtp_correct: dict[int, int] = {offset: 0 for offset in mtp_config.offsets}

    with torch.no_grad():
        for batch in _evaluation_batches(examples, batch_size):
            batch = batch.to(device)
            output = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
            labels = build_offset_targets(
                batch.input_ids,
                1,
                attention_mask=batch.attention_mask,
                target_mask=batch.target_mask,
                ignore_index=mtp_config.ignore_index,
            )
            selected = labels.ne(mtp_config.ignore_index)
            losses = F.cross_entropy(
                output.logits.float().reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
                ignore_index=mtp_config.ignore_index,
            ).view_as(labels)
            batch_predictions = output.logits.argmax(dim=-1)
            total_loss += float(losses[selected].sum().item())
            count = int(selected.sum().item())
            total_tokens += count
            correct += int((batch_predictions[selected] == labels[selected]).sum().item())
            predictions.update(int(value) for value in batch_predictions[selected].detach().cpu().tolist())
            for row, family in enumerate(batch.families):
                row_selected = selected[row]
                row_count = int(row_selected.sum().item())
                family_total[family] = family_total.get(family, 0) + row_count
                family_correct[family] = family_correct.get(family, 0) + int(
                    (batch_predictions[row][row_selected] == labels[row][row_selected]).sum().item()
                )

            for offset in mtp_config.offsets:
                offset_labels = build_offset_targets(
                    batch.input_ids,
                    offset,
                    attention_mask=batch.attention_mask,
                    target_mask=batch.target_mask,
                    ignore_index=mtp_config.ignore_index,
                )
                offset_selected = offset_labels.ne(mtp_config.ignore_index)
                offset_count = int(offset_selected.sum().item())
                if offset_count == 0:
                    continue
                logits = output.mtp_logits[offset]
                offset_losses = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    offset_labels.reshape(-1),
                    reduction="none",
                    ignore_index=mtp_config.ignore_index,
                ).view_as(offset_labels)
                mtp_loss_sum[offset] += float(offset_losses[offset_selected].sum().item())
                mtp_tokens[offset] += offset_count
                mtp_correct[offset] += int(
                    (logits.argmax(dim=-1)[offset_selected] == offset_labels[offset_selected])
                    .sum()
                    .item()
                )
    if total_tokens == 0:
        raise ValueError("evaluation selected no main targets")
    return {
        "main_nll": total_loss / total_tokens,
        "main_accuracy": correct / total_tokens,
        "supervised_tokens": total_tokens,
        "distinct_predictions": len(predictions),
        "per_family_accuracy": {
            family: family_correct[family] / family_total[family]
            for family in sorted(family_total)
        },
        "mtp": {
            str(offset): {
                "nll": mtp_loss_sum[offset] / mtp_tokens[offset] if mtp_tokens[offset] else None,
                "accuracy": mtp_correct[offset] / mtp_tokens[offset] if mtp_tokens[offset] else None,
                "supervised_tokens": mtp_tokens[offset],
            }
            for offset in mtp_config.offsets
        },
    }


def evaluate_generation(
    model: TinyTGSAForCausalLMWithMTP,
    examples: Sequence[Phase2SequenceExample],
    *,
    count: int,
    eos_token_id: int,
    min_confidence: float,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = list(examples[: min(count, len(examples))])
    if not selected:
        raise ValueError("generation evaluation requires at least one example")
    rows: list[dict[str, object]] = []
    total_generated = 0
    target_calls = 0
    proposed = 0
    accepted = 0
    corrected = 0
    exact_decode = True
    task_exact = 0
    for example in selected:
        prompt = torch.tensor([example.tokens[: example.prompt_length]], dtype=torch.long, device=device)
        target = list(example.target_tokens)
        greedy = greedy_generate(
            model,
            prompt,
            max_new_tokens=len(target),
            eos_token_id=eos_token_id,
        )
        verified = verified_mtp_generate(
            model,
            prompt,
            max_new_tokens=len(target),
            eos_token_id=eos_token_id,
            min_confidence=min_confidence,
        )
        greedy_tokens = greedy.generated_ids[0].detach().cpu().tolist()
        verified_tokens = verified.generated_ids[0].detach().cpu().tolist()
        same = greedy_tokens == verified_tokens
        exact_decode = exact_decode and same
        matches_target = greedy_tokens == target
        task_exact += int(matches_target)
        total_generated += verified.stats.generated_tokens
        target_calls += verified.stats.target_forward_calls
        proposed += verified.stats.proposed_auxiliary_tokens
        accepted += verified.stats.accepted_auxiliary_tokens
        corrected += verified.stats.corrected_tokens
        rows.append(
            {
                "example_id": example.id,
                "family": example.family,
                "prompt": list(example.tokens[: example.prompt_length]),
                "target": target,
                "greedy": greedy_tokens,
                "verified_mtp": verified_tokens,
                "exact_greedy_equivalence": same,
                "task_exact_match": matches_target,
                "greedy_stats": greedy.stats.to_dict(),
                "verified_mtp_stats": verified.stats.to_dict(),
            }
        )
    if not exact_decode:
        bad = [row["example_id"] for row in rows if not row["exact_greedy_equivalence"]]
        raise RuntimeError(f"verified MTP decoder diverged from greedy on {bad}")
    metrics = {
        "examples": len(rows),
        "exact_greedy_equivalence": True,
        "task_exact_match": task_exact / len(rows),
        "generated_tokens": total_generated,
        "target_forward_calls": target_calls,
        "tokens_per_target_forward": total_generated / target_calls if target_calls else 0.0,
        "serial_round_speedup_vs_ar": total_generated / target_calls if target_calls else 0.0,
        "serial_round_reduction_fraction_vs_ar": (
            1.0 - (target_calls / total_generated) if total_generated else 0.0
        ),
        "proposed_auxiliary_tokens": proposed,
        "accepted_auxiliary_tokens": accepted,
        "auxiliary_acceptance_rate": accepted / proposed if proposed else 0.0,
        "corrected_tokens": corrected,
        "efficiency_scope": (
            "serial target-model rounds only; wall-clock speed, FLOPs, memory traffic, "
            "and fused-kernel performance are not inferred"
        ),
    }
    return metrics, rows


def _dataset_target_statistics(
    examples: Sequence[Phase2SequenceExample],
    *,
    vocab_size: int,
) -> dict[str, object]:
    counts: dict[int, int] = {}
    total = 0
    for example in examples:
        for token in example.target_tokens:
            counts[token] = counts.get(token, 0) + 1
            total += 1
    majority = max(counts.values()) / total
    return {
        "target_tokens": total,
        "majority_accuracy": majority,
        "uniform_vocab_accuracy": 1.0 / vocab_size,
        "uniform_vocab_nll": math.log(vocab_size),
        "target_histogram": {str(key): value for key, value in sorted(counts.items())},
    }


def _build_competence_report(
    *,
    run_records: Sequence[Mapping[str, object]],
    competence_config: Mapping[str, object],
    target_statistics: Mapping[str, object],
    primary_families: Sequence[str],
) -> dict[str, object]:
    arms: list[ArmCompetence] = []
    for arm in PRIMARY_ARMS:
        rows = [record for record in run_records if record["arm"] == arm]
        if not rows:
            raise ValueError(f"competence report has no rows for {arm}")
        holdout = [float(cast(Mapping[str, object], row["holdout_metrics"])["main_accuracy"]) for row in rows]
        seen = [float(cast(Mapping[str, object], row["train_metrics"])["main_accuracy"]) for row in rows]
        nll = [float(cast(Mapping[str, object], row["holdout_metrics"])["main_nll"]) for row in rows]
        distinct = [int(cast(Mapping[str, object], row["holdout_metrics"])["distinct_predictions"]) for row in rows]
        per_family = {
            family: sum(
                float(
                    cast(
                        Mapping[str, object],
                        cast(Mapping[str, object], row["holdout_metrics"])["per_family_accuracy"],
                    )[family]
                )
                for row in rows
            )
            / len(rows)
            for family in primary_families
        }
        arms.append(
            ArmCompetence(
                arm=arm,
                holdout_accuracy=sum(holdout) / len(holdout),
                seen_accuracy=sum(seen) / len(seen),
                answer_nll=sum(nll) / len(nll),
                distinct_predictions=min(distinct),
                per_family_accuracy=per_family,
                per_seed_holdout_accuracy=holdout,
            )
        )
    thresholds = CompetenceThresholds(
        minimum_margin_over_trivial=float(
            competence_config.get("minimum_margin_over_trivial", 0.02)
        ),
        minimum_absolute_accuracy=float(
            competence_config.get("minimum_absolute_accuracy", 0.12)
        ),
        maximum_answer_nll_fraction_of_uniform=float(
            competence_config.get("maximum_answer_nll_fraction_of_uniform", 1.0)
        ),
        minimum_distinct_predictions=int(
            competence_config.get("minimum_distinct_predictions", 4)
        ),
        maximum_family_accuracy=float(competence_config.get("maximum_family_accuracy", 0.98)),
        maximum_generalization_gap=float(
            competence_config.get("maximum_generalization_gap", 0.15)
        ),
        maximum_seed_coefficient_of_variation=float(
            competence_config.get("maximum_seed_coefficient_of_variation", 0.50)
        ),
        minimum_seed_runs=int(competence_config.get("minimum_seed_runs", 3)),
    )
    report = evaluate_competence(
        arms,
        thresholds=thresholds,
        trivial_baselines={
            "majority_target": float(target_statistics["majority_accuracy"]),
            "uniform_vocab": float(target_statistics["uniform_vocab_accuracy"]),
        },
        uniform_answer_nll=float(target_statistics["uniform_vocab_nll"]),
        primary_families=primary_families,
    )
    report["phase2_scope"] = (
        "arm-blind development competence for recurrence tasks; this only decides "
        "whether the factorial contrast is interpretable on this synthetic regime"
    )
    return report


def run_phase2_study(
    *,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    force_data: bool = False,
) -> dict[str, object]:
    study = load_phase2_config(config_path)
    if bool(study["confirmatory"]):
        require_go(Path(str(study["competence_report"])))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use a fresh root to avoid mixing evidence"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = materialize_phase2_dataset(
        dataset_config=cast(Mapping[str, object], study["dataset"]),
        output_dir=dataset_dir,
        force=force_data,
    )
    validate_phase2_dataset(dataset_dir / "dataset_manifest.json")
    split_entries = cast(Mapping[str, Mapping[str, object]], dataset_manifest["splits"])
    splits = {
        split: load_phase2_examples(dataset_dir / str(split_entries[split]["path"]))
        for split in ("train", "dev", "test")
    }
    vocabulary = cast(Mapping[str, object], dataset_manifest["vocabulary"])
    max_sequence = max(int(entry["max_tokens"]) for entry in split_entries.values())
    base_config = _model_config(
        study,
        vocab_size=int(vocabulary["vocab_size"]),
        max_sequence_length=max_sequence,
    )
    mtp_config = _mtp_config(
        study,
        hidden_size=base_config.embed_dim,
        vocab_size=base_config.vocab_size,
    )
    runtime = cast(Mapping[str, object], study["runtime"])
    device = resolve_device(runtime)
    configure_runtime(runtime, min(int(seed) for seed in study["seeds"]))
    environment = environment_snapshot(device)
    config_hash = sha256_file(config_path)
    dataset_hash = sha256_file(dataset_dir / "dataset_manifest.json")
    training = cast(Mapping[str, object], study["training"])
    evaluation = cast(Mapping[str, object], study["evaluation"])
    holdout_name = "test" if bool(study["confirmatory"]) else "dev"
    run_records: list[dict[str, object]] = []
    initialization_audits: dict[str, object] = {}

    for seed_value in study["seeds"]:
        seed = int(seed_value)
        configure_runtime(runtime, seed)
        schedule = deterministic_batch_indices(
            len(splits["train"]),
            batch_size=int(training["batch_size"]),
            steps=int(training["steps"]),
            seed=seed,
        )
        schedule_sha = _schedule_hash(schedule)
        bundle = build_factorial_models(
            base_config,
            mtp_config,
            initialization_seed=seed,
            tgsa_arm=cast(Any, str(study.get("tgsa_arm", "full"))),
            include_auxiliary_controls=bool(study.get("include_auxiliary_controls", False)),
        )
        initialization_audits[str(seed)] = bundle.initialization_audit
        seed_dir = output_dir / "initialization" / f"seed_{seed}"
        atomic_json(seed_dir / "audit.json", cast(Mapping[str, object], bundle.initialization_audit))

        for arm, model in bundle.models.items():
            spec = bundle.specs[arm]
            run_dir = output_dir / "runs" / arm / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            initial_hash = state_dict_sha256(model)
            try:
                training_record = train_one_cell(
                    model=model,
                    spec=spec,
                    examples=splits["train"],
                    schedule=schedule,
                    training=training,
                    mtp_config=mtp_config,
                    device=device,
                    run_dir=run_dir,
                )
                train_metrics = evaluate_token_metrics(
                    model,
                    splits["train"][: int(evaluation.get("train_metric_examples", 256))],
                    batch_size=int(evaluation.get("batch_size", 32)),
                    mtp_config=mtp_config,
                    device=device,
                )
                holdout_metrics = evaluate_token_metrics(
                    model,
                    splits[holdout_name],
                    batch_size=int(evaluation.get("batch_size", 32)),
                    mtp_config=mtp_config,
                    device=device,
                )
                generation_metrics, generation_rows = evaluate_generation(
                    model,
                    splits[holdout_name],
                    count=int(evaluation.get("generation_examples", 8)),
                    eos_token_id=int(vocabulary["eos_token_id"]),
                    min_confidence=mtp_config.min_proposal_confidence,
                    device=device,
                )
                write_jsonl(run_dir / "generation_predictions.jsonl", generation_rows)
                record: dict[str, object] = {
                    "schema_version": PHASE2_RUN_SCHEMA,
                    "status": "complete",
                    "study_id": str(study["study_id"]),
                    "confirmatory": bool(study["confirmatory"]),
                    "arm": arm,
                    "seed": seed,
                    "spec": spec.to_dict(),
                    "config_sha256": config_hash,
                    "dataset_manifest_sha256": dataset_hash,
                    "batch_schedule_sha256": schedule_sha,
                    "initial_state_sha256": initial_hash,
                    "training": training_record,
                    "train_metrics": train_metrics,
                    "holdout_split": holdout_name,
                    "holdout_metrics": holdout_metrics,
                    "main_accuracy": holdout_metrics["main_accuracy"],
                    "main_nll": holdout_metrics["main_nll"],
                    "generation_metrics": generation_metrics,
                    "serial_round_reduction_fraction_vs_ar": generation_metrics[
                        "serial_round_reduction_fraction_vs_ar"
                    ],
                    "generation_predictions": "generation_predictions.jsonl",
                    "generation_predictions_sha256": sha256_file(
                        run_dir / "generation_predictions.jsonl"
                    ),
                    "environment": environment,
                    "claim_scope": (
                        "one cell of a paired synthetic factorial workflow; no broad Rineya "
                        "quality or wall-clock claim follows from this run"
                    ),
                }
                atomic_json(run_dir / "run_manifest.json", record)
                run_records.append(record)
            except BaseException as error:
                failure = {
                    "schema_version": PHASE2_RUN_SCHEMA,
                    "status": "failed",
                    "arm": arm,
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "config_sha256": config_hash,
                    "dataset_manifest_sha256": dataset_hash,
                    "batch_schedule_sha256": schedule_sha,
                }
                atomic_json(run_dir / "failure.json", failure)
                raise
            finally:
                model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Explicit expected set for the complete paired primary matrix.
    expected = {
        (arm, int(seed))
        for arm in PRIMARY_ARMS
        for seed in study["seeds"]
    }
    actual = {(str(record["arm"]), int(record["seed"])) for record in run_records}
    if not expected.issubset(actual):
        raise RuntimeError(f"primary matrix incomplete: missing {sorted(expected - actual)}")

    target_statistics = _dataset_target_statistics(
        splits[holdout_name],
        vocab_size=base_config.vocab_size,
    )
    families = sorted({example.family for example in splits[holdout_name]})
    if bool(study["confirmatory"]):
        competence_report = require_go(Path(str(study["competence_report"])))
    else:
        competence_report = _build_competence_report(
            run_records=run_records,
            competence_config=cast(Mapping[str, object], study["competence"]),
            target_statistics=target_statistics,
            primary_families=families,
        )
    atomic_json(output_dir / "competence_report.json", competence_report)

    analysis_cfg = cast(Mapping[str, object], study["analysis"])
    confidence = float(analysis_cfg.get("confidence", 0.95))
    samples = int(analysis_cfg.get("bootstrap_samples", 5000))
    quality = paired_factorial_effects(
        run_records,
        metric="main_accuracy",
        higher_is_better=True,
        confidence=confidence,
        bootstrap_samples=samples,
    )
    nll = paired_factorial_effects(
        run_records,
        metric="main_nll",
        higher_is_better=False,
        confidence=confidence,
        bootstrap_samples=samples,
    )
    efficiency_records = [
        {
            "arm": record["arm"],
            "seed": record["seed"],
            "serial_round_reduction_fraction_vs_ar": cast(
                Mapping[str, object], record["generation_metrics"]
            )["serial_round_reduction_fraction_vs_ar"],
        }
        for record in run_records
    ]
    efficiency = paired_factorial_effects(
        efficiency_records,
        metric="serial_round_reduction_fraction_vs_ar",
        higher_is_better=True,
        confidence=confidence,
        bootstrap_samples=samples,
    )
    analysis = {
        "schema_version": "tgsa.mtp-factorial-analysis-bundle.v1",
        "quality_main_accuracy": quality,
        "quality_main_nll": nll,
        "efficiency_serial_rounds": efficiency,
        "secondary_controls": [
            record
            for record in run_records
            if str(record["arm"]) not in PRIMARY_ARMS
        ],
        "target_statistics": target_statistics,
        "claim_scope": (
            "paired effects on deterministic recurrence tasks; serial-round metrics are an "
            "algorithmic proxy and are not wall-clock throughput"
        ),
    }
    atomic_json(output_dir / "analysis.json", analysis)

    gate_inputs = {
        "schema_version": "tgsa.mtp-gate-inputs.v1",
        "exact_decode_equivalence": all(
            bool(cast(Mapping[str, object], record["generation_metrics"])["exact_greedy_equivalence"])
            for record in run_records
            if str(record["arm"]) in PRIMARY_ARMS
        ),
        "main_path_initialization_neutral": all(
            bool(cast(Mapping[str, object], audit["checks"])["parallel_ar_mtp_exact_state"])
            and bool(cast(Mapping[str, object], audit["checks"])["tgsa_ar_mtp_exact_state"])
            for audit in initialization_audits.values()
        ),
        "competence_go": competence_report.get("verdict") == "GO",
        "quality_noninferiority_margin": float(
            analysis_cfg.get("quality_noninferiority_margin", 0.01)
        ),
        "minimum_serial_round_gain": float(
            analysis_cfg.get("minimum_serial_round_gain", 0.05)
        ),
        "require_positive_interaction": bool(
            analysis_cfg.get("require_positive_interaction", False)
        ),
        "quality_analysis": "analysis.json#/quality_main_accuracy",
        "efficiency_analysis": "analysis.json#/efficiency_serial_rounds",
    }
    atomic_json(output_dir / "gate_inputs.json", gate_inputs)

    matrix = {
        "schema_version": PHASE2_MATRIX_SCHEMA,
        "status": "complete",
        "study_id": str(study["study_id"]),
        "confirmatory": bool(study["confirmatory"]),
        "expected_base_commit": str(study["expected_base_commit"]),
        "config": str(config_path),
        "config_sha256": config_hash,
        "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
        "dataset_manifest_sha256": dataset_hash,
        "holdout_split": holdout_name,
        "seeds": [int(seed) for seed in study["seeds"]],
        "primary_arms": list(PRIMARY_ARMS),
        "runs": [
            {
                "arm": record["arm"],
                "seed": record["seed"],
                "manifest": str(
                    Path("runs")
                    / str(record["arm"])
                    / f"seed_{int(record['seed'])}"
                    / "run_manifest.json"
                ),
            }
            for record in sorted(run_records, key=lambda row: (str(row["arm"]), int(row["seed"])))
        ],
        "artifacts": {
            "analysis": "analysis.json",
            "competence": "competence_report.json",
            "gate_inputs": "gate_inputs.json",
        },
        "environment": environment,
        "claim_scope": (
            "complete executable Phase-2 workflow; until the evidence verifier passes and a "
            "real pretrained Rineya run is completed, this is implementation validation only"
        ),
    }
    atomic_json(output_dir / "matrix_manifest.json", matrix)
    return matrix
