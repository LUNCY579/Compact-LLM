"""Resumable execution and evidence orchestration for TGSA Phase 1 v5.

The runner executes one registered cell at a time so a laptop or shared GPU can
progress without holding the whole campaign in memory.  It never interprets a
software-successful ``NO_GO`` as an execution failure; scientific verdicts are
data.  Promotion is handled separately by :mod:`tgsa.phase1_v5`.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import traceback
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

import torch
from torch import Tensor

from .phase1_competence import ArmCompetence
from .phase1_data import (
    Phase1Batch,
    Phase1Example,
    SyntheticVocabulary,
    collate_examples,
    derive_seed,
    load_examples,
    materialize_dataset,
    sha256_file,
    validate_dataset_manifest,
)
from .phase1_model import (
    ModelArm,
    TinyLMConfig,
    answer_logits,
    build_model,
    masked_answer_loss,
    paired_initialization_audit,
    state_dict_sha256,
)
from .phase1_runner import (
    atomic_torch_save,
    atomic_write_json,
    configure_runtime,
    environment_snapshot,
    git_snapshot,
    resolve_device,
    utc_now,
    write_jsonl,
)
from .phase1_v5 import (
    V5_FROZEN_BAND_SCHEMA_VERSION,
    V5_PRIMARY_FAMILIES,
    V5_REQUIRED_ARMS,
    CapacityRung,
    ExposureCell,
    V5ContractError,
    canonical_sha256,
    capacity_ladder,
    competence_thresholds_from_plan,
    evaluate_prerequisite_probe,
    evaluate_v5_qualification,
    family_null_values,
    hop_count_modulation_diagnostic,
    load_v5_plan,
    plan_sha256,
    probe_exposure,
    registered_family_null_report,
    resolve_exposure,
    resolve_rung,
    source_plan,
    stage_dataset_config,
)

V5_CELL_SCHEMA_VERSION = "tgsa.phase1-v5-cell.v1"
V5_EXECUTION_SCHEMA_VERSION = "tgsa.phase1-v5-execution.v1"


def _atomic_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _load_torch(path: Path, *, map_location: torch.device) -> object:
    # ``weights_only`` is explicit on newer PyTorch releases but absent on some
    # supported 2.1 wheels.  Keep one audited compatibility fallback.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _exact_full_batch_stream(
    examples: Sequence[Phase1Example],
    *,
    batch_size: int,
    seed: int,
    start_step: int = 0,
) -> Iterator[tuple[int, int, Phase1Batch]]:
    """Yield deterministic full batches and support exact step resumption.

    The historical stream emitted a short final batch each epoch, making
    ``optimizer_steps * batch_size`` differ from the registered presentation
    budget.  V5 drops that short remainder and reshuffles each epoch, so every
    optimizer step consumes exactly ``batch_size`` examples.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    steps_per_epoch = len(examples) // batch_size
    if steps_per_epoch <= 0:
        raise ValueError(
            f"training split has {len(examples)} examples, fewer than batch_size={batch_size}"
        )
    if start_step < 0:
        raise ValueError("start_step must be non-negative")

    epoch = start_step // steps_per_epoch
    step_in_epoch = start_step % steps_per_epoch
    while True:
        indices = list(range(len(examples)))
        random.Random(derive_seed(seed, "v5-full-batch-epoch", epoch)).shuffle(indices)
        usable = indices[: steps_per_epoch * batch_size]
        for local_step in range(step_in_epoch, steps_per_epoch):
            offset = local_step * batch_size
            selection = [
                examples[index] for index in usable[offset : offset + batch_size]
            ]
            global_step = epoch * steps_per_epoch + local_step
            yield global_step, epoch, collate_examples(selection)
        epoch += 1
        step_in_epoch = 0


def _model_config(
    plan: Mapping[str, object],
    *,
    rung: CapacityRung,
    arm: ModelArm,
    vocab_size: int,
    max_sequence_length: int,
) -> TinyLMConfig:
    training = cast(Mapping[str, object], plan["training"])
    return TinyLMConfig(
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embed_dim=rung.embed_dim,
        num_heads=rung.num_heads,
        num_layers=rung.num_layers,
        ffn_multiplier=float(training.get("ffn_multiplier", 4.0)),
        dropout=float(training.get("dropout", 0.0)),
        bias=bool(training.get("bias", False)),
        trajectory_rank=rung.trajectory_rank,
        arm=arm,
        tie_embeddings=bool(training.get("tie_embeddings", True)),
    )


def _learning_rate(
    *,
    step: int,
    total_steps: int,
    peak: float,
    warmup_fraction: float,
) -> float:
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    if step < warmup_steps:
        return peak * float(step + 1) / float(warmup_steps)
    denominator = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _measure_model(
    model: torch.nn.Module,
    examples: Sequence[Phase1Example],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not examples:
        raise ValueError("cannot measure an empty example sequence")
    total = 0
    correct = 0
    nll_sum = 0.0
    predictions: list[int] = []
    by_family: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    model.eval()

    for offset in range(0, len(examples), batch_size):
        selection = list(examples[offset : offset + batch_size])
        batch = collate_examples(selection).to(device)
        output = model(
            batch.input_ids,
            key_padding_mask=batch.key_padding_mask,
            telemetry_level="none",
        )
        logits = answer_logits(output.logits, batch.answer_input_indices).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        predicted = logits.argmax(dim=-1)
        token_nll = -log_probs.gather(1, batch.answer_tokens[:, None]).squeeze(1)
        hits = predicted.eq(batch.answer_tokens)

        for index, example in enumerate(selection):
            prediction = int(predicted[index].item())
            target = int(batch.answer_tokens[index].item())
            hit = bool(hits[index].item())
            loss = float(token_nll[index].item())
            family = example.family
            bucket = by_family.setdefault(
                family,
                {
                    "examples": 0,
                    "correct": 0,
                    "nll_sum": 0.0,
                    "predictions": [],
                },
            )
            bucket["examples"] = int(bucket["examples"]) + 1
            bucket["correct"] = int(bucket["correct"]) + int(hit)
            bucket["nll_sum"] = float(bucket["nll_sum"]) + loss
            cast(list[int], bucket["predictions"]).append(prediction)
            rows.append(
                {
                    "schema_version": "tgsa.phase1-v5-prediction.v1",
                    "example_id": example.id,
                    "split": example.split,
                    "family": family,
                    "stratum": example.stratum,
                    "target_token": target,
                    "predicted_token": prediction,
                    "correct": hit,
                    "answer_nll": loss,
                    "metadata": dict(example.metadata),
                }
            )
            total += 1
            correct += int(hit)
            nll_sum += loss
            predictions.append(prediction)

    per_family: dict[str, object] = {}
    for family, values in sorted(by_family.items()):
        count = int(values["examples"])
        family_predictions = cast(list[int], values["predictions"])
        per_family[family] = {
            "examples": count,
            "accuracy": int(values["correct"]) / count,
            "answer_nll": float(values["nll_sum"]) / count,
            "distinct_predictions": len(set(family_predictions)),
        }

    return (
        {
            "examples": total,
            "accuracy": correct / total,
            "answer_nll": nll_sum / total,
            "distinct_predictions": len(set(predictions)),
            "per_family": per_family,
        },
        rows,
    )


def _seen_subset(
    train: Sequence[Phase1Example],
    *,
    limit: int,
) -> list[Phase1Example]:
    """Take a deterministic, role-preserving seen sample."""

    if limit <= 0:
        raise ValueError("seen limit must be positive")
    by_family: dict[str, list[Phase1Example]] = {}
    for example in train:
        by_family.setdefault(example.family, []).append(example)
    names = sorted(by_family)
    per_family = max(1, limit // max(1, len(names)))
    selected: list[Phase1Example] = []
    for family in names:
        selected.extend(by_family[family][:per_family])
    if len(selected) < limit:
        chosen = {example.id for example in selected}
        selected.extend(
            example for example in train if example.id not in chosen
        )
    return selected[: min(limit, len(train))]


def _cell_identity(
    *,
    plan_hash: str,
    stage: str,
    rung: CapacityRung,
    exposure: ExposureCell,
    dataset_manifest_sha256: str,
    arm: str,
    seed: int,
    git: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": V5_CELL_SCHEMA_VERSION,
        "plan_sha256": plan_hash,
        "stage": stage,
        "rung": rung.to_dict(),
        "exposure": exposure.to_dict(),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "arm": arm,
        "seed": int(seed),
        "git_head": git.get("head"),
        "environment": {
            "torch": environment.get("torch"),
            "device": environment.get("device"),
            "cuda_version": environment.get("cuda_version"),
            "gpu_name": environment.get("gpu_name"),
            "platform": environment.get("platform"),
            # Determines cuBLAS GEMM workspace behaviour, and `configure_runtime`
            # only `setdefault`s it, so an inherited value wins. Without it in the
            # identity a cell could resume from a checkpoint produced under
            # different CUDA arithmetic and still match its fingerprint.
            "cublas_workspace_config": environment.get("cublas_workspace_config"),
        },
    }


def _validate_artifact(path: Path, expected_sha256: object, *, label: str) -> None:
    if not path.is_file():
        raise V5ContractError(f"reusable cell is missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise V5ContractError(
            f"reusable cell {label} hash mismatch: {actual} != {expected_sha256}"
        )


def _load_complete_cell(
    cell_dir: Path,
    *,
    expected_fingerprint: str,
) -> dict[str, object] | None:
    manifest_path = cell_dir / "cell_manifest.json"
    if not manifest_path.is_file():
        return None
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise V5ContractError(f"{manifest_path} is not a JSON object")
    manifest = dict(value)
    if manifest.get("schema_version") != V5_CELL_SCHEMA_VERSION:
        raise V5ContractError(f"{manifest_path} has an unsupported schema")
    if manifest.get("status") != "complete":
        raise V5ContractError(
            f"{manifest_path} exists but is not complete; preserve it and resume "
            "from training_state.pt or use a fresh output root"
        )
    if manifest.get("execution_fingerprint") != expected_fingerprint:
        raise V5ContractError(
            f"{manifest_path} was produced under a different execution identity"
        )
    files = cast(Mapping[str, object], manifest.get("files", {}))
    for name in ("checkpoint", "predictions_dev", "predictions_test"):
        entry = cast(Mapping[str, object], files.get(name, {}))
        _validate_artifact(
            cell_dir / str(entry.get("path", "")),
            entry.get("sha256"),
            label=name,
        )
    return manifest


def _load_training_state(
    path: Path,
    *,
    expected_fingerprint: str,
    device: torch.device,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    value = _load_torch(path, map_location=device)
    if not isinstance(value, Mapping):
        raise V5ContractError(f"{path} is not a training-state mapping")
    state = dict(value)
    if state.get("schema_version") != "tgsa.phase1-v5-training-state.v1":
        raise V5ContractError(f"{path} has an unsupported training-state schema")
    if state.get("execution_fingerprint") != expected_fingerprint:
        raise V5ContractError(
            f"{path} belongs to a different execution identity; do not mix runs"
        )
    return state


def _save_training_state(
    path: Path,
    *,
    fingerprint: str,
    next_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_curve: Sequence[Sequence[float]],
) -> None:
    atomic_torch_save(
        path,
        {
            "schema_version": "tgsa.phase1-v5-training-state.v1",
            "execution_fingerprint": fingerprint,
            "next_step": int(next_step),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss_curve": [list(pair) for pair in loss_curve],
        },
    )


#: Emergency snapshot written when a cell dies mid-training. Deliberately NOT
#: the resume source; see the exception handler in `_train_cell`.
INTERRUPTED_STATE_NAME = "interrupted_state.pt"


def _optimizer_applied_steps(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    """Read the optimizer's own per-parameter step counters.

    AdamW increments ``state[p]["step"]`` inside ``optimizer.step()``. If every
    parameter reports the same count then the step either completed for all of
    them or had not begun, and the tensors are mutually consistent. If they
    disagree, ``optimizer.step()`` was interrupted part-way through and no
    amount of bookkeeping can make them consistent again — that state is not a
    model at any point in the registered trajectory.

    Reported for diagnosis only. It is not used to decide where to resume.
    """

    counts: set[int] = set()
    for entry in optimizer.state.values():
        step = entry.get("step") if isinstance(entry, Mapping) else None
        if step is None:
            continue
        counts.add(int(step.item()) if hasattr(step, "item") else int(step))
    return {
        "per_parameter_step_counts": sorted(counts),
        "all_parameters_agree": len(counts) <= 1,
        "applied_steps": max(counts) if counts else None,
    }


def _train_cell(
    *,
    plan: Mapping[str, object],
    stage: str,
    rung: CapacityRung,
    exposure: ExposureCell,
    arm: ModelArm,
    seed: int,
    train: Sequence[Phase1Example],
    dev: Sequence[Phase1Example],
    test: Sequence[Phase1Example],
    vocab: SyntheticVocabulary,
    max_sequence_length: int,
    device: torch.device,
    identity: Mapping[str, object],
    cell_dir: Path,
    resume: bool,
) -> dict[str, object]:
    fingerprint = canonical_sha256(identity)
    if resume:
        complete = _load_complete_cell(
            cell_dir, expected_fingerprint=fingerprint
        )
        if complete is not None:
            return complete
    elif cell_dir.exists() and any(cell_dir.iterdir()):
        raise V5ContractError(
            f"{cell_dir} already contains evidence; --no-resume never overwrites it"
        )

    cell_dir.mkdir(parents=True, exist_ok=True)
    runtime = cast(Mapping[str, object], plan["runtime"])
    training = cast(Mapping[str, object], plan["training"])
    configure_runtime(runtime, seed)
    model_config = _model_config(
        plan,
        rung=rung,
        arm=arm,
        vocab_size=vocab.vocab_size,
        max_sequence_length=max_sequence_length,
    )
    initialization_seed = derive_seed(seed, "v5-model-initialization")
    model = build_model(
        model_config,
        initialization_seed=initialization_seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        weight_decay=float(training["weight_decay"]),
    )

    state_path = cell_dir / "training_state.pt"
    start_step = 0
    loss_curve: list[list[float]] = []
    resumed_from_step: int | None = None
    if resume:
        saved = _load_training_state(
            state_path,
            expected_fingerprint=fingerprint,
            device=device,
        )
        if saved is not None:
            start_step = int(saved["next_step"])
            if not 0 <= start_step <= exposure.optimizer_steps:
                raise V5ContractError(
                    f"{state_path}: next_step is outside the registered budget"
                )
            model.load_state_dict(cast(Mapping[str, Tensor], saved["model_state"]))
            optimizer.load_state_dict(
                cast(Mapping[str, object], saved["optimizer_state"])
            )
            loss_curve = [
                [float(pair[0]), float(pair[1])]
                for pair in cast(Sequence[Sequence[float]], saved["loss_curve"])
            ]
            resumed_from_step = start_step

    stream = _exact_full_batch_stream(
        train,
        batch_size=exposure.batch_size,
        seed=derive_seed(seed, "v5-training-order"),
        start_step=start_step,
    )
    warmup_fraction = float(training["warmup_fraction"])
    peak_lr = float(training["learning_rate"])
    gradient_clip = float(training["gradient_clip_norm"])
    checkpoint_interval = max(
        1, int(training.get("checkpoint_interval_steps", 500))
    )
    curve_interval = max(1, exposure.optimizer_steps // 20)
    next_step = start_step

    try:
        model.train()
        for expected_step in range(start_step, exposure.optimizer_steps):
            stream_step, _epoch, batch = next(stream)
            if stream_step != expected_step:
                raise AssertionError(
                    f"deterministic stream step {stream_step} != {expected_step}"
                )
            lr = _learning_rate(
                step=expected_step,
                total_steps=exposure.optimizer_steps,
                peak=peak_lr,
                warmup_fraction=warmup_fraction,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = batch.to(device)
            output = model(
                batch.input_ids,
                key_padding_mask=batch.key_padding_mask,
                telemetry_level="none",
            )
            loss = masked_answer_loss(
                output.logits,
                batch.labels,
                batch.loss_mask,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            next_step = expected_step + 1
            if (
                expected_step % curve_interval == 0
                or next_step == exposure.optimizer_steps
            ):
                loss_curve.append([float(expected_step), float(loss.detach())])
            if (
                next_step % checkpoint_interval == 0
                and next_step < exposure.optimizer_steps
            ):
                _save_training_state(
                    state_path,
                    fingerprint=fingerprint,
                    next_step=next_step,
                    model=model,
                    optimizer=optimizer,
                    loss_curve=loss_curve,
                )
    except BaseException as error:
        # The emergency snapshot is DIAGNOSTIC ONLY, and is deliberately written
        # somewhere other than `state_path`.
        #
        # `next_step` is assigned on the line after `optimizer.step()` returns.
        # A signal can be delivered between any two bytecodes, so a Ctrl-C
        # landing in that window leaves the parameters advanced by one step while
        # `next_step` still names that same step. Saving the pair to `state_path`
        # would make resume re-apply it: one optimizer step applied twice, and a
        # cell that silently diverges from an uninterrupted run while still
        # producing a well-formed, fingerprint-matching report. A CUDA OOM raised
        # *inside* `optimizer.step()` is worse -- some parameters are updated and
        # some are not, and no bookkeeping can repair that.
        #
        # So the handler does not nominate a resume point at all. Resume uses the
        # last periodic checkpoint, which is written inside the loop after the
        # bookkeeping, through `atomic_torch_save`, and is therefore always a
        # point the registered trajectory actually passes through. The cost is at
        # most `checkpoint_interval_steps` of recomputation; the alternative is
        # silent divergence, which no downstream verifier would catch.
        interrupted_path = cell_dir / INTERRUPTED_STATE_NAME
        applied = _optimizer_applied_steps(optimizer)
        _save_training_state(
            interrupted_path,
            fingerprint=fingerprint,
            next_step=next_step,
            model=model,
            optimizer=optimizer,
            loss_curve=loss_curve,
        )
        atomic_write_json(
            cell_dir / "failure.json",
            {
                "schema_version": "tgsa.phase1-v5-cell-failure.v2",
                "execution_fingerprint": fingerprint,
                "failed_at_utc": utc_now(),
                "next_step": next_step,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "optimizer_applied_steps": applied,
                "interrupted_state": interrupted_path.name,
                "interrupted_state_is_resumable": False,
                "resume_source": (
                    state_path.name if state_path.is_file() else None
                ),
                "resume_note": (
                    "interrupted_state.pt records where this cell died and is not a "
                    "resume source: the interrupt may have landed between "
                    "optimizer.step() and the step bookkeeping, or part-way through "
                    "the step itself. Re-running resumes from training_state.pt, the "
                    "last periodic checkpoint, which is consistent by construction. "
                    "To read which window it was, compare "
                    "optimizer_applied_steps.applied_steps with the next_step "
                    "recorded here. applied_steps == next_step means the snapshot is "
                    "untorn -- the step bookkeeping had already run, or no step had "
                    "begun. applied_steps == next_step + 1 is the stale-counter "
                    "window this file exists for: the optimizer advanced and the "
                    "bookkeeping did not, so resuming from it would apply that step "
                    "twice. all_parameters_agree=false means the step itself was "
                    "torn part-way through, which no bookkeeping can repair."
                ),
            },
        )
        raise

    qualification = cast(Mapping[str, object], plan["qualification"])
    evaluation_batch_size = int(qualification["evaluation_batch_size"])
    seen_examples = _seen_subset(
        train,
        limit=int(qualification["seen_examples"]),
    )
    model.eval()
    seen_metrics, _ = _measure_model(
        model,
        seen_examples,
        device=device,
        batch_size=evaluation_batch_size,
    )
    dev_metrics, dev_rows = _measure_model(
        model,
        dev,
        device=device,
        batch_size=evaluation_batch_size,
    )
    test_metrics, test_rows = _measure_model(
        model,
        test,
        device=device,
        batch_size=evaluation_batch_size,
    )

    dev_path = cell_dir / "predictions_dev.jsonl"
    test_path = cell_dir / "predictions_test.jsonl"
    checkpoint_path = cell_dir / "checkpoint.pt"
    write_jsonl(dev_path, dev_rows)
    write_jsonl(test_path, test_rows)
    atomic_torch_save(
        checkpoint_path,
        {
            "schema_version": "tgsa.phase1-v5-checkpoint.v1",
            "execution_fingerprint": fingerprint,
            "arm": arm,
            "seed": seed,
            "optimizer_steps": exposure.optimizer_steps,
            "sample_presentations": exposure.sample_presentations,
            "model_config": {
                "vocab_size": model_config.vocab_size,
                "max_sequence_length": model_config.max_sequence_length,
                "embed_dim": model_config.embed_dim,
                "num_heads": model_config.num_heads,
                "num_layers": model_config.num_layers,
                "ffn_multiplier": model_config.ffn_multiplier,
                "dropout": model_config.dropout,
                "bias": model_config.bias,
                "trajectory_rank": model_config.trajectory_rank,
                "arm": model_config.arm,
                "tie_embeddings": model_config.tie_embeddings,
            },
            "model_state": model.state_dict(),
        },
    )

    manifest: dict[str, object] = {
        "schema_version": V5_CELL_SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "execution_fingerprint": fingerprint,
        "identity": dict(identity),
        "arm": arm,
        "seed": seed,
        "training": {
            "optimizer_steps": exposure.optimizer_steps,
            "batch_size": exposure.batch_size,
            "sample_presentations": exposure.sample_presentations,
            "unique_train_examples": exposure.unique_train_examples,
            "effective_epochs": exposure.effective_epochs,
            "fixed_final_checkpoint": True,
            "early_stopping": False,
            "resumed_from_step": resumed_from_step,
            "loss_curve": loss_curve,
            "final_loss": loss_curve[-1][1] if loss_curve else None,
        },
        "metrics": {
            "seen": seen_metrics,
            "dev": dev_metrics,
            "test": test_metrics,
        },
        "model_state_sha256": state_dict_sha256(model),
        "files": {
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
            },
            "predictions_dev": {
                "path": dev_path.name,
                "sha256": sha256_file(dev_path),
            },
            "predictions_test": {
                "path": test_path.name,
                "sha256": sha256_file(test_path),
            },
        },
        "claim_scope": (
            "one apparatus-qualification cell; no TGSA mechanism inference"
        ),
    }
    atomic_write_json(cell_dir / "cell_manifest.json", manifest)
    _atomic_unlink(state_path)
    _atomic_unlink(cell_dir / "failure.json")
    if device.type == "cuda":
        del model
        torch.cuda.empty_cache()
    return manifest


def _read_prediction_rows(cell_dir: Path, filename: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (cell_dir / filename).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise V5ContractError(
                    f"{filename}:{line_number} is not a JSON object"
                )
            rows.append(dict(value))
    return rows


def _family_metric(
    metrics: Mapping[str, object],
    family: str,
    key: str,
) -> float:
    per_family = cast(Mapping[str, Mapping[str, object]], metrics["per_family"])
    if family not in per_family:
        raise V5ContractError(f"metrics are missing family {family!r}")
    return float(per_family[family][key])


def _aggregate_qualification_arms(
    cells: Sequence[Mapping[str, object]],
) -> list[ArmCompetence]:
    arms: list[ArmCompetence] = []
    for arm in V5_REQUIRED_ARMS:
        subset = [cell for cell in cells if cell["arm"] == arm]
        if not subset:
            raise V5ContractError(f"no completed cells for arm {arm}")
        per_family: dict[str, float] = {}
        family_names = sorted(
            {
                family
                for cell in subset
                for family in cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], cell["metrics"])["test"],
                )["per_family"]
            }
        )
        for family in family_names:
            per_family[family] = statistics.fmean(
                _family_metric(
                    cast(
                        Mapping[str, object],
                        cast(Mapping[str, object], cell["metrics"])["test"],
                    ),
                    family,
                    "accuracy",
                )
                for cell in subset
            )

        primary = V5_PRIMARY_FAMILIES[0]
        arms.append(
            ArmCompetence(
                arm=arm,
                holdout_accuracy=statistics.fmean(
                    _family_metric(
                        cast(
                            Mapping[str, object],
                            cast(Mapping[str, object], cell["metrics"])["test"],
                        ),
                        primary,
                        "accuracy",
                    )
                    for cell in subset
                ),
                seen_accuracy=statistics.fmean(
                    _family_metric(
                        cast(
                            Mapping[str, object],
                            cast(Mapping[str, object], cell["metrics"])["seen"],
                        ),
                        primary,
                        "accuracy",
                    )
                    for cell in subset
                ),
                answer_nll=statistics.fmean(
                    _family_metric(
                        cast(
                            Mapping[str, object],
                            cast(Mapping[str, object], cell["metrics"])["test"],
                        ),
                        primary,
                        "answer_nll",
                    )
                    for cell in subset
                ),
                distinct_predictions=int(
                    min(
                        _family_metric(
                            cast(
                                Mapping[str, object],
                                cast(Mapping[str, object], cell["metrics"])["test"],
                            ),
                            primary,
                            "distinct_predictions",
                        )
                        for cell in subset
                    )
                ),
                per_family_accuracy=per_family,
                per_seed_holdout_accuracy=[
                    _family_metric(
                        cast(
                            Mapping[str, object],
                            cast(Mapping[str, object], cell["metrics"])["test"],
                        ),
                        primary,
                        "accuracy",
                    )
                    for cell in subset
                ],
            )
        )
    return arms


def _canary_metrics(
    cells: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    family = "state_update"
    result: dict[str, object] = {
        "family": family,
        "gating": False,
        "per_arm_accuracy": {},
    }
    per_arm = cast(dict[str, float], result["per_arm_accuracy"])
    for arm in V5_REQUIRED_ARMS:
        subset = [cell for cell in cells if cell["arm"] == arm]
        per_arm[arm] = statistics.fmean(
            _family_metric(
                cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], cell["metrics"])["test"],
                ),
                family,
                "accuracy",
            )
            for cell in subset
        )
    return result


def _validate_frozen_band(
    *,
    plan: Mapping[str, object],
    frozen_band_path: Path | None,
    rung: CapacityRung,
) -> dict[str, object] | None:
    if bool(plan["development_only"]):
        return None
    if frozen_band_path is None:
        raise V5ContractError(
            "non-development qualification requires --frozen-band produced "
            "from the passing prerequisite probe"
        )
    value = json.loads(frozen_band_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise V5ContractError("frozen band must be a JSON object")
    band = dict(value)
    if band.get("schema_version") != V5_FROZEN_BAND_SCHEMA_VERSION:
        raise V5ContractError("frozen band has an incompatible schema")
    if band.get("plan_sha256") != plan_sha256(plan):
        raise V5ContractError("frozen band was produced from a different v5 plan")
    selected = cast(Mapping[str, object], band.get("selected_rung", {}))
    if selected.get("id") != rung.rung_id:
        raise V5ContractError(
            f"qualification rung {rung.rung_id} is not frozen rung {selected.get('id')}"
        )
    if band.get("mechanism_test_authorized") is not False:
        raise V5ContractError("frozen apparatus band may not pre-authorize mechanism tests")
    return band


def run_v5_stage(
    *,
    plan_path: Path,
    stage: str,
    rung_id: str,
    output_dir: Path,
    repo_root: Path,
    exposure_id: str | None = None,
    frozen_band_path: Path | None = None,
    resume: bool = True,
) -> dict[str, object]:
    """Execute one probe rung or one qualification exposure cell."""

    if stage not in {"probe", "qualification"}:
        raise V5ContractError("stage must be probe or qualification")
    plan = load_v5_plan(plan_path)
    plan_hash = plan_sha256(plan)
    rung = resolve_rung(plan, rung_id)
    if stage == "probe":
        if exposure_id is not None:
            raise V5ContractError("probe exposure is fixed by the plan")
        exposure = probe_exposure(plan)
        arms: tuple[ModelArm, ...] = ("cascade",)
        seeds = tuple(int(value) for value in plan["probe"]["seeds"])
        frozen_band = None
    else:
        if exposure_id is None:
            raise V5ContractError("qualification requires --exposure")
        exposure = resolve_exposure(plan, exposure_id)
        arms = cast(tuple[ModelArm, ...], V5_REQUIRED_ARMS)
        seeds = tuple(int(value) for value in plan["qualification"]["seeds"])
        frozen_band = _validate_frozen_band(
            plan=plan,
            frozen_band_path=frozen_band_path,
            rung=rung,
        )

    runtime = cast(Mapping[str, object], plan["runtime"])
    configure_runtime(runtime, seeds[0])
    device = resolve_device(runtime)
    environment = environment_snapshot(device)
    expected_base = str(plan.get("expected_base_commit", ""))
    git = git_snapshot(repo_root, expected_base)
    if not bool(plan["development_only"]):
        if git.get("head") is None:
            raise V5ContractError(
                "non-development v5 execution requires a Git checkout with a recorded HEAD"
            )
        if git.get("dirty") is not False:
            raise V5ContractError(
                "non-development v5 execution requires a clean Git checkout"
            )
        if expected_base and git.get("expected_base_is_ancestor") is not True:
            raise V5ContractError(
                "the plan's expected base commit is not an ancestor of this checkout"
            )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "plan_snapshot.json"
    if snapshot_path.is_file():
        existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != plan_hash:
            raise V5ContractError(
                f"{output_dir} already belongs to a different v5 plan"
            )
    else:
        atomic_write_json(snapshot_path, source_plan(plan))

    dataset_config = stage_dataset_config(
        plan,
        stage=stage,
        exposure=exposure,
    )
    dataset_dir = output_dir / "dataset"
    manifest = materialize_dataset(
        dataset_config=dataset_config,
        output_dir=dataset_dir,
        force=False,
    )
    manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_manifest_sha = sha256_file(manifest_path)
    validate_dataset_manifest(manifest_path)
    split_entries = cast(Mapping[str, Mapping[str, object]], manifest["splits"])
    examples = {
        split: load_examples(dataset_dir / str(split_entries[split]["path"]))
        for split in ("train", "dev", "test")
    }
    vocabulary = cast(Mapping[str, object], manifest["vocabulary"])
    vocab = SyntheticVocabulary(
        domain_size=int(vocabulary["domain_size"]),
        max_hops=int(vocabulary["max_hops"]),
        max_abs_step=int(vocabulary["max_abs_step"]),
    )
    max_sequence_length = max(
        len(example.tokens) - 1
        for split in examples.values()
        for example in split
    )

    base_config = _model_config(
        plan,
        rung=rung,
        arm="learned_delta",
        vocab_size=vocab.vocab_size,
        max_sequence_length=max_sequence_length,
    )
    initialization_audit = paired_initialization_audit(
        base_config=base_config,
        initialization_seed=derive_seed(seeds[0], "v5-model-initialization"),
    )

    cells: list[dict[str, object]] = []
    cell_index: list[dict[str, object]] = []
    for arm in arms:
        for seed in seeds:
            cell_dir = output_dir / "cells" / str(arm) / f"seed_{seed}"
            identity = _cell_identity(
                plan_hash=plan_hash,
                stage=stage,
                rung=rung,
                exposure=exposure,
                dataset_manifest_sha256=dataset_manifest_sha,
                arm=arm,
                seed=seed,
                git=git,
                environment=environment,
            )
            cell = _train_cell(
                plan=plan,
                stage=stage,
                rung=rung,
                exposure=exposure,
                arm=arm,
                seed=seed,
                train=examples["train"],
                dev=examples["dev"],
                test=examples["test"],
                vocab=vocab,
                max_sequence_length=max_sequence_length,
                device=device,
                identity=identity,
                cell_dir=cell_dir,
                resume=resume,
            )
            cells.append(cell)
            manifest_file = cell_dir / "cell_manifest.json"
            cell_index.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "path": str(manifest_file.relative_to(output_dir)),
                    "sha256": sha256_file(manifest_file),
                    "execution_fingerprint": cell["execution_fingerprint"],
                }
            )

    family_specs = cast(Mapping[str, object], dataset_config["families"])
    null_report = registered_family_null_report(
        examples["train"],
        examples["test"],
        family_specs=family_specs,
        vocab=vocab,
    )
    if stage == "probe":
        pointer_null = family_null_values(null_report)["pointer_chase"]
        holdout_accuracy = [
            _family_metric(
                cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], cell["metrics"])["test"],
                ),
                "pointer_chase",
                "accuracy",
            )
            for cell in cells
        ]
        seen_accuracy = [
            _family_metric(
                cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], cell["metrics"])["seen"],
                ),
                "pointer_chase",
                "accuracy",
            )
            for cell in cells
        ]
        rung_order = capacity_ladder(plan)
        rung_index = [item.rung_id for item in rung_order].index(rung.rung_id)
        gate = evaluate_prerequisite_probe(
            rung_id=rung.rung_id,
            rung_index=rung_index,
            rung_count=len(rung_order),
            per_seed_holdout_accuracy=holdout_accuracy,
            per_seed_seen_accuracy=seen_accuracy,
            family_null=pointer_null,
            thresholds=cast(Mapping[str, object], plan["probe"]["thresholds"]),
            development_only=bool(plan["development_only"]),
        )
        report = dict(gate)
    else:
        arm_evidence = _aggregate_qualification_arms(cells)
        hop_by_arm: dict[str, object] = {}
        for arm in V5_REQUIRED_ARMS:
            rows: list[dict[str, object]] = []
            for seed in seeds:
                cell_dir = output_dir / "cells" / arm / f"seed_{seed}"
                rows.extend(
                    _read_prediction_rows(cell_dir, "predictions_test.jsonl")
                )
            hop_by_arm[arm] = hop_count_modulation_diagnostic(rows)
        hop_diagnostic = {
            "gating": False,
            "pooled_across_arms": False,
            "per_arm": hop_by_arm,
        }
        gate = evaluate_v5_qualification(
            arms=arm_evidence,
            thresholds=competence_thresholds_from_plan(plan),
            family_null_report=null_report,
            uniform_answer_nll=math.log(vocab.domain_size),
            guardrail_minimum_margin_over_null=float(
                plan["qualification"]["guardrail_minimum_margin_over_null"]
            ),
            canary_metrics=_canary_metrics(cells),
            hop_diagnostic=hop_diagnostic,
            development_only=bool(plan["development_only"]),
        )
        report = dict(gate)

    report.update(
        {
            "execution_schema_version": V5_EXECUTION_SCHEMA_VERSION,
            "study_id": plan["study_id"],
            "plan_sha256": plan_hash,
            "plan_path": str(plan_path.resolve()),
            "registration_status": plan["registration_status"],
            "stage_execution": stage,
            "rung_id": rung.rung_id,
            "rung": rung.to_dict(),
            "exposure_id": exposure.exposure_id,
            "exposure": exposure.to_dict(),
            "dataset_manifest_sha256": dataset_manifest_sha,
            "dataset_manifest": str(manifest_path.relative_to(output_dir)),
            "dataset_config_sha256": canonical_sha256(dataset_config),
            "git": git,
            "environment": environment,
            "initialization_audit": initialization_audit,
            "cells": cell_index,
            "frozen_band_sha256": (
                None if frozen_band is None else canonical_sha256(frozen_band)
            ),
            "completed_at_utc": utc_now(),
            "status": "complete",
            "software_success": True,
            # A valid NO_GO is still a successfully completed scientific run.
            "exit_semantics": (
                "software success is independent of scientific GO/NO_GO"
            ),
        }
    )
    report_name = (
        "probe_report.json"
        if stage == "probe"
        else "qualification_report.json"
    )
    report_path = output_dir / report_name
    atomic_write_json(report_path, report)
    return report
