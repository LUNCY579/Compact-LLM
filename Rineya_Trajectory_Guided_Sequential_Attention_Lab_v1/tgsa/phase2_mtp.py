"""Competence-gated TGSA x multi-token-prediction experiment orchestration.

Phase 2 is deliberately layered on top of, rather than folded into, the frozen
Phase-1 mechanism study.  A non-confirmatory rehearsal may run immediately to
validate mechanics, losses, gradients, artifact production, and estimators.  A
confirmatory invocation fails closed unless a current Phase-1 competence report
returns ``GO`` and the Phase-2 registration is explicitly frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .metrics import count_parameters
from .mtp import (
    MTPConfig,
    MTPMode,
    TinyTGSAWithMTP,
    answer_horizon_metrics,
    build_tgsa_mtp_model,
    combined_tgsa_mtp_loss,
    horizon_token_metrics,
    multi_token_prediction_loss,
    teacher_forced_prefix_metrics,
)
from .phase1_competence import require_go
from .phase1_data import (
    Phase1Batch,
    Phase1Example,
    canonical_json_bytes,
    derive_seed,
    deterministic_batch_stream,
    fixed_batches,
    load_examples,
    materialize_dataset,
    sha256_bytes,
    sha256_file,
    validate_dataset_manifest,
)
from .phase1_model import (
    ModelArm,
    TinyLMConfig,
    answer_logits,
    build_canonical_supernet,
    state_dict_sha256,
)

PHASE2_SCHEMA_VERSION = "tgsa.phase2-mtp-complementarity.v1"
PHASE2_REPORT_VERSION = "tgsa.phase2-mtp-report.v1"
PHASE2_RUN_VERSION = "tgsa.phase2-mtp-run.v1"
SUPPORTED_MTP_MODES: tuple[MTPMode, ...] = ("dormant", "backbone", "trajectory")
SUPPORTED_TGSA_ARMS: tuple[ModelArm, ...] = (
    "parallel_mha",
    "cascade",
    "history_control",
    "raw_delta",
    "learned_delta",
    "qkv_conditioned",
    "predictor_corrector",
    "full",
)


@dataclass(frozen=True, slots=True)
class Phase2Cell:
    tgsa_arm: ModelArm
    mtp_mode: MTPMode

    @property
    def key(self) -> str:
        return f"{self.tgsa_arm}__mtp_{self.mtp_mode}"

    def to_dict(self) -> dict[str, str]:
        return {"tgsa_arm": self.tgsa_arm, "mtp_mode": self.mtp_mode, "key": self.key}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
    _atomic_write_text(path, payload)


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


def _git(args: Sequence[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_snapshot(repo_root: Path, expected_base_commit: str) -> dict[str, object]:
    head = _git(["rev-parse", "HEAD"], cwd=repo_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    dirty_text = _git(["status", "--porcelain"], cwd=repo_root)
    base_is_ancestor: bool | None = None
    if head is not None:
        try:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", expected_base_commit, "HEAD"],
                cwd=repo_root,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            base_is_ancestor = result.returncode == 0
        except OSError:
            base_is_ancestor = None
    return {
        "head": head,
        "branch": branch,
        "dirty": None if dirty_text is None else bool(dirty_text),
        "expected_base_commit": expected_base_commit,
        "expected_base_is_ancestor": base_is_ancestor,
    }


def environment_snapshot(device: torch.device) -> dict[str, object]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "cpu_threads": torch.get_num_threads(),
    }


def _configure_runtime(runtime: Mapping[str, object], seed: int) -> torch.device:
    requested = str(runtime.get("device", "auto"))
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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
    return device


def _parse_cells(value: object) -> tuple[Phase2Cell, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("cells must be a list")
    cells: list[Phase2Cell] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise TypeError("every cell must be an object")
        tgsa_arm = str(entry.get("tgsa_arm"))
        mtp_mode = str(entry.get("mtp_mode"))
        if tgsa_arm not in SUPPORTED_TGSA_ARMS:
            raise ValueError(f"unsupported TGSA arm in Phase-2 cell: {tgsa_arm}")
        if mtp_mode not in SUPPORTED_MTP_MODES:
            raise ValueError(f"unsupported MTP mode in Phase-2 cell: {mtp_mode}")
        if tgsa_arm == "parallel_mha" and mtp_mode == "trajectory":
            raise ValueError("parallel_mha has no TGSA trajectory control to expose")
        cells.append(Phase2Cell(cast(ModelArm, tgsa_arm), cast(MTPMode, mtp_mode)))
    if not cells:
        raise ValueError("at least one Phase-2 cell is required")
    keys = [cell.key for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("Phase-2 cells must be unique")
    return tuple(cells)


def load_phase2_config(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Phase-2 config must be a JSON object")
    if value.get("schema_version") != PHASE2_SCHEMA_VERSION:
        raise ValueError("unsupported Phase-2 config schema")
    for key in (
        "study_id",
        "confirmatory",
        "expected_base_commit",
        "cells",
        "seeds",
        "dataset",
        "model",
        "mtp",
        "training",
        "evaluation",
        "runtime",
        "registration",
    ):
        if key not in value:
            raise ValueError(f"Phase-2 config is missing {key!r}")
    if not isinstance(value["confirmatory"], bool):
        raise TypeError("confirmatory must be an explicit boolean")
    cells = _parse_cells(value["cells"])
    value["_parsed_cells"] = cells
    seeds = tuple(int(seed) for seed in value["seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if int(value["training"]["steps"]) <= 0:
        raise ValueError("training.steps must be positive")
    if int(value["training"]["batch_size"]) <= 0:
        raise ValueError("training.batch_size must be positive")
    if float(value["model"].get("dropout", 0.0)) != 0.0:
        raise ValueError("paired Phase-2 studies require dropout=0")
    if str(value["evaluation"].get("split", "dev")) != "dev":
        raise ValueError(
            "Phase-2 feasibility/calibration may evaluate only the development split; "
            "the test bank remains sealed until a later frozen confirmatory protocol"
        )

    mtp = value["mtp"]
    MTPConfig(
        max_horizon=int(mtp.get("max_horizon", 3)),
        transition_rank=int(mtp.get("transition_rank", 64)),
        loss_weight=float(mtp.get("loss_weight", 0.20)),
        mode="backbone",
        horizon_weights=tuple(float(item) for item in mtp.get("horizon_weights", [])),
        control_block_index=int(mtp.get("control_block_index", -1)),
        pad_token_id=int(mtp.get("pad_token_id", 0)),
        bias=bool(mtp.get("bias", False)),
    )

    registration = value["registration"]
    if not isinstance(registration, Mapping):
        raise TypeError("registration must be an object")
    if bool(value["confirmatory"]):
        if str(registration.get("status")) != "frozen":
            raise ValueError("confirmatory Phase-2 requires registration.status='frozen'")
        competence_report = registration.get("phase1_competence_report")
        if not isinstance(competence_report, str) or not competence_report:
            raise ValueError("confirmatory Phase-2 requires a frozen Phase-1 competence report")
        # The report path is resolved by ``run_phase2_study`` relative to the
        # config file.  Validation here intentionally cannot soften this rule.
    return value


def _model_config(
    study: Mapping[str, object],
    *,
    arm: ModelArm,
    vocabulary: Mapping[str, object],
    max_sequence_length: int,
) -> TinyLMConfig:
    model = cast(Mapping[str, object], study["model"])
    configured_max = int(model.get("max_sequence_length", max_sequence_length))
    if configured_max < max_sequence_length:
        raise ValueError("model.max_sequence_length is smaller than the materialized data")
    return TinyLMConfig(
        vocab_size=int(vocabulary["vocab_size"]),
        max_sequence_length=configured_max,
        embed_dim=int(model["embed_dim"]),
        num_heads=int(model["num_heads"]),
        num_layers=int(model["num_layers"]),
        ffn_multiplier=float(model.get("ffn_multiplier", 4.0)),
        dropout=float(model.get("dropout", 0.0)),
        bias=bool(model.get("bias", False)),
        trajectory_rank=int(model.get("trajectory_rank", 16)),
        arm=arm,
        tie_embeddings=bool(model.get("tie_embeddings", True)),
    )


def _mtp_config(study: Mapping[str, object], *, mode: MTPMode) -> MTPConfig:
    value = cast(Mapping[str, object], study["mtp"])
    return MTPConfig(
        max_horizon=int(value.get("max_horizon", 3)),
        transition_rank=int(value.get("transition_rank", 64)),
        loss_weight=float(value.get("loss_weight", 0.20)),
        mode=mode,
        horizon_weights=tuple(float(item) for item in value.get("horizon_weights", [])),
        control_block_index=int(value.get("control_block_index", -1)),
        pad_token_id=int(value.get("pad_token_id", 0)),
        bias=bool(value.get("bias", False)),
    )


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm().item())
        total += value * value
    return math.sqrt(total)


def _auxiliary_gradient_audit(
    model: TinyTGSAWithMTP,
    batch: Phase1Batch,
) -> dict[str, object]:
    model.zero_grad(set_to_none=True)
    output = model(
        batch.input_ids,
        labels=batch.labels,
        key_padding_mask=batch.key_padding_mask,
    )
    if output.trajectory_control.requires_grad:
        output.trajectory_control.retain_grad()
    mtp_loss, _, token_counts = multi_token_prediction_loss(
        output.mtp_logits,
        batch.labels,
        horizon_weights=model.mtp_config.weights,
    )
    mtp_loss.backward()
    control_grad = output.trajectory_control.grad
    report = {
        "mtp_loss": float(mtp_loss.detach().item()),
        "per_horizon_tokens": {str(key): value for key, value in token_counts.items()},
        "trajectory_control_requires_grad": bool(output.trajectory_control.requires_grad),
        "trajectory_control_gradient_norm": (
            None if control_grad is None else float(control_grad.detach().float().norm().item())
        ),
        "mtp_head_gradient_norm": _gradient_norm(model.mtp_head.parameters()),
        "backbone_gradient_norm": _gradient_norm(model.backbone.parameters()),
        "control_projection_gradient_norm": _gradient_norm(
            model.mtp_head.control_projection.parameters()
        ),
        "claim_scope": "mechanical gradient-path audit; not capability evidence",
    }
    model.zero_grad(set_to_none=True)
    return report


def _optimizer(model: nn.Module, training: Mapping[str, object]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1.0e-3)),
        betas=(float(training.get("beta1", 0.9)), float(training.get("beta2", 0.95))),
        eps=float(training.get("epsilon", 1.0e-8)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def _train_cell(
    model: TinyTGSAWithMTP,
    *,
    train_examples: Sequence[Phase1Example],
    training: Mapping[str, object],
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], float]:
    model.train()
    optimizer = _optimizer(model, training)
    steps = int(training["steps"])
    batch_size = int(training["batch_size"])
    clip_norm = float(training.get("gradient_clip_norm", 1.0))
    stream = deterministic_batch_stream(
        train_examples,
        batch_size=batch_size,
        seed=derive_seed(seed, "phase2_mtp_training_order"),
    )
    logs: list[dict[str, object]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        epoch, cpu_batch = next(stream)
        batch = cpu_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.input_ids,
            labels=batch.labels,
            key_padding_mask=batch.key_padding_mask,
        )
        objective = combined_tgsa_mtp_loss(
            output,
            batch.labels,
            batch.loss_mask,
            mtp_config=model.mtp_config,
        )
        if not torch.isfinite(objective.total_loss):
            raise FloatingPointError(f"non-finite Phase-2 loss at step {step}")
        objective.total_loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm).item()
        )
        optimizer.step()
        logs.append(
            {
                "step": step,
                "epoch": epoch,
                "total_loss": float(objective.total_loss.detach().item()),
                "autoregressive_loss": float(objective.autoregressive_loss.detach().item()),
                "mtp_loss": float(objective.mtp_loss.detach().item()),
                "gradient_norm_before_clip": gradient_norm,
                "batch_example_ids_sha256": sha256_bytes(
                    canonical_json_bytes(list(batch.example_ids))
                ),
            }
        )
    return logs, time.perf_counter() - started


@torch.no_grad()
def _evaluate_cell(
    model: TinyTGSAWithMTP,
    *,
    examples: Sequence[Phase1Example],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]], float]:
    model.eval()
    total_examples = 0
    answer_correct = 0
    answer_nll_sum = 0.0
    family_total: dict[str, int] = defaultdict(int)
    family_correct: dict[str, int] = defaultdict(int)
    horizon_tokens: dict[int, int] = defaultdict(int)
    horizon_correct_weighted: dict[int, float] = defaultdict(float)
    horizon_nll_weighted: dict[int, float] = defaultdict(float)
    answer_horizon_examples: dict[int, int] = defaultdict(int)
    answer_horizon_correct_weighted: dict[int, float] = defaultdict(float)
    answer_horizon_nll_weighted: dict[int, float] = defaultdict(float)
    prefix_eligible = 0
    prefix_consecutive_sum = 0.0
    prefix_full_sum = 0.0
    control_square_sum = 0.0
    control_elements = 0
    prediction_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for cpu_batch in fixed_batches(examples, batch_size=batch_size):
        batch = cpu_batch.to(device)
        output = model(
            batch.input_ids,
            labels=batch.labels,
            key_padding_mask=batch.key_padding_mask,
        )
        selected_logits = answer_logits(output.logits, batch.answer_input_indices)
        predictions = selected_logits.argmax(dim=-1)
        correct = predictions.eq(batch.answer_tokens)
        losses = F.cross_entropy(
            selected_logits.float(), batch.answer_tokens, reduction="none"
        )
        count = int(batch.answer_tokens.shape[0])
        total_examples += count
        answer_correct += int(correct.sum().item())
        answer_nll_sum += float(losses.sum().item())
        for row, family in enumerate(batch.families):
            family_total[family] += 1
            family_correct[family] += int(correct[row].item())
            prediction_rows.append(
                {
                    "example_id": batch.example_ids[row],
                    "family": family,
                    "stratum": batch.strata[row],
                    "target": int(batch.answer_tokens[row].item()),
                    "prediction": int(predictions[row].item()),
                    "correct": bool(correct[row].item()),
                    "answer_nll": float(losses[row].item()),
                }
            )

        for horizon, metric in horizon_token_metrics(output.mtp_logits, batch.labels).items():
            tokens = int(metric["tokens"])
            horizon_tokens[horizon] += tokens
            horizon_correct_weighted[horizon] += float(metric["accuracy"]) * tokens
            horizon_nll_weighted[horizon] += float(metric["nll"]) * tokens

        for horizon, metric in answer_horizon_metrics(
            output,
            answer_input_indices=batch.answer_input_indices,
            answer_tokens=batch.answer_tokens,
        ).items():
            eligible = int(metric["examples"])
            answer_horizon_examples[horizon] += eligible
            answer_horizon_correct_weighted[horizon] += float(metric["accuracy"]) * eligible
            answer_horizon_nll_weighted[horizon] += float(metric["nll"]) * eligible

        prefix = teacher_forced_prefix_metrics(output, batch.labels)
        eligible = int(prefix["eligible_positions"])
        prefix_eligible += eligible
        prefix_consecutive_sum += float(prefix["mean_consecutive_correct"]) * eligible
        prefix_full_sum += float(prefix["full_horizon_fraction"]) * eligible

        valid = ~batch.key_padding_mask
        control = output.trajectory_control[valid]
        if control.numel() > 0:
            control_square_sum += float(control.float().square().sum().item())
            control_elements += int(control.numel())

    elapsed = time.perf_counter() - started
    if total_examples == 0:
        raise ValueError("Phase-2 evaluation split is empty")
    summary: dict[str, object] = {
        "examples": total_examples,
        "answer_accuracy": answer_correct / total_examples,
        "answer_nll": answer_nll_sum / total_examples,
        "per_family_accuracy": {
            family: family_correct[family] / family_total[family]
            for family in sorted(family_total)
        },
        "mtp_horizons": {
            str(horizon): {
                "tokens": horizon_tokens[horizon],
                "accuracy": (
                    horizon_correct_weighted[horizon] / horizon_tokens[horizon]
                    if horizon_tokens[horizon]
                    else 0.0
                ),
                "nll": (
                    horizon_nll_weighted[horizon] / horizon_tokens[horizon]
                    if horizon_tokens[horizon]
                    else 0.0
                ),
            }
            for horizon in sorted(horizon_tokens)
        },
        "answer_horizons": {
            str(horizon): {
                "examples": answer_horizon_examples[horizon],
                "accuracy": (
                    answer_horizon_correct_weighted[horizon]
                    / answer_horizon_examples[horizon]
                    if answer_horizon_examples[horizon]
                    else 0.0
                ),
                "nll": (
                    answer_horizon_nll_weighted[horizon]
                    / answer_horizon_examples[horizon]
                    if answer_horizon_examples[horizon]
                    else 0.0
                ),
            }
            for horizon in sorted(answer_horizon_examples)
        },
        "teacher_forced_future_prefix": {
            "eligible_positions": prefix_eligible,
            "mean_consecutive_correct": (
                prefix_consecutive_sum / prefix_eligible if prefix_eligible else 0.0
            ),
            "full_horizon_fraction": (
                prefix_full_sum / prefix_eligible if prefix_eligible else 0.0
            ),
            "claim_scope": "teacher-forced quality surrogate; not throughput",
        },
        "trajectory_control_rms": (
            math.sqrt(control_square_sum / control_elements) if control_elements else 0.0
        ),
        "elapsed_seconds": elapsed,
        "examples_per_second": total_examples / elapsed if elapsed > 0 else None,
        "efficiency_is_gating": False,
    }
    return summary, prediction_rows, elapsed


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _paired_contrast(
    runs: Sequence[Mapping[str, object]],
    *,
    left: str,
    right: str,
    metric: str,
) -> dict[str, object]:
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for run in runs:
        cell = str(run["cell_key"])
        if cell not in {left, right}:
            continue
        evaluation = cast(Mapping[str, object], run["evaluation"])
        by_seed[int(run["seed"])][cell] = float(evaluation[metric])
    paired = [
        {"seed": seed, "difference": values[left] - values[right]}
        for seed, values in sorted(by_seed.items())
        if left in values and right in values
    ]
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "paired_seeds": len(paired),
        "differences": paired,
        "mean_difference": _mean([float(item["difference"]) for item in paired]),
    }


def analyze_factorial(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compute preregistration-ready paired contrasts without a success label."""

    keys = {str(run["cell_key"]) for run in runs}
    contrasts: dict[str, object] = {}
    for arm in ("parallel_mha", "cascade", "history_control", "learned_delta"):
        dormant = f"{arm}__mtp_dormant"
        backbone = f"{arm}__mtp_backbone"
        if {dormant, backbone}.issubset(keys):
            contrasts[f"mtp_effect__{arm}"] = _paired_contrast(
                runs,
                left=backbone,
                right=dormant,
                metric="answer_accuracy",
            )
        trajectory = f"{arm}__mtp_trajectory"
        if {backbone, trajectory}.issubset(keys):
            contrasts[f"trajectory_conditioning_effect__{arm}"] = _paired_contrast(
                runs,
                left=trajectory,
                right=backbone,
                metric="answer_accuracy",
            )

    def interaction(
        name: str,
        a_left: str,
        a_right: str,
        b_left: str,
        b_right: str,
        metric: str,
    ) -> None:
        required = {a_left, a_right, b_left, b_right}
        if not required.issubset(keys):
            return
        by_seed: dict[int, dict[str, float]] = defaultdict(dict)
        for run in runs:
            cell = str(run["cell_key"])
            if cell in required:
                evaluation = cast(Mapping[str, object], run["evaluation"])
                by_seed[int(run["seed"])][cell] = float(evaluation[metric])
        rows = []
        for seed, values in sorted(by_seed.items()):
            if not required.issubset(values):
                continue
            first = values[a_left] - values[a_right]
            second = values[b_left] - values[b_right]
            rows.append(
                {
                    "seed": seed,
                    "first_difference": first,
                    "second_difference": second,
                    "difference_in_differences": first - second,
                }
            )
        contrasts[name] = {
            "metric": metric,
            "paired_seeds": len(rows),
            "rows": rows,
            "mean_difference_in_differences": _mean(
                [float(row["difference_in_differences"]) for row in rows]
            ),
        }

    interaction(
        "mtp_x_trajectory_mechanism",
        "learned_delta__mtp_backbone",
        "learned_delta__mtp_dormant",
        "history_control__mtp_backbone",
        "history_control__mtp_dormant",
        "answer_accuracy",
    )
    interaction(
        "trajectory_readout_specificity",
        "learned_delta__mtp_trajectory",
        "learned_delta__mtp_backbone",
        "history_control__mtp_trajectory",
        "history_control__mtp_backbone",
        "answer_accuracy",
    )
    return {
        "contrasts": contrasts,
        "decision": "descriptive_only",
        "reason": (
            "This implementation validates the factorial estimands but does not "
            "invent confirmatory thresholds after observing a rehearsal."
        ),
    }


def _relative_or_absolute(path: str, *, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def run_phase2_study(
    *,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    repo_root: Path,
    overwrite: bool = False,
    force_data: bool = False,
) -> dict[str, object]:
    """Run a competence-gated TGSA x MTP development study."""

    config_path = config_path.resolve()
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    repo_root = repo_root.resolve()
    study = load_phase2_config(config_path)
    cells = cast(tuple[Phase2Cell, ...], study.pop("_parsed_cells"))

    # Fail closed before creating either the output root or the materialized
    # dataset.  A missing/NO_GO competence report must leave no directory that
    # could be mistaken for a partially dispatched confirmatory experiment.
    registration = cast(Mapping[str, object], study["registration"])
    competence: dict[str, object] | None = None
    if bool(study["confirmatory"]):
        report_path = _relative_or_absolute(
            str(registration["phase1_competence_report"]),
            base=config_path.parent,
        )
        competence = require_go(report_path)

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite or a fresh root"
        )
    if overwrite and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = materialize_dataset(
        dataset_config=cast(Mapping[str, object], study["dataset"]),
        output_dir=dataset_dir,
        force=force_data,
    )
    validate_dataset_manifest(dataset_dir / "dataset_manifest.json")
    split_entries = cast(Mapping[str, Mapping[str, object]], manifest["splits"])
    train_examples = load_examples(dataset_dir / str(split_entries["train"]["path"]))
    dev_examples = load_examples(dataset_dir / str(split_entries["dev"]["path"]))
    max_sequence = max(
        int(split_entries[split]["max_tokens"]) - 1 for split in ("train", "dev", "test")
    )
    vocabulary = cast(Mapping[str, object], manifest["vocabulary"])
    seeds = tuple(int(seed) for seed in study["seeds"])
    training = cast(Mapping[str, object], study["training"])
    evaluation = cast(Mapping[str, object], study["evaluation"])
    runtime = cast(Mapping[str, object], study["runtime"])
    config_sha = sha256_file(config_path)
    dataset_sha = sha256_file(dataset_dir / "dataset_manifest.json")
    git = git_snapshot(repo_root, str(study["expected_base_commit"]))

    runs: list[dict[str, object]] = []
    initialization_audits: list[dict[str, object]] = []
    for seed in seeds:
        device = _configure_runtime(runtime, seed)
        base_config = _model_config(
            study,
            arm="learned_delta",
            vocabulary=vocabulary,
            max_sequence_length=max_sequence,
        )
        canonical = build_canonical_supernet(base_config, initialization_seed=seed)
        canonical_state = {
            key: value.detach().clone() for key, value in canonical.state_dict().items()
        }
        del canonical

        initial_by_cell: dict[str, dict[str, str]] = {}
        built: list[tuple[Phase2Cell, TinyTGSAWithMTP]] = []
        for cell in cells:
            lm_config = _model_config(
                study,
                arm=cell.tgsa_arm,
                vocabulary=vocabulary,
                max_sequence_length=max_sequence,
            )
            model = build_tgsa_mtp_model(
                lm_config,
                _mtp_config(study, mode=cell.mtp_mode),
                initialization_seed=seed,
                canonical_state=canonical_state,
            )
            initial_by_cell[cell.key] = {
                "backbone_sha256": state_dict_sha256(model.backbone),
                "mtp_head_sha256": state_dict_sha256(model.mtp_head),
            }
            built.append((cell, model))

        # MTP parameters must be byte-identical across every cell.  Backbones
        # must also be identical across MTP modes within a TGSA arm.
        mtp_hashes = {entry["mtp_head_sha256"] for entry in initial_by_cell.values()}
        if len(mtp_hashes) != 1:
            raise RuntimeError("MTP heads did not receive identical paired initialization")
        by_arm_hashes: dict[str, set[str]] = defaultdict(set)
        for cell in cells:
            by_arm_hashes[cell.tgsa_arm].add(
                initial_by_cell[cell.key]["backbone_sha256"]
            )
        if any(len(values) != 1 for values in by_arm_hashes.values()):
            raise RuntimeError("an MTP mode changed its TGSA backbone initialization")
        initialization_audits.append(
            {
                "seed": seed,
                "all_mtp_heads_identical": True,
                "mtp_head_sha256": next(iter(mtp_hashes)),
                "backbone_identical_within_tgsa_arm": True,
                "cells": initial_by_cell,
            }
        )

        for cell, model in built:
            model = model.to(device)
            audit_batch = next(
                deterministic_batch_stream(
                    train_examples,
                    batch_size=min(int(training["batch_size"]), len(train_examples)),
                    seed=derive_seed(seed, "phase2_gradient_audit"),
                )
            )[1].to(device)
            gradient_audit = _auxiliary_gradient_audit(model, audit_batch)
            if float(gradient_audit["mtp_head_gradient_norm"]) <= 0.0:
                raise RuntimeError(f"MTP head received no gradient in cell {cell.key}")
            if cell.mtp_mode == "trajectory":
                control_grad = gradient_audit["trajectory_control_gradient_norm"]
                if control_grad is None or float(control_grad) <= 0.0:
                    raise RuntimeError(
                        f"trajectory-conditioned cell {cell.key} has no control gradient"
                    )

            logs, train_seconds = _train_cell(
                model,
                train_examples=train_examples,
                training=training,
                seed=seed,
                device=device,
            )
            summary, predictions, eval_seconds = _evaluate_cell(
                model,
                examples=dev_examples,
                batch_size=int(evaluation.get("batch_size", 64)),
                device=device,
            )
            run_dir = output_dir / "runs" / cell.key / f"seed_{seed}"
            checkpoint_path = run_dir / "final_model.pt"
            atomic_torch_save(
                checkpoint_path,
                {
                    "schema_version": PHASE2_RUN_VERSION,
                    "study_id": study["study_id"],
                    "cell": cell.to_dict(),
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "lm_config": asdict(model.backbone.config),
                    "mtp_config": asdict(model.mtp_config),
                },
            )
            write_jsonl(run_dir / "training.jsonl", logs)
            write_jsonl(run_dir / "predictions_dev.jsonl", predictions)
            run: dict[str, object] = {
                "schema_version": PHASE2_RUN_VERSION,
                "status": "complete",
                "study_id": study["study_id"],
                "cell_key": cell.key,
                "cell": cell.to_dict(),
                "seed": seed,
                "initialization": initial_by_cell[cell.key],
                "gradient_audit": gradient_audit,
                "training": {
                    "steps": int(training["steps"]),
                    "elapsed_seconds": train_seconds,
                    "final_log": logs[-1],
                    "fixed_final_checkpoint": True,
                    "early_stopping": False,
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                },
                "evaluation": summary,
                "artifacts": {
                    "training_log": "training.jsonl",
                    "training_log_sha256": sha256_file(run_dir / "training.jsonl"),
                    "predictions_dev": "predictions_dev.jsonl",
                    "predictions_dev_sha256": sha256_file(run_dir / "predictions_dev.jsonl"),
                },
                "parameters": {
                    "total_trainable": count_parameters(model),
                    "backbone_trainable": count_parameters(model.backbone),
                    "mtp_head_trainable": count_parameters(model.mtp_head),
                },
                "source": {
                    "study_config_sha256": config_sha,
                    "dataset_manifest_sha256": dataset_sha,
                    "git": git,
                },
                "claim_scope": (
                    "development-only TGSA x MTP feasibility evidence; no sealed test-bank, "
                    "speculative-throughput, or Rineya production-quality claim"
                ),
                "efficiency": {
                    "train_seconds": train_seconds,
                    "evaluation_seconds": eval_seconds,
                    "gating": False,
                },
            }
            atomic_write_json(run_dir / "run_manifest.json", run)
            runs.append(run)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    analysis = analyze_factorial(runs)
    report: dict[str, object] = {
        "schema_version": PHASE2_REPORT_VERSION,
        "study_id": study["study_id"],
        "status": "complete",
        "confirmatory": bool(study["confirmatory"]),
        "decision": "descriptive_only" if bool(study["confirmatory"]) else "smoke_only",
        "competence_precondition": (
            {
                "required": True,
                "verdict": competence["verdict"],
                "schema_version": competence["schema_version"],
            }
            if competence is not None
            else {
                "required": False,
                "verdict": "NOT_EVALUATED_FOR_NON_CONFIRMATORY_REHEARSAL",
            }
        ),
        "cells": [cell.to_dict() for cell in cells],
        "seeds": list(seeds),
        "runs": runs,
        "initialization_audits": initialization_audits,
        "analysis": analysis,
        "source": {
            "config": str(config_path),
            "study_config_sha256": config_sha,
            "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
            "dataset_manifest_sha256": dataset_sha,
            "git": git,
            "environment": environment_snapshot(_configure_runtime(runtime, seeds[0])),
        },
        "claim_scope": (
            "This report validates a development-only factorial TGSA x MTP workflow. "
            "It does not establish TGSA efficacy, MTP quality improvement, speculative "
            "speedup, or production Rineya performance."
        ),
        "next_gate": (
            "Obtain a frozen Phase-1 v4 competence GO, freeze the Phase-2 calibration "
            "and estimators without reading a sealed test bank, then run a powered paired "
            "matrix. Runtime speed requires a separate cache-aware speculative verifier."
        ),
    }
    atomic_write_json(output_dir / "phase2_report.json", report)
    return report


__all__ = [
    "PHASE2_REPORT_VERSION",
    "PHASE2_SCHEMA_VERSION",
    "Phase2Cell",
    "analyze_factorial",
    "load_phase2_config",
    "run_phase2_study",
]
