"""Training, evaluation, and evidence orchestration for TGSA Phase 1.

The runner is intentionally conservative:

* every arm consumes the same materialized examples in the same order;
* learned-delta and history-control use byte-identical initialization;
* the final checkpoint is selected by a fixed step count, never by dev score;
* every prediction is preserved for paired re-analysis;
* exceptions emit a failure bundle with traceback and the last batch identity;
* trajectory interventions are evaluated only after training and never tune it.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import torch
from torch import Tensor

from .metrics import telemetry_summary
from .phase1_data import (
    Phase1Batch,
    Phase1Example,
    canonical_json_bytes,
    collate_examples,
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
    ControlCapture,
    ControlIntervention,
    InterventionMode,
    ModelArm,
    ReplayIntervention,
    TinyLMConfig,
    TinyTGSAForCausalLM,
    answer_logits,
    build_canonical_supernet,
    build_model,
    gradient_group_norms,
    length_matched_derangement,
    masked_answer_loss,
    model_debug_snapshot,
    paired_initialization_audit,
    parameter_audit,
    state_dict_sha256,
)

STUDY_SCHEMA_VERSION = "tgsa.phase1-model-identifiability-study.v1"
RUN_MANIFEST_VERSION = "tgsa.phase1-model-run.v1"
MATRIX_MANIFEST_VERSION = "tgsa.phase1-model-matrix.v1"
PREDICTION_SCHEMA_VERSION = "tgsa.phase1-prediction.v1"
REQUIRED_ARMS: tuple[ModelArm, ...] = ("cascade", "history_control", "learned_delta")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_study_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("study config must be a JSON object")
    if value.get("schema_version") != STUDY_SCHEMA_VERSION:
        raise ValueError("unsupported study config schema")
    for key in (
        "study_id",
        "confirmatory",
        "expected_base_commit",
        "arms",
        "seeds",
        "dataset",
        "model",
        "training",
        "evaluation",
        "analysis",
        "runtime",
    ):
        if key not in value:
            raise ValueError(f"study config is missing {key!r}")
    # `confirmatory` must be an explicit boolean. A missing or coercible value is
    # how a smoke configuration silently acquired an evidentiary decision before
    # the launch audit (finding P0-4).
    if not isinstance(value["confirmatory"], bool):
        raise TypeError("study config field 'confirmatory' must be a boolean")
    arms = tuple(str(arm) for arm in value["arms"])
    if len(arms) != len(set(arms)):
        raise ValueError("study arms must be unique")
    if not set(REQUIRED_ARMS).issubset(arms):
        raise ValueError(f"study must include required arms: {REQUIRED_ARMS}")
    seeds = tuple(int(seed) for seed in value["seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("study seeds must be non-empty and unique")
    if int(value["training"]["steps"]) <= 0:
        raise ValueError("training.steps must be positive")
    if int(value["training"]["batch_size"]) <= 0:
        raise ValueError("training.batch_size must be positive")
    if float(value["model"].get("dropout", 0.0)) != 0.0:
        raise ValueError("registered Phase-1 configs require dropout=0 for paired determinism")
    return value


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


def append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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
    dirty = _git(["status", "--porcelain"], cwd=repo_root)
    base_is_ancestor: bool | None = None
    if head is not None:
        try:
            completed = subprocess.run(
                ["git", "merge-base", "--is-ancestor", expected_base_commit, "HEAD"],
                cwd=repo_root,
                check=False,
            )
            base_is_ancestor = completed.returncode == 0
        except OSError:
            base_is_ancestor = None
    return {
        "head": head,
        "branch": branch,
        "dirty": None if dirty is None else bool(dirty),
        "expected_base_commit": expected_base_commit,
        "expected_base_is_ancestor": base_is_ancestor,
    }


def environment_snapshot(device: torch.device) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(),
    }


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


def _model_config_from_study(
    study: Mapping[str, object],
    *,
    arm: ModelArm,
    vocabulary: Mapping[str, object],
    max_sequence_length: int,
) -> TinyLMConfig:
    model = cast(Mapping[str, object], study["model"])
    configured_max = int(model.get("max_sequence_length", max_sequence_length))
    if configured_max < max_sequence_length:
        raise ValueError(
            f"model.max_sequence_length={configured_max} is smaller than materialized data "
            f"maximum {max_sequence_length}"
        )
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


def prepare_study_dataset(
    study: Mapping[str, object],
    *,
    dataset_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    dataset = cast(Mapping[str, object], study["dataset"])
    manifest = materialize_dataset(dataset_config=dataset, output_dir=dataset_dir, force=force)
    return cast(dict[str, object], manifest)


def load_study_examples(dataset_dir: Path) -> dict[str, list[Phase1Example]]:
    manifest = validate_dataset_manifest(dataset_dir / "dataset_manifest.json")
    split_entries = cast(Mapping[str, Mapping[str, object]], manifest["splits"])
    return {
        split: load_examples(dataset_dir / str(split_entries[split]["path"]))
        for split in ("train", "dev", "test")
    }


def _evaluation_chunks(count: int, batch_size: int) -> list[tuple[int, int]]:
    if count <= 0 or batch_size <= 0:
        raise ValueError("evaluation count and batch size must be positive")
    if count == 1:
        return [(0, 1)]
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = min(start + batch_size, count)
        if count - end == 1:
            end -= 1
        if end <= start:
            end = count
        chunks.append((start, end))
        start = end
    if chunks[-1][1] != count:
        raise AssertionError("evaluation chunking did not consume all examples")
    return chunks


def _batch_signature(batch: Phase1Batch) -> str:
    return sha256_bytes(canonical_json_bytes(list(batch.example_ids)))


REUSE_IDENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("study_config_sha256", "study_config_sha256"),
    ("dataset_manifest_sha256", "dataset_manifest_sha256"),
    ("git.head", "current_git_head"),
)


def _manifest_field(manifest: Mapping[str, object], dotted: str) -> object:
    value: object = manifest
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def validate_completed_run_reuse(
    *,
    output_root: Path,
    arms: Sequence[str],
    seeds: Sequence[int],
    study_config_sha256: str,
    dataset_manifest_sha256: str,
    current_git_head: str | None,
) -> dict[str, object]:
    """Refuse to reuse a finished run that does not match this invocation.

    ``train_one_run`` short-circuits on an existing complete ``run_manifest.json``.
    Without this guard that short-circuit silently mixes runs produced by a
    different configuration, a different materialized dataset, or a different
    source commit into one matrix (``PHASE1_LAUNCH_AUDIT.md`` finding P0-3).

    Incomplete runs are left alone: they are resumable, not stale. Only runs
    that already claim ``trained`` or ``complete`` are checked, and every
    identity field must match exactly.
    """

    expected = {
        "study_config_sha256": str(study_config_sha256),
        "dataset_manifest_sha256": str(dataset_manifest_sha256),
        "current_git_head": None if current_git_head is None else str(current_git_head),
    }
    inspected: list[dict[str, object]] = []
    for arm in arms:
        for seed in seeds:
            manifest_path = output_root / "runs" / str(arm) / f"seed_{int(seed)}" / "run_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = str(manifest.get("status", ""))
            if status not in {"trained", "complete"}:
                continue
            if str(manifest.get("arm")) != str(arm) or int(manifest.get("seed", -1)) != int(seed):
                raise ValueError(
                    f"stale completed run {manifest_path}: arm/seed identity mismatch "
                    f"(manifest={manifest.get('arm')!r}/{manifest.get('seed')!r}, "
                    f"expected={arm!r}/{seed!r})"
                )
            for dotted, expected_key in REUSE_IDENTITY_FIELDS:
                actual = _manifest_field(manifest, dotted)
                wanted = expected[expected_key]
                if wanted is None:
                    continue
                if actual != wanted:
                    raise ValueError(
                        f"stale completed run {manifest_path}: {dotted} mismatch "
                        f"(manifest={actual!r}, current={wanted!r}); rerun into a fresh "
                        f"output root instead of mixing evidence"
                    )
            inspected.append(
                {
                    "arm": str(arm),
                    "seed": int(seed),
                    "status": status,
                    "manifest": str(manifest_path),
                }
            )
    return {
        "checked_runs": inspected,
        "reusable_run_count": len(inspected),
        "study_config_sha256": expected["study_config_sha256"],
        "dataset_manifest_sha256": expected["dataset_manifest_sha256"],
        "git_head": expected["current_git_head"],
    }


def _truncate_train_log(path: Path, *, completed_steps: int) -> dict[str, object]:
    """Drop log records beyond the restored checkpoint boundary.

    A crash between an appended log record and the next checkpoint leaves rows
    for steps the restored model never performed. Resuming then appends those
    steps a second time, so the evidentiary log stops being a unique increasing
    step sequence (``PHASE1_LAUNCH_AUDIT.md`` finding P1-1). The model
    trajectory is unaffected; the log is what needs repair.
    """

    if not path.is_file():
        return {"truncated": False, "removed_records": 0, "retained_records": 0}
    retained: list[Mapping[str, object]] = []
    removed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("step", 0)) <= completed_steps:
                retained.append(row)
            else:
                removed += 1
    steps = [int(row.get("step", 0)) for row in retained]
    if len(steps) != len(set(steps)):
        raise ValueError(f"train log contains duplicate step records: {path}")
    if steps != sorted(steps):
        raise ValueError(f"train log step sequence is not increasing: {path}")
    if removed:
        write_jsonl(path, retained)
    return {
        "truncated": bool(removed),
        "removed_records": removed,
        "retained_records": len(retained),
        "boundary_step": completed_steps,
    }


def _record_failed_attempt(run_dir: Path, failure: Mapping[str, object]) -> Path:
    """Write an immutable, never-overwritten record of one failed attempt.

    The preregistration requires failed attempts to be preserved. Writing every
    failure to a single ``failure_bundle.json`` that is unlinked at run start
    destroyed the original evidence on retry (finding P1-2).
    """

    failures_dir = run_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(failure, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    digest = sha256_bytes(payload.encode("utf-8"))
    index = 1 + sum(1 for entry in failures_dir.glob("attempt_*.json"))
    path = failures_dir / f"attempt_{index:04d}_{digest[:16]}.json"
    if path.exists():
        return path
    _atomic_write_text(path, payload)
    return path


def _summary_from_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty predictions")

    def summarize(selection: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
        count = len(selection)
        return {
            "examples": count,
            "accuracy": sum(float(bool(row["correct"])) for row in selection) / count,
            "mean_answer_nll": sum(float(row["answer_nll"]) for row in selection) / count,
            "mean_target_margin": sum(float(row["target_margin"]) for row in selection) / count,
        }

    by_family: dict[str, object] = {}
    by_stratum: dict[str, object] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        by_family[family] = summarize([row for row in rows if row["family"] == family])
    for stratum in sorted({str(row["stratum"]) for row in rows}):
        by_stratum[stratum] = summarize([row for row in rows if row["stratum"] == stratum])
    return {"overall": summarize(rows), "by_family": by_family, "by_stratum": by_stratum}


def evaluate_model(
    model: TinyTGSAForCausalLM,
    examples: Sequence[Phase1Example],
    *,
    device: torch.device,
    batch_size: int,
    split: str,
    arm: ModelArm,
    seed: int,
    intervention: InterventionMode = "identity",
    capture_intervention_records: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    model.eval()
    rows: list[dict[str, object]] = []
    intervention_records: list[dict[str, object]] = []
    context = (
        nullcontext()
        if intervention == "identity"
        else ControlIntervention(
            model,
            intervention,
            capture_records=capture_intervention_records,
            max_records=128,
        )
    )
    with context as intervention_context, torch.inference_mode():
        for start, end in _evaluation_chunks(len(examples), batch_size):
            batch = fixed_batches(examples[start:end], batch_size=end - start).__next__().to(device)
            if intervention == "roll_batch" and batch.input_ids.shape[0] < 2:
                raise ValueError("roll_batch requires at least two evaluation examples")
            output = model(
                batch.input_ids,
                key_padding_mask=batch.key_padding_mask,
                telemetry_level="none",
            )
            selected_logits = answer_logits(output.logits, batch.answer_input_indices).float()
            log_probabilities = torch.log_softmax(selected_logits, dim=-1)
            predictions = selected_logits.argmax(dim=-1)
            target_log_probabilities = log_probabilities.gather(
                dim=-1,
                index=batch.answer_tokens[:, None],
            ).squeeze(-1)
            competing = selected_logits.clone()
            competing.scatter_(1, batch.answer_tokens[:, None], float("-inf"))
            best_competing = competing.max(dim=-1).values
            target_values = selected_logits.gather(1, batch.answer_tokens[:, None]).squeeze(1)
            target_margin = target_values - best_competing
            for index, example_id in enumerate(batch.example_ids):
                rows.append(
                    {
                        "schema_version": PREDICTION_SCHEMA_VERSION,
                        "example_id": example_id,
                        "split": split,
                        "arm": arm,
                        "seed": seed,
                        "intervention": intervention,
                        "family": batch.families[index],
                        "stratum": batch.strata[index],
                        "answer_token": int(batch.answer_tokens[index].item()),
                        "prediction_token": int(predictions[index].item()),
                        "correct": bool(predictions[index].eq(batch.answer_tokens[index]).item()),
                        "answer_nll": float((-target_log_probabilities[index]).item()),
                        "target_probability": float(target_log_probabilities[index].exp().item()),
                        "target_margin": float(target_margin[index].item()),
                    }
                )
        if isinstance(intervention_context, ControlIntervention):
            intervention_records.extend(intervention_context.records)
    if len(rows) != len(examples):
        raise AssertionError("evaluation did not produce exactly one row per example")
    if [row["example_id"] for row in rows] != [example.id for example in examples]:
        raise AssertionError("evaluation order changed")
    return rows, _summary_from_rows(rows), intervention_records


def _replay_groups(
    examples: Sequence[Phase1Example],
    *,
    batch_size: int,
) -> list[list[int]]:
    """Chunk examples into batches that are homogeneous in stratum and length.

    The replay falsifier may only donate a control between examples with the
    same valid sequence length, otherwise a valid position can receive a
    padded-position control. Grouping by the registered stratum as well keeps
    donor and receiver inside the same registered difficulty cell.
    """

    buckets: dict[tuple[str, int], list[int]] = {}
    for position, example in enumerate(examples):
        buckets.setdefault((example.stratum, len(example.tokens)), []).append(position)
    chunks: list[list[int]] = []
    for key in sorted(buckets):
        rows = buckets[key]
        if len(rows) == 1:
            chunks.append(rows)
            continue
        for start, end in _evaluation_chunks(len(rows), batch_size):
            chunks.append(rows[start:end])
    return chunks


def evaluate_replay_derangement(
    model: TinyTGSAForCausalLM,
    examples: Sequence[Phase1Example],
    *,
    device: torch.device,
    batch_size: int,
    split: str,
    arm: ModelArm,
    seed: int,
    derangement_index: int = 0,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Registered alignment falsifier: replay another example's aligned control.

    Two passes per batch. The first is unperturbed and captures every
    trajectory-encoder input. The second replaces each example's control with a
    donor's *captured* control, so head ``j`` never receives a signal that was
    itself produced under the intervention at earlier heads. Donors come from a
    fixed length-matched derangement, so no valid position is ever handed a
    padded-position control.

    Together these repair both halves of ``PHASE1_LAUNCH_AUDIT.md`` finding P1-4.
    """

    model.eval()
    rows_by_position: dict[int, dict[str, object]] = {}
    applied_count = 0
    encoder_records: list[dict[str, object]] = []
    with torch.inference_mode():
        for positions in _replay_groups(examples, batch_size=batch_size):
            selection = [examples[position] for position in positions]
            batch = collate_examples(selection).to(device)
            donor, applied = length_matched_derangement(
                [(example.stratum, len(example.tokens)) for example in selection],
                derangement_index=derangement_index,
            )
            with ControlCapture(model) as capture:
                model(
                    batch.input_ids,
                    key_padding_mask=batch.key_padding_mask,
                    telemetry_level="none",
                )
            with ReplayIntervention(
                model,
                captured=capture.captured,
                donor_index=donor,
                applied=applied,
            ) as replay:
                output = model(
                    batch.input_ids,
                    key_padding_mask=batch.key_padding_mask,
                    telemetry_level="none",
                )
            if len(encoder_records) < 128:
                encoder_records.extend(replay.records)
            selected_logits = answer_logits(output.logits, batch.answer_input_indices).float()
            log_probabilities = torch.log_softmax(selected_logits, dim=-1)
            predictions = selected_logits.argmax(dim=-1)
            target_log_probabilities = log_probabilities.gather(
                dim=-1,
                index=batch.answer_tokens[:, None],
            ).squeeze(-1)
            competing = selected_logits.clone()
            competing.scatter_(1, batch.answer_tokens[:, None], float("-inf"))
            best_competing = competing.max(dim=-1).values
            target_values = selected_logits.gather(1, batch.answer_tokens[:, None]).squeeze(1)
            target_margin = target_values - best_competing
            for row, position in enumerate(positions):
                applied_count += int(applied[row])
                rows_by_position[position] = {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "example_id": selection[row].id,
                    "split": split,
                    "arm": arm,
                    "seed": seed,
                    "intervention": "replay_derangement",
                    "derangement_index": derangement_index,
                    "intervention_applied": bool(applied[row]),
                    "donor_example_id": selection[donor[row]].id,
                    "family": batch.families[row],
                    "stratum": batch.strata[row],
                    "answer_token": int(batch.answer_tokens[row].item()),
                    "prediction_token": int(predictions[row].item()),
                    "correct": bool(predictions[row].eq(batch.answer_tokens[row]).item()),
                    "answer_nll": float((-target_log_probabilities[row]).item()),
                    "target_probability": float(target_log_probabilities[row].exp().item()),
                    "target_margin": float(target_margin[row].item()),
                }
    if len(rows_by_position) != len(examples):
        raise AssertionError("replay evaluation did not produce exactly one row per example")
    rows = [rows_by_position[position] for position in range(len(examples))]
    if [row["example_id"] for row in rows] != [example.id for example in examples]:
        raise AssertionError("replay evaluation order changed")
    for row in rows:
        if row["intervention_applied"] and row["donor_example_id"] == row["example_id"]:
            raise AssertionError("derangement assigned an example its own control")
    coverage = {
        "derangement_index": derangement_index,
        "examples": len(examples),
        "intervened_examples": applied_count,
        "coverage": applied_count / len(examples),
        "unmatched_examples": len(examples) - applied_count,
        "encoder_records": encoder_records,
    }
    return rows, _summary_from_rows(rows), coverage


def telemetry_snapshot(
    model: TinyTGSAForCausalLM,
    examples: Sequence[Phase1Example],
    *,
    device: torch.device,
    count: int,
) -> dict[str, object]:
    if model.config.arm == "parallel_mha":
        return {"available": False, "reason": "parallel control has no TGSA telemetry"}
    selected = examples[: max(1, min(count, len(examples)))]
    batch = fixed_batches(selected, batch_size=len(selected)).__next__().to(device)
    model.eval()
    with torch.inference_mode():
        output = model(
            batch.input_ids,
            key_padding_mask=batch.key_padding_mask,
            telemetry_level="summary",
        )
    blocks: list[dict[str, object]] = []
    for index, telemetry in enumerate(output.block_telemetry):
        blocks.append(
            {
                "block_index": index,
                "summary": None if telemetry is None else telemetry_summary(telemetry),
                "raw": None if telemetry is None else telemetry.to_cpu_dict(),
            }
        )
    return {
        "available": True,
        "example_ids": list(batch.example_ids),
        "batch_signature": _batch_signature(batch),
        "blocks": blocks,
    }


def _checkpoint_payload(
    *,
    model: TinyTGSAForCausalLM,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    examples_seen: int,
    tokens_seen: int,
    study_config_sha256: str,
    dataset_manifest_sha256: str,
    arm: ModelArm,
    seed: int,
    initialization_seed: int,
    data_order_seed: int,
) -> dict[str, object]:
    return {
        "schema_version": "tgsa.phase1-checkpoint.v1",
        "created_at_utc": utc_now(),
        "arm": arm,
        "seed": seed,
        "initialization_seed": initialization_seed,
        "data_order_seed": data_order_seed,
        "completed_steps": completed_steps,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
        "study_config_sha256": study_config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _load_checkpoint(
    path: Path,
    *,
    model: TinyTGSAForCausalLM,
    optimizer: torch.optim.Optimizer,
    expected: Mapping[str, object],
) -> dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint mismatch for {key}: {payload.get(key)!r} != {value!r}")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return {
        "completed_steps": int(payload["completed_steps"]),
        "examples_seen": int(payload.get("examples_seen", 0)),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
    }


def train_one_run(
    *,
    study: Mapping[str, object],
    study_config_path: Path,
    dataset_dir: Path,
    output_root: Path,
    repo_root: Path,
    arm: ModelArm,
    seed: int,
    resume: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    run_dir = output_root / "runs" / arm / f"seed_{seed}"
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") in {"trained", "complete"}:
            return cast(dict[str, object], manifest)
        if not resume:
            raise FileExistsError(f"incomplete run exists and resume is disabled: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Failed attempts are immutable. Nothing is unlinked here; see
    # `_record_failed_attempt`.
    last_batch: Phase1Batch | None = None
    last_step = 0
    started_at = time.perf_counter()

    try:
        dataset_manifest_path = dataset_dir / "dataset_manifest.json"
        dataset_manifest = validate_dataset_manifest(dataset_manifest_path)
        examples = load_study_examples(dataset_dir)
        vocabulary = cast(Mapping[str, object], dataset_manifest["vocabulary"])
        max_sequence_length = max(
            int(entry["max_tokens"]) - 1
            for entry in cast(Mapping[str, Mapping[str, object]], dataset_manifest["splits"]).values()
        )
        model_config = _model_config_from_study(
            study,
            arm=arm,
            vocabulary=vocabulary,
            max_sequence_length=max_sequence_length,
        )
        training = cast(Mapping[str, object], study["training"])
        evaluation = cast(Mapping[str, object], study["evaluation"])
        runtime = cast(Mapping[str, object], study["runtime"])
        study_config_sha = sha256_file(study_config_path)
        dataset_manifest_sha = sha256_file(dataset_manifest_path)
        initialization_seed = derive_seed(seed, "model-initialization")
        data_order_seed = derive_seed(seed, "training-order")
        runtime_seed = derive_seed(seed, arm, "runtime")
        configure_runtime(runtime, runtime_seed)
        device = resolve_device(runtime)
        model = build_model(model_config, initialization_seed=initialization_seed).to(device)
        initial_model_sha = state_dict_sha256(model)
        pre_training_telemetry = telemetry_snapshot(
            model,
            examples["dev"],
            device=device,
            count=int(evaluation.get("telemetry_examples", 16)),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            betas=(float(training.get("beta1", 0.9)), float(training.get("beta2", 0.95))),
            eps=float(training.get("epsilon", 1.0e-8)),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        total_steps = int(training["steps"])
        batch_size = int(training["batch_size"])
        checkpoint_every = int(training.get("checkpoint_every", 0))
        log_every = max(1, int(training.get("log_every", 25)))
        gradient_clip = float(training.get("gradient_clip_norm", 1.0))
        checkpoint_path = run_dir / "checkpoint_last.pt"
        start_step = 0
        examples_seen = 0
        tokens_seen = 0
        if resume and checkpoint_path.exists() and not overwrite:
            restored = _load_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                expected={
                    "arm": arm,
                    "seed": seed,
                    "initialization_seed": initialization_seed,
                    "data_order_seed": data_order_seed,
                    "study_config_sha256": study_config_sha,
                    "dataset_manifest_sha256": dataset_manifest_sha,
                },
            )
            start_step = restored["completed_steps"]
            examples_seen = restored["examples_seen"]
            tokens_seen = restored["tokens_seen"]
            if start_step > total_steps:
                raise ValueError("checkpoint is beyond configured total steps")

        train_log_path = run_dir / "train_log.jsonl"
        if overwrite and start_step == 0:
            train_log_path.unlink(missing_ok=True)
        # Rewind the evidentiary log to the restored checkpoint boundary so the
        # resumed run cannot append a second copy of already-logged steps.
        train_log_repair = _truncate_train_log(train_log_path, completed_steps=start_step)
        stream = deterministic_batch_stream(
            examples["train"],
            batch_size=batch_size,
            seed=data_order_seed,
        )
        for _ in range(start_step):
            next(stream)

        model.train()
        for step_index in range(start_step, total_steps):
            epoch, batch = next(stream)
            last_batch = batch
            last_step = step_index + 1
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch.input_ids,
                key_padding_mask=batch.key_padding_mask,
                telemetry_level="none",
            )
            loss = masked_answer_loss(output.logits, batch.labels, batch.loss_mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step_index + 1}")
            loss.backward()
            unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
                error_if_nonfinite=True,
            )
            gradient_groups = gradient_group_norms(model) if (step_index + 1) % log_every == 0 or step_index == start_step else None
            optimizer.step()
            valid_tokens = int((~batch.key_padding_mask).sum().item())
            examples_seen += batch.input_ids.shape[0]
            tokens_seen += valid_tokens

            if (step_index + 1) % log_every == 0 or step_index == start_step or step_index + 1 == total_steps:
                append_jsonl(
                    train_log_path,
                    {
                        "schema_version": "tgsa.phase1-train-log.v1",
                        "created_at_utc": utc_now(),
                        "arm": arm,
                        "seed": seed,
                        "step": step_index + 1,
                        "epoch": epoch,
                        "loss": float(loss.detach().item()),
                        "unclipped_gradient_norm": float(unclipped_grad_norm.detach().item()),
                        "gradient_group_norms": gradient_groups,
                        "batch_signature": _batch_signature(batch),
                        "example_ids": list(batch.example_ids),
                        "examples_seen": examples_seen,
                        "tokens_seen": tokens_seen,
                    },
                )

            if checkpoint_every > 0 and (step_index + 1) % checkpoint_every == 0:
                atomic_torch_save(
                    checkpoint_path,
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        completed_steps=step_index + 1,
                        examples_seen=examples_seen,
                        tokens_seen=tokens_seen,
                        study_config_sha256=study_config_sha,
                        dataset_manifest_sha256=dataset_manifest_sha,
                        arm=arm,
                        seed=seed,
                        initialization_seed=initialization_seed,
                        data_order_seed=data_order_seed,
                    ),
                )

        final_checkpoint_path = run_dir / "checkpoint_final.pt"
        atomic_torch_save(
            final_checkpoint_path,
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                completed_steps=total_steps,
                examples_seen=examples_seen,
                tokens_seen=tokens_seen,
                study_config_sha256=study_config_sha,
                dataset_manifest_sha256=dataset_manifest_sha,
                arm=arm,
                seed=seed,
                initialization_seed=initialization_seed,
                data_order_seed=data_order_seed,
            ),
        )
        final_model_sha = state_dict_sha256(model)
        post_training_telemetry = telemetry_snapshot(
            model,
            examples["dev"],
            device=device,
            count=int(evaluation.get("telemetry_examples", 16)),
        )

        # Confirmatory test outcomes are NOT produced here. Training and test
        # evaluation are separate phases so that no test result from an early
        # arm/seed cell can be observed while later cells are still training
        # (`PHASE1_LAUNCH_AUDIT.md` finding P0-8). Development evaluation stays
        # available for operational debugging.
        evaluation_batch_size = int(evaluation.get("batch_size", batch_size))
        evaluation_records: dict[str, object] = {}
        dev_rows, dev_summary, _ = evaluate_model(
            model,
            examples["dev"],
            device=device,
            batch_size=evaluation_batch_size,
            split="dev",
            arm=arm,
            seed=seed,
            intervention="identity",
        )
        dev_path = run_dir / "predictions_dev__identity.jsonl"
        write_jsonl(dev_path, dev_rows)
        evaluation_records["dev__identity"] = {
            "path": dev_path.name,
            "sha256": sha256_file(dev_path),
            "summary": dev_summary,
        }

        elapsed_seconds = time.perf_counter() - started_at
        manifest: dict[str, object] = {
            "schema_version": RUN_MANIFEST_VERSION,
            "status": "trained",
            "phase": "train",
            "confirmatory": bool(study["confirmatory"]),
            "test_evaluation_deferred": True,
            "study_id": study["study_id"],
            "created_at_utc": utc_now(),
            "arm": arm,
            "seed": seed,
            "claim_scope": (
                "trained tiny-decoder mechanism-identification evidence only; not a Qwen/GQA, "
                "natural-language, long-context, efficiency, novelty, or paper-primary claim"
            ),
            "git": git_snapshot(repo_root, str(study["expected_base_commit"])),
            "environment": environment_snapshot(device),
            "study_config_path": str(study_config_path),
            "study_config_sha256": study_config_sha,
            "dataset_manifest_path": str(dataset_manifest_path),
            "dataset_manifest_sha256": dataset_manifest_sha,
            "model_config": asdict(model_config),
            "parameter_audit": parameter_audit(model),
            "initialization_seed": initialization_seed,
            "data_order_seed": data_order_seed,
            "runtime_seed": runtime_seed,
            "initial_model_state_sha256": initial_model_sha,
            "final_model_state_sha256": final_model_sha,
            "training": {
                "fixed_final_checkpoint": True,
                "early_stopping": False,
                "completed_steps": total_steps,
                "examples_seen": examples_seen,
                "tokens_seen": tokens_seen,
                "elapsed_seconds": elapsed_seconds,
                "train_log": train_log_path.name,
                "train_log_sha256": sha256_file(train_log_path),
                "train_log_repair": train_log_repair,
                "resumed_from_step": start_step,
                "final_checkpoint": final_checkpoint_path.name,
                "final_checkpoint_sha256": sha256_file(final_checkpoint_path),
            },
            "telemetry": {
                "pre_training": pre_training_telemetry,
                "post_training": post_training_telemetry,
            },
            "evaluations": evaluation_records,
            "debug_snapshot": model_debug_snapshot(model),
        }
        atomic_write_json(manifest_path, manifest)
        checkpoint_path.unlink(missing_ok=True)
        return manifest
    except BaseException as exc:
        failure: dict[str, object] = {
            "schema_version": "tgsa.phase1-failure-bundle.v1",
            "status": "failed",
            "created_at_utc": utc_now(),
            "arm": arm,
            "seed": seed,
            "last_step": last_step,
            "last_batch_signature": None if last_batch is None else _batch_signature(last_batch),
            "last_example_ids": None if last_batch is None else list(last_batch.example_ids),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "git": git_snapshot(repo_root, str(study["expected_base_commit"])),
        }
        _record_failed_attempt(run_dir, failure)
        raise


def evaluate_one_run(
    *,
    study: Mapping[str, object],
    study_config_path: Path,
    dataset_dir: Path,
    output_root: Path,
    repo_root: Path,
    arm: ModelArm,
    seed: int,
) -> dict[str, object]:
    """Frozen test-evaluation phase for one already-trained run.

    Runs only after every registered cell has a hashed final checkpoint. The
    checkpoint is re-verified against the hash the training phase recorded, so
    the evaluated weights are provably the fixed final checkpoint and not a
    later mutation.
    """

    run_dir = output_root / "runs" / arm / f"seed_{seed}"
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "complete":
        return cast(dict[str, object], manifest)
    if manifest.get("status") != "trained":
        raise ValueError(f"run is not in the trained state: {manifest_path}")

    try:
        training_record = cast(Mapping[str, object], manifest["training"])
        checkpoint_path = run_dir / str(training_record["final_checkpoint"])
        recorded_hash = str(training_record["final_checkpoint_sha256"])
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != recorded_hash:
            raise ValueError(
                f"final checkpoint hash changed since training: {checkpoint_path} "
                f"({actual_hash} != {recorded_hash})"
            )
        recorded_log_hash = str(training_record["train_log_sha256"])
        actual_log_hash = sha256_file(run_dir / str(training_record["train_log"]))
        if actual_log_hash != recorded_log_hash:
            raise ValueError(f"train log hash changed since training: {run_dir}")

        dataset_manifest_path = dataset_dir / "dataset_manifest.json"
        dataset_manifest = validate_dataset_manifest(dataset_manifest_path)
        if sha256_file(dataset_manifest_path) != str(manifest["dataset_manifest_sha256"]):
            raise ValueError("dataset manifest changed between training and evaluation")
        if sha256_file(study_config_path) != str(manifest["study_config_sha256"]):
            raise ValueError("study config changed between training and evaluation")

        examples = load_study_examples(dataset_dir)
        evaluation = cast(Mapping[str, object], study["evaluation"])
        runtime = cast(Mapping[str, object], study["runtime"])
        configure_runtime(runtime, derive_seed(seed, arm, "evaluation"))
        device = resolve_device(runtime)
        vocabulary = cast(Mapping[str, object], dataset_manifest["vocabulary"])
        max_sequence_length = max(
            int(entry["max_tokens"]) - 1
            for entry in cast(Mapping[str, Mapping[str, object]], dataset_manifest["splits"]).values()
        )
        model_config = _model_config_from_study(
            study,
            arm=arm,
            vocabulary=vocabulary,
            max_sequence_length=max_sequence_length,
        )
        model = build_model(
            model_config,
            initialization_seed=int(manifest["initialization_seed"]),
        ).to(device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        if state_dict_sha256(model) != str(manifest["final_model_state_sha256"]):
            raise ValueError("restored model state does not match the recorded final state")

        evaluation_batch_size = int(evaluation.get("batch_size", 64))
        evaluation_records = dict(cast(Mapping[str, object], manifest["evaluations"]))
        rows, summary, _ = evaluate_model(
            model,
            examples["test"],
            device=device,
            batch_size=evaluation_batch_size,
            split="test",
            arm=arm,
            seed=seed,
            intervention="identity",
        )
        path = run_dir / "predictions_test__identity.jsonl"
        write_jsonl(path, rows)
        evaluation_records["test__identity"] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "summary": summary,
        }

        replay_coverage: dict[str, object] = {}
        if arm == "learned_delta":
            for index in [int(value) for value in evaluation.get("derangement_indices", [0])]:
                rows, summary, coverage = evaluate_replay_derangement(
                    model,
                    examples["test"],
                    device=device,
                    batch_size=evaluation_batch_size,
                    split="test",
                    arm=arm,
                    seed=seed,
                    derangement_index=index,
                )
                name = "replay_derangement" if index == 0 else f"replay_derangement_{index}"
                path = run_dir / f"predictions_test__{name}.jsonl"
                write_jsonl(path, rows)
                evaluation_records[f"test__{name}"] = {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "summary": summary,
                }
                replay_coverage[name] = coverage

            for mode_text in [str(mode) for mode in evaluation.get("interventions", [])]:
                mode = cast(InterventionMode, mode_text)
                rows, summary, records = evaluate_model(
                    model,
                    examples["test"],
                    device=device,
                    batch_size=evaluation_batch_size,
                    split="test",
                    arm=arm,
                    seed=seed,
                    intervention=mode,
                    capture_intervention_records=True,
                )
                path = run_dir / f"predictions_test__{mode}.jsonl"
                write_jsonl(path, rows)
                evaluation_records[f"test__{mode}"] = {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "summary": summary,
                    "intervention_records": records,
                    "interpretation": (
                        "descriptive diagnostic only; not the registered alignment falsifier"
                    ),
                }

        manifest["status"] = "complete"
        manifest["phase"] = "evaluate"
        manifest["test_evaluation_deferred"] = True
        manifest["evaluations"] = evaluation_records
        manifest["replay_coverage"] = replay_coverage
        manifest["evaluated_at_utc"] = utc_now()
        manifest["evaluation_git"] = git_snapshot(repo_root, str(study["expected_base_commit"]))
        manifest["verified_final_checkpoint_sha256"] = actual_hash
        atomic_write_json(manifest_path, manifest)
        return cast(dict[str, object], manifest)
    except BaseException as exc:
        _record_failed_attempt(
            run_dir,
            {
                "schema_version": "tgsa.phase1-failure-bundle.v1",
                "status": "failed",
                "phase": "evaluate",
                "created_at_utc": utc_now(),
                "arm": arm,
                "seed": seed,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "git": git_snapshot(repo_root, str(study["expected_base_commit"])),
            },
        )
        raise


def run_study_matrix(
    *,
    study_config_path: Path,
    dataset_dir: Path,
    output_root: Path,
    repo_root: Path,
    arms: Sequence[ModelArm] | None = None,
    seeds: Sequence[int] | None = None,
    resume: bool = True,
    overwrite: bool = False,
    force_data: bool = False,
) -> dict[str, object]:
    study = load_study_config(study_config_path)
    confirmatory = bool(study["confirmatory"])
    if confirmatory and force_data:
        raise ValueError(
            "--force-data regenerates the materialized dataset and is forbidden for a "
            "confirmatory study; use a fresh output root instead"
        )
    dataset_manifest = prepare_study_dataset(study, dataset_dir=dataset_dir, force=force_data)
    configured_arms = tuple(cast(ModelArm, str(arm)) for arm in study["arms"])
    configured_seeds = tuple(int(seed) for seed in study["seeds"])
    selected_arms = configured_arms if arms is None else tuple(arms)
    selected_seeds = configured_seeds if seeds is None else tuple(int(seed) for seed in seeds)
    if not selected_arms or not selected_seeds:
        raise ValueError("selected matrix must contain at least one arm and seed")
    if not set(selected_arms).issubset(configured_arms):
        raise ValueError("requested arm is not registered in the study config")
    if not set(selected_seeds).issubset(configured_seeds):
        raise ValueError("requested seed is not registered in the study config")

    vocabulary = cast(Mapping[str, object], dataset_manifest["vocabulary"])
    max_sequence_length = max(
        int(entry["max_tokens"]) - 1
        for entry in cast(Mapping[str, Mapping[str, object]], dataset_manifest["splits"]).values()
    )
    base_model_config = _model_config_from_study(
        study,
        arm="learned_delta",
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
    )
    preflight: dict[str, object] = {
        "schema_version": "tgsa.phase1-preflight.v1",
        "created_at_utc": utc_now(),
        "required_arms_registered": set(REQUIRED_ARMS).issubset(configured_arms),
        "study_config_sha256": sha256_file(study_config_path),
        "dataset_manifest_sha256": sha256_file(dataset_dir / "dataset_manifest.json"),
        "dataset_manifest_validated": True,
        "git": git_snapshot(repo_root, str(study["expected_base_commit"])),
        "paired_initialization_by_seed": {},
    }
    paired = cast(dict[str, object], preflight["paired_initialization_by_seed"])
    for seed in selected_seeds:
        paired[str(seed)] = paired_initialization_audit(
            base_config=base_model_config,
            initialization_seed=derive_seed(seed, "model-initialization"),
        )
    preflight_path = output_root / "preflight.json"
    atomic_write_json(preflight_path, preflight)

    # Refuse to silently mix a finished run produced by a different config,
    # dataset, or source commit into this matrix (finding P0-3).
    current_git = git_snapshot(repo_root, str(study["expected_base_commit"]))
    reuse_audit = validate_completed_run_reuse(
        output_root=output_root,
        arms=[str(arm) for arm in selected_arms],
        seeds=[int(seed) for seed in selected_seeds],
        study_config_sha256=sha256_file(study_config_path),
        dataset_manifest_sha256=sha256_file(dataset_dir / "dataset_manifest.json"),
        current_git_head=None if current_git["head"] is None else str(current_git["head"]),
    )
    if confirmatory and current_git.get("dirty"):
        raise ValueError(
            "confirmatory execution requires a clean source checkout; commit or stash "
            "the working tree before dispatching the registered matrix"
        )

    cells = [(seed, arm) for seed in selected_seeds for arm in selected_arms]
    continue_on_error = bool(
        cast(Mapping[str, object], study["runtime"]).get("continue_on_error", False)
    )
    run_entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    def _abort(phase: str) -> None:
        atomic_write_json(
            output_root / "matrix_manifest.json",
            {
                "schema_version": MATRIX_MANIFEST_VERSION,
                "status": "failed",
                "phase": phase,
                "created_at_utc": utc_now(),
                "study_id": study["study_id"],
                "confirmatory": confirmatory,
                "selected_arms": list(selected_arms),
                "selected_seeds": list(selected_seeds),
                "runs": run_entries,
                "failures": failures,
            },
        )

    # ---- Phase 1: train every registered cell. No test outcome is produced. ----
    for seed, arm in cells:
        try:
            train_one_run(
                study=study,
                study_config_path=study_config_path,
                dataset_dir=dataset_dir,
                output_root=output_root,
                repo_root=repo_root,
                arm=arm,
                seed=seed,
                resume=resume,
                overwrite=overwrite,
            )
        except BaseException as exc:
            failures.append({"phase": "train", "arm": arm, "seed": seed, "exception": repr(exc)})
            if not continue_on_error:
                _abort("train")
                raise

    # ---- Barrier: every cell must be trained and hash-verified before any
    # confirmatory test outcome becomes visible (finding P0-8). ----
    trained: dict[str, object] = {}
    for seed, arm in cells:
        manifest_path = output_root / "runs" / arm / f"seed_{seed}" / "run_manifest.json"
        if not manifest_path.is_file():
            failures.append({"phase": "barrier", "arm": arm, "seed": seed, "exception": "missing run manifest"})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"trained", "complete"}:
            failures.append({"phase": "barrier", "arm": arm, "seed": seed, "exception": "run is not trained"})
            continue
        training_record = cast(Mapping[str, object], manifest["training"])
        if int(training_record["completed_steps"]) != int(
            cast(Mapping[str, object], study["training"])["steps"]
        ):
            failures.append(
                {"phase": "barrier", "arm": arm, "seed": seed, "exception": "completed steps differ from registered steps"}
            )
            continue
        trained[f"{arm}:{seed}"] = {
            "final_checkpoint_sha256": training_record["final_checkpoint_sha256"],
            "train_log_sha256": training_record["train_log_sha256"],
            "git_head": _manifest_field(manifest, "git.head"),
        }

    matrix_is_complete = len(trained) == len(cells) and not failures
    heads = {str(entry["git_head"]) for entry in trained.values()}  # type: ignore[index]
    if matrix_is_complete and len(heads) != 1:
        failures.append({"phase": "barrier", "exception": f"runs span multiple source heads: {sorted(heads)}"})
        matrix_is_complete = False
    if not matrix_is_complete and not continue_on_error:
        _abort("barrier")
        raise ValueError(
            "refusing to begin the frozen test-evaluation phase: the registered training "
            f"matrix is incomplete or inconsistent ({len(trained)}/{len(cells)} cells)"
        )

    # ---- Phase 2: one frozen test-evaluation sweep over the fixed checkpoints. ----
    for seed, arm in cells:
        try:
            manifest = evaluate_one_run(
                study=study,
                study_config_path=study_config_path,
                dataset_dir=dataset_dir,
                output_root=output_root,
                repo_root=repo_root,
                arm=arm,
                seed=seed,
            )
            run_entries.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "status": manifest["status"],
                    "manifest": str(Path("runs") / arm / f"seed_{seed}" / "run_manifest.json"),
                }
            )
        except BaseException as exc:
            failures.append({"phase": "evaluate", "arm": arm, "seed": seed, "exception": repr(exc)})
            if not continue_on_error:
                _abort("evaluate")
                raise

    matrix: dict[str, object] = {
        "schema_version": MATRIX_MANIFEST_VERSION,
        "status": "complete" if not failures else "partial_failure",
        "phase": "evaluate",
        "created_at_utc": utc_now(),
        "study_id": study["study_id"],
        "confirmatory": confirmatory,
        "claim_scope": (
            "execution manifest only; conclusions require the frozen paired analyzer and registered "
            "full arm/seed matrix"
        ),
        "study_config_sha256": sha256_file(study_config_path),
        "dataset_manifest_sha256": sha256_file(dataset_dir / "dataset_manifest.json"),
        "preflight": preflight_path.name,
        "preflight_sha256": sha256_file(preflight_path),
        "configured_arms": list(configured_arms),
        "configured_seeds": list(configured_seeds),
        "selected_arms": list(selected_arms),
        "selected_seeds": list(selected_seeds),
        "is_full_registered_matrix": set(selected_arms) == set(configured_arms)
        and set(selected_seeds) == set(configured_seeds),
        "deferred_test_evaluation": True,
        "trained_before_evaluation": sorted(trained),
        "source_head": sorted(heads),
        "reuse_audit": reuse_audit,
        "runs": run_entries,
        "failures": failures,
        "git": git_snapshot(repo_root, str(study["expected_base_commit"])),
    }
    atomic_write_json(output_root / "matrix_manifest.json", matrix)
    return matrix
