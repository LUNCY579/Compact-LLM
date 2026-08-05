from __future__ import annotations

"""Run Phase 2C importance-guided targeted recovery on an approved Phase 2B model."""

import argparse
import hashlib
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import (
    approved_content_only_migration,
    atomic_write_text,
    read_json,
    replace_path_with_retry,
    run,
    write_json,
)
from scripts.phase_2_layer_selection.resolve_phase_2_profile import resolve_profile
from scripts.phase_2c_targeted_recovery.phase2c_common import (
    assess_training_candidate,
    candidate_specs,
    relative_to_root,
    select_profiles,
)

PYTHON = sys.executable
TRAIN = ROOT / "scripts/phase_1_dense_sft/train_dense_sft.py"
EVALUATE = ROOT / "scripts/phase_1_dense_sft/evaluate_model.py"
NLL = ROOT / "scripts/phase_2_layer_selection/evaluate_teacher_forcing.py"
COMPARE = ROOT / "scripts/phase_2_layer_selection/compare_outputs.py"
BUILD_DATA = ROOT / "scripts/phase_2c_targeted_recovery/build_phase_2c_dataset.py"
RANK = ROOT / "scripts/phase_2c_targeted_recovery/rank_surviving_layers.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 2C targeted selected-layer recovery.")
    p.add_argument("--phase2b-profile", choices=("balanced", "min_memory"), default="balanced")
    p.add_argument("--phase2b-catalog", type=Path, default=ROOT / "artifacts/phase_2_progressive_reduction/model_catalog.json")
    p.add_argument("--phase2b-gate", type=Path, default=ROOT / "reports/PHASE_2B_STOP_GATE.md")
    p.add_argument("--phase2b-registry", type=Path, default=ROOT / "artifacts/phase_2_progressive_reduction/candidate_registry.json")
    p.add_argument("--train-file", type=Path, default=ROOT / "data/phase_2c_targeted_recovery/train.jsonl")
    p.add_argument("--dev-file", type=Path, default=ROOT / "data/phase_2c_targeted_recovery/dev.jsonl")
    p.add_argument("--phase-benchmark", type=Path, default=ROOT / "data/benchmark/shortchatbench_pilot.jsonl")
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery")
    p.add_argument("--report", type=Path, default=ROOT / "reports/PHASE_2C_TARGETED_RECOVERY_REPORT.md")
    p.add_argument("--gate-report", type=Path, default=ROOT / "reports/PHASE_2C_STOP_GATE.md")
    p.add_argument("--calibration-limit", type=int, default=64)
    p.add_argument("--eval-limit", type=int, default=180)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--base-seed", type=int, default=11)
    p.add_argument("--repeat-seeds", default="23,37")
    p.add_argument("--repeat-top-k", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-sequence-length", type=int, default=1024)
    p.add_argument("--gradient-weight", type=float, default=0.35)
    p.add_argument("--ablation-weight", type=float, default=0.45)
    p.add_argument("--structural-weight", type=float, default=0.20)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--quality-profile",
        choices=("balanced", "logic_oriented", "conversational_permissive"),
        default="balanced",
    )
    for deprecated in (
        "--max-quality-index-loss", "--max-frozen-quality-loss",
        "--max-heuristic-loss", "--max-semantic-pass-loss",
        "--max-reference-f1-loss", "--max-category-loss",
        "--max-frozen-category-loss", "--max-nll-increase",
        "--min-complete-utterance-rate", "--minimum-trainable-reduction",
        "--balanced-quality-tolerance",
    ):
        p.add_argument(deprecated, type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--max-new-major-issues", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument(
        "--keep-training-checkpoints",
        action="store_true",
        help="Retain Trainer checkpoint-* directories after a candidate finishes.",
    )
    p.add_argument(
        "--minimum-free-disk-gb",
        type=float,
        default=1.0,
        help="Minimum free space required before starting a fresh candidate training run.",
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _valid_json_sentinel(path: Path, required_keys: tuple[str, ...] = ()) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    return all(key in payload for key in required_keys)


def _quarantine_invalid_sentinel(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_name(f"{path.name}.invalid-{stamp}")
    try:
        replace_path_with_retry(path, quarantine)
        print(f"[quarantine invalid sentinel] {path} -> {quarantine}")
    except OSError:
        path.unlink(missing_ok=True)
        print(f"[remove invalid sentinel] {path}")


def run_if_needed(
    command: list[str],
    sentinel: Path,
    resume: bool,
    *,
    required_json_keys: tuple[str, ...] = (),
) -> None:
    if resume and sentinel.exists():
        valid = (
            _valid_json_sentinel(sentinel, required_json_keys)
            if sentinel.suffix.lower() == ".json"
            else sentinel.stat().st_size > 0
        )
        if valid:
            print(f"[reuse] {sentinel}")
            return
        _quarantine_invalid_sentinel(sentinel)
    run(command, cwd=ROOT)


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on invalid or internally inconsistent experiment settings."""

    positive_ints = {
        "--calibration-limit": args.calibration_limit,
        "--eval-limit": args.eval_limit,
        "--max-steps": args.max_steps,
        "--repeat-top-k": args.repeat_top_k,
        "--lora-rank": args.lora_rank,
        "--lora-alpha": args.lora_alpha,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--max-sequence-length": args.max_sequence_length,
        "--max-new-tokens": args.max_new_tokens,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive; received {value!r}")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.minimum_free_disk_gb < 0.0:
        raise ValueError("--minimum-free-disk-gb cannot be negative")

    weights = {
        "gradient": float(args.gradient_weight),
        "ablation": float(args.ablation_weight),
        "structural": float(args.structural_weight),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError(f"importance weights must be finite and non-negative: {weights}")
    if sum(weights.values()) <= 0.0:
        raise ValueError("at least one importance weight must be positive")

    repeat_values: list[int] = []
    for raw in args.repeat_seeds.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            repeat_values.append(int(raw))
        except ValueError as exc:
            raise ValueError(f"invalid integer in --repeat-seeds: {raw!r}") from exc
    if not repeat_values:
        raise ValueError("--repeat-seeds must contain at least one integer")
    if len(repeat_values) != len(set(repeat_values)):
        raise ValueError("--repeat-seeds contains duplicate values")
    if set(repeat_values) == {int(args.base_seed)}:
        raise ValueError("--repeat-seeds must include a seed different from --base-seed")

    for label, path in {
        "Phase 2B catalog": args.phase2b_catalog,
        "Phase 2B gate": args.phase2b_gate,
        "Phase 2B registry": args.phase2b_registry,
        "training data": args.train_file,
        "development data": args.dev_file,
        "phase benchmark": args.phase_benchmark,
    }.items():
        resolved = resolved_repo_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"{label} is missing: {resolved}")


def resolved_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = resolved_repo_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": relative_to_root(resolved, ROOT),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def config_differences(previous: Any, requested: Any, prefix: str = "") -> list[str]:
    """Return precise nested differences for a rejected resume request."""

    if isinstance(previous, dict) and isinstance(requested, dict):
        differences: list[str] = []
        for key in sorted(set(previous) | set(requested)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous:
                differences.append(f"{path}: missing from recorded run")
            elif key not in requested:
                differences.append(f"{path}: no longer present in requested run")
            else:
                differences.extend(config_differences(previous[key], requested[key], path))
        return differences
    if previous != requested:
        return [f"{prefix}: recorded={previous!r}, requested={requested!r}"]
    return []


def approved_run_config_contract_migration(
    previous: dict[str, Any],
    requested: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Approve the v1.4.6→v1.4.7 contract-only Phase 2C data migration."""

    previous_normalized = dict(previous)
    migrations: dict[str, Any] = {}
    for key, path in (("train_file", args.train_file), ("dev_file", args.dev_file)):
        old_fingerprint = previous.get(key)
        new_fingerprint = requested.get(key)
        if old_fingerprint == new_fingerprint:
            continue
        if not isinstance(old_fingerprint, dict) or not isinstance(new_fingerprint, dict):
            return None
        migration = approved_content_only_migration(
            path,
            previous_sha256=old_fingerprint.get("sha256"),
            current_sha256=new_fingerprint.get("sha256"),
        )
        if migration is None:
            return None
        previous_normalized[key] = new_fingerprint
        migrations[key] = migration

    if not migrations or config_differences(previous_normalized, requested):
        return None
    return migrations


def model_fingerprint(model_id: str) -> dict[str, Any]:
    """Fingerprint local reduced weights so resume cannot mix checkpoints."""

    candidate = resolved_repo_path(Path(model_id))
    if not candidate.exists():
        return {"kind": "hub_or_external_id", "model_id": model_id}
    if not candidate.is_dir():
        return {"kind": "local_file", **file_fingerprint(candidate)}
    important_names = {
        "config.json",
        "generation_config.json",
        "phase_2b_stage_manifest.json",
        "depth_reduction_manifest.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    files = [
        path for path in candidate.rglob("*")
        if path.is_file()
        and (
            path.name in important_names
            or path.suffix in {".safetensors", ".bin"}
        )
    ]
    if not files:
        raise RuntimeError(f"No model weights/configuration files found in {candidate}")
    digest = hashlib.sha256()
    total_bytes = 0
    entries: list[dict[str, Any]] = []
    for path in sorted(files, key=lambda value: value.relative_to(candidate).as_posix()):
        relative = path.relative_to(candidate).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        digest.update(file_hash.encode("ascii"))
        entries.append({"path": relative, "bytes": size, "sha256": file_hash})
    return {
        "kind": "local_model_directory",
        "path": relative_to_root(candidate, ROOT),
        "file_count": len(entries),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "files": entries,
    }


def adapter_directory_complete(path: Path) -> bool:
    if not (path / "adapter_config.json").exists():
        return False
    weights = (path / "adapter_model.safetensors", path / "adapter_model.bin")
    return any(candidate.exists() and candidate.stat().st_size > 0 for candidate in weights)


def cleanup_intermediate_checkpoints(training: Path, *, keep: bool) -> None:
    if keep or not training.exists():
        return
    removed = 0
    for checkpoint in training.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint, ignore_errors=True)
            removed += 1
    if removed:
        print(f"[cleanup] removed {removed} intermediate checkpoint director{'y' if removed == 1 else 'ies'} under {training}")


def format_optional(value: Any, spec: str, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return format(number, spec)


def assessment_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {"quality_profile": args.quality_profile}


def prepare_contract_migration_resume(
    *,
    data_path: Path,
    manifest_path: Path,
    summary_path: Path,
    hash_field: str,
) -> None:
    """Force a cheap child resume when a completed sentinel predates contracts."""

    if not manifest_path.exists() or not summary_path.exists():
        return
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        return
    previous_hash = manifest.get(hash_field)
    current_hash = sha256_file(data_path)
    migration = approved_content_only_migration(
        data_path,
        previous_sha256=previous_hash,
        current_sha256=current_hash,
    )
    if migration is not None:
        _quarantine_invalid_sentinel(summary_path)
        print(
            f"[contract migration resume] final sentinel moved so {manifest_path.parent} "
            "can update/rescore durable rows"
        )


def evaluate_model(model_id: str, adapter: Path | None, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    generation = root / "generation"
    nll = root / "teacher_forcing"
    eval_cmd = [
        PYTHON, str(EVALUATE), "--model-id", model_id,
        "--benchmark", str(args.dev_file), "--output-dir", str(generation),
        "--max-new-tokens", str(args.max_new_tokens), "--limit", str(args.eval_limit),
    ]
    nll_cmd = [
        PYTHON, str(NLL), "--model-id", model_id,
        "--input", str(args.dev_file), "--output-dir", str(nll),
        "--limit", str(args.eval_limit),
    ]
    if adapter is not None:
        eval_cmd += ["--adapter-path", str(adapter)]
        nll_cmd += ["--adapter-path", str(adapter)]
    legacy_dev_input_sha256 = getattr(args, "legacy_dev_input_sha256", None)
    if legacy_dev_input_sha256:
        nll_cmd += ["--legacy-input-sha256", str(legacy_dev_input_sha256)]
    if args.resume:
        eval_cmd.append("--resume")
        nll_cmd.append("--resume")
        prepare_contract_migration_resume(
            data_path=args.dev_file,
            manifest_path=generation / "evaluation_manifest.json",
            summary_path=generation / "summary.json",
            hash_field="benchmark_sha256",
        )
        prepare_contract_migration_resume(
            data_path=args.dev_file,
            manifest_path=nll / "evaluation_manifest.json",
            summary_path=nll / "summary.json",
            hash_field="input_sha256",
        )
    run_if_needed(
        eval_cmd,
        generation / "summary.json",
        args.resume,
        required_json_keys=("case_count", "mean_quality_score"),
    )
    legacy_nll_without_manifest = (
        args.resume
        and (nll / "summary.json").exists()
        and (nll / "results.jsonl").exists()
        and not (nll / "evaluation_manifest.json").exists()
    )
    if legacy_nll_without_manifest:
        if not legacy_dev_input_sha256:
            raise RuntimeError(
                f"Legacy NLL evidence lacks a manifest and no approved dataset migration is recorded: {nll}"
            )
        print(f"[legacy NLL migration] validating and manifesting durable rows under {nll}")
        run(nll_cmd, cwd=ROOT)
    else:
        run_if_needed(
            nll_cmd,
            nll / "summary.json",
            args.resume,
            required_json_keys=("case_count", "mean_target_nll"),
        )
    return {
        "generation_summary": read_json(generation / "summary.json"),
        "nll_summary": read_json(nll / "summary.json"),
        "generation_results": generation / "results.jsonl",
    }


def deviation_sentinel_current(path: Path, reference_results: Path, candidate_results: Path) -> bool:
    if not _valid_json_sentinel(path, ("summary",)):
        return False
    payload = read_json(path)
    summary = payload.get("summary") or {}
    return (
        summary.get("reference_results_sha256") == sha256_file(reference_results)
        and summary.get("candidate_results_sha256") == sha256_file(candidate_results)
    )


def train_candidate(
    *,
    name: str,
    model_id: str,
    layers: str | list[int],
    module_group: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    root = args.output_root / "candidates" / name / f"seed_{seed}"
    training = root / "training"
    adapter = training / "adapter"
    layer_text = layers if isinstance(layers, str) else ",".join(str(value) for value in layers)
    command = [
        PYTHON, str(TRAIN), "--model-id", model_id,
        "--train-file", str(args.train_file), "--dev-file", str(args.dev_file),
        "--output-dir", str(training), "--selected-layers", layer_text,
        "--target-module-group", module_group, "--max-steps", str(args.max_steps),
        "--seed", str(seed), "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--learning-rate", str(args.learning_rate),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--max-sequence-length", str(args.max_sequence_length),
        "--run-label", f"phase2c:{name}:seed_{seed}",
    ]
    if args.qlora:
        command.append("--qlora")
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if not args.keep_training_checkpoints:
        command.append("--disable-intermediate-checkpoints")
    command += ["--minimum-free-disk-gb", str(args.minimum_free_disk_gb)]
    training_manifest_path = training / "training_manifest.json"
    complete_training = (
        _valid_json_sentinel(training_manifest_path, ("trainable_parameter_count",))
        and adapter_directory_complete(adapter)
    )
    if args.resume and training.exists() and not complete_training:
        print(f"[clean incomplete training] {training}")
        shutil.rmtree(training)
    run_if_needed(
        command,
        training_manifest_path,
        args.resume,
        required_json_keys=("trainable_parameter_count",),
    )
    if not adapter_directory_complete(adapter):
        raise RuntimeError(f"Candidate training did not produce a complete adapter: {adapter}")
    cleanup_intermediate_checkpoints(training, keep=args.keep_training_checkpoints)
    evidence = evaluate_model(model_id, adapter, root / "evaluation", args)
    baseline_results = args.output_root / "reference/reduced_frozen/generation/results.jsonl"
    deviation_json = root / "evaluation/output_deviation.json"
    candidate_results = Path(evidence["generation_results"])
    if args.resume and deviation_json.exists() and not deviation_sentinel_current(
        deviation_json, baseline_results, candidate_results
    ):
        _quarantine_invalid_sentinel(deviation_json)
    run_if_needed(
        [
            PYTHON, str(COMPARE), "--reference", str(baseline_results),
            "--candidate", str(evidence["generation_results"]),
            "--out-json", str(deviation_json),
            "--out-report", str(root / "evaluation/OUTPUT_DEVIATION.md"),
        ],
        deviation_json,
        args.resume,
        required_json_keys=("summary",),
    )
    manifest = read_json(training / "training_manifest.json")
    row = {
        "name": name,
        "seed": seed,
        "layers": layers,
        "selected_layers": layer_text,
        "module_group": module_group,
        "model_id": model_id,
        "adapter": str(adapter),
        "artifact_root": str(root),
        "training_manifest": manifest,
        "generation_summary": evidence["generation_summary"],
        "nll_summary": evidence["nll_summary"],
        "output_deviation": read_json(deviation_json)["summary"],
    }
    write_json(root / "candidate_result.json", row)
    return row


def profile_payload(name: str, row: dict[str, Any], source_model: str, phase_summary: str) -> dict[str, Any]:
    return {
        "profile": name,
        "candidate": row["name"],
        "seed": row["seed"],
        "training_model_id": source_model,
        "training_adapter": row["adapter"],
        "inference_model": source_model,
        "evaluation_summary": phase_summary,
        "selected_layers": row["selected_layers"],
        "module_group": row["module_group"],
        "trainable_parameter_count": int(row["training_manifest"].get("trainable_parameter_count", 0)),
        "seed_stability_pass": bool(row.get("seed_stability_pass", False)),
        "repeat_seed_count": len(row.get("repeat_results") or []),
        "assessment": row["assessment"],
    }


def render_report(
    source_profile: dict[str, Any], ranking: dict[str, Any], rows: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]], args: argparse.Namespace,
) -> str:
    rank_lines = [
        f"| {r['importance_rank']} | {r['current_layer_id']} | {r['original_layer_id']} | "
        f"{format_optional(r.get('combined_importance_score'), '.4f')} | "
        f"{format_optional(r.get('gradient_taylor_score'), '.3e')} | "
        f"{format_optional(r.get('ablation_nll_increase'), '+.4f')} |"
        for r in ranking["ranking"]
    ]
    candidate_lines = []
    for row in rows:
        a = row["assessment"]
        candidate_lines.append(
            f"| {row['name']} | {row['selected_layers']} | {row['module_group']} | "
            f"{format_optional(a.get('quality_index'), '.4f')} | "
            f"{format_optional(a.get('resource_gain_fraction'), '.3%')} | "
            f"{format_optional(a.get('quality_opportunity_loss_fraction'), '.3%')} | "
            f"{format_optional(a.get('opportunity_margin_across_seeds', a.get('opportunity_margin')), '+.3%')} | "
            f"{row['generation_summary'].get('critical_failure_count', 0)} | "
            f"{len(row.get('repeat_results') or [])} | "
            f"{'WORTHWHILE' if a['eligible'] else 'NOT WORTHWHILE'} |"
        )
    profile_lines = [
        f"| {name} | {row['name']} | {row['selected_layers']} | "
        f"{row['training_manifest'].get('trainable_parameter_count', 0):,} | "
        f"{format_optional((row.get('assessment') or {}).get('quality_index'), '.4f')} |"
        for name, row in profiles.items()
    ]
    return f"""# PHASE 2C — IMPORTANCE-GUIDED TARGETED RECOVERY REPORT

## Purpose

Preserve the approved Phase 2B reduced architecture while finding the smallest surviving-layer LoRA placement whose proportional trainable-state saving is greater than its proportional quality opportunity loss.

## Evidence boundary

- Source Phase 2B profile: `{args.phase2b_profile}`
- Source reduced model: `{source_profile['training_model_id']}`
- Sealed evaluation used: **No**
- Layer-ranking calibration cases: {ranking['calibration_case_count']}
- Development evaluation cases per candidate: {args.eval_limit}
- Base seed: {args.base_seed}; repeat seeds: {args.repeat_seeds}

## Surviving-layer importance ranking

| Rank | Current layer | Original layer | Combined | Taylor | Ablation NLL delta |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rank_lines)}

The ranking combines first-order target-loss sensitivity, temporary layer-bypass NLL damage, and optional final-stage Phase 2B structural evidence. It ranks where limited updates may be useful; it does not authorize deleting high- or low-ranked layers.

## Candidate frontier

| Candidate | Current layers trained | Modules | Quality index | Trainable reduction | Quality opportunity loss | Opportunity margin | Critical | Repeat seeds | Gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(candidate_lines)}

## Recommended retained profiles

| Profile | Candidate | Trained layers | Trainable parameters | Quality index |
|---|---|---|---:|---:|
{chr(10).join(profile_lines)}

## Selection rule

A selective candidate is compared with a same-seed all-surviving-layer control. Proportional trainable-state savings must be strictly greater than combined proportional quality opportunity loss. Semantics, content anchors, completion, categories, issues, and target NLL contribute continuously and never act as private absolute vetoes. Peak training VRAM and wall time remain measured secondary evidence.

## Handoff

After human approval, Phase 3 loads the same reduced Phase 2B checkpoint plus the selected Phase 2C SFT adapter. Phase 3 must write profile-specific artifacts so incompatible architectures or adapter placements are never mixed in Phase 4 soups.
"""


def main() -> None:
    args = parse_args()
    if not args.train_file.exists() or not args.dev_file.exists():
        run([PYTHON, str(BUILD_DATA)], cwd=ROOT)
    validate_args(args)
    if not args.phase2b_catalog.exists():
        raise FileNotFoundError(args.phase2b_catalog)
    gate_text = args.phase2b_gate.read_text(encoding="utf-8") if args.phase2b_gate.exists() else ""
    source_profile = resolve_profile(read_json(args.phase2b_catalog), args.phase2b_profile, gate_text)
    source_model = str(source_profile["training_model_id"])

    run_config = {
        "manifest_version": 1,
        "source_profile": args.phase2b_profile,
        "source_model": source_model,
        "source_model_fingerprint": model_fingerprint(source_model),
        "phase2b_catalog": file_fingerprint(args.phase2b_catalog),
        "phase2b_registry": file_fingerprint(args.phase2b_registry),
        "train_file": file_fingerprint(args.train_file),
        "dev_file": file_fingerprint(args.dev_file),
        "phase_benchmark": file_fingerprint(args.phase_benchmark),
        "calibration_limit": args.calibration_limit,
        "eval_limit": args.eval_limit,
        "max_steps": args.max_steps,
        "base_seed": args.base_seed,
        "repeat_seeds": args.repeat_seeds,
        "repeat_top_k": args.repeat_top_k,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_sequence_length": args.max_sequence_length,
        "importance_weights": {
            "gradient_taylor": args.gradient_weight,
            "temporary_ablation_nll": args.ablation_weight,
            "phase2b_structural": args.structural_weight,
        },
        "max_new_tokens": args.max_new_tokens,
        "relative_opportunity_policy": {
            "quality_profile": args.quality_profile,
            "decision_rule": "trainable-state gain > combined proportional quality loss",
            "absolute_quality_floors": "prohibited",
        },
        "qlora": args.qlora,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    config_path = args.output_root / "run_config.json"
    if args.resume:
        if not config_path.exists():
            raise RuntimeError(f"--resume requested but run configuration is missing: {config_path}")
        previous_config = read_json(config_path)
        differences = config_differences(previous_config, run_config)
        if differences:
            migrations = approved_run_config_contract_migration(previous_config, run_config, args)
            if migrations is not None:
                write_json(
                    args.output_root / "RUN_CONFIG_CONTRACT_MIGRATION_v1.4.7.json",
                    {
                        "schema_version": 1,
                        "migration_type": "phase2c_contract_only_dataset_upgrade",
                        "migrations": migrations,
                        "training_and_generation_messages_unchanged": True,
                        "existing_adapters_and_raw_replies_reusable": True,
                        "existing_generation_quality_rows_require_rescore": True,
                    },
                )
                write_json(config_path, run_config)
                print(
                    "[approved Phase 2C migration] dataset messages/targets are unchanged; "
                    "existing adapters and raw replies will be reused and rescored"
                )
            else:
                detail = "\n  - ".join(differences[:40])
                suffix = "\n  - ... additional differences omitted" if len(differences) > 40 else ""
                raise RuntimeError(
                    "Resume settings differ from the recorded Phase 2C run. Use the original settings "
                    "or choose a new output root for a different experiment.\n  - "
                    + detail
                    + suffix
                )
    elif args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(
            f"Output root already contains files: {args.output_root}. Use --resume for the exact "
            "same run, or choose/delete the output root for a clean experiment."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_json(config_path, run_config)

    migration_audit_path = args.output_root / "RUN_CONFIG_CONTRACT_MIGRATION_v1.4.7.json"
    args.legacy_dev_input_sha256 = None
    if migration_audit_path.exists():
        migration_audit = read_json(migration_audit_path)
        dev_migration = (migration_audit.get("migrations") or {}).get("dev_file") or {}
        previous_hash = dev_migration.get("previous_sha256")
        current_hash = dev_migration.get("current_sha256")
        if previous_hash and current_hash == run_config["dev_file"]["sha256"]:
            args.legacy_dev_input_sha256 = str(previous_hash)

    ranking_path = args.output_root / "importance/combined_layer_ranking.json"
    run_if_needed(
        [
            PYTHON, str(RANK), "--model-id", source_model,
            "--calibration-file", str(args.dev_file), "--output", str(ranking_path),
            "--phase2b-registry", str(args.phase2b_registry),
            "--limit", str(args.calibration_limit),
            "--gradient-weight", str(args.gradient_weight),
            "--ablation-weight", str(args.ablation_weight),
            "--structural-weight", str(args.structural_weight),
            "--require-cuda",
        ],
        ranking_path,
        args.resume,
        required_json_keys=("ranking", "layer_count"),
    )
    ranking = read_json(ranking_path)

    baseline_root = args.output_root / "reference/reduced_frozen"
    baseline_eval = evaluate_model(source_model, None, baseline_root, args)
    reduced_frozen = {
        "name": "reduced_frozen",
        "generation_summary": baseline_eval["generation_summary"],
        "nll_summary": baseline_eval["nll_summary"],
    }

    specs = candidate_specs(ranking["ranking"], args.base_seed)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            train_candidate(
                name=spec["name"], model_id=source_model, layers=spec["layers"],
                module_group=spec["module_group"], seed=args.base_seed, args=args,
            )
        )
    by_name = {row["name"]: row for row in rows}
    all_remaining = by_name["all_remaining"]
    for row in rows:
        row["assessment"] = assess_training_candidate(
            candidate=row,
            all_remaining=all_remaining,
            reduced_frozen=reduced_frozen,
            **assessment_kwargs(args),
        )

    selective = [row for row in rows if row["name"] != "all_remaining" and row["assessment"]["eligible"]]
    selective.sort(
        key=lambda row: (
            float(row["assessment"]["quality_index"]),
            float(row["assessment"]["trainable_parameter_reduction_vs_all_remaining"]),
        ),
        reverse=True,
    )
    provisional_profiles = select_profiles(rows)
    repeat_names = {row["name"] for row in selective[: args.repeat_top_k]}
    repeat_names.update(row["name"] for row in provisional_profiles.values())
    repeat_seeds = sorted(
        {int(value.strip()) for value in args.repeat_seeds.split(",") if value.strip()}
        - {args.base_seed}
    )
    if not repeat_seeds:
        raise ValueError("Phase 2C requires at least one repeat seed different from the base seed.")

    repeated: list[dict[str, Any]] = []
    spec_by_name = {spec["name"]: spec for spec in specs}
    all_remaining_by_seed: dict[int, dict[str, Any]] = {}

    # Every repeated selective candidate is compared with an independently
    # trained all-surviving-layer control using the exact same seed.
    all_spec = spec_by_name["all_remaining"]
    for seed in repeat_seeds:
        same_seed_all = train_candidate(
            name="all_remaining", model_id=source_model, layers=all_spec["layers"],
            module_group=all_spec["module_group"], seed=seed, args=args,
        )
        same_seed_all["assessment"] = assess_training_candidate(
            candidate=same_seed_all,
            all_remaining=same_seed_all,
            reduced_frozen=reduced_frozen,
            **assessment_kwargs(args),
        )
        all_remaining_by_seed[seed] = same_seed_all
        repeated.append(same_seed_all)

    repeated_names: set[str] = set()

    def repeat_and_gate(base_row: dict[str, Any]) -> None:
        name = str(base_row["name"])
        if name in repeated_names:
            return
        base_row["repeat_results"] = []
        spec = spec_by_name[name]
        margins = [float(base_row["assessment"].get("opportunity_margin", 0.0))]
        gains = [float(base_row["assessment"].get("resource_gain_fraction", 0.0))]
        losses = [float(base_row["assessment"].get("quality_opportunity_loss_fraction", 0.0))]
        technical_valid = bool(base_row["assessment"].get("technical_valid", True))
        for seed in repeat_seeds:
            same_seed_all = all_remaining_by_seed[seed]
            if name == "all_remaining":
                repeat = same_seed_all
            else:
                repeat = train_candidate(
                    name=name, model_id=source_model, layers=spec["layers"],
                    module_group=spec["module_group"], seed=seed, args=args,
                )
                repeat["assessment"] = assess_training_candidate(
                    candidate=repeat,
                    all_remaining=same_seed_all,
                    reduced_frozen=reduced_frozen,
                    **assessment_kwargs(args),
                )
                repeated.append(repeat)
            base_row["repeat_results"].append(repeat)
            assessment = repeat["assessment"]
            technical_valid = technical_valid and bool(assessment.get("technical_valid", True))
            margins.append(float(assessment.get("opportunity_margin", 0.0)))
            gains.append(float(assessment.get("resource_gain_fraction", 0.0)))
            losses.append(float(assessment.get("quality_opportunity_loss_fraction", 0.0)))
        mean_gain = sum(gains) / len(gains)
        mean_loss = sum(losses) / len(losses)
        mean_margin = mean_gain - mean_loss
        base_row["repeat_mean_resource_gain"] = mean_gain
        base_row["repeat_mean_quality_opportunity_loss"] = mean_loss
        base_row["repeat_mean_opportunity_margin"] = mean_margin
        base_row["seed_stability_pass"] = technical_valid and (name == "all_remaining" or mean_margin > 0.0)
        if name != "all_remaining":
            base_row["assessment"]["eligible"] = base_row["seed_stability_pass"]
            base_row["assessment"]["opportunity_margin_across_seeds"] = mean_margin
            base_row["assessment"]["decision_explanation_across_seeds"] = (
                f"mean resource gain {mean_gain:.6f} versus mean quality loss {mean_loss:.6f}"
            )
            if not base_row["seed_stability_pass"]:
                base_row["assessment"]["rejection_reasons"].append(
                    "mean proportional opportunity margin across recorded seeds is not positive"
                )
        repeated_names.add(name)

    for name in sorted(repeat_names):
        repeat_and_gate(by_name[name])

    # A failed finalist can expose a different recommendation. Repeat any newly
    # selected profile and reselect until all final profiles have seed evidence.
    while True:
        provisional_profiles = select_profiles(rows)
        missing = {row["name"] for row in provisional_profiles.values()} - repeated_names
        if not missing:
            break
        for name in sorted(missing):
            repeat_and_gate(by_name[name])

    profiles = select_profiles(rows)
    phase_summaries: dict[str, str] = {}
    catalog_profiles: dict[str, dict[str, Any]] = {}
    for profile_name, row in profiles.items():
        eval_root = args.output_root / "profile_validation" / profile_name
        adapter = Path(row["adapter"])
        generation = eval_root / "generation"
        profile_command = [
            PYTHON, str(EVALUATE), "--model-id", source_model,
            "--adapter-path", str(adapter), "--benchmark", str(args.phase_benchmark),
            "--output-dir", str(generation), "--max-new-tokens", str(args.max_new_tokens),
        ]
        if args.resume:
            profile_command.append("--resume")
        run_if_needed(
            profile_command,
            generation / "summary.json",
            args.resume,
            required_json_keys=("case_count", "mean_quality_score"),
        )
        phase_summaries[profile_name] = str(generation / "summary.json")
        catalog_profiles[profile_name] = profile_payload(profile_name, row, source_model, phase_summaries[profile_name])

    write_json(args.output_root / "candidate_registry.json", {"base_seed_candidates": rows, "repeat_candidates": repeated})
    catalog = {
        "manifest_version": 1,
        "status": "PENDING_HUMAN_REVIEW",
        "source_phase2b_profile": args.phase2b_profile,
        "source_phase2b_catalog": str(args.phase2b_catalog),
        "source_model": source_model,
        "train_file": relative_to_root(args.train_file, ROOT),
        "dev_file": relative_to_root(args.dev_file, ROOT),
        "importance_ranking": str(ranking_path),
        "base_seed": args.base_seed,
        "repeat_seeds": repeat_seeds,
        "profiles": catalog_profiles,
        "candidate_registry": str(args.output_root / "candidate_registry.json"),
        "gate_report": str(args.gate_report),
        "downstream_rule": (
            "Phase 3 may consume only a named Phase 2C profile explicitly approved in the Phase 2C gate. "
            "The source reduced checkpoint and Phase 2C adapter must remain paired."
        ),
    }
    write_json(args.output_root / "model_catalog.json", catalog)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.report, render_report(source_profile, ranking, rows, profiles, args))
    atomic_write_text(
        args.gate_report,
        "\n".join(
            [
                "# PHASE 2C STOP GATE", "", "## Automated status", "", "`PENDING_HUMAN_REVIEW`", "",
                f"- Model catalog: `{args.output_root / 'model_catalog.json'}`",
                f"- Report: `{args.report}`",
                f"- Maximum-quality recommendation: `{profiles['maximum_quality']['name']}`",
                f"- Balanced-training recommendation: `{profiles['balanced_training']['name']}`",
                f"- Minimum-training-memory recommendation: `{profiles['minimum_training_memory']['name']}`",
                "", "## Required human decision", "",
                "Review base-seed and repeated-seed replies, especially logic, clarification, safety, malformed output, and the weakest category. Confirm that the selected adapter remains paired with the exact reduced checkpoint.",
                "", "- Human decision: `PENDING`", "- Reviewer:", "- Date:",
                "- Approved profiles: `maximum_quality`", "- Notes:", "",
            ]
        ),
    )
    print(args.output_root / "model_catalog.json")
    print(args.report)
    print(args.gate_report)


if __name__ == "__main__":
    main()
