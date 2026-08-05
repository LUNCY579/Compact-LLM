from __future__ import annotations

"""Read-only structural and provenance audit for a Phase 2C artifact tree."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, read_jsonl, write_json

EXPECTED_PROFILES = {"maximum_quality", "balanced_training", "minimum_training_memory"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase 2C artifacts without modifying experiment evidence.")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/phase_2c_targeted_recovery")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_object(path: Path, errors: list[str], *, label: str) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        errors.append(f"missing or empty {label}: {path}")
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        errors.append(f"invalid {label} {path}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label} is not a JSON object: {path}")
        return None
    return payload


def adapter_complete(path: Path) -> bool:
    if not (path / "adapter_config.json").is_file():
        return False
    return any(
        candidate.is_file() and candidate.stat().st_size > 0
        for candidate in (path / "adapter_model.safetensors", path / "adapter_model.bin")
    )


def validate_summary(
    path: Path,
    errors: list[str],
    *,
    label: str,
    required_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    payload = load_object(path, errors, label=label)
    if payload is None:
        return None
    missing = [field for field in required_fields if field not in payload]
    if missing:
        errors.append(f"{label} is missing fields {missing}: {path}")
    for field in ("case_count", "input_case_count", "evaluated_case_count"):
        if field in payload:
            try:
                if int(payload[field]) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{label} has invalid {field}={payload[field]!r}: {path}")
    for field in ("mean_quality_score", "mean_target_nll"):
        if field in payload and payload[field] is not None:
            try:
                if not math.isfinite(float(payload[field])):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{label} has invalid {field}={payload[field]!r}: {path}")
    return payload


def validate_candidate(row: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    name = str(row.get("name", "<unnamed>"))
    seed = row.get("seed", "?")
    prefix = f"{name} seed {seed}"
    artifact_value = row.get("artifact_root")
    if not isinstance(artifact_value, str) or not artifact_value:
        errors.append(f"{prefix}: missing artifact_root")
        return
    candidate_root = repo_path(artifact_value)
    training = candidate_root / "training"
    adapter = training / "adapter"
    if not adapter_complete(adapter):
        errors.append(f"{prefix}: incomplete adapter directory {adapter}")
    manifest = load_object(training / "training_manifest.json", errors, label=f"{prefix} training manifest")
    if manifest is not None:
        count = manifest.get("trainable_parameter_count")
        try:
            if int(count) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{prefix}: invalid trainable_parameter_count={count!r}")

    generation_dir = candidate_root / "evaluation/generation"
    generation = validate_summary(
        generation_dir / "summary.json",
        errors,
        label=f"{prefix} generation summary",
        required_fields=("case_count", "mean_quality_score"),
    )
    results_path = generation_dir / "results.jsonl"
    if not results_path.exists():
        errors.append(f"{prefix}: missing generation results {results_path}")
    else:
        try:
            rows = read_jsonl(results_path)
            ids = [str(item.get("id")) for item in rows]
            if len(ids) != len(set(ids)):
                errors.append(f"{prefix}: generation results contain duplicate IDs")
            if generation is not None and len(rows) != int(generation.get("case_count", -1)):
                errors.append(
                    f"{prefix}: generation summary case_count={generation.get('case_count')} "
                    f"but results contain {len(rows)} rows"
                )
        except (OSError, ValueError) as exc:
            errors.append(f"{prefix}: invalid generation results: {type(exc).__name__}: {exc}")

    validate_summary(
        candidate_root / "evaluation/teacher_forcing/summary.json",
        errors,
        label=f"{prefix} teacher-forcing summary",
        required_fields=("case_count", "mean_target_nll"),
    )
    deviation = load_object(
        candidate_root / "evaluation/output_deviation.json",
        errors,
        label=f"{prefix} output deviation",
    )
    if deviation is not None:
        summary = deviation.get("summary")
        if not isinstance(summary, dict) or not summary.get("technical_valid", True):
            errors.append(f"{prefix}: output deviation is missing a technically valid summary")

    result = load_object(candidate_root / "candidate_result.json", errors, label=f"{prefix} candidate result")
    if result is not None:
        if str(result.get("name")) != name or result.get("seed") != row.get("seed"):
            errors.append(f"{prefix}: candidate_result identity does not match registry row")
        if str(result.get("adapter")) != str(row.get("adapter")):
            errors.append(f"{prefix}: candidate_result adapter differs from registry row")

    checkpoints = sorted(training.glob("checkpoint-*"))
    if checkpoints:
        warnings.append(f"{prefix}: {len(checkpoints)} intermediate checkpoint director(ies) remain")


def main() -> None:
    args = parse_args()
    root = args.artifact_root
    errors: list[str] = []
    warnings: list[str] = []

    run_config = load_object(root / "run_config.json", errors, label="run configuration")
    ranking = load_object(root / "importance/combined_layer_ranking.json", errors, label="importance ranking")
    registry = load_object(root / "candidate_registry.json", errors, label="candidate registry")
    catalog = load_object(root / "model_catalog.json", errors, label="model catalog")
    validate_summary(
        root / "reference/reduced_frozen/generation/summary.json",
        errors,
        label="frozen reference generation summary",
        required_fields=("case_count", "mean_quality_score"),
    )
    validate_summary(
        root / "reference/reduced_frozen/teacher_forcing/summary.json",
        errors,
        label="frozen reference teacher-forcing summary",
        required_fields=("case_count", "mean_target_nll"),
    )

    rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    if registry is not None:
        base_rows = list(registry.get("base_seed_candidates") or [])
        repeat_rows = list(registry.get("repeat_candidates") or [])
        rows = [*base_rows, *repeat_rows]
        identities = [(str(row.get("name")), row.get("seed")) for row in rows]
        if len(identities) != len(set(identities)):
            errors.append("candidate registry contains duplicate (name, seed) rows")
        if not any(str(row.get("name")) == "all_remaining" for row in base_rows):
            errors.append("base candidate registry is missing all_remaining")
        for row in rows:
            if not isinstance(row, dict):
                errors.append("candidate registry contains a non-object row")
                continue
            validate_candidate(row, errors, warnings)

    if ranking is not None:
        ranked = ranking.get("ranking") or []
        try:
            layer_count = int(ranking.get("layer_count", -1))
        except (TypeError, ValueError):
            layer_count = -1
        if len(ranked) != layer_count:
            errors.append("importance ranking count does not match layer_count")
        try:
            ranks = sorted(int(row.get("importance_rank", 0)) for row in ranked)
        except (TypeError, ValueError):
            ranks = []
        if ranks != list(range(1, len(ranked) + 1)):
            errors.append("importance ranks are not contiguous")
        current_ids = [row.get("current_layer_id") for row in ranked]
        if len(current_ids) != len(set(current_ids)):
            errors.append("importance ranking contains duplicate current_layer_id values")

    if catalog is not None:
        profiles = catalog.get("profiles") or {}
        if set(profiles) != EXPECTED_PROFILES:
            errors.append(f"profile names are {sorted(profiles)}, expected {sorted(EXPECTED_PROFILES)}")
        expected_repeat_seeds = {int(value) for value in (catalog.get("repeat_seeds") or [])}
        base_by_name = {str(row.get("name")): row for row in base_rows}
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict):
                errors.append(f"profile {profile_name}: payload is not an object")
                continue
            candidate = str(profile.get("candidate"))
            source = base_by_name.get(candidate)
            if source is None:
                errors.append(f"profile {profile_name}: candidate {candidate!r} is missing from base registry")
                continue
            assessment = source.get("assessment") or {}
            if not assessment.get("technical_valid", True):
                errors.append(f"profile {profile_name}: candidate {candidate!r} is technically invalid")
            if not assessment.get("eligible"):
                errors.append(f"profile {profile_name}: candidate {candidate!r} is ineligible")
            if not source.get("seed_stability_pass"):
                errors.append(f"profile {profile_name}: candidate {candidate!r} failed or lacks seed stability")
            observed = {int(item.get("seed")) for item in (source.get("repeat_results") or [])}
            if expected_repeat_seeds and observed != expected_repeat_seeds:
                errors.append(
                    f"profile {profile_name}: repeat seeds {sorted(observed)} do not match "
                    f"{sorted(expected_repeat_seeds)}"
                )
            adapter_value = profile.get("training_adapter")
            if not isinstance(adapter_value, str) or not adapter_complete(repo_path(adapter_value)):
                errors.append(f"profile {profile_name}: incomplete training adapter {adapter_value!r}")
            summary_value = profile.get("evaluation_summary")
            if not isinstance(summary_value, str):
                errors.append(f"profile {profile_name}: missing evaluation_summary")
            else:
                validate_summary(
                    repo_path(summary_value),
                    errors,
                    label=f"profile {profile_name} evaluation summary",
                    required_fields=("case_count", "mean_quality_score"),
                )

    if run_config is not None and catalog is not None:
        if str(run_config.get("source_model")) != str(catalog.get("source_model")):
            errors.append("run_config and model_catalog disagree on source_model")

    status = "CLEAN_COMPLETE" if not errors else "NEEDS_ATTENTION"
    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": status,
        "artifact_root": str(root),
        "base_candidate_count": len(base_rows),
        "candidate_seed_count": len(rows),
        "errors": errors,
        "warnings": warnings,
    }
    print("PHASE 2C ARTIFACT AUDIT")
    print("=" * 72)
    print(f"Artifact root: {root}")
    print(f"Candidates:    {len(base_rows)} base configurations / {len(rows)} total seeds")
    print(f"Errors:        {len(errors)}")
    print(f"Warnings:      {len(warnings)}")
    for error in errors:
        print(f"[ERROR] {error}")
    for warning in warnings:
        print(f"[WARNING] {warning}")
    print(f"\nFINAL STATUS: {status}")
    if args.json_out:
        write_json(args.json_out, payload)
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
