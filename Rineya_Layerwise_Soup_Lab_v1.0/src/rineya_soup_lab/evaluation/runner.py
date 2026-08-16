from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import ExperimentConfig
from ..errors import EvaluationError
from .external import run_external_evaluator, run_official_phase_v45
from .mock import run_mock_evaluator
from .native_v45 import evaluate_model_bundled_v45
from .dataset import load_dataset
from ..model.layer_map import MergeRecipe
from ..utils import (
    canonical_json,
    read_json,
    sha256_file,
    utc_timestamp,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class EvaluationEvidence:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    output_dir: Path
    backend: str
    protocol_name: str


def run_evaluation(
    config: ExperimentConfig,
    *,
    model_path: Path,
    tokenizer_path: Path | None = None,
    dataset_path: Path,
    output_dir: Path,
    protocol_name: str,
    resume: bool = True,
    recipe: MergeRecipe | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> EvaluationEvidence:
    evaluation = config.get("evaluation")
    protocols = evaluation["protocols"]
    if protocol_name not in protocols:
        raise EvaluationError(f"Unknown evaluation protocol: {protocol_name}")
    protocol = dict(protocols[protocol_name])
    protocol.setdefault("seed", int(config.get("project.seed", 0)))
    if protocol.get("enabled", True) is False:
        raise EvaluationError(f"Evaluation protocol is disabled: {protocol_name}")
    backend = str(evaluation["backend"])
    tokenizer_path = Path(
        tokenizer_path
        or config.get("models.tokenizer_path")
        or config.get("models.base_model_path")
        or model_path
    )
    if backend == "official_phase_v45":
        quality_config_path = (
            Path(evaluation["official_repo_root"])
            / "configs"
            / "quality_evaluation.json"
        )
    else:
        quality_config_path = Path(evaluation["quality_config"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "lab_evaluation_manifest.json"
    manifest_identity = {
        "schema_version": "1.0",
        "backend": backend,
        "model_path": str(model_path.expanduser().resolve()),
        "tokenizer_path": str(tokenizer_path.expanduser().resolve()),
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "quality_config_path": str(quality_config_path.expanduser().resolve()),
        "quality_config_sha256": sha256_file(quality_config_path),
        "protocol_name": protocol_name,
        "protocol": dict(protocol),
        "recipe": recipe.identity_dict() if recipe is not None else None,
        "provenance": dict(provenance or {}),
    }
    if manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        existing_identity = dict(existing_manifest)
        existing_identity.pop("created_at", None)
        if canonical_json(existing_identity) != canonical_json(manifest_identity):
            raise EvaluationError(
                f"Evaluation directory provenance mismatch: {output_dir}. "
                "Do not reuse evidence for a different model, source checkpoint, dataset, or protocol."
            )
    else:
        if (output_dir / "results.jsonl").exists() or (output_dir / "summary.json").exists():
            raise EvaluationError(
                f"Evaluation evidence exists without a lab provenance manifest: {output_dir}."
            )
        write_json(manifest_path, {**manifest_identity, "created_at": utc_timestamp()})

    if backend == "bundled_v45":
        summary, rows = evaluate_model_bundled_v45(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            quality_config_path=quality_config_path,
            protocol_name=protocol_name,
            protocol=protocol,
            trust_remote_code=bool(config.get("models.trust_remote_code", False)),
            resume=resume,
        )
    elif backend == "official_phase_v45":
        summary, rows = run_official_phase_v45(
            official_repo_root=Path(evaluation["official_repo_root"]),
            python_executable=(Path(evaluation["python_executable"]) if evaluation.get("python_executable") else None),
            model_path=model_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            protocol=protocol,
            resume=resume,
        )
    elif backend == "external":
        values = {
            "python": str(evaluation.get("python_executable") or sys.executable),
            "model": str(model_path),
            "dataset": str(dataset_path),
            "output_dir": str(output_dir),
            "quality_config": str(quality_config_path),
            "protocol": protocol_name,
            "seed": str(int(protocol.get("seed", 0))),
        }
        summary, rows = run_external_evaluator(
            template=evaluation["external_command"],
            values=values,
            cwd=config.root,
            output_dir=output_dir,
            summary_path_template=str(evaluation["external_summary_path"]),
            results_path_template=str(evaluation["external_results_path"]),
        )
    elif backend == "mock":
        if not evaluation.get("mock_allowed"):
            raise EvaluationError("Mock backend is disabled.")
        summary, rows = run_mock_evaluator(
            model_path=model_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            protocol_name=protocol_name,
            recipe=recipe,
        )
    else:  # pragma: no cover - validated earlier
        raise EvaluationError(f"Unsupported evaluation backend: {backend}")

    validate_evidence(summary, rows, dataset_path)
    # Normalize every backend into the same run-owned evidence contract. This also
    # makes external commands that write to custom locations safely resumable.
    # Do not atomically replace an already-written results.jsonl: Windows scanners
    # lock that file and the native evaluator already appended every completed row.
    write_json(output_dir / "summary.json", summary)
    results_path = output_dir / "results.jsonl"
    if not results_path.exists():
        write_jsonl(results_path, rows)
    write_json(
        output_dir / "lab_evaluation_result.json",
        {
            "schema_version": "1.0",
            "completed_at": utc_timestamp(),
            "manifest_sha256": sha256_file(manifest_path),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "results_sha256": sha256_file(output_dir / "results.jsonl"),
            "case_count": len(rows),
        },
    )
    return EvaluationEvidence(
        summary=summary,
        rows=rows,
        output_dir=output_dir,
        backend=backend,
        protocol_name=protocol_name,
    )


def validate_evidence(
    summary: Mapping[str, Any], rows: list[Mapping[str, Any]], dataset_path: Path
) -> None:
    if str(summary.get("quality_evaluation_version")) != "4.5":
        raise EvaluationError("Evaluation evidence is not Quality Evaluator v4.5.")
    if int(summary.get("case_count", -1)) != len(rows):
        raise EvaluationError("Evaluation summary case_count does not match results rows.")
    expected_ids = [str(row["id"]) for row in load_dataset(dataset_path)]
    actual_ids = [str(row.get("id")) for row in rows]
    if actual_ids != expected_ids:
        raise EvaluationError("Evaluation rows are not in exact dataset ID order.")
    for row in rows:
        if row.get("quality_score") is None:
            raise EvaluationError(f"Result row {row.get('id')} has no quality_score.")
