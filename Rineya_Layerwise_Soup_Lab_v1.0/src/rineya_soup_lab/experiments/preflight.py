from __future__ import annotations

from pathlib import Path
import json

from ..evaluation.dataset import audit_splits, require_split_audit, write_split_audit
from ..model.compatibility import compare_checkpoints, require_compatible, write_compatibility_report
from ..model.sources import ResolvedSources, resolve_model_sources
from ..run_context import RunContext
from ..utils import sha256_file, write_csv, write_json
from ..vendor.official_v45.quality_evaluation import (
    QUALITY_EVALUATION_VERSION,
    load_quality_config,
)


def run_preflight(context: RunContext) -> ResolvedSources:
    sources = resolve_model_sources(context)
    report = compare_checkpoints(
        sources.anchor,
        sources.greedy,
        layer_key_regex=str(context.config.get("models.layer_key_regex")),
        expected_layers=context.config.get("models.expected_layers"),
        allow_quantized_sources=bool(context.config.get("models.allow_quantized_sources", False)),
        require_non_floating_equal=bool(
            context.config.get("models.require_non_floating_equal", True)
        ),
        lineage_acknowledged=bool(
            context.config.get("models.lineage_acknowledged", False)
        ),
        tokenizer_path=sources.tokenizer,
    )
    compatibility_path = context.path("manifests", "compatibility.json")
    write_compatibility_report(compatibility_path, report)
    write_csv(
        context.path("manifests", "parameter_delta_statistics.csv"),
        report.get("parameter_delta_statistics") or [],
    )
    require_compatible(report)

    split_report = audit_splits(
        Path(context.config.get("datasets.dev")),
        Path(context.config.get("datasets.test"))
        if context.config.get("datasets.test")
        else None,
        allow_id_overlap=bool(context.config.get("datasets.allow_id_overlap", False)),
        allow_prompt_overlap=bool(
            context.config.get("datasets.allow_prompt_overlap", False)
        ),
    )
    split_path = context.path("manifests", "dataset_split_audit.json")
    write_split_audit(split_path, split_report)
    require_split_audit(split_report)

    backend = str(context.config.get("evaluation.backend"))
    if backend == "official_phase_v45":
        anchor_tokenizer_files = (report.get("anchor_tokenizer") or {}).get("files") or []
        greedy_tokenizer_files = (report.get("greedy_tokenizer") or {}).get("files") or []
        if not anchor_tokenizer_files or not greedy_tokenizer_files:
            raise ValueError(
                "official_phase_v45 loads the tokenizer from each model directory. "
                "Both resolved endpoints must therefore contain tokenizer files; use "
                "bundled_v45 when relying on a separate models.tokenizer_path."
            )
    official_root: Path | None = None
    if backend == "official_phase_v45":
        official_root = Path(context.config.get("evaluation.official_repo_root"))
        quality_path = official_root / "configs" / "quality_evaluation.json"
    else:
        quality_path = Path(context.config.get("evaluation.quality_config"))
    if not quality_path.exists():
        raise FileNotFoundError(f"Quality Evaluator configuration does not exist: {quality_path}")
    if backend == "official_phase_v45":
        quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
        if str(quality_payload.get("schema_version")) != QUALITY_EVALUATION_VERSION:
            raise ValueError(
                f"Official quality evaluator is not v{QUALITY_EVALUATION_VERSION}: "
                f"{quality_payload.get('schema_version')!r}"
            )
        taxonomy_value = Path(
            str(
                quality_payload.get(
                    "issue_taxonomy_path",
                    "configs/quality_issue_taxonomy_v4_5.json",
                )
            )
        )
        taxonomy_path = (
            taxonomy_value
            if taxonomy_value.is_absolute()
            else (official_root / taxonomy_value).resolve()
        )
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Official v4.5 issue taxonomy is missing: {taxonomy_path}")
        taxonomy_payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        if str(taxonomy_payload.get("schema_version")) != QUALITY_EVALUATION_VERSION:
            raise ValueError("Official issue taxonomy is not v4.5.")
        quality_config = None
    else:
        quality_config = load_quality_config(quality_path)
        # Resolve and validate the taxonomy now rather than failing after candidates have
        # already been materialized and evaluated.
        _ = quality_config.issue_taxonomy
        taxonomy_path = quality_config.issue_taxonomy_path
    if backend == "official_phase_v45":
        assert official_root is not None
        expected_script = official_root / "scripts" / "phase_1_dense_sft" / "evaluate_model.py"
        expected_quality = official_root / "configs" / "quality_evaluation.json"
        if not expected_script.exists() or not expected_quality.exists():
            raise FileNotFoundError(
                "official_phase_v45 is configured, but its evaluate_model.py or "
                f"quality_evaluation.json is missing under {official_root}."
            )
    write_json(
        context.path("manifests", "preflight.json"),
        {
            "schema_version": "1.0",
            "compatible": True,
            "source_resolution": sources.report,
            "compatibility_report": str(compatibility_path),
            "dataset_split_audit": str(split_path),
            "quality_config": str(quality_path),
            "quality_config_sha256": sha256_file(quality_path),
            "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
            "quality_taxonomy": str(taxonomy_path),
            "quality_taxonomy_sha256": sha256_file(taxonomy_path),
            "evaluation_backend": context.config.get("evaluation.backend"),
            "selection_protocol": context.config.get("evaluation.selection_protocol"),
            "source_mutation_policy": "read-only",
        },
    )
    return sources
