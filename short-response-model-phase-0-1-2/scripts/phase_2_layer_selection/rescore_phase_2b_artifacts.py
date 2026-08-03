from __future__ import annotations

"""Re-score stored Phase 2B generations under the current evaluator.

The migration performs no model inference.  In ``--in-place --prepare-resume``
mode it:

1. backs up every existing generation result and summary;
2. re-scores the stored replies under Evaluator v4.5;
3. rebuilds full-reference output deviations;
4. archives stale parent deviations, combined metrics, catalogs, registries, and
   stage decisions;
5. writes lineage provenance for each existing candidate; and
6. prepares the original Phase 2B command to be rerun with ``-Resume``.

The progressive runner then replays decisions from Stage 1 using the preserved
raw evidence.  Generations are reused while the corrected frontier follows the
same parent lineage.  If a corrected stage chooses a different parent, evidence
from the old downstream branch is automatically invalidated and only work from
that divergence point is regenerated.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.quality_evaluation import QUALITY_EVALUATION_VERSION
from scripts.common.rescore_quality_artifacts import rescore

COMPARE = ROOT / "scripts/phase_2_layer_selection/compare_outputs.py"
OPPORTUNITY_CONFIG = ROOT / "configs/relative_opportunity_evaluation.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))



def validate_explicit_contract_benchmark(path: Path) -> dict[str, int]:
    """Require a complete Evaluator-v4.5 contract on every Phase 2B case."""

    case_count = 0
    missing: list[str] = []
    invalid: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case_count += 1
            row = json.loads(line)
            case_id = str(row.get("id", f"line-{line_number}"))
            contract = row.get("response_contract")
            if not isinstance(contract, dict):
                missing.append(case_id)
            elif (
                str(contract.get("schema_version")) != QUALITY_EVALUATION_VERSION
                or contract.get("contract_origin") != "freshly_authored_from_visible_messages_v4_5"
                or contract.get("contract_category_used") is not False
                or contract.get("contract_reference_used") is not False
                or contract.get("contract_inherited_requirements_used") is not False
            ):
                invalid.append(case_id)
    if case_count == 0:
        raise ValueError(f"Benchmark contains no cases: {path}")
    if missing or invalid:
        parts = []
        if missing:
            parts.append(f"missing response_contract: {missing[:8]}")
        if invalid:
            parts.append(f"wrong schema or contract provenance: {invalid[:8]}")
        raise ValueError(
            f"Official Phase 2B rescoring requires an explicit Evaluator-v4.5 benchmark ({'; '.join(parts)}). "
            "Use probe_150_thorough_v4_5.jsonl or author a contracted benchmark first."
        )
    return {"case_count": case_count, "contracted_case_count": case_count}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normal_path(value: str | Path) -> str:
    raw = str(value).strip().replace("\\", "/")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve(strict=False).as_posix().rstrip("/").casefold()


def repo_path(value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    return candidate if candidate.is_absolute() else ROOT / candidate


def backup_file(path: Path, suffix: str) -> Path:
    backup = path.with_name(path.name + suffix)
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")
    path.replace(backup)
    return backup


def archive_if_exists(path: Path, suffix: str) -> str | None:
    if not path.exists():
        return None
    archived = path.with_name(path.name + suffix)
    if archived.exists():
        raise FileExistsError(archived)
    shutil.move(str(path), str(archived))
    return str(archived)


def write_candidate_provenance(
    *,
    rows: list[dict[str, Any]],
    benchmark: Path,
    quality_config: Path,
    run_configuration: dict[str, Any],
) -> list[str]:
    written: list[str] = []
    probe_sha = sha256(benchmark)
    quality_sha = sha256(quality_config)
    opportunity_sha = sha256(OPPORTUNITY_CONFIG)
    max_new_tokens = int(run_configuration.get("max_new_tokens", 32))
    recovery_steps = int(run_configuration.get("recovery_steps", 50))
    recovery_seed = int(run_configuration.get("recovery_seed", 11))
    qlora = bool(run_configuration.get("qlora", True))
    gradient_checkpointing = bool(run_configuration.get("gradient_checkpointing", True))

    for row in rows:
        evidence_value = row.get("evidence_dir")
        if not evidence_value:
            continue
        evidence_dir = repo_path(str(evidence_value))
        if not evidence_dir.exists():
            continue
        variant_kind = str(row.get("variant_kind", "raw deletion"))
        recovered = "recovery" in variant_kind.lower()
        payload = {
            "schema_version": 1,
            "stage": int(row.get("stage", 0)),
            "variant_id": str(row.get("variant_id", evidence_dir.name)),
            "variant_kind": variant_kind,
            "parent_model": normal_path(str(row.get("parent_model", ""))),
            "dropped_original_layer_id": int(row.get("dropped_original_layer_id", -1)),
            "retained_original_layer_ids": [int(value) for value in row.get("retained_original_layer_ids", [])],
            "probe_file": normal_path(benchmark),
            "probe_file_sha256": probe_sha,
            "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
            "quality_config_sha256": quality_sha,
            "opportunity_policy_sha256": opportunity_sha,
            "max_new_tokens": max_new_tokens,
            "recovery_steps": recovery_steps if recovered else 0,
            "recovery_seed": recovery_seed if recovered else None,
            "qlora": qlora if recovered else False,
            "gradient_checkpointing": gradient_checkpointing if recovered else False,
        }
        path = evidence_dir / "evidence_provenance.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score stored Phase 2B generations without model inference.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--quality-config", type=Path, default=ROOT / "configs/quality_evaluation.json")
    parser.add_argument("--in-place", action="store_true", help="Replace prior generation evidence after timestamped backups.")
    parser.add_argument(
        "--prepare-resume",
        action="store_true",
        help="Archive derived decisions and prepare the same Phase 2B command to continue with -Resume.",
    )
    args = parser.parse_args()
    if args.prepare_resume and not args.in_place:
        raise SystemExit("--prepare-resume requires --in-place.")

    run_root = args.run_root.resolve()
    benchmark = args.benchmark.resolve()
    quality_config = args.quality_config.resolve()
    if not run_root.exists():
        raise FileNotFoundError(run_root)
    if not benchmark.exists():
        raise FileNotFoundError(benchmark)
    benchmark_contract_audit = validate_explicit_contract_benchmark(benchmark)

    catalog_path = run_root / "model_catalog.json"
    registry_path = run_root / "candidate_registry.json"
    old_catalog = read_json(catalog_path) if catalog_path.exists() else {}
    old_registry_payload = read_json(registry_path) if registry_path.exists() else {}
    old_registry = list(old_registry_payload.get("candidates", [])) if isinstance(old_registry_payload, dict) else []
    run_configuration = dict(old_catalog.get("run_configuration") or {})

    sources = sorted(path for path in run_root.rglob("results.jsonl") if path.parent.name == "generation")
    if not sources:
        raise SystemExit(f"No generation/results.jsonl files found under {run_root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tag = QUALITY_EVALUATION_VERSION.replace(".", "_")
    backup_suffix = f".pre_quality_v{version_tag}_{stamp}"
    rows: list[dict[str, str]] = []

    for source in sources:
        if args.in_place:
            summary = source.parent / "summary.json"
            backup_results = backup_file(source, backup_suffix)
            backup_summary = backup_file(summary, backup_suffix) if summary.exists() else None
            result_out, summary_out = rescore(benchmark, backup_results, source.parent, quality_config)
        else:
            backup_results = source
            backup_summary = None
            output = source.parent.with_name(f"generation_quality_v{version_tag}")
            result_out, summary_out = rescore(benchmark, source, output, quality_config)
        rows.append({
            "source": str(source),
            "source_backup": str(backup_results) if args.in_place else "",
            "summary_backup": str(backup_summary) if backup_summary else "",
            "rescored_results": str(result_out),
            "rescored_summary": str(summary_out),
        })
        print(f"rescored: {source}")

    rebuilt_deviations: list[str] = []
    provenance_files: list[str] = []
    archived_derived: list[str] = []

    if args.prepare_resume:
        provenance_files = write_candidate_provenance(
            rows=old_registry,
            benchmark=benchmark,
            quality_config=quality_config,
            run_configuration=run_configuration,
        )

        reference_results = run_root / "reference/evidence/generation/results.jsonl"
        if not reference_results.exists():
            raise FileNotFoundError(f"Missing rescored Phase 2B reference results: {reference_results}")
        candidate_results = sorted(
            path for path in run_root.glob("evidence/stage_*/**/generation/results.jsonl") if path.is_file()
        )
        for candidate in candidate_results:
            evidence_dir = candidate.parent.parent
            out_json = evidence_dir / "output_deviation.json"
            out_report = evidence_dir / "OUTPUT_DEVIATION.md"
            subprocess.run(
                [
                    sys.executable, str(COMPARE),
                    "--reference", str(reference_results),
                    "--candidate", str(candidate),
                    "--out-json", str(out_json),
                    "--out-report", str(out_report),
                ],
                check=True,
                cwd=ROOT,
            )
            rebuilt_deviations.append(str(out_json))

            # Parent-relative decisions depend on the corrected frontier.  They
            # are cheap to rebuild, so never preserve the old parent comparison.
            for name in (
                "parent_output_deviation.json", "PARENT_OUTPUT_DEVIATION.md", "combined_metrics.json"
            ):
                archived = archive_if_exists(evidence_dir / name, backup_suffix)
                if archived:
                    archived_derived.append(archived)

        for path in run_root.glob("frontier/stage_*/stage_result.json"):
            archived = archive_if_exists(path, backup_suffix)
            if archived:
                archived_derived.append(archived)
        for path in (catalog_path, registry_path):
            archived = archive_if_exists(path, backup_suffix)
            if archived:
                archived_derived.append(archived)

    manifest = {
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "run_root": str(run_root),
        "benchmark": str(benchmark),
        "benchmark_contract_audit": benchmark_contract_audit,
        "quality_config": str(quality_config),
        "in_place": args.in_place,
        "prepare_resume": args.prepare_resume,
        "backup_suffix": backup_suffix if args.in_place else None,
        "rescored_count": len(rows),
        "artifacts": rows,
        "candidate_provenance_files": provenance_files,
        "rebuilt_output_deviations": rebuilt_deviations,
        "archived_derived_decisions": archived_derived,
        "reuse_boundary": (
            "Stored generation/NLL/footprint evidence is reused while the corrected frontier follows the same parent lineage. "
            "At the first changed parent, stale downstream evidence is invalidated and regenerated from that divergence only."
        ),
        "next_action": (
            "Rerun the exact original Phase 2B command with -Resume. The runner will reconstruct decisions from Stage 1 "
            "without re-generating matching stored replies and will continue from the corrected frontier."
            if args.prepare_resume else
            "Review sibling rescored summaries. Use --in-place --prepare-resume for the official migration."
        ),
    }
    out = run_root / f"PHASE2B_QUALITY_V{version_tag}_MIGRATION.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
