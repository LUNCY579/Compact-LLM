from __future__ import annotations

"""Migrate stored Phase 2B generations to Evaluator v3 without inference.

Non-destructive mode writes sibling ``generation_quality_v3`` directories for
inspection. ``--in-place --prepare-resume`` backs up legacy generation evidence,
rescoring in place, rebuilds every output-deviation comparison against the
rescored full reference, and archives derived catalogs so the original Phase 2B
command can be rerun with ``-Resume``. The progressive script then reuses model,
NLL, footprint, and generation evidence, recomputes opportunity assessments, and
continues from the newly selected frontier parent.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.quality_evaluation import QUALITY_EVALUATION_VERSION
from scripts.common.rescore_quality_artifacts import rescore

COMPARE = ROOT / "scripts/phase_2_layer_selection/compare_outputs.py"


def backup_file(path: Path, suffix: str) -> Path:
    backup = path.with_name(path.name + suffix)
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")
    path.replace(backup)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score stored Phase 2B generations without model inference.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--quality-config", type=Path, default=ROOT / "configs/quality_evaluation.json")
    parser.add_argument("--in-place", action="store_true", help="Replace legacy generation evidence after timestamped backups.")
    parser.add_argument(
        "--prepare-resume",
        action="store_true",
        help="Rebuild deviations and archive derived catalogs so the same Phase 2B command can continue with -Resume.",
    )
    args = parser.parse_args()
    if args.prepare_resume and not args.in_place:
        raise SystemExit("--prepare-resume requires --in-place so the progressive runner sees v3 summaries.")
    run_root = args.run_root.resolve()
    sources = sorted(path for path in run_root.rglob("results.jsonl") if path.parent.name == "generation")
    if not sources:
        raise SystemExit(f"No generation/results.jsonl files found under {run_root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_suffix = f".pre_quality_v3_{stamp}"
    rows: list[dict[str, str]] = []
    for source in sources:
        if args.in_place:
            summary = source.parent / "summary.json"
            backup_results = backup_file(source, backup_suffix)
            backup_summary = backup_file(summary, backup_suffix) if summary.exists() else None
            result_out, summary_out = rescore(args.benchmark, backup_results, source.parent, args.quality_config)
        else:
            backup_results = source
            backup_summary = None
            output = source.parent.with_name("generation_quality_v3")
            result_out, summary_out = rescore(args.benchmark, source, output, args.quality_config)
        rows.append({
            "source": str(source),
            "source_backup": str(backup_results) if args.in_place else "",
            "summary_backup": str(backup_summary) if backup_summary else "",
            "results_v3": str(result_out),
            "summary_v3": str(summary_out),
        })
        print(f"rescored: {source}")

    rebuilt_deviations: list[str] = []
    archived_derived: list[str] = []
    if args.prepare_resume:
        reference_results = run_root / "reference/evidence/generation/results.jsonl"
        if not reference_results.exists():
            raise FileNotFoundError(f"Missing rescored Phase 2B reference results: {reference_results}")
        candidate_results = sorted(
            path for path in run_root.glob("evidence/stage_*/**/generation/results.jsonl")
            if path.is_file()
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

        # These files contain decisions derived from the superseded evaluator.
        # Archive rather than delete them. ``progressive_reduction.py -Resume``
        # will regenerate them from preserved raw evidence.
        for name in ("model_catalog.json", "candidate_registry.json"):
            path = run_root / name
            if path.exists():
                archived = path.with_name(path.name + backup_suffix)
                shutil.move(str(path), str(archived))
                archived_derived.append(str(archived))

    manifest = {
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "run_root": str(run_root),
        "benchmark": str(args.benchmark.resolve()),
        "in_place": args.in_place,
        "prepare_resume": args.prepare_resume,
        "backup_suffix": backup_suffix if args.in_place else None,
        "rescored_count": len(rows),
        "artifacts": rows,
        "rebuilt_output_deviations": rebuilt_deviations,
        "archived_derived_decisions": archived_derived,
        "next_action": (
            "Rerun the exact original Phase 2B command with -Resume. It will reuse model checkpoints, "
            "generation, NLL, and footprint evidence; recompute combined metrics under relative opportunity policy; "
            "select the corrected Stage 1 parent; and continue to later stages."
            if args.prepare_resume else
            "Review the sibling v3 summaries. Use --in-place --prepare-resume when ready to continue the official run."
        ),
    }
    out = run_root / "PHASE2B_QUALITY_V3_MIGRATION.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
