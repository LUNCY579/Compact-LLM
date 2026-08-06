from __future__ import annotations

"""Package the human-approved Phase 11 production RAG release."""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.check_human_gate import has_human_pass
from scripts.common.experiment_utils import read_json, sha256, sha256_tree, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the approved Phase 11 release candidate.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "artifacts" / "phase_11_production_rag" / "release_candidate_registry.json",
    )
    parser.add_argument("--gate", type=Path, default=ROOT / "reports" / "PHASE_11_STOP_GATE.md")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--phase10-reference", type=Path, default=ROOT / "artifacts" / "references" / "phase_10_student_reference.json")
    parser.add_argument("--mode", choices=("manifest", "copy"), default="manifest")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "phase_11_release")
    parser.add_argument("--zip", type=Path, default=ROOT / "artifacts" / "phase_11_release.zip")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hash_entry(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return {"path": str(path), "type": "directory", "tree_sha256": sha256_tree(path)}
    return {"path": str(path), "type": "file", "sha256": sha256(path), "bytes": path.stat().st_size}


def copy_artifact(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if not has_human_pass(args.gate):
        raise SystemExit(f"Phase 11 gate lacks explicit human PASS: {args.gate}")
    registry = read_json(args.registry)
    candidate = (registry.get("candidates") or {}).get(args.candidate)
    if not candidate:
        raise ValueError(f"Unknown Phase 11 candidate: {args.candidate}")
    runtime_summary = Path(str(candidate.get("summary") or ""))
    if not runtime_summary.is_absolute():
        runtime_summary = ROOT / runtime_summary
    if not runtime_summary.exists():
        raise FileNotFoundError(runtime_summary)
    artifact = Path(str(candidate.get("model") or ""))
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    snapshot_manifest = Path(str(registry.get("snapshot_manifest") or ""))
    if not snapshot_manifest.is_absolute():
        snapshot_manifest = ROOT / snapshot_manifest
    calibration = Path(str(registry.get("answerability_calibration") or ""))
    if not calibration.is_absolute():
        calibration = ROOT / calibration
    for path in (snapshot_manifest, calibration, args.phase10_reference):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Release directory exists; use --overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    if args.zip.exists():
        if not args.overwrite:
            raise FileExistsError(f"Release ZIP exists; use --overwrite: {args.zip}")
        args.zip.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_snapshot_dir = snapshot_manifest.parent
    copied: dict[str, str] = {}
    if args.mode == "copy":
        model_destination = args.output_dir / "model"
        copy_artifact(artifact, model_destination)
        snapshot_destination = args.output_dir / "knowledge_snapshot"
        copy_artifact(source_snapshot_dir, snapshot_destination)
        copied = {
            "model": str(model_destination),
            "knowledge_snapshot": str(snapshot_destination),
        }

    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (args.registry, "release_candidate_registry.json"),
        (args.gate, "PHASE_11_STOP_GATE.md"),
        (runtime_summary, "runtime_summary.json"),
        (calibration, "answerability_calibration.json"),
        (snapshot_manifest, "snapshot_manifest.json"),
        (args.phase10_reference, "phase_10_student_reference.json"),
    ):
        shutil.copy2(source, evidence_dir / name)
    runtime_report = runtime_summary.parent / "REPORT.md"
    if runtime_report.exists():
        shutil.copy2(runtime_report, evidence_dir / "runtime_report.md")

    release_manifest = {
        "schema_version": "1.0",
        "phase": "11_production_rag",
        "candidate": args.candidate,
        "package_mode": args.mode,
        "artifact": hash_entry(artifact),
        "runtime_backend": candidate.get("runtime_backend"),
        "tokenizer_source": candidate.get("tokenizer_source"),
        "runtime_summary": hash_entry(runtime_summary),
        "knowledge_snapshot": hash_entry(source_snapshot_dir),
        "answerability_calibration": hash_entry(calibration),
        "phase10_reference": hash_entry(args.phase10_reference),
        "source_registry": hash_entry(args.registry),
        "source_gate": hash_entry(args.gate),
        "copied_artifacts": copied,
        "rollback": read_json(args.phase10_reference).get("rollback"),
        "installation_boundary": (
            "Manifest mode contains provenance/evidence only. Copy mode additionally embeds the selected model artifact "
            "and immutable knowledge snapshot. Application binaries and private user-memory databases remain separate."
        ),
    }
    write_json(args.output_dir / "release_manifest.json", release_manifest)
    (args.output_dir / "README.md").write_text(
        "# Phase 11 release package\n\n"
        f"Candidate: `{args.candidate}`\n\n"
        f"Package mode: `{args.mode}`\n\n"
        "Validate `release_manifest.json`, load the exact model/runtime backend, open the immutable knowledge snapshot, "
        "and preserve the Phase 10 rollback checkpoint before distribution.\n",
        encoding="utf-8",
    )

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.output_dir.parent))
    with zipfile.ZipFile(args.zip, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Release ZIP integrity failed at {bad}")
    print(args.output_dir / "release_manifest.json")
    print(args.zip)


if __name__ == "__main__":
    main()
