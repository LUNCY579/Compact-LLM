from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.architecture import (  # noqa: E402
    ArchitectureAuditError,
    compatibility_report,
    inspect_checkpoint,
    render_compatibility_markdown,
)
from rineya_cross_scale.hashing import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


def parse_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"expected NAME=PATH, received {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise argparse.ArgumentTypeError(f"expected NAME=PATH, received {value!r}")
        result[name] = Path(raw_path)
    return result


def guard_path(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    forbidden = (
        "rineya_invisible_limit_lab_v1/data/test2_answer_key.jsonl",
        "rineya_invisible_limit_lab_v1/data/test2",
        "rineya_invisible_limit_lab_v1/outputs/stage6",
    )
    if any(fragment in normalized for fragment in forbidden):
        raise ArchitectureAuditError(f"leakage firewall rejected path: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=LAB_ROOT / "configs" / "architecture_audit.json"
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override a model path (repeatable)",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override a release-manifest path (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LAB_ROOT / "cross_scale_evidence" / "02_architecture",
    )
    parser.add_argument(
        "--skip-weight-hashes",
        action="store_true",
        help="development-only shortcut; full evidence runs must omit this flag",
    )
    args = parser.parse_args()

    guard_path(args.spec)
    with args.spec.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    overrides = parse_paths(args.path)
    manifest_overrides = parse_paths(args.manifest)
    expected_names = {model["name"] for model in spec["models"]}
    unknown = set(overrides) - expected_names
    if unknown:
        raise ArchitectureAuditError(f"unknown model path overrides: {sorted(unknown)}")
    unknown_manifests = set(manifest_overrides) - expected_names
    if unknown_manifests:
        raise ArchitectureAuditError(
            f"unknown manifest overrides: {sorted(unknown_manifests)}"
        )

    audits = {}
    for model in spec["models"]:
        name = model["name"]
        raw_path = overrides.get(name)
        if raw_path is None:
            env_value = os.environ.get(model["path_env"])
            if env_value:
                raw_path = Path(env_value)
        if raw_path is None:
            raise ArchitectureAuditError(
                f"no path for {name}; pass --path {name}=PATH or set {model['path_env']}"
            )
        guard_path(raw_path)
        manifest_path = manifest_overrides.get(name)
        if manifest_path is not None:
            guard_path(manifest_path)
        audits[name] = inspect_checkpoint(
            name=name,
            role=model["role"],
            source_id=model["source_id"],
            revision=model["revision"],
            model_path=raw_path,
            hash_weights=not args.skip_weight_hashes,
            release_manifest_path=manifest_path,
        )

    compatibility = compatibility_report(audits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, audit in audits.items():
        atomic_write_json(args.output_dir / f"{name}.json", audit)
    depth_maps = {
        name: details["depth_mappings"]
        for name, details in compatibility["donors"].items()
    }
    atomic_write_json(args.output_dir / "depth_mappings.json", depth_maps)
    atomic_write_json(args.output_dir / "compatibility.json", compatibility)
    atomic_write_text(
        args.output_dir / "compatibility_report.md",
        render_compatibility_markdown(audits, compatibility),
    )

    evidence_files = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "audit_manifest.json"
    )
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": str(args.spec.resolve()),
        "spec_sha256": sha256_file(args.spec),
        "code_files": {
            str(path.relative_to(LAB_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (
                Path(__file__),
                LAB_ROOT / "src" / "rineya_cross_scale" / "architecture.py",
                LAB_ROOT / "src" / "rineya_cross_scale" / "checkpoint.py",
                LAB_ROOT / "src" / "rineya_cross_scale" / "hashing.py",
            )
        },
        "weight_hashes_included": not args.skip_weight_hashes,
        "projection_gate": compatibility["projection_gate"],
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in evidence_files
        },
    }
    atomic_write_json(args.output_dir / "audit_manifest.json", manifest)
    print(json.dumps({"projection_gate": compatibility["projection_gate"], "output": str(args.output_dir)}, indent=2))
    return 0 if compatibility["projection_gate"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
