#!/usr/bin/env python3
"""Anchor a completed run to exact code/config and artifact byte hashes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.integrity import (  # noqa: E402
    artifact_hash_map,
    content_tree_sha256,
    hash_tree,
)
from rineya_significance_lab.manifest import atomic_write_json  # noqa: E402


def finalize(
    run_dir: Path,
    *,
    runner: Path,
    config: Path,
    extra_code: list[Path] | None = None,
) -> dict[str, object]:
    run = run_dir.resolve(strict=True)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_dir = LAB_ROOT / "src" / "rineya_significance_lab"
    # Anchor the complete local package as portable source, never
    # interpreter-specific cache artifacts.  Selecting a directory directly
    # would also capture ``__pycache__/*.pyc`` and make a recorded run
    # unverifiable from a clean checkout.
    package_sources = sorted(
        path for path in package_dir.rglob("*") if path.is_file() and path.suffix in {".py", ".pyi"}
    )
    selected = [
        *package_sources,
        runner.resolve(strict=True),
        config.resolve(strict=True),
        LAB_ROOT / "pyproject.toml",
    ]
    selected.extend(path.resolve(strict=True) for path in (extra_code or []))
    code_files = hash_tree(REPO_ROOT, selected)
    artifacts = artifact_hash_map(run)
    # The run manifest refers to the integrity root and is therefore excluded
    # from that root alongside the self-referential content manifest.
    artifacts.pop("manifest.json", None)
    content = {
        "schema_version": "1.0",
        "code_files": code_files,
        "code_tree_sha256": content_tree_sha256(code_files),
        "artifact_files": artifacts,
        "artifact_tree_sha256": content_tree_sha256(artifacts),
        "excluded": ["manifest.json", "content_manifest.json"],
    }
    atomic_write_json(run / "content_manifest.json", content)
    manifest["code_integrity"] = {
        "algorithm": "sha256-canonical-path-map-v1",
        "tree_sha256": content["code_tree_sha256"],
        "file_count": len(code_files),
    }
    manifest["artifact_integrity"] = {
        "algorithm": "sha256-canonical-path-map-v1",
        "tree_sha256": content["artifact_tree_sha256"],
        "file_count": len(artifacts),
        "content_manifest": "content_manifest.json",
    }
    manifest.setdefault("artifacts", {})["content_manifest"] = "content_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--extra-code", type=Path, action="append", default=[])
    args = parser.parse_args()
    content = finalize(
        args.run,
        runner=args.runner,
        config=args.config,
        extra_code=args.extra_code,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "code_tree_sha256": content["code_tree_sha256"],
                "artifact_tree_sha256": content["artifact_tree_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
