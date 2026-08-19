from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections.abc import Iterator
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".tox",
    ".nox",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
}
EXCLUDED_FILE_NAMES = {
    ".coverage",
    "coverage.xml",
    ".env",
    ".DS_Store",
}
MANIFEST_PATH = "artifacts/release/repository_manifest.json"


def _is_sensitive_local_file(relative: str) -> bool:
    lowered = relative.lower()
    name = Path(relative).name.lower()
    if name in {"secrets.yaml", "secrets.yml", "secrets.json"}:
        return True
    if name.endswith(".local.yaml") or name.endswith(".local.yml"):
        return True
    if lowered == "configs/local-rineya.yaml":
        return True
    return False


def files_for(
    root: Path,
    *,
    for_manifest: bool = False,
    excluded_paths: set[Path] | None = None,
) -> Iterator[tuple[Path, str]]:
    """Yield deterministic, release-safe repository files.

    ``for_manifest`` excludes the manifest itself to avoid self-referential hashing. Release
    artifacts such as the validation report remain included and therefore become integrity-bound.
    ``excluded_paths`` is used by the packager to prevent an output ZIP from packaging itself.
    """

    root = root.resolve()
    excluded = {path.resolve() for path in (excluded_paths or set())}
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in excluded:
            continue
        relative_path = path.relative_to(root)
        relative = relative_path.as_posix()
        if any(part in EXCLUDED_DIRS for part in relative_path.parts):
            continue
        if path.name in EXCLUDED_FILE_NAMES:
            continue
        if for_manifest and relative == MANIFEST_PATH:
            continue
        if relative.startswith("artifacts/runtime/") and not relative.endswith(".gitkeep"):
            continue
        if path.suffix.lower() == ".zip" or path.name.lower().endswith(".zip.sha256"):
            continue
        if _is_sensitive_local_file(relative):
            continue
        yield path, relative


def project_version(root: Path) -> str:
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def build_manifest(root: Path) -> dict[str, object]:
    records = []
    for path, relative in files_for(root, for_manifest=True):
        content = path.read_bytes()
        records.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    aggregate = hashlib.sha256(
        "\n".join(f"{item['sha256']}  {item['path']}" for item in records).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.0",
        "project": "rineya-context-harness",
        "version": project_version(root),
        "file_count": len(records),
        "aggregate_sha256": aggregate,
        "files": records,
        "notes": [
            "The manifest excludes itself to avoid self-referential hashing.",
            "Runtime databases, caches, local secrets, and prior ZIP artifacts are excluded.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path(MANIFEST_PATH)
    )
    args = parser.parse_args()
    root = args.root.resolve()
    payload = build_manifest(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: payload[key] for key in ("file_count", "aggregate_sha256")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
