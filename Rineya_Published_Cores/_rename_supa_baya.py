#!/usr/bin/env python3
"""Transactionally rename the published Taria/Lumina pair to Supa/Baya."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VAULT = Path(__file__).resolve().parent
CATALOG = VAULT / "CATALOG.json"


@dataclass(frozen=True)
class RenameSpec:
    old_codename: str
    new_codename: str
    old_directory: str
    new_directory: str
    old_core_id: str
    new_core_id: str
    weight_sha256: str

    @property
    def old_path(self) -> Path:
        return VAULT / self.old_directory

    @property
    def new_path(self) -> Path:
        return VAULT / self.new_directory


SPECS = (
    RenameSpec(
        old_codename="Taria",
        new_codename="Supa",
        old_directory="Rineya-Taria-1.7.0-devfreeze",
        new_directory="Rineya-Supa-1.7.0-devfreeze",
        old_core_id=(
            "rineya.core.taria.v1.7.0-devfreeze.28L."
            "qwen3-nimbus-lcr-stage-a-step200-scale-0p8125"
        ),
        new_core_id=(
            "rineya.core.supa.v1.7.0-devfreeze.28L."
            "qwen3-nimbus-lcr-stage-a-step200-scale-0p8125"
        ),
        weight_sha256="962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
    ),
    RenameSpec(
        old_codename="Lumina",
        new_codename="Baya",
        old_directory="Rineya-Lumina-1.7.0-devfreeze",
        new_directory="Rineya-Baya-1.7.0-devfreeze",
        old_core_id=(
            "rineya.core.lumina.v1.7.0-devfreeze.28L."
            "qwen3-nimbus-lcr-stage-a-step200-scale-1p25"
        ),
        new_core_id=(
            "rineya.core.baya.v1.7.0-devfreeze.28L."
            "qwen3-nimbus-lcr-stage-a-step200-scale-1p25"
        ),
        weight_sha256="1737cd1ee2bdde714c259adf2012ae44c89ad82a06797653343bad7e3cf88bba",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def transform_core(
    core: dict[str, Any], spec: RenameSpec, sibling: RenameSpec, renamed_at: str
) -> dict[str, Any]:
    if core.get("core_id") != spec.old_core_id:
        raise RuntimeError(f"Unexpected source core id for {spec.old_codename}")
    if core.get("codename") != spec.old_codename:
        raise RuntimeError(f"Unexpected source codename for {spec.old_codename}")
    if core.get("directory_name") != spec.old_directory:
        raise RuntimeError(f"Unexpected source directory for {spec.old_codename}")
    result = copy.deepcopy(core)
    result["core_id"] = spec.new_core_id
    result["codename"] = spec.new_codename
    result["directory_name"] = spec.new_directory
    result["former_identity"] = {
        "codename": spec.old_codename,
        "core_id": spec.old_core_id,
        "directory": spec.old_directory,
        "renamed_at_utc": renamed_at,
        "reason": "User-requested public model rename; weights and frozen evidence unchanged.",
    }
    pair = dict(result.get("pair") or {})
    pair["sibling_core_id"] = sibling.new_core_id
    pair["sibling_directory"] = sibling.new_directory
    result["pair"] = pair
    claim = str(result.get("public_claim") or "")
    result["public_claim"] = re.sub(
        rf"\b{re.escape(spec.old_codename)}\b", spec.new_codename, claim
    )
    result["note"] = (
        f"Current public identity is {spec.new_codename}; frozen merge/parity evidence "
        f"retains the former {spec.old_codename} identity for auditability."
    )
    return result


def transform_catalog(
    catalog: dict[str, Any], cores: dict[str, dict[str, Any]], renamed_at: str
) -> dict[str, Any]:
    result = copy.deepcopy(catalog)
    entries = result.get("cores")
    if not isinstance(entries, list):
        raise TypeError("CATALOG.json cores must be a list")
    for spec in SPECS:
        matches = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and (
                entry.get("core_id") == spec.old_core_id
                or entry.get("directory") == spec.old_directory
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one catalog entry for {spec.old_codename}, found {len(matches)}"
            )
        entry = matches[0]
        entry["core_id"] = spec.new_core_id
        entry["codename"] = spec.new_codename
        entry["directory"] = spec.new_directory
        entry["present"] = True
        entry["former_identity"] = {
            "codename": spec.old_codename,
            "core_id": spec.old_core_id,
            "directory": spec.old_directory,
        }
        entry["core"] = cores[spec.new_codename]
    ids = [entry.get("core_id") for entry in entries if isinstance(entry, dict)]
    directories = [
        entry.get("directory") for entry in entries if isinstance(entry, dict)
    ]
    if len(ids) != len(set(ids)) or len(directories) != len(set(directories)):
        raise RuntimeError("Rename would create duplicate catalog identities")
    result["updated_at_utc"] = renamed_at
    result["last_written"] = [spec.new_directory for spec in SPECS]
    return result


def snapshot_immutable(path: Path) -> dict[str, str]:
    names = (
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "LICENSE",
        ".gitattributes",
        "PARITY.v2.json",
        "merge_manifest.json",
    )
    return {name: sha256_file(path / name) for name in names}


def preflight() -> dict[str, dict[str, str]]:
    snapshots: dict[str, dict[str, str]] = {}
    catalog = load_json(CATALOG)
    for spec in SPECS:
        if not spec.old_path.is_dir():
            raise FileNotFoundError(spec.old_path)
        if spec.new_path.exists():
            raise FileExistsError(spec.new_path)
        if sha256_file(spec.old_path / "model.safetensors") != spec.weight_sha256:
            raise RuntimeError(f"Weight hash mismatch for {spec.old_codename}")
        core = load_json(spec.old_path / "CORE.json")
        if core.get("core_id") != spec.old_core_id:
            raise RuntimeError(f"CORE identity mismatch for {spec.old_codename}")
        snapshots[spec.old_codename] = snapshot_immutable(spec.old_path)
        if not any(
            isinstance(entry, dict) and entry.get("core_id") == spec.old_core_id
            for entry in catalog.get("cores", [])
        ):
            raise RuntimeError(f"Catalog is missing {spec.old_codename}")
    return snapshots


def rename_manifest(
    spec: RenameSpec, snapshots: dict[str, str], renamed_at: str
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "operation": "metadata_only_public_identity_rename",
        "renamed_at_utc": renamed_at,
        "old": {
            "codename": spec.old_codename,
            "core_id": spec.old_core_id,
            "directory": spec.old_directory,
        },
        "new": {
            "codename": spec.new_codename,
            "core_id": spec.new_core_id,
            "directory": spec.new_directory,
        },
        "weight_sha256": spec.weight_sha256,
        "immutable_file_sha256": snapshots,
        "evidence_policy": (
            "PARITY.v2.json and merge_manifest.json remain byte-identical and use "
            "the former identity because they describe the evaluated publication event."
        ),
    }


def render_readme(original: str, spec: RenameSpec) -> str:
    text = original
    for rename in SPECS:
        text = re.sub(
            rf"\b{re.escape(rename.old_codename)}\b", rename.new_codename, text
        )
        text = text.replace(rename.old_directory, rename.new_directory)
        text = text.replace(rename.old_core_id, rename.new_core_id)
    marker = (
        f"> Formerly published internally as **{spec.old_codename}**. "
        "Weight and parity evidence are unchanged.\n\n"
    )
    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines and lines[0].startswith("---") else 0
    if insert_at:
        for index in range(1, len(lines)):
            if lines[index].startswith("---"):
                insert_at = index + 1
                break
    lines.insert(insert_at, marker)
    return "".join(lines)


def verify_applied(snapshots: dict[str, dict[str, str]] | None = None) -> None:
    catalog = load_json(CATALOG)
    for index, spec in enumerate(SPECS):
        sibling = SPECS[1 - index]
        if spec.old_path.exists() or not spec.new_path.is_dir():
            raise RuntimeError(f"Directory rename incomplete for {spec.new_codename}")
        if sha256_file(spec.new_path / "model.safetensors") != spec.weight_sha256:
            raise RuntimeError(f"Weight changed for {spec.new_codename}")
        if (
            snapshots is not None
            and snapshot_immutable(spec.new_path) != snapshots[spec.old_codename]
        ):
            raise RuntimeError(f"Immutable evidence changed for {spec.new_codename}")
        rename = load_json(spec.new_path / "RENAME.json")
        recorded_hashes = rename.get("immutable_file_sha256")
        if not isinstance(recorded_hashes, dict):
            raise TypeError(f"Rename manifest hashes missing for {spec.new_codename}")
        if snapshot_immutable(spec.new_path) != recorded_hashes:
            raise RuntimeError(f"Rename manifest hash mismatch for {spec.new_codename}")
        if rename.get("old", {}).get("core_id") != spec.old_core_id:
            raise RuntimeError(f"Former identity mismatch for {spec.new_codename}")
        if rename.get("new", {}).get("core_id") != spec.new_core_id:
            raise RuntimeError(
                f"Current rename identity mismatch for {spec.new_codename}"
            )
        core = load_json(spec.new_path / "CORE.json")
        if (
            core.get("core_id") != spec.new_core_id
            or core.get("codename") != spec.new_codename
            or core.get("directory_name") != spec.new_directory
        ):
            raise RuntimeError(
                f"Current CORE identity mismatch for {spec.new_codename}"
            )
        pair = core.get("pair") or {}
        if (
            pair.get("sibling_core_id") != sibling.new_core_id
            or pair.get("sibling_directory") != sibling.new_directory
        ):
            raise RuntimeError(f"Sibling identity mismatch for {spec.new_codename}")
        matches = [
            entry
            for entry in catalog.get("cores", [])
            if isinstance(entry, dict) and entry.get("core_id") == spec.new_core_id
        ]
        if len(matches) != 1 or matches[0].get("directory") != spec.new_directory:
            raise RuntimeError(f"Catalog mismatch for {spec.new_codename}")


def apply_rename() -> None:
    snapshots = preflight()
    renamed_at = utc_now()
    catalog_original = CATALOG.read_text(encoding="utf-8")
    mutable_originals = {
        spec.old_codename: {
            "CORE.json": (spec.old_path / "CORE.json").read_text(encoding="utf-8"),
            "README.md": (spec.old_path / "README.md").read_text(encoding="utf-8"),
        }
        for spec in SPECS
    }
    moved: list[RenameSpec] = []
    try:
        for spec in SPECS:
            os.replace(spec.old_path, spec.new_path)
            moved.append(spec)
        transformed_cores: dict[str, dict[str, Any]] = {}
        for index, spec in enumerate(SPECS):
            sibling = SPECS[1 - index]
            core = json.loads(mutable_originals[spec.old_codename]["CORE.json"])
            transformed = transform_core(core, spec, sibling, renamed_at)
            transformed_cores[spec.new_codename] = transformed
            atomic_json(spec.new_path / "CORE.json", transformed)
            atomic_write(
                spec.new_path / "README.md",
                render_readme(mutable_originals[spec.old_codename]["README.md"], spec),
            )
            atomic_json(
                spec.new_path / "RENAME.json",
                rename_manifest(spec, snapshots[spec.old_codename], renamed_at),
            )
        catalog = transform_catalog(
            json.loads(catalog_original), transformed_cores, renamed_at
        )
        atomic_json(CATALOG, catalog)
        verify_applied(snapshots)
    except Exception:
        atomic_write(CATALOG, catalog_original)
        for spec in reversed(moved):
            current = spec.new_path if spec.new_path.exists() else spec.old_path
            if current.exists() and current != spec.old_path:
                os.replace(current, spec.old_path)
            if spec.old_path.is_dir():
                for name, content in mutable_originals[spec.old_codename].items():
                    atomic_write(spec.old_path / name, content)
                rename_file = spec.old_path / "RENAME.json"
                if rename_file.exists():
                    rename_file.unlink()
        raise


def status() -> dict[str, Any]:
    return {
        spec.new_codename.lower(): {
            "old_exists": spec.old_path.exists(),
            "new_exists": spec.new_path.exists(),
            "weight_sha256": (
                sha256_file(spec.new_path / "model.safetensors")
                if (spec.new_path / "model.safetensors").is_file()
                else None
            ),
        }
        for spec in SPECS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "apply", "verify", "status"))
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2))
    elif args.command == "apply":
        apply_rename()
        print(json.dumps(status(), indent=2))
    elif args.command == "verify":
        verify_applied()
        print(json.dumps(status(), indent=2))
    else:
        print(json.dumps(status(), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
