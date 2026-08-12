from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from consensus_lab.io import sha256_file, tree_manifest_sha256, write_json_atomic
from consensus_lab.paths import LabPaths


V22 = Path("artifacts/phase_9_capability_composition_v22_lr1e4_s1200")
IMPORTANT_SOURCES = [
    V22 / "frozen_anchor/config.json",
    V22 / "frozen_anchor/model.safetensors",
    V22 / "frozen_anchor/tokenizer.json",
    V22 / "track_d_greedy_soup/final_model/config.json",
    V22 / "track_d_greedy_soup/final_model/model.safetensors",
    V22 / "track_d_greedy_soup/final_model/tokenizer.json",
    V22 / "capability_composition_v22_selection.json",
    V22 / "track_d_greedy_soup/GREEDY_SOUP_RESULT.json",
    Path("data/phase_9_student_distillation/multidomain_reasoning_v22/reviewed_splits/dev.reviewed.jsonl"),
    Path("data/phase_9_student_distillation/general_reasoning_v212/reviewed_splits/dev.reviewed.jsonl"),
]
for _name in ("general", "grounding", "procedure", "abstention", "social"):
    IMPORTANT_SOURCES.extend(
        [
            V22 / f"track_c_{_name}/adapter/adapter_config.json",
            V22 / f"track_c_{_name}/adapter/adapter_model.safetensors",
        ]
    )
for _track in ("track_a_broad_reasoning", "track_b_anchored_reasoning"):
    IMPORTANT_SOURCES.extend(
        [V22 / _track / "adapter/adapter_config.json", V22 / _track / "adapter/adapter_model.safetensors"]
    )


def _git_revision(workspace: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=workspace, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "status_porcelain": None}


def build_source_manifest(paths: LabPaths) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative in IMPORTANT_SOURCES:
        source = paths.assert_official_source(paths.official_root / relative)
        if not source.is_file():
            raise FileNotFoundError(f"Required source artifact is missing: {source}")
        entries.append(
            {
                "relative_path": str(relative).replace("\\", "/"),
                "absolute_path": str(source),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_root": str(paths.official_root),
        "official_policy": "read_only",
        "entry_count": len(entries),
        "entries": entries,
        "manifest_sha256": tree_manifest_sha256(entries),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_revision(paths.workspace_root),
    }


def write_source_manifest(paths: LabPaths, output: Path) -> dict[str, Any]:
    output = paths.assert_output_path(output)
    packet = build_source_manifest(paths)
    write_json_atomic(output, packet)
    return packet


def compare_source_manifests(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    a = {entry["relative_path"]: entry["sha256"] for entry in before.get("entries", [])}
    b = {entry["relative_path"]: entry["sha256"] for entry in after.get("entries", [])}
    return sorted(path for path in set(a) | set(b) if a.get(path) != b.get(path))

