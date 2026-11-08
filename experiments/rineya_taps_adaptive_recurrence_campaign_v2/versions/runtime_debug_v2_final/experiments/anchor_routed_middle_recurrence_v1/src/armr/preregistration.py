"""Validate the sole authorized change in the CPU-budget replication."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from .provenance import sha256


def function_hash(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def budget_only_config(base: dict, effective: dict) -> dict:
    differences = {k for k in set(base) | set(effective) if base.get(k) != effective.get(k)}
    if differences != {"max_prompt_seconds"}:
        raise ValueError("CPU replication permits only max_prompt_seconds to differ")
    if base["max_prompt_seconds"] != 90 or effective["max_prompt_seconds"] != 900:
        raise ValueError("preregistered CPU deadline must change from 90 to 900 seconds")
    return dict(effective)


def load_budget_preregistration(path: Path, root: Path, base: dict, atlas: Path, selected: dict):
    path = path.resolve()
    if not path.is_relative_to((root / "preregistrations").resolve()):
        raise ValueError("preregistration must be inside this experiment")
    registration = json.loads(path.read_text(encoding="utf-8"))
    if registration["status"] != "FROZEN" or registration["id"] != "fixed_cpu_budget_v2":
        raise ValueError("frozen CPU-budget preregistration required")
    prior = (root / registration["prior_run"]).resolve()
    if not prior.is_relative_to((root / "evidence/02_fixed_recurrence").resolve()):
        raise ValueError("unexpected prior fixed-run path")
    if sha256(prior / "run_manifest.json") != registration["prior_manifest_sha256"]:
        raise ValueError("prior receipt changed")
    prior_manifest = json.loads((prior / "run_manifest.json").read_text())
    if not prior_manifest["canonical_unchanged"] or prior_manifest["config"] != base:
        raise ValueError("prior canonical/configuration integrity failed")
    if (root / registration["atlas"]).resolve() != atlas or sha256(atlas / "run_manifest.json") != registration["atlas_manifest_sha256"]:
        raise ValueError("candidate atlas changed")
    normalized = json.loads(json.dumps(selected))
    if normalized != registration["selected_configuration"] or normalized != prior_manifest["selected_configuration"]:
        raise ValueError("fixed arm selection changed")
    repo = root.parents[1]
    for relative, expected in registration["preserved_source_sha256"].items():
        source = (repo / relative).resolve()
        if not source.is_relative_to(repo.resolve()) or sha256(source) != expected:
            raise ValueError(f"frozen execution/data source changed: {relative}")
    if function_hash(root / "scripts/run_fixed_recurrence.py", "development_result") != registration["comparison_function_sha256"]:
        raise ValueError("development scoring/gates changed")
    effective = budget_only_config(base, registration["effective_config"])
    return effective, registration, prior_manifest
