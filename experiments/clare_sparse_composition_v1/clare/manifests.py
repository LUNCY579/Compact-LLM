"""Machine-readable manifests for every discovered mask and trained skill delta.

Every skill artifact records (at least) the fields the CLARE-RINEYA-001 protocol requires:
experiment ID, branch, git commit, parent commit, model/tokenizer identity, skill, skill order,
seed, rho/budget, eligible/selected coordinates, discovery lambda/steps, training steps,
optimizer, LR, dtype, delta norm, mask checksum, delta checksum, dataset split IDs, timestamps.
Allocated capacity (mask size) and ACTIVE non-zero coordinates are reported separately.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import subprocess
from pathlib import Path
from typing import Mapping

from . import EXPERIMENT_ID, PACKAGE_VERSION

BRANCH = "research/titan-clare-sparse-composable-skills-v1"
PARENT_BRANCH = "research/titan-model-soup-mini-v1"
PARENT_COMMIT = "816a1e8f4d1ab1c9d1ab847697d962a141521bac"
MANIFEST_FORMAT = "clare-skill-manifest-v1"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def _git(*args: str, cwd: Path = EXPERIMENT_DIR) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def git_identity(cwd: Path = EXPERIMENT_DIR) -> dict:
    """HEAD, branch, dirty state of the EXPERIMENT directory, and whether the parent is an ancestor."""
    head = _git("rev-parse", "HEAD", cwd=cwd)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    dirty = _git("status", "--porcelain", "--untracked-files=all", "--", ".", cwd=cwd)
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_COMMIT, head], cwd=cwd, check=True,
                       capture_output=True)
        parent_ok = True
    except subprocess.CalledProcessError:
        parent_ok = False
    return {"commit": head, "branch": branch, "experiment_dir_dirty": bool(dirty),
            "dirty_entries": dirty.splitlines()[:50], "parent_branch": PARENT_BRANCH,
            "parent_commit": PARENT_COMMIT, "parent_is_ancestor": parent_ok}


def code_digest(root: Path = EXPERIMENT_DIR / "clare") -> str:
    """sha256 over the package's .py files (sorted relative paths + bytes): the code identity."""
    h = hashlib.sha256(b"clare-code-v1")
    for p in sorted(root.rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def skill_manifest(*, arm: str, skill: str, order: str, position: int, seed: int, rho: str,
                   footprint_summary: Mapping, eligible_before: int | None, selected: int | None,
                   budget: int | None, delta_side: Mapping, mask_checksum: str | None,
                   recipe: Mapping, train_log: Mapping | None, discovery: Mapping | None,
                   model_identity: Mapping, dataset: Mapping, runtime: Mapping, git: Mapping,
                   frozen_checksum_before: str | None, frozen_checksum_after: str | None,
                   code: str, started_utc: str | None = None, finished_utc: str | None = None) -> dict:
    total = int(footprint_summary["total_coordinates"])
    return {
        "format": MANIFEST_FORMAT, "experiment": EXPERIMENT_ID, "package_version": PACKAGE_VERSION,
        "branch": git.get("branch"), "git_commit": git.get("commit"), "experiment_dir_dirty": git.get("experiment_dir_dirty"),
        "parent_branch": PARENT_BRANCH, "parent_commit": PARENT_COMMIT, "code_digest": code,
        "model": dict(model_identity), "dataset": dict(dataset), "runtime": dict(runtime),
        "arm": arm, "skill": skill, "skill_order": order, "position": position, "seed": seed,
        "rho": rho, "budget": budget,
        "footprint": dict(footprint_summary), "total_eligible_coordinates": total,
        "free_coordinates_before_skill": eligible_before,
        "selected_coordinates": selected,
        "selected_fraction_of_free": (selected / eligible_before) if (selected is not None and eligible_before) else None,
        "selected_fraction_of_footprint": (selected / total) if selected is not None else None,
        "active_nonzero_coordinates": delta_side.get("nonzero"),
        "discovery_lambda": (discovery or {}).get("l1_lambda"),
        "discovery_updates": (discovery or {}).get("discovery_updates"),
        "discovery_nonzero_free_after_probe": (discovery or {}).get("nonzero_free_after_probe"),
        "discovery_selected_with_zero_importance": (discovery or {}).get("selected_with_zero_importance"),
        "selection_rule": (discovery or {}).get("rule", "dense" if mask_checksum is None else None),
        "training_updates": (train_log or {}).get("updates"),
        "training_processed_tokens": (train_log or {}).get("processed_tokens"),
        "training_supervised_tokens": (train_log or {}).get("supervised_tokens"),
        "optimizer": "torch.optim.AdamW(foreach=False, fused=False)",
        "recipe": dict(recipe), "peak_lr": recipe.get("peak_lr"), "dtype": "float32 (BF16-stored base, exact upcast)",
        "delta_l2_norm": delta_side.get("l2_norm"), "delta_checksum": delta_side.get("delta_checksum"),
        "delta_payload_sha256": delta_side.get("payload_sha256"), "delta_storage": delta_side.get("storage"),
        "mask_checksum": mask_checksum,
        "frozen_checksum_before": frozen_checksum_before, "frozen_checksum_after": frozen_checksum_after,
        "frozen_unchanged": (frozen_checksum_before == frozen_checksum_after) if frozen_checksum_before else None,
        "started_utc": started_utc, "finished_utc": finished_utc or utc_now(),
        "claims": {"quality_claim": False, "note": "a manifest records provenance, not a result"},
    }
