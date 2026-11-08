"""MRS-2 identity, and the hash-bound reuse of the MRS-1 frozen bank.

MRS-2 does NOT regenerate the scout bank. It binds MRS-1's frozen bank by hash and
reads the identical bytes, so paired MRS-1 / MRS-2 / future C-D comparisons remain on
one population. Regenerating the scout is forbidden by the MRS-1 posttrain contract.
"""
from __future__ import annotations
from pathlib import Path
import subprocess

from experiments.model_quality_gpu_v1.integrity import (  # noqa: F401  (re-exported)
    canonical, decode, digest, file_hash, new_dir, read_json, write_new,
)
from experiments.multi_reasoning_scout_v1 import core as mrs1
from experiments.multi_reasoning_scout_v1.core import PublicItem, verify_files

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
RECORDS = ROOT / "research/multi_reasoning_scout_v2"
MRS1_RECORDS = mrs1.RECORDS
BRANCH = "research/multi-reasoning-scout-v2"
CLASSIFICATION = "OPEN / DEVELOPMENT / SELECTION-EXPOSED FOREVER"
ANSWERS = ("0", "1", "2", "3", "UNKNOWN")

# The probe that selected this instrument's output contract.
PROBE = {"id": "MRS-FP-1", "branch": "research/mrs-format-probe-v1",
         "contract": "C1_BARE",
         "records": "research/mrs_format_probe_v1"}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_files():
    own = sorted(PACKAGE.glob("*.py")) + sorted((PACKAGE / "tests").glob("*.py"))
    reused = [ROOT / f"experiments/model_quality_gpu_v1/{n}.py"
              for n in ("integrity", "supervisor", "resources")]
    reused += [ROOT / f"experiments/multi_reasoning_scout_v1/{n}.py"
               for n in ("core", "budget", "native", "analysis")]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in own + reused}


def source_identity():
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("wrong isolated branch")
    return {"repository": "LUNCY579/ChatBubble", "branch": BRANCH,
            "sha": git("rev-parse", "HEAD"), "files": source_files()}


def mrs1_binding():
    """Verify every MRS-1 frozen artifact byte-for-byte and confirm lineage."""
    freeze = read_json(MRS1_RECORDS / "FREEZE.json")
    verify_files(mrs1.ROOT, freeze["files"])
    if mrs1.source_files() != freeze["source_files"]:
        raise ValueError("stale MRS-1 executable source")
    frozen_sha = freeze["source"]["sha"]
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                       frozen_sha, "HEAD"]).returncode:
        raise ValueError("frozen MRS-1 source is not an ancestor of MRS-2")
    return {"mrs1_freeze_sha256": file_hash(MRS1_RECORDS / "FREEZE.json"),
            "mrs1_frozen_source_sha": frozen_sha,
            "bank": {k: v for k, v in freeze["files"].items() if "/bank/" in k}}


def public_bank():
    """The MRS-1 frozen public bank, byte-identical. No alternative bank reader exists."""
    mrs1_binding()
    rows = [PublicItem(**decode(line)) for line
            in (MRS1_RECORDS / "bank/public.jsonl").read_text(encoding="utf-8").splitlines()]
    if len({r.id for r in rows}) != len(rows) or len({r.scenario_id for r in rows}) != len(rows):
        raise ValueError("duplicate item/world")
    return rows


def gold_labels():
    """Gold is loaded ONLY by analysis, after generation. Never importable into a prompt."""
    mrs1_binding()
    rows = [decode(line) for line
            in (MRS1_RECORDS / "bank/gold.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["id"]: r for r in rows}


def load_frozen():
    freeze = read_json(RECORDS / "FREEZE.json")
    verify_files(ROOT, freeze["files"])
    if source_files() != freeze["source_files"]:
        raise ValueError("stale frozen executable source")
    current = source_identity()
    frozen_source = freeze["source"]
    if (any(current[k] != frozen_source[k] for k in ("repository", "branch", "files"))
            or git("merge-base", frozen_source["sha"], current["sha"]) != frozen_source["sha"]):
        raise ValueError("stale source lineage; frozen source must be an ancestor "
                         "with identical executable bytes")
    return freeze


def model_assets(model):
    return mrs1.model_assets(model)


def validate_receipt(receipt, binding, stage):
    return mrs1.validate_receipt(receipt, binding, stage)
