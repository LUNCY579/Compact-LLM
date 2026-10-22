"""Small, torch-free integrity and protocol surface. Evidence writes are exclusive."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

from experiments.model_quality_gpu_v1.integrity import (
    canonical, decode, digest, file_hash, new_dir, read_json, write_new,
)

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
RECORDS = ROOT / "research/multi_reasoning_scout_v1"
BRANCH = "codex/multi-reasoning-frozen-scout-v1-completion"
PARENT = "fbec90a0d0653aeb745d5223137ae3cfdd9c48dd"
CLASSIFICATION = "OPEN / DEVELOPMENT / SELECTION-EXPOSED FOREVER"
ANSWERS = ("0", "1", "2", "3", "UNKNOWN")


@dataclass(frozen=True)
class PublicItem:
    id: str
    scenario_id: str
    family: str
    split: str
    problem: str


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_files():
    own = sorted(PACKAGE.glob("*.py")) + sorted((PACKAGE / "tests").glob("*.py"))
    reused = [ROOT / f"experiments/model_quality_gpu_v1/{n}.py"
              for n in ("integrity", "supervisor", "resources")]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in own + reused}


def source_identity():
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("wrong isolated branch")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", PARENT, "HEAD"], check=True)
    return {"repository": "LUNCY579/ChatBubble", "branch": BRANCH,
            "sha": git("rev-parse", "HEAD"), "files": source_files()}


def verify_files(base, expected):
    for name, expected_hash in expected.items():
        path = Path(base) / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe manifest member")
        if file_hash(path) != expected_hash:
            raise ValueError(f"frozen artifact changed: {name}")


def load_frozen():
    freeze = read_json(RECORDS / "FREEZE.json")
    verify_files(ROOT, freeze["files"])
    if source_files() != freeze["source_files"]:
        raise ValueError("stale frozen executable source")
    current = source_identity()
    frozen_source = freeze["source"]
    if (any(current[k] != frozen_source[k] for k in ("repository", "branch", "files"))
            or git("merge-base", frozen_source["sha"], current["sha"]) != frozen_source["sha"]):
        raise ValueError("stale source lineage; frozen source must be an ancestor with identical executable bytes")
    return freeze


def public_bank():
    # There is intentionally no arbitrary --bank argument or historical-bank reader.
    load_frozen()
    path = RECORDS / "bank/public.jsonl"
    rows = [PublicItem(**decode(line)) for line in path.read_text(encoding="utf-8").splitlines()]
    if len({r.id for r in rows}) != len(rows) or len({r.scenario_id for r in rows}) != len(rows):
        raise ValueError("duplicate item/world")
    return rows


def model_assets(model):
    root = Path(model["path"])
    verify_files(root, model["assets"])
    weights = {p.name for p in root.iterdir() if p.suffix in (".safetensors", ".bin", ".pt", ".gguf")}
    if weights != {"model.safetensors"}:
        raise ValueError("unregistered weight layout")
    return digest(model)


def validate_receipt(receipt, binding, stage):
    if receipt.get("status") != "PASS" or receipt.get("stage") != stage or receipt.get("binding") != binding:
        raise ValueError("stale, failed or incompatible receipt")


def dump_lines_new(path, rows):
    with Path(path).open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")


def make_freeze():
    if (RECORDS / "FREEZE.json").exists():
        raise FileExistsError("freeze is immutable")
    paths = [RECORDS / n for n in (
        "protocol.json", "MODEL_MANIFEST.json", "MODEL_LINEAGE.json", "SOURCE_PINS.json", "BANK_MANIFEST.json",
        "PROMPT_MANIFEST.json", "MULTI_REASONING_PROTOCOL.md", "MULTI_REASONING_PREREG.md",
        "MULTI_REASONING_POSTTRAIN_CONTRACT.md", "FUTURE_2X2_ANALYSIS_PLAN.md",
        "ACCESS_MANIFEST.json", "SOURCE_REAUDIT.json", "CANDIDATE_REAUDIT.json", "CURRENT_STATE_AUDIT.md",
        "SCOUT_BANK_AUDIT.md", "BANK_REAUDIT.json", "PHASE_II_CONTINUATION.md",
        "bank/public.jsonl", "bank/gold.jsonl", "bank/worlds.jsonl")]
    write_new(RECORDS / "FREEZE.json", {
        "version": "MRS-1", "classification": CLASSIFICATION,
        "source_files": source_files(), "source": source_identity(),
        "files": {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in paths},
        "model_results_observed": False, "future_protocol_changes": "NEW EXPERIMENT REQUIRED",
    })
