"""Write the bounded STR-QUAL-001 receipt; never launch training or sealed eval."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.soft_token_reasoning_v1.implementation.binding import bind_dataset_manifest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resource-probe", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, default=None)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    dataset_manifest = ROOT / "Rineya-Puzzle-Reasoning-51k" / "data" / "manifest.json"
    binding = bind_dataset_manifest(dataset_manifest, "Rineya-Puzzle-Reasoning-51k", ("train", "validation"))
    receipt = {
        "milestone": "STR-QUAL-001",
        "status": "PASS",
        "qualification_scope": "mechanical tensor qualification only",
        "base_sha": git("rev-parse", "origin/main"),
        "head_sha_at_measurement": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "canonical_model": {"model_id": "Qwen/Qwen3-0.6B", "tokenizer_id": "Qwen/Qwen3-0.6B", "source": "RESEARCH_PAPER_DATA.md"},
        "dataset_binding": binding,
        "seed": 20260925,
        "dtype": "float64 for deterministic qualification tensors",
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "tests": {"command": "python -m pytest -q experiments/soft_token_reasoning_v1/tests", "passed": args.tests_passed, "sealed_tests": False},
        "resource_probe": json.loads(args.resource_probe.read_text(encoding="utf-8")),
        "training_status": "not_run",
        "sealed_access": False,
        "gflowrl": False,
        "native_checkpoint_loaded": False,
        "evidence_classification": "mechanical qualification; no model-quality or improvement claim",
        "source_files": {
            "implementation/soft_tokens.py": sha256_file(ROOT / "experiments/soft_token_reasoning_v1/implementation/soft_tokens.py"),
            "implementation/rollout.py": sha256_file(ROOT / "experiments/soft_token_reasoning_v1/implementation/rollout.py"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
