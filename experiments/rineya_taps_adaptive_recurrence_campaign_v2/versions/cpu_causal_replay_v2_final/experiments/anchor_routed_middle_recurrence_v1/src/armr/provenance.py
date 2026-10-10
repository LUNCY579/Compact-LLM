"""Exclusive evidence files and explicit non-sealed input allowlist."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
import transformers

from rineya_cross_scale.leakage import guard_non_sealed_path

EXPERIMENT = Path(__file__).resolve().parents[2]
REPO = EXPERIMENT.parents[1]
LAB = REPO / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1"
DATA = LAB / "data/evaluation"
HISTORY = LAB / "cross_scale_evidence/15_supa_layer_ablation_screen"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash(model: Any) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def cases_from(path: Path) -> list[dict]:
    path = guard_non_sealed_path(path)
    allowed = {(DATA / name).resolve() for name in
               ("supa_layer_ablation_smoke.jsonl", "mergedev_fast.jsonl")}
    if path not in allowed:
        raise ValueError("input is not an explicitly allowlisted ARMR dataset")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate case IDs")
    return rows


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def new_run(stage: str, config: dict, model_path: Path, dataset: Path) -> tuple[Path, dict]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = EXPERIMENT / "evidence" / stage / stamp
    out.mkdir(parents=True, exist_ok=False)
    import inspect
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    impl = Path(inspect.getfile(Qwen3DecoderLayer))
    files = [*EXPERIMENT.glob("src/**/*.py"), *EXPERIMENT.glob("scripts/*.py"),
             EXPERIMENT / "README.md", EXPERIMENT / "configs/reapplication_screen.json"]
    info = {
        "status": "STARTED", "start_utc": stamp, "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"), "git_status": git("status", "--short"),
        "model_path": str(model_path), "config": config, "dataset": str(dataset),
        "dataset_sha256": sha256(dataset), "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in files},
        "qwen_implementation_sha256": sha256(impl), "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "os": platform.platform(), "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
        "physical_memory_bytes": psutil.virtual_memory().total, "device": "cpu", "dtype": "float32",
        "kv_strategy": "full_prefix_recompute_no_kv", "kv_cache_bytes": 0,
        "gpu_memory_bytes": None, "model_loads": 1, "training": False,
    }
    write_json(out / "started_manifest.json", info)
    return out, info


def checkpoint_files(path: Path) -> dict[str, str]:
    names = ("model.safetensors", "config.json", "generation_config.json", "modeling_supa_a11.py",
             "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "CORE.json", "FILES.json")
    return {name: sha256(path / name) for name in names if (path / name).is_file()}
