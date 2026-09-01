"""Filesystem anchors. Writers must call ``assert_readonly_target`` first."""

from __future__ import annotations

from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LAB_ROOT.parent
SRC_ROOT = LAB_ROOT / "src"
CONFIGS = LAB_ROOT / "configs"
REPORTS = LAB_ROOT / "reports"
DATA = LAB_ROOT / "data"
ARTIFACTS = REPO_ROOT / "artifacts" / "supa_a1_1_fast_weight_attention_v1"
SUPA_PATH = REPO_ROOT / "Rineya_Published_Cores" / "Rineya-Supa-A1.1-1.1.0-devfreeze"

PUBLISHED_CORES = REPO_ROOT / "Rineya_Published_Cores"
PSA_SEALED = REPO_ROOT / "rineya_positional_sequential_attention_lab" / "data" / "sealed"
CC0_SEALED = REPO_ROOT / "rineya_long_context_cc0_v1_main" / "sealed_eval"

FORBIDDEN_READ_HINTS = (
    "showcase_prompts.jsonl",
    "showcase_answer_key.jsonl",
    "test2_answer_key.jsonl",
)


def assert_readonly_target(path: Path) -> None:
    resolved = Path(path).resolve()
    protected = [PUBLISHED_CORES, PSA_SEALED, CC0_SEALED]
    qwen = REPO_ROOT / "Qwen3-0.6B"
    if qwen.is_dir():
        protected.append(qwen)
    for root in protected:
        if not root.exists():
            continue
        root_r = root.resolve()
        if resolved == root_r or root_r in resolved.parents:
            raise PermissionError(f"refusing to write inside protected tree {root}: {resolved}")


def assert_unsealed_source(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    if "sealed_eval" in text or "/data/sealed/" in text or "test2" in text:
        raise PermissionError(f"refusing sealed path {path}")
    name = path.name.lower()
    if name in FORBIDDEN_READ_HINTS:
        raise PermissionError(f"refusing reserved evaluation file {path}")
