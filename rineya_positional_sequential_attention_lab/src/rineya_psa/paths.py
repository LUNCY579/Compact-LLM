"""Filesystem anchors and the read-only model registry.

Checkpoint paths are discovered relative to the repository root rather than
hard-coded absolutely, so the lab moves with the repo. Nothing in this module
opens a checkpoint for writing; see ``tests/test_immutability.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from rineya_psa.types import ModelSpec

LAB_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LAB_ROOT.parent

# Output trees. Created on demand by ``ensure_dirs``; never checked into git as
# binaries (see the repo .gitignore vault rules).
CONFIGS = LAB_ROOT / "configs"
DATA = LAB_ROOT / "data"
DATA_GENERATED = DATA / "generated"
DATA_CALIBRATION = DATA / "calibration"
DATA_VALIDATION = DATA / "validation"
DATA_SEALED = DATA / "sealed"
MANIFESTS = LAB_ROOT / "manifests"
RESULTS = LAB_ROOT / "results"
FIGURES = LAB_ROOT / "figures"
REPORTS = LAB_ROOT / "reports"
EXPERIMENTS = LAB_ROOT / "experiments"

_OUTPUT_DIRS = (
    CONFIGS,
    DATA_GENERATED,
    DATA_CALIBRATION,
    DATA_VALIDATION,
    DATA_SEALED,
    MANIFESTS,
    RESULTS,
    FIGURES,
    REPORTS,
    EXPERIMENTS,
)


def ensure_dirs() -> None:
    for d in _OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


_CORES = REPO_ROOT / "Rineya_Published_Cores"

MODELS: dict[str, ModelSpec] = {
    "qwen3": ModelSpec(
        key="qwen3",
        display_name="Qwen3-0.6B",
        path=REPO_ROOT / "Qwen3-0.6B",
        store_dtype="bfloat16",
        trust_remote_code=False,
        expected_arch="Qwen3ForCausalLM",
        ungatable=False,
    ),
    "halo": ModelSpec(
        key="halo",
        display_name="Rineya-Halo-1.2.0",
        path=_CORES / "Rineya-Halo-1.2.0-devfreeze",
        store_dtype="float32",
        trust_remote_code=False,
        expected_arch="Qwen3ForCausalLM",
        ungatable=False,
    ),
    "nimbus": ModelSpec(
        key="nimbus",
        display_name="Rineya-Nimbus-1.6.0",
        path=_CORES / "Rineya-Nimbus-1.6.0-devfreeze",
        store_dtype="bfloat16",
        trust_remote_code=False,
        expected_arch="Qwen3ForCausalLM",
        ungatable=False,
    ),
    "supa_a11": ModelSpec(
        key="supa_a11",
        display_name="Rineya-Supa-A1.1-1.1.0",
        path=_CORES / "Rineya-Supa-A1.1-1.1.0-devfreeze",
        store_dtype="float16",
        trust_remote_code=True,
        expected_arch="Qwen3AbstainForCausalLM",
        ungatable=True,
    ),
}

PANEL: tuple[str, ...] = ("qwen3", "halo", "nimbus", "supa_a11")

# Reference model for every delta computation in the audit.
BASE_KEY = "qwen3"


def model_spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(MODELS)}") from None


def assert_readonly_target(path: Path) -> None:
    """Raise if ``path`` lies inside a source checkpoint tree.

    Every writer in the lab calls this before opening a file for write. It is the
    programmatic form of the 'checkpoints are immutable' rule.
    """
    resolved = Path(path).resolve()
    for spec in MODELS.values():
        root = spec.path.resolve()
        if resolved == root or root in resolved.parents:
            raise PermissionError(
                f"refusing to write inside source checkpoint {spec.key}: {resolved}"
            )


def python_executable() -> str:
    """The interpreter the lab is expected to run under."""
    return os.environ.get("PSA_PYTHON", str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"))
