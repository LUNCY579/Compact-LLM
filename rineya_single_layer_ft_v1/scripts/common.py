"""Shared paths and tiny I/O helpers for the single-layer FT lab."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
REPO_ROOT = LAB_ROOT.parent

# --- Abstention gate control -------------------------------------------------
# Supa-A1.1's Qwen3AbstainForCausalLM.forward reads ``abstain_logit_bias`` off
# ``self.config`` and adds it to logits[..., abstain_token_id] at EVERY decode
# step. The published devfreeze ships +5.156249985098839 on token 51378
# ("UNKNOWN"), so every load of that core silently gates toward abstention in
# training, generation AND scoring alike.
#
# This lab pins that addition to exactly 0.0. The override is applied at load
# time via from_pretrained kwargs; the canonical config.json is hash-covered by
# the frozen sidecar contract (sha256 89c4ae73...) and is never modified.
# Measured: the override removes exactly 5.15625 at token 51378 and 0.0 at
# every other token id.
ABSTAIN_TOKEN_ID = 51378
ABSTAIN_BIAS_DISABLED = 0.0
ABSTAIN_BIAS_PUBLISHED = 5.156249985098839


def abstention_overrides(bias: float = ABSTAIN_BIAS_DISABLED) -> dict[str, Any]:
    """``from_pretrained`` kwargs pinning the abstention logit bias.

    Defaults to fully disabled. Every model load in this lab must pass these so
    the gate cannot differ between training and evaluation.
    """
    value = float(bias)
    if value != 0.0:
        raise ValueError(
            "this lab is registered with the abstention gate disabled; "
            f"refusing a nonzero abstain_logit_bias ({value!r})"
        )
    return {"abstain_logit_bias": value}


def _base_config(model: Any) -> Any:
    """Reach the underlying HF config through PEFT / wrapper layers."""
    seen = set()
    node = model
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        config = getattr(node, "config", None)
        if config is not None and hasattr(config, "abstain_token_id"):
            return config
        node = getattr(node, "base_model", None) or getattr(node, "model", None)
    return getattr(model, "config", None)


def assert_gate_disabled(model: Any) -> dict[str, Any]:
    """Fail closed unless token 51378 receives exactly 0.0 additional bias.

    Called after any adapter is attached, so a wrapper cannot reintroduce the
    published bias. Returns a receipt fragment for the run manifest.
    """
    config = _base_config(model)
    bias = float(getattr(config, "abstain_logit_bias", 0.0) or 0.0)
    token_id = getattr(config, "abstain_token_id", None)
    if bias != 0.0:
        raise RuntimeError(
            "abstention gate is active: abstain_logit_bias="
            f"{bias!r} on token {token_id!r}; expected exactly 0.0"
        )
    if token_id is not None and int(token_id) != ABSTAIN_TOKEN_ID:
        raise RuntimeError(
            f"unexpected abstain_token_id {token_id!r}; expected {ABSTAIN_TOKEN_ID}"
        )
    return {
        "abstain_token_id": None if token_id is None else int(token_id),
        "abstain_logit_bias_effective": bias,
        "abstain_gate_disabled": True,
        "abstain_bias_published": ABSTAIN_BIAS_PUBLISHED,
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()
