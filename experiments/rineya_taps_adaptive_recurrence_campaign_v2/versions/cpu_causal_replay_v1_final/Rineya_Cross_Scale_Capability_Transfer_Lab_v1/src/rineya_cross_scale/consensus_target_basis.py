"""Consensus direction vs target basis (plan Exp C)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import atomic_write_json, sha256_file


STOP_MARGIN = 0.02


def analyze_consensus_vs_target(e2_path: Path, output_dir: Path) -> dict[str, Any]:
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    global_stats = e2.get("global") or {}

    def cosine(key: str, block: dict[str, Any] | None = None) -> float | None:
        source = block if block is not None else global_stats
        if key in source and isinstance(source[key], dict) and "cosine" in source[key]:
            return float(source[key]["cosine"])
        return None

    shared = cosine("delta_mean")
    d4 = cosine("delta_4b")
    d8 = cosine("delta_8b")
    random_c = cosine("control_norm_matched_random")
    sign_c = cosine("control_sign_shuffled")
    layer_c = cosine(
        "control_layer_shuffled", e2.get("layer_shuffled_control") or {}
    )

    controls = [c for c in (random_c, sign_c, layer_c) if c is not None]
    best_control = max(controls) if controls else 0.0
    margin = None if shared is None else float(shared) - float(best_control)
    distinguishable = margin is not None and margin > STOP_MARGIN

    report = {
        "schema_version": "3.0-diagnostics-index",
        "hypothesis": "Shared transported direction aligns with Δ_Supa beyond controls",
        "source": {
            "path": str(e2_path).replace("\\", "/"),
            "sha256": sha256_file(e2_path),
        },
        "cosines_vs_delta_supa": {
            "shared_mean": shared,
            "delta_4b": d4,
            "delta_8b": d8,
            "norm_matched_random": random_c,
            "sign_shuffled": sign_c,
            "layer_shuffled": layer_c,
        },
        "best_control": best_control,
        "margin_shared_minus_best_control": margin,
        "stop_margin": STOP_MARGIN,
        "distinguishable_from_controls": distinguishable,
        "verdict": "PASS" if distinguishable else "FAIL",
        "stop_rule": (
            "Shared direction indistinguishable from controls in target space; "
            "do not inject."
            if not distinguishable
            else "Shared direction beats controls; injection may be considered."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "target_basis_report.json", report)
    return report
