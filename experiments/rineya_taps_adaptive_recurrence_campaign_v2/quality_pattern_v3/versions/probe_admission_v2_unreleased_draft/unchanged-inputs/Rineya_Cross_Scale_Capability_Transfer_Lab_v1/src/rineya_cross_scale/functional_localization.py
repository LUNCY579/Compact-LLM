"""Functional/parameter localization (plan Exp B).

Assembles coarse early/middle/late and module-family tables from executed E3
(and optional E5) without creating one-layer candidate checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import atomic_write_json, sha256_file


CONTROL_MARGIN = 0.02
ZERO_BAND = 0.002  # E3: all cells within ±0.0016 of zero vs Δ_Supa


def _bucket_rows(localization: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_depth = localization.get("by_depth_bucket") or {}
    for key, stats in by_depth.items():
        if not isinstance(stats, dict):
            continue
        parts = key.split("::")
        rows.append(
            {
                "region": key,
                "family": parts[0] if parts else key,
                "bucket": parts[1] if len(parts) > 1 else "",
                "cosine_vs_delta_supa": stats.get("cosine"),
                "sign_agreement": stats.get("sign_agreement"),
                "norm_ratio": stats.get("norm_ratio"),
                "coordinates": stats.get("coordinates"),
            }
        )
    by_module = localization.get("by_module") or localization.get("by_module_family") or {}
    for key, stats in by_module.items():
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "region": f"module::{key}",
                "family": key,
                "bucket": "module",
                "cosine_vs_delta_supa": stats.get("cosine"),
                "sign_agreement": stats.get("sign_agreement"),
                "norm_ratio": stats.get("norm_ratio"),
                "coordinates": stats.get("coordinates"),
            }
        )
    return rows


def select_promising_regions(
    rows: list[dict[str, Any]],
    *,
    control_cosine: float = 0.0,
) -> list[dict[str, Any]]:
    promising = []
    for row in rows:
        cosine = row.get("cosine_vs_delta_supa")
        if cosine is None:
            continue
        if abs(float(cosine)) <= ZERO_BAND:
            continue
        if float(cosine) <= float(control_cosine) + CONTROL_MARGIN:
            continue
        promising.append(row)
    return promising


def run_functional_localization(
    *,
    e3_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    localization = json.loads(e3_path.read_text(encoding="utf-8"))
    rows = _bucket_rows(localization)
    promising = select_promising_regions(rows)
    report = {
        "schema_version": "3.0-diagnostics-index",
        "hypothesis": "Transferable signal is localized by depth/module",
        "source": {
            "path": str(e3_path).replace("\\", "/"),
            "sha256": sha256_file(e3_path),
        },
        "coarse_rows": rows,
        "promising_regions": promising,
        "subdivision_authorised": bool(promising),
        "verdict": "FAIL" if not promising else "PASS",
        "interpretation": (
            "No depth bucket or module family exceeds the zero-band vs Δ_Supa; "
            "do not subdivide; do not micro-inject."
            if not promising
            else "Promising regions found; subdivision authorised."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "coarse_heatmap.json", report)
    atomic_write_json(output_dir / "promising_regions.json", {
        "promising_regions": promising,
        "count": len(promising),
        "verdict": report["verdict"],
    })
    return report
