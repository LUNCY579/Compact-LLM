"""Functional alignment atlas helpers (plan Phase 2 / Exp B inputs).

Full donor-scale captures live in streaming_forward + v3 E4/E5 scripts.
This module provides the cheap summary/assembly layer used by the plan-index
tree so new work can consume existing disk-backed atlases without reloading
4B/8B weights.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .architecture import uniform_depth_map
from .hashing import atomic_write_json, sha256_file


RELATIVE_DEPTHS = (0.0, 0.17, 0.33, 0.50, 0.67, 0.83, 1.0)
PLAN_DEPTHS = (0.0, 0.25, 0.50, 0.75, 1.0)


def layer_at_fraction(n_layers: int, fraction: float) -> int:
    if n_layers <= 1:
        return 0
    return int(round(fraction * (n_layers - 1)))


def matched_depth_pairs(
    donor_layers: int,
    target_layers: int,
    fractions: tuple[float, ...] = PLAN_DEPTHS,
) -> list[dict[str, Any]]:
    """Uniform relative-depth correspondence (validated by E5 as diagonal)."""
    mapping = uniform_depth_map(donor_layers, target_layers)
    rows = []
    for fraction in fractions:
        donor_layer = layer_at_fraction(donor_layers, fraction)
        target_layer = mapping[donor_layer] if donor_layer < len(mapping) else layer_at_fraction(
            target_layers, fraction
        )
        rows.append(
            {
                "fraction": fraction,
                "donor_layer": donor_layer,
                "target_layer": int(target_layer),
            }
        )
    return rows


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / max(na * nb, 1e-30)


def summarise_existing_atlas(source_json: Path) -> dict[str, Any]:
    """Load an executed atlas JSON and extract plan-facing summary fields."""
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    return {
        "source": str(source_json).replace("\\", "/"),
        "source_sha256": sha256_file(source_json),
        "keys": sorted(payload.keys()),
        "payload_excerpt": {
            key: payload[key]
            for key in (
                "experiment",
                "schema_version",
                "pairs",
                "cka",
                "logit",
                "summary",
                "verdict",
            )
            if key in payload
        },
        "full": payload,
    }


def write_heatmap_markdown(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", "", "| comparison | metric | value | control | lift |", "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| {comparison} | {metric} | {value} | {control} | {lift} |".format(
                comparison=row.get("comparison", ""),
                metric=row.get("metric", ""),
                value=row.get("value", ""),
                control=row.get("control", ""),
                lift=row.get("lift", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_layer_depth_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["fraction", "donor_layer", "target_layer"]
        )
        writer.writeheader()
        for row in pairs:
            writer.writerow(row)


def assemble_plan_atlas(
    *,
    e4_path: Path,
    e5_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Curate plan Phase-2 atlas artifacts from executed E4/E5 JSON."""
    e4 = summarise_existing_atlas(e4_path)
    e5 = summarise_existing_atlas(e5_path) if e5_path and e5_path.is_file() else None
    depth_pairs = matched_depth_pairs(36, 28)
    write_layer_depth_csv(output_dir / "layer_depth_table.csv", depth_pairs)

    heatmap_rows: list[dict[str, Any]] = []
    if e5 and "full" in e5:
        full = e5["full"]
        decision = full.get("decision") or {}
        if decision:
            heatmap_rows.append(
                {
                    "comparison": "E5_decision",
                    "metric": "best_donor_target_matched_lift",
                    "value": decision.get("best_donor_target_matched_lift"),
                    "control": decision.get("cka_lift_threshold"),
                    "lift": decision.get("best_donor_target_matched_lift"),
                }
            )
        surfaces = full.get("depth_surfaces") or {}
        for pair_name, surface in surfaces.items():
            if not isinstance(surface, dict):
                continue
            cka = surface.get("cka")
            shuffled = surface.get("cka_prompt_shuffled")
            if not isinstance(cka, list) or not cka:
                continue
            # Diagonal-ish summary: max over matched cells excluding embedding row/col 0.
            matched_vals = []
            control_vals = []
            for i in range(1, min(len(cka), 6)):
                row = cka[i]
                if isinstance(row, list) and i < len(row):
                    matched_vals.append(float(row[i]))
                if (
                    isinstance(shuffled, list)
                    and i < len(shuffled)
                    and isinstance(shuffled[i], list)
                    and i < len(shuffled[i])
                ):
                    control_vals.append(float(shuffled[i][i]))
            if matched_vals:
                value = sum(matched_vals) / len(matched_vals)
                control = (
                    sum(control_vals) / len(control_vals) if control_vals else None
                )
                heatmap_rows.append(
                    {
                        "comparison": pair_name,
                        "metric": "mean_diag_cka",
                        "value": round(value, 6),
                        "control": None if control is None else round(control, 6),
                        "lift": None
                        if control is None
                        else round(value - control, 6),
                    }
                )

    if not heatmap_rows:
        # Fall back to documenting sources when schema differs.
        heatmap_rows.append(
            {
                "comparison": "see_source_json",
                "metric": "pointer",
                "value": e4["source"],
                "control": e5["source"] if e5 else "",
                "lift": "",
            }
        )

    write_heatmap_markdown(output_dir / "heatmap.md", "Functional alignment atlas", heatmap_rows)

    matrix = {
        "schema_version": "3.0-diagnostics-index",
        "hypothesis": "Locate where 4B↔8B agreement lives vs what ports to 0.6B/Supa",
        "e4": {"path": e4["source"], "sha256": e4["source_sha256"]},
        "e5": (
            {"path": e5["source"], "sha256": e5["source_sha256"]} if e5 else None
        ),
        "depth_pairs_36_to_28": depth_pairs,
        "heatmap_rows": heatmap_rows,
        "interpretation_pointer": (
            "13_v3_transport_diagnosis/report/v3_report.md and "
            "13_v3_transport_diagnosis/report/v3_1_donor_side_report.md"
        ),
    }
    atomic_write_json(output_dir / "similarity_matrix.json", matrix)
    return matrix
