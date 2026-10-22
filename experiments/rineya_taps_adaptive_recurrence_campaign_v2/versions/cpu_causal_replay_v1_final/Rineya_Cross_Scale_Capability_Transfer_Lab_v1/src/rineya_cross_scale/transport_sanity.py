"""Transport sanity tests (plan Exp A).

Consumes executed E1/E1c/E2 JSON. Early-reject criteria match the plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import atomic_write_json, sha256_file


EARLY_REJECT = {
    "cosine_drop_vs_raw": 0.15,
    "norm_ratio_lo": 0.5,
    "norm_ratio_hi": 2.0,
    "target_margin_over_control": 0.02,
}


def _arm_cosine(rows: list[dict[str, Any]], needle: str) -> float | None:
    for row in rows:
        arm = str(row.get("arm", ""))
        if needle in arm:
            return float(row["cosine"])
    return None


def _mean_arm_cosine(rows: list[dict[str, Any]], needle: str) -> float | None:
    values = [
        float(row["cosine"])
        for row in rows
        if needle in str(row.get("arm", "")) and "cosine" in row
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _verdict_from_e1(e1: dict[str, Any], e1c: dict[str, Any] | None) -> dict[str, Any]:
    controls = e1.get("controls") or []
    real = _arm_cosine(controls, "real_4b_vs_real_8b")
    shape_null = None
    if e1c:
        shape_null = _mean_arm_cosine(e1c.get("arms") or [], "shape_matched_gaussian")
        if shape_null is None and isinstance(e1c.get("verdict"), dict):
            shape_null = e1c["verdict"].get("null_cosine") or e1c["verdict"].get(
                "shape_matched_mean_cosine"
            )

    checks: list[dict[str, Any]] = []
    if real is not None and shape_null is not None:
        operator_induced = float(shape_null) >= float(real)
        checks.append(
            {
                "name": "consensus_vs_shape_null",
                "real_cosine": real,
                "shape_matched_null_cosine": shape_null,
                "pass": not operator_induced,
                "note": (
                    "shape-matched Gaussians match or beat real donors"
                    if operator_induced
                    else "real donors exceed shape null"
                ),
            }
        )

    channel = e1.get("channel") or {}
    # P4 transmits singular-value profiles only; directions are zero by construction.
    checks.append(
        {
            "name": "donor_directions_transmitted",
            "donor_scalars": channel.get("donor_scalars_transmitted"),
            "effective_scalars": channel.get("effective_donor_scalars_transmitted"),
            "pass": False,
            "note": "P4 keeps target singular vectors; donor directions are discarded",
        }
    )

    failed = [check for check in checks if not check["pass"]]
    return {
        "transformation": "P4_spectral_s010",
        "checks": checks,
        "verdict": "FAIL" if failed else "PASS",
        "failed": failed,
    }


def evaluate_p5_from_e2(e2: dict[str, Any]) -> dict[str, Any]:
    global_block = e2.get("global") or {}
    p5 = None
    if "delta_p5_4b" in global_block:
        p5 = float(global_block["delta_p5_4b"]["cosine"])
    controls = []
    for key in (
        "control_norm_matched_random",
        "control_sign_shuffled",
        "control_layer_shuffled",
    ):
        if key in global_block:
            controls.append(float(global_block[key]["cosine"]))
    layer = e2.get("layer_shuffled_control") or {}
    if "control_layer_shuffled" in layer:
        controls.append(float(layer["control_layer_shuffled"]["cosine"]))

    best_control = max(controls) if controls else 0.0
    margin = None if p5 is None else float(p5) - best_control
    passed = margin is not None and margin > EARLY_REJECT["target_margin_over_control"]
    return {
        "transformation": "P5_direction_transfer",
        "cosine_vs_delta_supa": p5,
        "best_control_cosine": best_control,
        "margin": margin,
        "verdict": "FAIL" if p5 is not None and not passed else ("PASS" if passed else "INCONCLUSIVE"),
        "note": "Must beat shuffle/random controls in target basis before injection",
    }


def run_transport_sanity(
    *,
    e1_path: Path,
    e1c_path: Path | None,
    e2_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    e1c = (
        json.loads(e1c_path.read_text(encoding="utf-8"))
        if e1c_path and e1c_path.is_file()
        else None
    )
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))

    families = {
        "P4_spectral_s010": _verdict_from_e1(e1, e1c),
        "P5_direction_transfer": evaluate_p5_from_e2(e2),
        "controls_required": [
            "norm_matched_random",
            "layer_shuffled",
            "sign_shuffled",
            "shape_matched_gaussian_donors",
        ],
        "sources": {
            "e1": {"path": str(e1_path).replace("\\", "/"), "sha256": sha256_file(e1_path)},
            "e1c": (
                {
                    "path": str(e1c_path).replace("\\", "/"),
                    "sha256": sha256_file(e1c_path),
                }
                if e1c_path and e1c_path.is_file()
                else None
            ),
            "e2": {"path": str(e2_path).replace("\\", "/"), "sha256": sha256_file(e2_path)},
        },
    }

    verdicts = [
        families["P4_spectral_s010"]["verdict"],
        families["P5_direction_transfer"]["verdict"],
    ]
    if any(v == "PASS" for v in verdicts):
        overall = "PASS"
    elif all(v == "FAIL" for v in verdicts):
        overall = "FAIL"
    else:
        overall = "FAIL" if "FAIL" in verdicts else "INCONCLUSIVE"

    report = {
        "schema_version": "3.0-diagnostics-index",
        "hypothesis": "T preserves known structure before capability transfer",
        "families": families,
        "overall_verdict": overall,
        "early_reject_thresholds": EARLY_REJECT,
        "next_action": (
            "Do not create merge candidates from P4/P5; prefer aligned-transport "
            "research only if functional atlas (E5) shows structure"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "ranked_table.json", report)
    atomic_write_json(
        output_dir / "P4_spectral_s010" / "report.json", families["P4_spectral_s010"]
    )
    atomic_write_json(
        output_dir / "P5_direction_transfer" / "report.json",
        families["P5_direction_transfer"],
    )
    return report
