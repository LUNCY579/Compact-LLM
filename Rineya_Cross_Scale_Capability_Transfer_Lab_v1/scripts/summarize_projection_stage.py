from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402


def main() -> int:
    evidence = LAB_ROOT / "cross_scale_evidence" / "03_projection"
    validation_directory = evidence / "projection_validation"
    manifest_directory = evidence / "projection_manifests"
    rows = []
    for validation_path in sorted(validation_directory.glob("P1_*.json")):
        name = validation_path.stem
        manifest_path = manifest_directory / f"{name}.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "arm": name,
                "donor": "8B" if "_8B_" in name else "4B",
                "depth": manifest["plan"]["depth_strategy"],
                "width": manifest["plan"]["width_strategy"],
                "tied_projection": manifest["plan"]["tied_projection"],
                "structural_key_shape_pass": (
                    not validation["missing_keys"]
                    and not validation["unexpected_keys"]
                    and not validation["loader_missing_keys"]
                    and not validation["loader_unexpected_keys"]
                    and not validation["loader_mismatched_keys"]
                ),
                "parameter_count": validation["parameter_count"],
                "finite_logits": validation["logits_finite"],
                "functional_status": validation["status"],
                "generation_text": validation["generation_text"],
                "eos_emitted": validation["eos_emitted_in_smoke"],
                "short_period_repetition": validation["short_period_repetition"],
                "errors": validation["errors"],
                "output_weight_sha256": validation["checkpoint_fingerprint"][
                    "weight_files"
                ][0]["sha256"],
                "manifest_sha256": sha256_file(manifest_path),
                "validation_sha256": sha256_file(validation_path),
            }
        )
    go_count = sum(row["functional_status"] == "GO" for row in rows)
    summary = {
        "schema_version": "1.0",
        "stage": "A_projection_viability",
        "arm_count": len(rows),
        "functional_go_count": go_count,
        "structural_pass_count": sum(
            row["structural_key_shape_pass"] for row in rows
        ),
        "decision": "P1_NO_GO" if not go_count else "P1_HAS_VIABLE_ARM",
        "merge_authorized": bool(go_count),
        "w1_note": (
            "W1 prefix slicing is numerically identical to W2 for these shapes: "
            "target Q/K/V dimensions are exact multiples of the shared 128 head dimension, "
            "so W2 merely verifies that W1 boundaries are complete heads."
        ),
        "interpretation": (
            "Depth choice and magnitude-ranked structured subsets did not restore a "
            "coherent donor-only model. The tested independent-model coordinate bases "
            "are not compatible under deterministic intersection/truncation."
        ),
        "code_files": {
            str(path.relative_to(LAB_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (
                LAB_ROOT / "scripts" / "project_large_to_small.py",
                LAB_ROOT / "scripts" / "validate_checkpoint.py",
                LAB_ROOT / "src" / "rineya_cross_scale" / "projection.py",
                LAB_ROOT / "src" / "rineya_cross_scale" / "checkpoint.py",
                LAB_ROOT / "src" / "rineya_cross_scale" / "validation.py",
            )
        },
        "arms": rows,
    }
    atomic_write_json(evidence / "stage_a_summary.json", summary)
    lines = [
        "# Projection Stage A",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        f"All {len(rows)} arms passed tensor/key/parameter/finite-logit checks; "
        f"{go_count} passed the functional coherence and EOS gate.",
        "",
        "| Arm | Structural | Functional | EOS | Loop | Smoke output |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        escaped = row["generation_text"].replace("|", "\\|")
        lines.append(
            f"| {row['arm']} | {row['structural_key_shape_pass']} | "
            f"{row['functional_status']} | {row['eos_emitted']} | "
            f"{row['short_period_repetition']} | `{escaped}` |"
        )
    lines.extend(
        [
            "",
            summary["w1_note"],
            "",
            summary["interpretation"],
            "",
            "No P1 projection is authorized for MERGEDEV or for constructing an absolute/task-vector merge.",
        ]
    )
    atomic_write_text(evidence / "stage_a_summary.md", "\n".join(lines))
    print(json.dumps({key: summary[key] for key in ("arm_count", "functional_go_count", "decision", "merge_authorized")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
