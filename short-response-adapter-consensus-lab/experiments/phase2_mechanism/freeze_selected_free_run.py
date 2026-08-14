from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


LAB_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from consensus_lab.io import canonical_json_sha256, read_json, sha256_file, write_json_atomic  # noqa: E402


ANALYSIS = LAB_ROOT / "results/phase2_mechanism/teacher_forced_dev_analysis.json"
CAPTURE_MANIFEST = LAB_ROOT / "artifacts/phase2_mechanism/teacher_forced_dev_capture_topk64_manifest.json"
SOURCE_CONFIG = LAB_ROOT / "configs/fusion_sweep.yaml"
OUTPUT_CONFIG = LAB_ROOT / "configs/phase2_selected_dev_free_run.yaml"
OUTPUT_MANIFEST = LAB_ROOT / "artifacts/phase2_mechanism/selected_dev_free_run_manifest.json"
PARITY_RESULT = LAB_ROOT / "results/summaries/cache_parity_phase2_selected_dev_free_run.json"
RANK_METHODS = ("top2_approval", "borda_top5", "reciprocal_rank_top5")
ANCHOR_METHODS = ("hard_plurality", "probability_mean", "calibrated_probability_mean")


def committee_row(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    matches = [row for row in rows if int(row["mask"]) == 127 and row["method"] == method]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one full-K7 row for {method}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze DEV-selected methods and calibration for Phase 2 free runs.")
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    args = parser.parse_args()
    analysis = read_json(args.analysis)
    if analysis.get("status") not in {
        "DEV_TEACHER_FORCED_ANALYSIS_COMPLETE",
        "DEV_TEACHER_FORCED_ANALYSIS_COMPLETE_WITH_UNCERTIFIED_SOFT_POSITIONS",
    }:
        raise RuntimeError("Refusing selection from a partial teacher-forced analysis")
    if int(analysis.get("example_count", 0)) != 4_050:
        raise RuntimeError("Selection requires exactly 4,050 DEV examples")

    rows = analysis["committee_results"]
    rank_rows = [committee_row(rows, method) for method in RANK_METHODS]
    selected_rank = min(
        rank_rows,
        key=lambda row: (-float(row["example_accuracy"]), -float(row["token_accuracy"]), str(row["method"])),
    )
    selected_methods = [*ANCHOR_METHODS, str(selected_rank["method"])]
    selected_rows = {method: committee_row(rows, method) for method in selected_methods}
    uncertified = {
        method: row["candidate_argmax_certification_rate"]
        for method, row in selected_rows.items()
        if float(row["candidate_argmax_certification_rate"]) < 1.0
    }
    if uncertified:
        raise RuntimeError(f"Refusing to freeze uncertified selected rules: {uncertified}")

    source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8")) or {}
    temperatures = {
        expert: float(packet["temperature"])
        for expert, packet in analysis["calibration"]["experts"].items()
    }
    experts = []
    for expert in source["experts"]:
        packet = dict(expert)
        packet["temperature"] = temperatures[str(packet["name"])]
        experts.append(packet)
    config = {
        "name": "phase2_selected_dev_free_run",
        "backbone": source["backbone"],
        "experts": experts,
        "committees": [
            {
                "name": "full_k7",
                "experts": [
                    "track_a",
                    "track_b",
                    "general",
                    "grounding",
                    "procedure",
                    "abstention",
                    "social",
                ],
                "role": "dev_selected_mechanism_confirmation",
            }
        ],
        "methods": selected_methods,
        "datasets": {
            "v22_dev": "data/phase_9_student_distillation/multidomain_reasoning_v22/reviewed_splits/dev.reviewed.jsonl"
        },
        "max_sequence_length": int(source.get("max_sequence_length", 512)),
        "max_new_tokens": int(source.get("max_new_tokens", 32)),
        "cache_mode": "reference",
        "eos_policy": "ordinary",
        "seed": int(source.get("seed", 20260811)),
    }
    config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFIG.write_text(config_text, encoding="utf-8")

    capture = read_json(CAPTURE_MANIFEST)
    parity = read_json(PARITY_RESULT) if PARITY_RESULT.is_file() else None
    manifest = {
        "schema_version": "1.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV MODEL-SELECTION / CALIBRATION",
        "selection_rule_declared_in": str(Path(__file__).resolve()),
        "selection_rule": {
            "anchors": list(ANCHOR_METHODS),
            "rank_family": list(RANK_METHODS),
            "rank_choice": "highest full-K7 all-gold-token example accuracy; then token accuracy; then method name",
            "certification_gate": "every selected cached rule must have candidate_argmax_certification_rate == 1.0",
            "temperature_fit": "independent per-expert scalar minimizing teacher-forced DEV NLL on the frozen grid",
        },
        "selected_methods": selected_methods,
        "selected_rank_method": selected_rank["method"],
        "selected_teacher_forced_rows": selected_rows,
        "fitted_temperatures": temperatures,
        "analysis_path": str(args.analysis.resolve()),
        "analysis_sha256": sha256_file(args.analysis),
        "analysis_example_count": analysis["example_count"],
        "analysis_token_count": analysis["token_count"],
        "capture_manifest_path": str(CAPTURE_MANIFEST.resolve()),
        "capture_manifest_sha256": sha256_file(CAPTURE_MANIFEST),
        "capture_config_sha256": capture["capture_config_sha256"],
        "source_config_path": str(SOURCE_CONFIG.resolve()),
        "source_config_sha256": sha256_file(SOURCE_CONFIG),
        "output_config_path": str(OUTPUT_CONFIG.resolve()),
        "output_config_sha256": sha256_file(OUTPUT_CONFIG),
        "output_config_canonical_sha256": canonical_json_sha256(config),
        "test_data_read_for_selection": False,
        "decoder_choice": {
            "cache_mode": "reference",
            "reason": (
                "The optimized separate-cache decoder is not used for paper-facing runs because its Phase 2 parity gate failed."
                if parity and parity.get("status") != "PASS"
                else "Exact full-prefix reference decoding is the conservative default."
            ),
            "parity_result_path": str(PARITY_RESULT.resolve()) if parity else None,
            "parity_result_sha256": sha256_file(PARITY_RESULT) if parity else None,
            "parity_status": parity.get("status") if parity else "NOT_RUN",
            "parity_token_match_rate": parity.get("token_match_rate") if parity else None,
            "parity_trace_match_rate": parity.get("decision_trace_match_rate") if parity else None,
        },
        "notes": [
            "Hard and raw probability are fixed anchors, not DEV-selected winners.",
            "Calibrated probability is included as the pre-specified calibration-mechanism test.",
            "Only the rank-family representative is selected by DEV outcome.",
            "The failed optimized-cache parity result is retained as negative evidence; reference decoding is frozen.",
            "Free-running exact match remains distinct from teacher-forced all-token example accuracy.",
        ],
    }
    write_json_atomic(OUTPUT_MANIFEST, manifest)
    print(
        json.dumps(
            {
                "status": "PHASE2_DEV_FREE_RUN_SELECTION_FROZEN",
                "methods": selected_methods,
                "temperatures": temperatures,
                "config": str(OUTPUT_CONFIG),
                "manifest": str(OUTPUT_MANIFEST),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
