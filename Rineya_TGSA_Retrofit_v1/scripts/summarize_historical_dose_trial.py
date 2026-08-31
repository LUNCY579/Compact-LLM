#!/usr/bin/env python3
"""Evaluate the frozen 1.0625 dose candidate against exact A1.1 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_historical_soup_trial import (
    DATA,
    EVALUATOR,
    filter_predictions,
    generation_health,
    key_ids,
    report_metrics,
    run_evaluator,
    sha256_file,
)

LAB = Path(__file__).resolve().parents[1]
FIRST = LAB / "artifacts" / "historical_soup_a11_baya_midpoint_v1" / "evidence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB / "configs" / "historical_dose_a11_1p0625_trial.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evidence = (LAB / config["outputs"]["evidence"]).resolve()
    candidate_dir = (LAB / config["outputs"]["candidate"]).resolve()
    candidate_all = evidence / "candidate_all256.jsonl"
    if not candidate_all.is_file():
        raise FileNotFoundError(candidate_all)
    candidate_all_report = run_evaluator(
        DATA / "all_answer_key.jsonl",
        candidate_all,
        "candidate_all256",
        evidence_dir=evidence,
    )
    candidate = {"all256": report_metrics(candidate_all_report)}
    parent = {"all256": report_metrics(FIRST / "parent_all256_report.json")}
    for split in ("showcase", "stability"):
        key = DATA / f"{split}_answer_key.jsonl"
        filtered = evidence / f"candidate_{split}.jsonl"
        filter_predictions(candidate_all, key_ids(key), filtered)
        candidate[split] = report_metrics(
            run_evaluator(key, filtered, f"candidate_{split}", evidence_dir=evidence)
        )
        parent[split] = report_metrics(FIRST / f"parent_{split}_report.json")

    fidelity = json.loads((evidence / "control_fidelity.json").read_text(encoding="utf-8"))
    smoke = json.loads((evidence / "fresh_process_smoke.json").read_text(encoding="utf-8"))
    checks = {
        "control_reconstruction_bitwise_exact": fidelity.get("passed") is True,
        "strict_all256_gain": (
            candidate["all256"]["strict_exact_hits"]
            >= parent["all256"]["strict_exact_hits"] + 1
        ),
        "stability_preserved": (
            candidate["stability"]["strict_exact_hits"]
            >= parent["stability"]["strict_exact_hits"]
        ),
        "all_four_packs_preserved": (
            candidate["all256"]["all_four_packs"] >= parent["all256"]["all_four_packs"]
        ),
        "base_paraphrase_packs_preserved": (
            candidate["all256"]["base_paraphrase_both_correct_packs"]
            >= parent["all256"]["base_paraphrase_both_correct_packs"]
        ),
        "coverage_and_identity": all(
            candidate[split]["coverage"] == 1.0
            and not candidate[split]["duplicate_prediction_ids"]
            and not candidate[split]["unexpected_prediction_ids"]
            for split in ("showcase", "stability", "all256")
        ),
        "candidate_generation_health": generation_health(candidate_all)["passed"],
        "fresh_process_reload": smoke.get("status") == "PASS",
        "artifact_provenance_complete": all(
            path.is_file()
            for path in (
                evidence / "preflight.json",
                evidence / "control_fidelity.json",
                evidence / "build_result.json",
                candidate_dir / "DOSE_ARTIFACT.json",
            )
        ),
    }
    deltas = {
        split: {
            key: candidate[split][key] - parent[split][key]
            for key in (
                "strict_exact_hits",
                "all_four_packs",
                "base_paraphrase_both_correct_packs",
                "counterfactual_pair_packs",
                "unanswerable_hits",
            )
        }
        for split in ("showcase", "stability", "all256")
    }
    passed = all(checks.values())
    outcome = {
        "schema_version": "rineya.historical-dose-transfer-outcome.v1",
        "trial_id": config["trial_id"],
        "registration_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(LAB.parent / config["evaluation"]["runner"]["path"]),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "parent_predictions_reused_from": config["evaluation"][
            "reuse_exact_parent_predictions_from_trial"
        ],
        "reports": {"parent": parent, "candidate": candidate},
        "deltas": deltas,
        "control_fidelity": fidelity,
        "candidate_generation_health": generation_health(candidate_all),
        "transfer_gate": checks,
        "transfer_result": "TRANSFER_POSITIVE" if passed else "TRANSFER_FAIL",
    }
    (evidence / "transfer_outcome.json").write_text(
        json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"result": outcome["transfer_result"], "checks": checks, "deltas": deltas},
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
