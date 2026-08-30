from __future__ import annotations

import json

from rineya_cross_scale.evidence import (
    build_evidence_manifest,
    verify_evidence_manifest,
)
from rineya_cross_scale.promotion import evaluate_candidate, evaluate_two_control


def summary(overall: float, family: float, *, dataset: str = "x"):
    return {
        "dataset_sha256": dataset,
        "items": 2,
        "overall_score": overall,
        "scenario_macro_score": overall,
        "worst_family_score": family,
        "families": {"f": {"score": family}},
        "invalid_generation_rate": 0.0,
        "completion_rate": 1.0,
        "format_adherence": {"overall": 1.0},
    }


def test_promotion_requires_gain_and_preservation() -> None:
    baseline = summary(0.5, 0.5)
    assert evaluate_candidate(baseline, summary(0.6, 0.6))["eligible"]
    tied = evaluate_candidate(baseline, summary(0.5, 0.5))
    assert not tied["eligible"]
    regressed = evaluate_candidate(baseline, summary(0.7, 0.3))
    assert not regressed["eligible"]


def test_two_control_promotion_beats_both_parents() -> None:
    supa = summary(0.6, 0.6)
    consensus = summary(0.65, 0.6)
    assert not evaluate_two_control(supa, consensus, summary(0.62, 0.62))["eligible"]
    assert evaluate_two_control(supa, consensus, summary(0.7, 0.7))["eligible"]


def test_evidence_manifest_excludes_private_and_verifies(tmp_path) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "00").mkdir(parents=True)
    (evidence / "00" / "a.txt").write_text("a", encoding="utf-8")
    private = evidence / "10_sealed_eval" / "private"
    private.mkdir(parents=True)
    (private / "secret.txt").write_text("secret", encoding="utf-8")
    output = evidence / "manifest.json"
    manifest = build_evidence_manifest(evidence, output, git_head="abc")
    assert manifest["sealed_private_directory_present_but_not_read"]
    assert "10_sealed_eval/private/secret.txt" not in manifest["files"]
    assert verify_evidence_manifest(output)["status"] == "verified"
    (evidence / "00" / "a.txt").write_text("changed", encoding="utf-8")
    assert verify_evidence_manifest(output)["status"] == "failed"
