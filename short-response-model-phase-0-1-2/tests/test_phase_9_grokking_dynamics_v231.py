from __future__ import annotations

import json
from pathlib import Path

from scripts.phase_9_student_distillation.compile_grokking_report_v231 import compile_markdown, build_methods_block, PATTERN_CATALOG
from scripts.phase_9_student_distillation.detect_grokking_transitions_v231 import (
    competing_evidence,
    dose_summary,
    temporal_summary,
)
from scripts.phase_9_student_distillation.grokking_utils_v231 import (
    critical_dose_trajectory,
    discover_adapters,
    temporal_readout_summary,
)
from scripts.phase_9_student_distillation.run_grokking_factor_sweep_v231 import anchor_cli_flags

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase_9_grokking_dynamics_v231.json"


def test_v231_config_has_all_declared_axes_and_no_sealed_access():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["schema_version"] == "2.3.1"
    axes = cfg["factor_sweep"]
    for key in (
        "data_fraction",
        "unique_residual_cases",
        "residual_fraction",
        "weight_decay",
        "anchor_kl_weight",
        "learning_rate",
        "lora_rank",
        "seed",
        "target_module_group",
        "layer_scope",
    ):
        assert key in axes and axes[key]
    assert cfg["controls"]["sealed_test_access"] is False
    assert max(cfg["dose_surface"]["doses"]) > 1.0
    td = cfg["transition_detection"]
    for key in (
        "memorization_minimum_train_gain",
        "critical_dose_progress_fraction",
        "critical_dose_absolute_floor",
        "temporal_readout_accuracy_jump_fraction_minimum",
        "readout_destination_dose_minimum",
    ):
        assert key in td


def test_temporal_detector_distinguishes_delayed_ood():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["transition_detection"]
    rows = [
        {"step": 0, "train_answerable_correctness": 0.0, "dev_answerable_correctness": 0.0, "ood_answerable_correctness": 0.0},
        {"step": 100, "train_answerable_correctness": 0.8, "dev_answerable_correctness": 0.0, "ood_answerable_correctness": 0.0},
        {"step": 200, "train_answerable_correctness": 0.95, "dev_answerable_correctness": 0.01, "ood_answerable_correctness": 0.0},
        {"step": 300, "train_answerable_correctness": 0.99, "dev_answerable_correctness": 0.01, "ood_answerable_correctness": 0.0},
        {"step": 400, "train_answerable_correctness": 1.0, "dev_answerable_correctness": 0.02, "ood_answerable_correctness": 0.0},
        {"step": 500, "train_answerable_correctness": 1.0, "dev_answerable_correctness": 0.10, "ood_answerable_correctness": 0.05},
        {"step": 700, "train_answerable_correctness": 1.0, "dev_answerable_correctness": 0.40, "ood_answerable_correctness": 0.35},
    ]
    out = temporal_summary(rows, cfg)
    assert out["train_progress_step"] == 100
    assert out["ood_progress_step"] == 700
    assert out["ood_to_train_delay_ratio"] == 7.0
    assert out["grokking_like_under_primary_thresholds"] is True


def test_dose_detector_can_flag_high_dose_readout_threshold():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["transition_detection"]
    rows = []
    for dose, acc, margin in [
        (0.0, 0.0, -2.0),
        (0.25, 0.0, -1.5),
        (0.5, 0.0, -1.0),
        (0.75, 0.001, -0.4),
        (1.0, 0.06, 0.1),
    ]:
        rows.append({"step": 1000, "dose": dose, "ood_answerable_correctness": acc, "ood_first_target_margin_mean": margin})
    out = dose_summary({"rows": rows}, cfg)
    assert out["largest_ood_accuracy_jump"]["to"] == 1.0
    assert out["readout_threshold_like"] is True


def test_critical_dose_detects_leftward_migration_vs_fixed_high():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["transition_detection"]
    migrating = []
    for step, critical in [(0, 1.0), (100, 0.9), (200, 0.5), (300, 0.25)]:
        for dose in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
            acc = 0.2 if dose >= critical else 0.0
            if dose >= 1.0:
                acc = 0.2
            migrating.append({"step": step, "dose": dose, "ood_answerable_correctness": acc})
    left = critical_dose_trajectory(
        migrating,
        progress_fraction=cfg["critical_dose_progress_fraction"],
        absolute_floor=cfg["critical_dose_absolute_floor"],
        leftward_shift_minimum=cfg["critical_dose_leftward_shift_minimum"],
        fixed_high_minimum=cfg["fixed_high_critical_dose_minimum"],
    )
    assert left["critical_dose_start"] == 1.0
    assert left["critical_dose_end"] == 0.25
    assert left["critical_dose_migrates_left"] is True
    assert left["fixed_high_critical_dose"] is False

    fixed = []
    for step in (0, 100, 200, 300):
        for dose, acc in [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.08)]:
            fixed.append({"step": step, "dose": dose, "ood_answerable_correctness": acc})
    high = critical_dose_trajectory(
        fixed,
        progress_fraction=cfg["critical_dose_progress_fraction"],
        absolute_floor=cfg["critical_dose_absolute_floor"],
        leftward_shift_minimum=cfg["critical_dose_leftward_shift_minimum"],
        fixed_high_minimum=cfg["fixed_high_critical_dose_minimum"],
    )
    assert high["critical_dose_end"] == 1.0
    assert high["fixed_high_critical_dose"] is True
    assert high["critical_dose_migrates_left"] is False


def test_temporal_readout_flags_smooth_logits_with_sharp_accuracy():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["transition_detection"]
    rows = [
        {"step": 0, "ood_answerable_correctness": 0.0, "ood_first_target_margin_mean": -3.0, "ood_target_token_nll_mean": 4.0},
        {"step": 100, "ood_answerable_correctness": 0.0, "ood_first_target_margin_mean": -2.0, "ood_target_token_nll_mean": 3.2},
        {"step": 200, "ood_answerable_correctness": 0.0, "ood_first_target_margin_mean": -1.0, "ood_target_token_nll_mean": 2.4},
        {"step": 300, "ood_answerable_correctness": 0.0, "ood_first_target_margin_mean": -0.2, "ood_target_token_nll_mean": 1.6},
        {"step": 400, "ood_answerable_correctness": 0.12, "ood_first_target_margin_mean": 0.4, "ood_target_token_nll_mean": 0.8},
    ]
    out = temporal_readout_summary(
        rows,
        accuracy_jump_fraction_minimum=cfg["temporal_readout_accuracy_jump_fraction_minimum"],
        margin_linear_r2_minimum=cfg["temporal_readout_margin_linear_r2_minimum"],
        nll_linear_r2_minimum=cfg["temporal_readout_nll_linear_r2_minimum"],
    )
    assert out["sharp_ood_accuracy_jump"] is True
    assert out["smooth_ood_margin_trajectory"] is True
    assert out["smooth_logits_with_sharp_accuracy"] is True


def test_dose_jump_alone_never_yields_grokking_label():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["transition_detection"]
    time_rows = [
        {"step": 0, "train_answerable_correctness": 0.0, "dev_answerable_correctness": 0.0, "ood_answerable_correctness": 0.0},
        {"step": 100, "train_answerable_correctness": 0.02, "dev_answerable_correctness": 0.01, "ood_answerable_correctness": 0.01},
        {"step": 200, "train_answerable_correctness": 0.03, "dev_answerable_correctness": 0.02, "ood_answerable_correctness": 0.02},
    ]
    temporal = temporal_summary(time_rows, cfg)
    assert temporal["grokking_like_under_primary_thresholds"] is False
    dose_rows = []
    for dose, acc, margin in [
        (0.0, 0.0, -2.0),
        (0.25, 0.0, -1.5),
        (0.5, 0.0, -1.0),
        (0.75, 0.001, -0.4),
        (1.0, 0.06, 0.1),
    ]:
        dose_rows.append({"step": 200, "dose": dose, "ood_answerable_correctness": acc, "ood_first_target_margin_mean": margin})
    dose = dose_summary({"rows": dose_rows}, cfg)
    assert dose["readout_threshold_like"] is True
    readout = temporal_readout_summary(time_rows)
    evidence = competing_evidence(
        label="READOUT_THRESHOLD_LIKELY",
        temporal=temporal,
        dose=dose,
        temporal_readout=readout,
        representation={},
        cfg=cfg,
    )
    assert "GROKKING_LIKE_TEMPORAL" not in evidence["supporting_patterns"]
    assert temporal["grokking_like_under_primary_thresholds"] is False


def test_compiled_report_contains_required_scientific_sections():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    transition_cfg = cfg["transition_detection"]
    temporal = {
        "train_observed_gain": 0.2,
        "dev_observed_gain": 0.05,
        "ood_observed_gain": 0.04,
        "train_progress_step": 100,
        "ood_progress_step": 400,
        "ood_to_train_delay_ratio": 4.0,
        "plateau_intervals": 3,
        "grokking_like_under_primary_thresholds": True,
        "final_train_answerable": 1.0,
        "final_dev_answerable": 0.2,
        "final_ood_answerable": 0.15,
    }
    dose = {
        "readout_threshold_like": False,
        "critical_dose_start": 1.0,
        "critical_dose_end": 0.5,
        "critical_dose_leftward_shift": 0.5,
        "critical_dose_migrates_left": True,
        "fixed_high_critical_dose": False,
        "critical_dose_by_step": [
            {"step": 0, "critical_dose": 1.0, "max_ood_accuracy": 0.0, "target_accuracy": None},
            {"step": 400, "critical_dose": 0.5, "max_ood_accuracy": 0.15, "target_accuracy": 0.075},
        ],
        "largest_ood_accuracy_jump": {"from": 0.75, "to": 1.0, "accuracy_jump": 0.01},
        "ood_logit_margin_monotonic_pair_fraction": 0.8,
    }
    readout = {
        "smooth_logits_with_sharp_accuracy": False,
        "largest_ood_accuracy_jump_fraction_of_gain": 0.2,
        "sharp_ood_accuracy_jump": False,
        "ood_margin_linear_r2": 0.5,
        "smooth_ood_margin_trajectory": False,
        "ood_nll_linear_r2": 0.4,
        "smooth_ood_nll_trajectory": False,
    }
    evidence = competing_evidence(
        label="GROKKING_LIKE_TRANSITION_DETECTED",
        temporal=temporal,
        dose=dose,
        temporal_readout=readout,
        representation={"representation_precedes_ood_transition": True, "representation_progress_step_50pct": 200},
        cfg=transition_cfg,
    )
    methods = build_methods_block(
        cfg,
        {"preregistered_thresholds": {k: transition_cfg[k] for k in (
            "train_progress_fraction",
            "ood_progress_fraction",
            "minimum_plateau_intervals",
            "minimum_observed_ood_gain_for_grokking_like",
            "minimum_delay_ratio_for_grokking_like",
            "reproduced_seed_count",
            "memorization_minimum_train_gain",
            "readout_destination_dose_minimum",
            "critical_dose_progress_fraction",
            "critical_dose_absolute_floor",
            "temporal_readout_accuracy_jump_fraction_minimum",
        )}},
        {"source_counts": {}, "variant_count": 0, "probe_manifest": {}},
    )
    payload = {
        "diagnostic_label": "GROKKING_LIKE_TRANSITION_DETECTED",
        "claim_policy": cfg["experiment"]["claim_policy"],
        "methods": methods,
        "temporal": temporal,
        "temporal_readout": readout,
        "dose_surface": dose,
        "sensitivity": {
            "trials": 27,
            "positive_trials": 20,
            "positive_fraction": 20 / 27,
            "fragile": True,
            "flipping_cells": [{"train_fraction": 0.9, "ood_fraction": 0.6, "delay_ratio_threshold": 3.0, "detected": False}],
        },
        "representation": {"representation_precedes_ood_transition": True, "representation_progress_step_50pct": 200},
        "adapter_dynamics": {},
        "optimizer_dynamics": {},
        "competing_evidence": evidence,
        "factor_rows": [
            {
                "axis": "seed",
                "value": 1,
                "train_progress_step": 100,
                "ood_progress_step": 400,
                "ood_to_train_delay_ratio": 4.0,
                "grokking_like_under_primary_thresholds": True,
                "train_observed_gain": 0.2,
                "ood_observed_gain": 0.04,
            }
        ],
        "procedure_long_horizon_control": {},
        "v23_teacher_ceiling_context": {},
        "artifact_paths": {"training_time_curve": "x.json"},
        "observation_pattern_catalog": PATTERN_CATALOG,
    }
    markdown = compile_markdown(cfg=cfg, payload=payload)
    for header in (
        "## Abstract / diagnostic verdict",
        "## Experimental question and mechanism taxonomy",
        "## Methods (preregistered)",
        "### Preregistered transition thresholds",
        "## Observation pattern catalog",
        "## Results with technique↔evidence mapping",
        "## Sensitivity analysis",
        "## Competing explanations",
        "## Limitations",
        "## Artifact index",
    ):
        assert header in markdown
    assert "train_progress_fraction" in markdown
    assert str(transition_cfg["train_progress_fraction"]) in markdown
    assert "LEFTWARD_CRITICAL_DOSE" in markdown or "Leftward critical-dose migration" in markdown


def test_zero_anchor_kl_omits_anchor_model_flag():
    off = anchor_cli_flags("greedy_soup", {"anchor_kl_weight": 0.0, "anchor_temperature": 1.5})
    assert off == []
    on = anchor_cli_flags("greedy_soup", {"anchor_kl_weight": 0.3, "anchor_temperature": 1.5})
    assert on[:4] == ["--anchor-model", "greedy_soup", "--anchor-kl-weight", "0.3"]
    assert "--anchor-model" in on


def test_snapshot_discovery_prefers_lightweight_dynamics_snapshot(tmp_path: Path):
    training = tmp_path / "training"
    ckpt = training / "checkpoints/checkpoint-100"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    snap = training / "dynamics_snapshots/step_000100"
    snap.mkdir(parents=True)
    (snap / "adapter_config.json").write_text("{}", encoding="utf-8")
    final = training / "adapter"
    final.mkdir(parents=True)
    (final / "adapter_config.json").write_text("{}", encoding="utf-8")
    (training / "training_manifest.json").write_text(json.dumps({"optimizer_steps": 200}), encoding="utf-8")
    found = discover_adapters(training)
    by_step = {int(item["step"]): item for item in found}
    assert by_step[100]["kind"] == "dynamics_snapshot"
    assert by_step[200]["kind"] == "final"


def test_train_student_has_opt_in_dynamics_without_changing_default_path():
    text = (ROOT / "scripts/phase_9_student_distillation/train_student.py").read_text(encoding="utf-8")
    for token in (
        "--dynamics-log-file",
        "--dynamics-snapshot-dir",
        "TrainingDynamicsCallback",
        "AdapterSnapshotCallback",
        "--save-only-model-checkpoints",
        "slow_to_total_gradient_norm_ratio",
    ):
        assert token in text
    assert 'default=None' in text  # dynamics features are opt-in


def test_master_runner_is_unattended_and_never_opens_sealed_test():
    text = (ROOT / "scripts/phase_9_student_distillation/run_phase_9_grokking_dynamics_v231.ps1").read_text(encoding="utf-8")
    for token in (
        "Core long-horizon anchored reasoning training",
        "Evaluate every core checkpoint",
        "step x dose surface",
        "Probe cross-format representation dynamics",
        "Run one-factor grokking phase screen",
        "Procedure long-horizon contrast control",
        "Detect delayed-generalization / readout-threshold signatures",
        "Compile v2.3.1 Grokking Dynamics Observatory",
    ):
        assert token.casefold() in text.casefold()
    assert "RunFreshSealedTest" not in text
    assert "fresh_sealed" not in text.casefold()


def test_observatory_document_refuses_dose_only_grokking_claim():
    text = (ROOT / "GROKKING_DYNAMICS_V2_3_1.md").read_text(encoding="utf-8")
    assert "capability-dose threshold is not grokking" in text.casefold()
    assert "unique examples" in text.casefold()
    assert "repeated exposure" in text.casefold()
    assert "critical_dose" in text.casefold() or "critical-dose" in text.casefold()
