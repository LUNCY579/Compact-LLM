from __future__ import annotations

import json
from pathlib import Path

from scripts.phase_9_student_distillation.detect_grokking_transitions_v231 import dose_summary, temporal_summary
from scripts.phase_9_student_distillation.grokking_utils_v231 import discover_adapters

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
    rows = []
    for dose, acc, margin in [
        (0.0, 0.0, -2.0),
        (0.25, 0.0, -1.5),
        (0.5, 0.0, -1.0),
        (0.75, 0.001, -0.4),
        (1.0, 0.06, 0.1),
    ]:
        rows.append({"step": 1000, "dose": dose, "ood_answerable_correctness": acc, "ood_first_target_margin_mean": margin})
    out = dose_summary({"rows": rows})
    assert out["largest_ood_accuracy_jump"]["to"] == 1.0
    assert out["readout_threshold_like"] is True


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
