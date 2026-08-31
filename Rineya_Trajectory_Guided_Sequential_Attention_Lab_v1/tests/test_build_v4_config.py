"""The v4 registered config must be derivable, not typed.

Every number in it comes from a frozen artifact: the band, data volume, update
budget and model from the competence report; the seed count from the power
report; the statistical design copied wholesale from v3. These tests pin the
properties that make that claim true — in particular that the builder fails
closed rather than emitting a config nobody should run.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB / "scripts") not in sys.path:
    sys.path.insert(0, str(LAB / "scripts"))

from build_phase1_v4_config import V3_ANALYSIS_KEYS, build  # noqa: E402

from tgsa.phase1_competence import (  # noqa: E402
    ArmCompetence,
    CompetenceThresholds,
    evaluate_competence,
)

V3_CONFIG = LAB / "configs" / "phase1_identifiability_v3.json"
BAND = {
    "direct_lookup": {"min_nodes": 5, "max_nodes": 8, "min_hops": 1, "max_hops": 1},
    "pointer_chase": {"min_nodes": 5, "max_nodes": 8, "min_hops": 2, "max_hops": 2,
                      "reject_trivial": True},
    "state_update": {"min_steps": 2, "max_steps": 3, "reject_trivial": True},
}


def _arm(name: str, accuracy: float = 0.30) -> ArmCompetence:
    return ArmCompetence(
        arm=name,
        holdout_accuracy=accuracy,
        seen_accuracy=accuracy + 0.02,
        answer_nll=2.0,
        distinct_predictions=30,
        per_family_accuracy={"pointer_chase": accuracy, "state_update": accuracy},
        per_seed_holdout_accuracy=[accuracy - 0.02, accuracy, accuracy + 0.02],
    )


def _competence(tmp_path: Path, *, verdict_arms=None, train_examples: int = 300) -> Path:
    arms = verdict_arms or [_arm(n) for n in ("cascade", "history_control", "learned_delta")]
    report = evaluate_competence(
        arms,
        thresholds=CompetenceThresholds(
            minimum_margin_over_trivial=0.030, minimum_absolute_accuracy=0.070,
            maximum_answer_nll_fraction_of_uniform=0.97, minimum_distinct_predictions=4,
            maximum_family_accuracy=0.95, maximum_generalization_gap=0.10,
            maximum_seed_coefficient_of_variation=0.35, minimum_seed_runs=3,
        ),
        trivial_baselines={"uniform": 0.03125, "eval_majority": 0.039},
        uniform_answer_nll=math.log(32),
        primary_families=("pointer_chase", "state_update"),
        family_null={"pointer_chase": 0.0312, "state_update": 0.0312},
    )
    report["regime"] = {
        "band": "easy_nontrivial", "band_spec": BAND,
        "train_examples": train_examples, "steps": 20_000,
        "embed_dim": 96, "num_layers": 2, "num_heads": 6,
        "batch_size": 64, "learning_rate": 1e-3,
        "development_seeds": [1, 2, 3], "train_split_seed": 1_234_567,
        "holdout_split_seed": 987_654_321, "holdout_examples": 300,
        "split_generation_order": ["holdout", "train"],
    }
    path = tmp_path / "competence.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _power(tmp_path: Path, *, seed_count: int | None = 20) -> Path:
    path = tmp_path / "power.json"
    path.write_text(json.dumps({"recommended_seed_count": seed_count}), encoding="utf-8")
    return path


def _build(tmp_path: Path, **overrides):
    kwargs = dict(
        study_id="tgsa-phase1-identifiability-v4-test", dataset_seed=4_150_927,
        test_per_family=40, dev_per_family=20, length_margin=4,
    )
    kwargs.update(overrides)
    return build(
        overrides.pop("competence", None) or _competence(tmp_path),
        overrides.pop("power", None) or _power(tmp_path),
        V3_CONFIG,
        **{k: v for k, v in kwargs.items()
           if k in {"study_id", "dataset_seed", "test_per_family",
                    "dev_per_family", "length_margin"}},
    )


# --------------------------------------------------------------------------
# Fails closed
# --------------------------------------------------------------------------


def test_a_no_go_competence_report_blocks_the_build(tmp_path: Path) -> None:
    arms = [_arm(n) for n in ("cascade", "history_control", "learned_delta")]
    arms[0] = _arm("cascade", accuracy=0.04)  # a control at floor
    competence = _competence(tmp_path, verdict_arms=arms)
    with pytest.raises(ValueError, match="NO_GO"):
        build(competence, _power(tmp_path), V3_CONFIG,
              study_id="x", dataset_seed=1, test_per_family=40,
              dev_per_family=20, length_margin=4)


def test_a_missing_competence_report_blocks_the_build(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="competence report not found"):
        build(tmp_path / "absent.json", _power(tmp_path), V3_CONFIG,
              study_id="x", dataset_seed=1, test_per_family=40,
              dev_per_family=20, length_margin=4)


def test_an_unpowered_design_blocks_the_build(tmp_path: Path) -> None:
    """No adequate seed count means the study must not be registered at all."""

    with pytest.raises(ValueError, match="not powered"):
        build(_competence(tmp_path), _power(tmp_path, seed_count=None), V3_CONFIG,
              study_id="x", dataset_seed=1, test_per_family=40,
              dev_per_family=20, length_margin=4)


# --------------------------------------------------------------------------
# The statistical design is carried over, not re-derived
# --------------------------------------------------------------------------


def test_the_analysis_block_is_copied_from_v3_verbatim(tmp_path: Path) -> None:
    """RULE 1: v3's statistical design was not what failed."""

    config = _build(tmp_path)
    v3 = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    assert config["analysis"] == {k: v3["analysis"][k] for k in V3_ANALYSIS_KEYS}
    assert config["analysis"]["interval_estimator"] == "crossed_cluster_t"
    assert config["analysis"]["primary_gate"] == "lower_bound_above_zero"


def test_arms_and_claim_scope_are_unchanged(tmp_path: Path) -> None:
    config = _build(tmp_path)
    v3 = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    assert config["arms"] == v3["arms"]
    assert config["claim_scope"] == v3["claim_scope"]


# --------------------------------------------------------------------------
# The regime comes from the competence report
# --------------------------------------------------------------------------


def test_the_regime_matches_the_certified_one(tmp_path: Path) -> None:
    competence = _competence(tmp_path)
    regime = json.loads(competence.read_text(encoding="utf-8"))["regime"]
    config = _build(tmp_path, competence=competence)

    assert config["dataset"]["families"] == regime["band_spec"]
    assert config["training"]["steps"] == regime["steps"]
    assert config["training"]["batch_size"] == regime["batch_size"]
    assert config["training"]["learning_rate"] == regime["learning_rate"]
    assert config["model"]["embed_dim"] == regime["embed_dim"]
    assert config["model"]["num_layers"] == regime["num_layers"]
    assert config["model"]["num_heads"] == regime["num_heads"]
    assert all(count == regime["train_examples"] // 3
               for count in config["dataset"]["splits"]["train"].values())


def test_v4_tests_in_band(tmp_path: Path) -> None:
    """v3 made dev and test harder than train; v4 must not."""

    config = _build(tmp_path)
    assert "split_overrides" not in config["dataset"]
    v3 = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    assert "split_overrides" in v3["dataset"], "v3 fixture no longer contrasts; revisit"


def test_sequence_length_is_measured_not_guessed(tmp_path: Path) -> None:
    """A guessed length silently truncates examples out of the dataset."""

    from tgsa.phase1_data import SyntheticVocabulary, generate_split

    config = _build(tmp_path)
    vocab = SyntheticVocabulary(**config["dataset"]["vocabulary"])
    examples = generate_split(
        split="train", count_by_family=config["dataset"]["splits"]["test"],
        dataset_seed=config["dataset"]["seed"], vocab=vocab,
        generation_spec={"families": config["dataset"]["families"]},
    )
    longest = max(len(example.tokens) - 1 for example in examples)
    assert config["model"]["max_sequence_length"] >= longest
    # ...and not absurdly larger, which would waste position embeddings.
    assert config["model"]["max_sequence_length"] <= longest + 32


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_both_inputs_are_hash_bound(tmp_path: Path) -> None:
    competence = _competence(tmp_path)
    power = _power(tmp_path)
    config = build(competence, power, V3_CONFIG, study_id="x", dataset_seed=1,
                   test_per_family=40, dev_per_family=20, length_margin=4)
    registration = config["registration"]
    assert registration["competence_report_sha256"] == hashlib.sha256(
        competence.read_bytes()).hexdigest()
    assert registration["power_report_sha256"] == hashlib.sha256(
        power.read_bytes()).hexdigest()
    assert registration["competence_verdict"] == "GO"
    assert registration["supersedes_study_id"] == json.loads(
        V3_CONFIG.read_text(encoding="utf-8"))["study_id"]


def test_the_config_points_the_runner_at_the_gate(tmp_path: Path) -> None:
    """Without this key the runner never consults the gate at all."""

    competence = _competence(tmp_path)
    config = _build(tmp_path, competence=competence)
    assert config["confirmatory"] is True
    assert config["competence_report"] == competence.name


def test_seed_count_comes_from_the_power_report(tmp_path: Path) -> None:
    config = _build(tmp_path, power=_power(tmp_path, seed_count=27))
    assert len(config["seeds"]) == 27
    assert len(set(config["seeds"])) == 27, "seeds must be distinct"


def test_the_build_is_deterministic(tmp_path: Path) -> None:
    competence, power = _competence(tmp_path), _power(tmp_path)
    kwargs = dict(study_id="x", dataset_seed=1, test_per_family=40,
                  dev_per_family=20, length_margin=4)
    assert build(competence, power, V3_CONFIG, **kwargs) == build(
        competence, power, V3_CONFIG, **kwargs)


def test_the_evaluation_splits_are_generated_first(tmp_path: Path) -> None:
    """Train-first generation left test with only three-step state_update items.

    Every split excludes the signatures already taken, so the first split
    generated is the only unbiased one. On this band a 200k train split exhausts
    the entire 7,680-item two-step space.
    """

    config = _build(tmp_path)
    assert config["dataset"]["split_generation_order"] == ["test", "dev", "train"]


def test_a_report_from_the_defective_harness_cannot_build_a_config(tmp_path: Path) -> None:
    """A holdout drawn from what training left over is not the study's distribution.

    Such a report is conservative -- its holdout is harder -- so it is valid
    evidence that a regime FAILS. It is not a certificate to dispatch on.
    """

    competence = _competence(tmp_path)
    payload = json.loads(competence.read_text(encoding="utf-8"))
    payload["regime"]["split_generation_order"] = ["train", "holdout"]
    competence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split_generation_order"):
        build(competence, _power(tmp_path), V3_CONFIG, study_id="x", dataset_seed=1,
              test_per_family=40, dev_per_family=20, length_margin=4)


def test_a_report_with_no_recorded_order_cannot_build_a_config(tmp_path: Path) -> None:
    """Silence is not a certificate. Older reports predate the field entirely."""

    competence = _competence(tmp_path)
    payload = json.loads(competence.read_text(encoding="utf-8"))
    payload["regime"].pop("split_generation_order")
    competence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split_generation_order"):
        build(competence, _power(tmp_path), V3_CONFIG, study_id="x", dataset_seed=1,
              test_per_family=40, dev_per_family=20, length_margin=4)


# --------------------------------------------------------------------------
# The development_only marker must travel, and must not be launderable
# --------------------------------------------------------------------------


def _mark(competence: Path) -> Path:
    payload = json.loads(competence.read_text(encoding="utf-8"))
    payload["regime"]["development_only"] = True
    competence.write_text(json.dumps(payload), encoding="utf-8")
    return competence


def test_a_marked_report_cannot_build_a_config_by_default(tmp_path: Path) -> None:
    competence = _mark(_competence(tmp_path))
    with pytest.raises(ValueError, match="development_only"):
        build(competence, _power(tmp_path), V3_CONFIG, study_id="x", dataset_seed=1,
              test_per_family=40, dev_per_family=20, length_margin=4)


def test_a_marked_report_produces_a_marked_config(tmp_path: Path) -> None:
    """The rehearsal opts in; the marker must then propagate, not vanish."""

    competence = _mark(_competence(tmp_path))
    config = build(competence, _power(tmp_path), V3_CONFIG, study_id="x", dataset_seed=1,
                   test_per_family=40, dev_per_family=20, length_margin=4,
                   allow_development_only=True)
    assert config["development_only"] is True
    assert "development_only_reason" in config["registration"]


def test_opting_in_cannot_launder_a_marked_report_into_a_clean_config(tmp_path: Path) -> None:
    """There must be no way to get an unmarked config from a marked report."""

    competence = _mark(_competence(tmp_path))
    config = build(competence, _power(tmp_path), V3_CONFIG, study_id="x", dataset_seed=1,
                   test_per_family=40, dev_per_family=20, length_margin=4,
                   allow_development_only=True)
    assert config.get("development_only") is True


def test_an_unmarked_report_never_marks_the_config(tmp_path: Path) -> None:
    """The opt-in flag must not mark a config that did not earn the marker."""

    config = build(_competence(tmp_path), _power(tmp_path), V3_CONFIG, study_id="x",
                   dataset_seed=1, test_per_family=40, dev_per_family=20,
                   length_margin=4, allow_development_only=True)
    assert "development_only" not in config
