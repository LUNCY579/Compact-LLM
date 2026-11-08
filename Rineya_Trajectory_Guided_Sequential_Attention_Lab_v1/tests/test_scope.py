"""A competence report must be shown to be about the study it certifies.

Before `tgsa/scope.py` the only gate was `require_go(external_report)`, which
checks that a report exists, parses, is not development-only, and says GO. It
cannot check what the report is *about*, so a Phase-1 GO earned on one band would
certify a Phase-2 study on a different generator, depth, or objective, and
nothing downstream could tell.

That is the v4 failure mode transplanted forward: v4's regime was rejected
because the families did not move *on the scope being measured*, and a
scope-blind gate cannot detect that condition in a later study however rigorous
the earlier one was.

Every test here is a way a report about different science could be accepted.
"""

from __future__ import annotations

import copy

import pytest

from tgsa.scope import (
    SCOPE_FIELDS,
    ScopeMismatchError,
    compute_scope,
    environment_lock,
    require_scope_match,
    sha256_json,
)

STUDY: dict = {
    "arms": ["cascade", "history_control", "learned_delta"],
    "dataset": {
        "vocabulary": {"domain_size": 32, "max_hops": 8, "max_abs_step": 8},
        "families": {
            "direct_lookup": {"min_nodes": 5, "max_nodes": 8, "min_hops": 1, "max_hops": 1},
            "pointer_chase": {"min_nodes": 5, "max_nodes": 8, "min_hops": 2, "max_hops": 2},
            "state_update": {"min_steps": 2, "max_steps": 3},
        },
        "split_generation_order": ["test", "dev", "train"],
        "splits": {"train": {"direct_lookup": 16666}, "test": {"direct_lookup": 2000}},
    },
    "model": {
        "embed_dim": 96, "num_heads": 6, "num_layers": 2, "ffn_multiplier": 4.0,
        "trajectory_rank": 16, "tie_embeddings": True, "bias": False,
        "dropout": 0.0, "max_sequence_length": 35,
    },
    "training": {
        "learning_rate": 1e-3, "batch_size": 64, "beta1": 0.9, "beta2": 0.95,
        "weight_decay": 0.01, "gradient_clip_norm": 1.0, "epsilon": 1e-8,
    },
    "analysis": {
        "primary_families": ["pointer_chase", "state_update"],
        "guardrail_family": "direct_lookup",
        "interval_estimator": "crossed_cluster_t",
        "primary_gate": "lower_bound_above_zero",
        "confidence": 0.95,
    },
}


def _report(scope: dict) -> dict:
    return {"verdict": "GO", "scope": scope}


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_a_report_on_the_same_scope_is_accepted() -> None:
    scope = compute_scope(STUDY)
    result = require_scope_match(_report(scope), scope)
    assert set(result["matched_fields"]) == set(SCOPE_FIELDS)
    assert result["environment_differs"] is False


def test_every_canonical_identifier_is_produced() -> None:
    """The canonical path names five; a subset is not acceptable."""

    scope = compute_scope(STUDY)
    for field in SCOPE_FIELDS:
        assert isinstance(scope[field], str) and len(scope[field]) == 64, field


def test_the_scope_is_stable_across_calls() -> None:
    assert compute_scope(STUDY)["task_scope_sha256"] == compute_scope(STUDY)["task_scope_sha256"]


# --------------------------------------------------------------------------
# Every scientific change must be detected
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,value,field", [
    # Task: the generator, the families, the band, the split policy.
    (("dataset", "families", "pointer_chase", "max_hops"), 4, "task_scope_sha256"),
    (("dataset", "vocabulary", "domain_size"), 64, "task_scope_sha256"),
    (("dataset", "split_generation_order"), ["train", "dev", "test"], "task_scope_sha256"),
    (("dataset", "splits", "train", "direct_lookup"), 66666, "task_scope_sha256"),
    # Model: depth and capacity, the axis v4 never ran.
    (("model", "num_layers"), 4, "model_scope_sha256"),
    (("model", "embed_dim"), 192, "model_scope_sha256"),
    (("model", "trajectory_rank"), 32, "model_scope_sha256"),
    # Objective: what is optimized.
    (("training", "learning_rate"), 3e-4, "objective_scope_sha256"),
    (("training", "batch_size"), 128, "objective_scope_sha256"),
    # Competence: the metric and the decision rule.
    (("analysis", "primary_families"), ["pointer_chase"], "competence_scope_sha256"),
    (("analysis", "interval_estimator"), "seed_cluster", "competence_scope_sha256"),
    (("analysis", "primary_gate"), "conjunction", "competence_scope_sha256"),
])
def test_a_scientific_change_changes_the_scope(path, value, field) -> None:
    changed = copy.deepcopy(STUDY)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    original = compute_scope(STUDY)
    altered = compute_scope(changed)
    assert altered[field] != original[field], f"{'.'.join(map(str, path))} left {field} unchanged"

    with pytest.raises(ScopeMismatchError, match="different science"):
        require_scope_match(_report(original), altered)


def test_adding_an_auxiliary_objective_changes_the_objective_scope() -> None:
    """An MTP head is a different objective. This is the substitution the
    canonical path explicitly forbids: a Phase-1 GO earned without the auxiliary
    loss cannot certify a Phase-2 study that adds one."""

    without = compute_scope(STUDY)
    with_mtp = compute_scope(STUDY, auxiliary={"mtp_horizons": 4, "weight": 0.5})
    assert with_mtp["objective_scope_sha256"] != without["objective_scope_sha256"]
    with pytest.raises(ScopeMismatchError, match="different science"):
        require_scope_match(_report(without), with_mtp)


def test_renaming_a_family_is_detected_even_at_identical_parameters() -> None:
    changed = copy.deepcopy(STUDY)
    changed["dataset"]["families"]["pointer_walk"] = changed["dataset"]["families"].pop("pointer_chase")
    assert compute_scope(changed)["task_scope_sha256"] != compute_scope(STUDY)["task_scope_sha256"]


def test_changing_the_arms_changes_the_objective_scope() -> None:
    changed = copy.deepcopy(STUDY)
    changed["arms"] = ["cascade", "learned_delta"]
    assert (compute_scope(changed)["objective_scope_sha256"]
            != compute_scope(STUDY)["objective_scope_sha256"])


def test_arm_order_alone_does_not_change_the_scope() -> None:
    """Reordering a list is not a scientific change; scope must not be noisy."""

    changed = copy.deepcopy(STUDY)
    changed["arms"] = list(reversed(STUDY["arms"]))
    assert (compute_scope(changed)["objective_scope_sha256"]
            == compute_scope(STUDY)["objective_scope_sha256"])


# --------------------------------------------------------------------------
# Fails closed
# --------------------------------------------------------------------------


def test_a_report_with_no_scope_block_is_refused() -> None:
    with pytest.raises(ScopeMismatchError, match="no `scope` block"):
        require_scope_match({"verdict": "GO"}, compute_scope(STUDY))


def test_a_report_missing_an_identifier_is_refused() -> None:
    """A subset of the five is not a scope."""

    scope = compute_scope(STUDY)
    partial = {k: v for k, v in scope.items() if k != "competence_scope_sha256"}
    with pytest.raises(ScopeMismatchError, match="competence_scope_sha256"):
        require_scope_match(_report(partial), scope)


def test_an_empty_identifier_is_refused_not_defaulted() -> None:
    scope = compute_scope(STUDY)
    blanked = dict(scope, model_scope_sha256="")
    with pytest.raises(ScopeMismatchError, match="model_scope_sha256"):
        require_scope_match(_report(blanked), scope)


@pytest.mark.parametrize("missing", ["dataset", "model", "training", "arms"])
def test_an_incomplete_study_cannot_produce_a_scope(missing: str) -> None:
    incomplete = {k: v for k, v in STUDY.items() if k != missing}
    with pytest.raises(ScopeMismatchError):
        compute_scope(incomplete)


# --------------------------------------------------------------------------
# The environment escape hatch must not relax the science
# --------------------------------------------------------------------------


def test_an_environment_difference_alone_can_be_acknowledged() -> None:
    ours = compute_scope(STUDY, device={"device_type": "cuda"})
    theirs = compute_scope(STUDY, device={"device_type": "cpu"})
    assert theirs["environment_lock_sha256"] != ours["environment_lock_sha256"]

    with pytest.raises(ScopeMismatchError, match="environment_lock_sha256"):
        require_scope_match(_report(theirs), ours)

    result = require_scope_match(_report(theirs), ours, ignore_environment=True)
    assert result["environment_differs"] is True
    assert result["environment_ignored"] is True


def test_ignoring_the_environment_does_not_ignore_the_science() -> None:
    """The escape hatch must not become a way past a real mismatch."""

    changed = copy.deepcopy(STUDY)
    changed["model"]["num_layers"] = 4
    with pytest.raises(ScopeMismatchError, match="model_scope_sha256"):
        require_scope_match(
            _report(compute_scope(STUDY, device={"device_type": "cpu"})),
            compute_scope(changed, device={"device_type": "cuda"}),
            ignore_environment=True,
        )


def test_the_environment_lock_records_the_interpreter_and_torch() -> None:
    lock = environment_lock()
    assert "python" in lock and "torch" in lock
    assert sha256_json(lock) == sha256_json(environment_lock())


# --------------------------------------------------------------------------
# Phase 1 declares `arms`; Phase 2 declares `cells`. Both are compared conditions.
# --------------------------------------------------------------------------


def test_phase2_cells_are_accepted_as_compared_conditions() -> None:
    """Imposing Phase 1's shape on a Phase 2 config would make the gate unusable
    exactly where it is most needed."""

    study = copy.deepcopy(STUDY)
    del study["arms"]
    study["cells"] = [
        {"tgsa_arm": "cascade", "mtp_mode": "dormant"},
        {"tgsa_arm": "cascade", "mtp_mode": "trajectory"},
    ]
    scope = compute_scope(study)
    assert len(scope["objective_scope_sha256"]) == 64


def test_changing_a_cell_changes_the_objective_scope() -> None:
    base = copy.deepcopy(STUDY)
    del base["arms"]
    base["cells"] = [{"tgsa_arm": "cascade", "mtp_mode": "dormant"}]
    changed = copy.deepcopy(base)
    changed["cells"] = [{"tgsa_arm": "cascade", "mtp_mode": "trajectory"}]
    assert (compute_scope(changed)["objective_scope_sha256"]
            != compute_scope(base)["objective_scope_sha256"])


def test_a_study_comparing_nothing_is_refused() -> None:
    study = copy.deepcopy(STUDY)
    del study["arms"]
    with pytest.raises(ScopeMismatchError, match="neither `arms` nor `cells`"):
        compute_scope(study)
