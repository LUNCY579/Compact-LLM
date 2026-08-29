"""Regression tests for every launch-audit finding remediated in study v2.

Each test names the finding it locks down. These are the negative tests the
pre-launch acceptance gates require: a guard that is never shown to *reject*
anything is not a guard.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import torch

from tgsa.config import TGSAConfig
from tgsa.module import TrajectoryEncoder, TrajectoryGuidedSequentialAttention
from tgsa.phase1_analysis import (
    BOOTSTRAP_SCHEMES,
    crossed_paired_bootstrap,
    resolve_bootstrap,
    seed_cluster_bootstrap,
)
from tgsa.phase1_model import (
    CANONICAL_SUPERNET_ARM,
    ControlCapture,
    ReplayIntervention,
    TinyLMConfig,
    build_canonical_supernet,
    build_model,
    length_matched_derangement,
    paired_initialization_audit,
    state_dict_sha256,
)
from tgsa.phase1_runner import (
    _record_failed_attempt,
    _replay_groups,
    _truncate_train_log,
    load_study_config,
    validate_completed_run_reuse,
    write_jsonl,
)

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
V2_CONFIG = LAB_ROOT / "configs" / "phase1_identifiability_v2.json"
V2_SMOKE_CONFIG = LAB_ROOT / "configs" / "phase1_identifiability_v2_smoke.json"


def _config(arm: str = "learned_delta") -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=64,
        max_sequence_length=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ffn_multiplier=2.0,
        dropout=0.0,
        bias=False,
        trajectory_rank=8,
        arm=arm,  # type: ignore[arg-type]
        tie_embeddings=True,
    )


# --------------------------------------------------------------------------
# P0-1 — the packaging extra CI installs must actually exist
# --------------------------------------------------------------------------


def test_p0_1_every_extra_the_workflow_installs_is_declared() -> None:
    workflow = REPO_ROOT / ".github" / "workflows" / "tgsa-phase1.yml"
    if not workflow.is_file():
        pytest.skip("lab consumed standalone; repo-root workflow is not present")
    declared = set(
        tomllib.loads((LAB_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]
    )
    requested = {
        extra.strip()
        for match in re.finditer(r"pip install[^\n]*\[([^\]]+)\]", workflow.read_text(encoding="utf-8"))
        for extra in match.group(1).split(",")
        # skip `--index-url .../whl/cpu "torch>=2.1"`-style lines with no extra
        if extra.strip() and "/" not in extra
    }
    assert requested, "workflow no longer installs the package with an extra"
    assert requested <= declared, f"workflow requests undeclared extras: {sorted(requested - declared)}"


def test_p0_1_runtime_dependencies_cover_the_evidence_hash_path() -> None:
    """`state_dict_sha256` hashes through `Tensor.numpy()`.

    torch declares numpy as optional and the CPU-only wheel index does not pull
    it, so the whole evidence chain silently depended on an undeclared package.
    CI surfaced this as `RuntimeError: Numpy is not available`.
    """

    project = tomllib.loads((LAB_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {re.split(r"[<>=!\[]", item)[0].strip() for item in project["dependencies"]}
    assert {"torch", "numpy"} <= names, f"runtime dependencies are incomplete: {sorted(names)}"


def test_p0_1_install_probe_covers_every_import_the_pipeline_needs() -> None:
    workflow = REPO_ROOT / ".github" / "workflows" / "tgsa-phase1.yml"
    if not workflow.is_file():
        pytest.skip("lab consumed standalone; repo-root workflow is not present")
    text = workflow.read_text(encoding="utf-8")
    for name in ("pytest", "jsonschema", "numpy", "torch"):
        assert f'"{name}"' in text, f"install probe does not fail closed on missing {name}"


def test_p0_1_dev_and_test_extras_both_provide_pytest() -> None:
    extras = tomllib.loads((LAB_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    for name in ("dev", "test"):
        assert name in extras, f"missing {name!r} extra"
        assert any(item.startswith("pytest") for item in extras[name])


# --------------------------------------------------------------------------
# P0-2 — every arm derives from one canonical supernet
# --------------------------------------------------------------------------


def test_p0_2_every_arm_is_a_byte_exact_restriction_of_the_canonical_supernet() -> None:
    seed = 4242
    canonical = build_canonical_supernet(_config(), initialization_seed=seed)
    canonical_state = canonical.state_dict()
    for arm in ("cascade", "history_control", "learned_delta"):
        model = build_model(_config(arm), initialization_seed=seed)
        state = model.state_dict()
        shared = [
            key
            for key in state.keys() & canonical_state.keys()
            if state[key].shape == canonical_state[key].shape
            and state[key].dtype == canonical_state[key].dtype
        ]
        assert shared
        assert all(torch.equal(state[key], canonical_state[key]) for key in shared), arm


def test_p0_2_shared_initialization_holds_for_all_five_registered_seeds() -> None:
    study = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    for seed in study["seeds"]:
        audit = paired_initialization_audit(base_config=_config(), initialization_seed=int(seed))
        assert audit["all_arms_match_canonical_shared_tensors"] is True
        assert audit["history_delta_exact_initialization"] is True
        assert audit["cascade_delta_shared_initialization"]["identical"] is True
        assert audit["canonical_supernet_arm"] == CANONICAL_SUPERNET_ARM


def test_p0_2_registration_order_change_cannot_silently_break_pairing() -> None:
    """A structural change must fail loudly, not diverge silently.

    The pre-remediation path relied on coincident RNG order, so adding a module
    reordered the random stream and quietly desynchronized the arms. Copying by
    key makes an incompatible structure visible instead.
    """

    seed = 77
    canonical = build_canonical_supernet(_config(), initialization_seed=seed)
    state = dict(canonical.state_dict())
    victim = next(key for key in state if key.endswith("trajectory_encoders.0.down.weight"))
    state[victim] = torch.zeros_like(state[victim])
    rebuilt = build_model(_config("learned_delta"), initialization_seed=seed, canonical_state=state)
    assert torch.equal(rebuilt.state_dict()[victim], torch.zeros_like(state[victim]))
    assert state_dict_sha256(rebuilt) != state_dict_sha256(canonical)


def test_p0_2_global_apply_does_not_clobber_the_trajectory_encoder_init() -> None:
    model = build_model(_config("learned_delta"), initialization_seed=11)
    ups = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("trajectory_encoders.0.up.weight")
    ]
    assert ups
    for parameter in ups:
        assert torch.count_nonzero(parameter) == 0, (
            "model-wide apply() overwrote the zero-initialized residual factor"
        )


# --------------------------------------------------------------------------
# P0-3 — stale completed runs are rejected
# --------------------------------------------------------------------------


def _write_completed_manifest(root: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "status": "complete",
        "arm": "learned_delta",
        "seed": 7,
        "study_config_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "git": {"head": "c" * 40},
    }
    payload.update(overrides)
    run_dir = root / "runs" / str(payload["arm"]) / f"seed_{payload['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_p0_3_incomplete_runs_are_left_alone_for_resume(tmp_path: Path) -> None:
    _write_completed_manifest(tmp_path, status="running")
    audit = validate_completed_run_reuse(
        output_root=tmp_path,
        arms=["learned_delta"],
        seeds=[7],
        study_config_sha256="z" * 64,
        dataset_manifest_sha256="z" * 64,
        current_git_head="z" * 40,
    )
    assert audit["reusable_run_count"] == 0


def test_p0_3_trained_but_unevaluated_runs_are_also_guarded(tmp_path: Path) -> None:
    _write_completed_manifest(tmp_path, status="trained")
    with pytest.raises(ValueError, match="stale completed run"):
        validate_completed_run_reuse(
            output_root=tmp_path,
            arms=["learned_delta"],
            seeds=[7],
            study_config_sha256="d" * 64,
            dataset_manifest_sha256="b" * 64,
            current_git_head="c" * 40,
        )


# --------------------------------------------------------------------------
# P0-4 — a non-confirmatory configuration can never be evidentiary
# --------------------------------------------------------------------------


def test_p0_4_study_config_requires_an_explicit_confirmatory_boolean(tmp_path: Path) -> None:
    study = json.loads(V2_SMOKE_CONFIG.read_text(encoding="utf-8"))
    del study["confirmatory"]
    path = tmp_path / "no_flag.json"
    path.write_text(json.dumps(study), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmatory"):
        load_study_config(path)

    study["confirmatory"] = "false"
    path.write_text(json.dumps(study), encoding="utf-8")
    with pytest.raises(TypeError, match="confirmatory"):
        load_study_config(path)


def test_p0_4_registered_and_smoke_configs_declare_opposite_intent() -> None:
    assert load_study_config(V2_CONFIG)["confirmatory"] is True
    assert load_study_config(V2_SMOKE_CONFIG)["confirmatory"] is False


def test_p0_4_analysis_schema_requires_the_confirmatory_field() -> None:
    schema = json.loads(
        (LAB_ROOT / "evidence" / "phase1_analysis.schema.json").read_text(encoding="utf-8")
    )
    assert "confirmatory_config" in schema["required"]


# --------------------------------------------------------------------------
# P0-6 — the control path is information-matched
# --------------------------------------------------------------------------


def test_p0_6_learned_encoder_is_exactly_the_identity_at_initialization() -> None:
    torch.manual_seed(0)
    encoder = TrajectoryEncoder(16, 8, "learned", False)
    value = torch.randn(3, 5, 16)
    assert torch.equal(encoder(value), value)


def test_p0_6_raw_control_survives_an_arbitrary_learned_residual() -> None:
    torch.manual_seed(1)
    encoder = TrajectoryEncoder(16, 8, "learned", False)
    with torch.no_grad():
        encoder.up.weight.normal_(0.0, 0.5)
        encoder.down.weight.normal_(0.0, 0.5)
    value = torch.randn(4, 6, 16)
    residual = encoder.up(encoder.activation(encoder.down(encoder.norm(value))))
    torch.testing.assert_close(encoder(value) - residual, value, atol=1.0e-6, rtol=1.0e-6)


def test_p0_6_history_mean_and_velocity_are_affine_bijections_given_the_carrier() -> None:
    """The exact property the preregistration claims.

    Given the current carrier ``c_j``, the raw velocity ``d_j`` and the raw
    history mean ``m_j`` each recover ``c_{j-1}`` exactly, so the two arms have
    algebraically equivalent access to the previous carrier.
    """

    torch.manual_seed(2)
    previous, current = torch.randn(5, 12), torch.randn(5, 12)
    velocity = current - previous
    history_mean = 0.5 * (current + previous)
    torch.testing.assert_close(current - velocity, previous, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(2.0 * history_mean - current, previous, atol=1.0e-6, rtol=1.0e-6)


def test_p0_6_both_arms_use_the_same_encoder_parameterization() -> None:
    history = TGSAConfig.for_ablation(embed_dim=32, num_heads=4, arm="history_control")
    delta = TGSAConfig.for_ablation(embed_dim=32, num_heads=4, arm="learned_delta")
    assert history.trajectory_mode == delta.trajectory_mode == "learned"
    assert history.control_source == "history_mean"
    assert delta.control_source == "velocity"
    left = TrajectoryGuidedSequentialAttention(history)
    right = TrajectoryGuidedSequentialAttention(delta)
    assert sorted(left.state_dict()) == sorted(right.state_dict())


# --------------------------------------------------------------------------
# P0-7 — crossed, not nested, resampling
# --------------------------------------------------------------------------


def _crossed_values(effect: float, *, seeds: int = 5, examples: int = 40) -> dict:
    return {
        seed: {
            "s0": {f"ex_{index:04d}": effect + (0.5 if index % 2 else -0.5) for index in range(examples)}
        }
        for seed in range(seeds)
    }


def test_p0_7_crossed_bootstrap_recovers_the_observed_effect() -> None:
    estimate = crossed_paired_bootstrap(
        _crossed_values(0.1), samples=400, confidence=0.95, seed=5
    )
    assert estimate.observed == pytest.approx(0.1, abs=1.0e-9)
    assert estimate.confidence_low <= estimate.observed <= estimate.confidence_high


def test_p0_7_crossed_bootstrap_rejects_non_crossed_input() -> None:
    values = _crossed_values(0.1)
    values[1]["s0"].pop("ex_0000")
    values[1]["s0"]["ex_9999"] = 0.0
    with pytest.raises(ValueError, match="same examples"):
        crossed_paired_bootstrap(values, samples=10, confidence=0.95, seed=1)


def test_p0_7_shared_example_effects_widen_the_crossed_interval() -> None:
    """The defect the audit named: independent per-seed resampling understates
    uncertainty when example-level effects are shared across seeds."""

    examples = 60
    # A pure example effect, identical in every seed: all uncertainty is
    # example-level, and every seed agrees exactly.
    shared = {
        seed: {"s0": {f"ex_{index:04d}": (1.0 if index < examples // 2 else -1.0) for index in range(examples)}}
        for seed in range(5)
    }
    crossed = crossed_paired_bootstrap(shared, samples=800, confidence=0.95, seed=3)
    clustered = seed_cluster_bootstrap(shared, samples=800, confidence=0.95, seed=3)
    crossed_width = crossed.confidence_high - crossed.confidence_low
    clustered_width = clustered.confidence_high - clustered.confidence_low
    # Seed-clustering sees zero variance here; the crossed scheme correctly
    # keeps the example-level uncertainty.
    assert clustered_width == pytest.approx(0.0, abs=1.0e-9)
    assert crossed_width > 0.1


def test_p0_7_only_registered_bootstrap_schemes_resolve() -> None:
    assert set(BOOTSTRAP_SCHEMES) == {"crossed_paired", "seed_cluster"}
    resolve_bootstrap("crossed_paired")
    with pytest.raises(ValueError, match="unsupported bootstrap scheme"):
        resolve_bootstrap("hierarchical_nested")


def test_p0_7_registered_config_selects_the_crossed_scheme() -> None:
    analysis = load_study_config(V2_CONFIG)["analysis"]
    assert analysis["bootstrap_scheme"] == "crossed_paired"
    assert analysis["alignment_condition"] == "replay_derangement"


# --------------------------------------------------------------------------
# P1-1 / P1-2 — resume log integrity and immutable failure evidence
# --------------------------------------------------------------------------


def test_p1_1_resume_truncates_the_log_to_the_checkpoint_boundary(tmp_path: Path) -> None:
    path = tmp_path / "train_log.jsonl"
    write_jsonl(path, [{"step": step} for step in (1, 2, 3, 4, 5)])
    repair = _truncate_train_log(path, completed_steps=3)
    assert repair["truncated"] is True
    assert repair["removed_records"] == 2
    steps = [json.loads(line)["step"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert steps == [1, 2, 3]


def test_p1_1_truncation_is_a_no_op_when_the_log_is_already_aligned(tmp_path: Path) -> None:
    path = tmp_path / "train_log.jsonl"
    write_jsonl(path, [{"step": step} for step in (1, 2, 3)])
    before = path.read_bytes()
    repair = _truncate_train_log(path, completed_steps=3)
    assert repair["truncated"] is False
    assert path.read_bytes() == before


def test_p1_1_duplicate_step_records_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train_log.jsonl"
    write_jsonl(path, [{"step": step} for step in (1, 2, 2, 3)])
    with pytest.raises(ValueError, match="duplicate step records"):
        _truncate_train_log(path, completed_steps=10)


def test_p1_2_failed_attempts_are_immutable_and_accumulate(tmp_path: Path) -> None:
    first = _record_failed_attempt(tmp_path, {"attempt": 1, "exception": "boom"})
    second = _record_failed_attempt(tmp_path, {"attempt": 2, "exception": "boom again"})
    assert first != second
    assert first.is_file() and second.is_file()
    assert json.loads(first.read_text(encoding="utf-8"))["attempt"] == 1
    assert len(list((tmp_path / "failures").glob("attempt_*.json"))) == 2


# --------------------------------------------------------------------------
# P1-4 — length-matched derangement and capture-and-replay
# --------------------------------------------------------------------------


def test_p1_4_derangement_never_crosses_a_group_or_fixes_a_point() -> None:
    keys = ["a", "a", "a", "b", "b", "c"]
    for index in range(4):
        donor, applied = length_matched_derangement(keys, derangement_index=index)
        assert all(keys[donor[row]] == keys[row] for row in range(len(keys)))
        assert all(donor[row] != row for row in range(len(keys)) if applied[row])
        assert applied[-1] is False, "a singleton group has no valid donor"


def test_p1_4_registered_derangement_indices_are_distinct_permutations() -> None:
    keys = ["a"] * 4
    donors = {tuple(length_matched_derangement(keys, derangement_index=i)[0]) for i in range(3)}
    assert len(donors) == 3


def test_p1_4_replay_uses_unperturbed_donor_controls() -> None:
    """Head j must not receive a control already perturbed at heads < j."""

    model = build_model(_config("learned_delta"), initialization_seed=9).eval()
    inputs = torch.randint(0, 64, (4, 10))
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    with torch.inference_mode():
        with ControlCapture(model) as capture:
            model(inputs, key_padding_mask=mask)
        aligned = {key: value.clone() for key, value in capture.captured.items()}
        donor, applied = length_matched_derangement(["g"] * 4, derangement_index=0)
        with ReplayIntervention(
            model, captured=aligned, donor_index=donor, applied=applied
        ) as replay:
            model(inputs, key_padding_mask=mask)
        # Every encoder was fed from the captured aligned pass, so the number of
        # substitutions equals the number of encoders, and each substitution used
        # the pre-intervention tensor.
        assert len(replay.records) == len(aligned)
        with ControlCapture(model) as second:
            model(inputs, key_padding_mask=mask)
    for key, value in aligned.items():
        torch.testing.assert_close(second.captured[key], value, atol=1.0e-6, rtol=1.0e-6)


def test_p1_4_unmatched_rows_keep_their_own_control() -> None:
    model = build_model(_config("learned_delta"), initialization_seed=13).eval()
    inputs = torch.randint(0, 64, (3, 9))
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    with torch.inference_mode():
        with ControlCapture(model) as capture:
            baseline = model(inputs, key_padding_mask=mask).logits
        donor, applied = length_matched_derangement(["a", "a", "b"], derangement_index=0)
        with ReplayIntervention(model, captured=capture.captured, donor_index=donor, applied=applied):
            perturbed = model(inputs, key_padding_mask=mask).logits
    assert applied == [True, True, False]
    torch.testing.assert_close(baseline[2], perturbed[2], atol=1.0e-6, rtol=1.0e-6)
    assert not torch.allclose(baseline[:2], perturbed[:2], atol=1.0e-6)


def test_p1_4_replay_batches_are_homogeneous_in_stratum_and_length() -> None:
    from tgsa.phase1_data import Phase1Example

    def example(identifier: str, stratum: str, length: int) -> Phase1Example:
        tokens = tuple(range(1, length + 1))
        return Phase1Example(
            schema_version="tgsa.phase1-example.v1",
            id=identifier,
            split="test",
            family="pointer_chase",
            stratum=stratum,
            tokens=tokens,
            answer_token=tokens[-1],
            answer_input_index=len(tokens) - 2,
            latent_signature=identifier,
            metadata={},
        )

    examples = [
        example("a", "s1", 8),
        example("b", "s1", 8),
        example("c", "s1", 10),
        example("d", "s2", 8),
        example("e", "s1", 10),
    ]
    for chunk in _replay_groups(examples, batch_size=8):
        keys = {(examples[row].stratum, len(examples[row].tokens)) for row in chunk}
        assert len(keys) == 1, "a replay batch mixed strata or sequence lengths"


# --------------------------------------------------------------------------
# P1-5 / P1-6 — task difficulty is measured and frozen
# --------------------------------------------------------------------------


def test_p1_5_primary_families_reject_trivial_items_in_the_registered_config() -> None:
    families = load_study_config(V2_CONFIG)["dataset"]["families"]
    assert families["pointer_chase"]["reject_trivial"] is True
    assert families["state_update"]["reject_trivial"] is True
    # The guardrail family is a one-hop lookup by construction; filtering it
    # would make the non-inferiority guardrail unrepresentative.
    assert "reject_trivial" not in families["direct_lookup"]


def test_p1_5_generator_records_the_difficulty_diagnostics() -> None:
    from tgsa.phase1_data import SyntheticVocabulary, generate_split

    vocab = SyntheticVocabulary(domain_size=16, max_hops=6, max_abs_step=4)
    spec = {
        "families": {
            "pointer_chase": {"min_nodes": 6, "max_nodes": 10, "min_hops": 2, "max_hops": 4},
            "state_update": {"min_steps": 3, "max_steps": 5},
        }
    }
    examples = generate_split(
        split="train",
        count_by_family={"pointer_chase": 20, "state_update": 20},
        dataset_seed=99,
        vocab=vocab,
        generation_spec=spec,
    )
    for example in examples:
        assert "trivial" in example.metadata
        assert "answer_equals_start" in example.metadata
        if example.family == "pointer_chase":
            assert example.metadata["effective_hops"] <= example.metadata["hops"]
        else:
            assert "cancels_to_start" in example.metadata


def test_p1_5_rejection_actually_removes_trivial_items() -> None:
    from tgsa.phase1_data import SyntheticVocabulary, generate_split

    vocab = SyntheticVocabulary(domain_size=16, max_hops=6, max_abs_step=4)
    spec = {
        "families": {
            "pointer_chase": {
                "min_nodes": 6,
                "max_nodes": 10,
                "min_hops": 2,
                "max_hops": 4,
                "reject_trivial": True,
            }
        }
    }
    examples = generate_split(
        split="train",
        count_by_family={"pointer_chase": 40},
        dataset_seed=123,
        vocab=vocab,
        generation_spec=spec,
    )
    assert examples
    assert all(example.metadata["trivial"] is False for example in examples)
    assert all(
        example.metadata["effective_hops"] == example.metadata["hops"] for example in examples
    )


def test_p1_6_held_out_range_overlap_is_reported_not_hidden() -> None:
    from tgsa.phase1_difficulty import _range_overlap

    dataset = load_study_config(V2_CONFIG)["dataset"]
    overlap = _range_overlap(
        dataset, family="state_update", low_key="min_steps", high_key="max_steps"
    )
    # The registered ranges genuinely overlap; the contract is that this is
    # surfaced rather than described as out-of-range extrapolation.
    assert overlap["overlapping_values"], "expected the registered overlap to be detected"
    assert overlap["fully_out_of_range"] is False
