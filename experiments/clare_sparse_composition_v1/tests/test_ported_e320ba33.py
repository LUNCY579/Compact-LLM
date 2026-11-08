"""CPU tests PORTED from the superseded parallel implementation of CLARE-RINEYA-001.

Source: branch research/titan-clare-sparse-composable-skills-v1 @ e320ba33 (author LuncyNever),
files experiments/clare_sparse_composition_v1/tests/test_core.py and tests/test_boundaries.py
(87 passing cases there). That implementation is superseded; its tree is NOT merged (the merge
commit uses strategy "ours"). Each of its test cases is re-expressed against the canonical API.
Where the two implementations deliberately differ, the invariant is kept and the difference is
stated. Mapping (theirs -> here):

test_core.py
  test_capacity_schedule                      -> test_capacity_schedule (identical expectations)
  test_budget_rounding                        -> test_budget_rounding (n=0 REFUSES here: fail-closed)
  test_bad_config                             -> test_bad_arm_spec / test_bad_recipe / test_bad_study_config
  test_config_roundtrip                       -> test_config_roundtrip
  test_exact_ties_and_unavailable_scores      -> test_exact_ties_and_unavailable_scores (ties: seeded hash, not index)
  test_random_determinism                     -> test_random_determinism
  test_selection_refusals                     -> test_selection_refusals
  test_allocation_protected_and_defensive_copy-> test_allocation_protected_and_defensive_copy
  test_arm_execution_reset_finite_and_accounting -> test_arm_execution_reset_finite_and_accounting (all arms)
  test_random_vs_learned_exact_capacity       -> test_random_vs_learned_exact_capacity
  test_adamw_frozen_coords_momentum_and_decay -> test_adamw_frozen_coords_momentum_and_decay
  test_reset_equivalent_to_fresh_masked_retrain -> test_reset_equivalent_to_fresh_masked_retrain
  test_previous_delta_is_unchanged_after_later_skills -> test_previous_delta_is_unchanged_after_later_skills
  test_training_boundary_refuses_without_mutation -> test_training_boundary_refuses_without_mutation
  test_failure_rolls_back_rng_modes_and_flags -> test_failure_leaves_host_clean_and_rerun_matches
  test_nonfinite_loss_and_guard_refused       -> test_nonfinite_loss_and_guard_refused
  test_arbitrary_order_and_reproducibility    -> test_arbitrary_order_and_reproducibility
  test_checkpoint_resume_matches_uninterrupted-> test_resume_matches_uninterrupted (all trainable arms)
  test_checkpoint_rejects_drift               -> test_artifacts_reject_drift
  test_coordinate_safety                      -> test_coordinate_safety
  test_geometry_zero_norm_and_overlap         -> test_geometry_zero_norm_and_overlap
  test_disjoint_geometry_does_not_imply_functional_additivity -> same name (+ a real tiny-model case)
  test_logit_constant_shift_diagnostic        -> test_logit_constant_shift_diagnostic
  test_rank_and_merges                        -> test_rank_and_merges (shared_private: no counterpart)
  test_tensor_hash_binds_dtype_shape_values   -> test_delta_checksum_binds_values_and_shape
  test_optimizer_sees_zero_frozen_gradients   -> test_optimizer_sees_zero_frozen_gradients
  test_l1_discovery_objective_has_effect      -> test_l1_discovery_objective_has_effect
  test_shared_private_refuses_broadcast       -> NOT APPLICABLE (no shared_private merge helper here)
  test_config_rejects_malformed_order         -> test_arm_spec_rejects_malformed_order
test_boundaries.py
  test_actual_corpus_schema_adapter_no_eval_access -> test_corpus_adapter_reads_train_only
  test_corpus_refusal                         -> test_corpus_refusal
  test_parent_cluster_pairing_and_undefined_retention -> same name
  test_evaluation_refuses_unmatched           -> test_evaluation_refuses_unmatched
  test_forgetting_matrix                      -> test_forgetting_matrix
  test_unbound_native_is_blocked_without_model_load -> test_blocked_native_never_loads_a_model
  test_cli_preflight_writes_refusal_and_never_overwrites -> test_cli_refusal_is_recorded_and_never_overwritten
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from clare.allocation import ARMS, ArmSpec, SkillData, SkillRecord, run_arm
from clare.data import AccessRefused, Corpus
from clare.deltas import DenseDelta, DiskDelta, compose, write_dense_stream
from clare.footprint import Footprint, qwen3_mlp_footprint
from clare.functional import interaction_rows, vocab_interaction
from clare.geometry import pair_geometry, pairwise, rank_energy
from clare.masks import AllocationState, SkillMask
from clare.recipe import Recipe, compile_row
from clare.schedule import budgets
from clare.select import seed_key, select_random, select_top_importance
from clare.training import (DeltaHost, base_checksum, load_stage_checkpoint, make_optimizer, run_stage,
                            save_stage_checkpoint)
from tinymodel import RECIPE, SKILLS, Tiny, problem

ORDER = tuple(SKILLS)
TRAINABLE = [a for a in ARMS if a != "C_magnitude_sparse"]
EXP = Path(__file__).resolve().parents[1]


def spec(arm="E_learned_disjoint", order=ORDER, **kw):
    if arm == "F_shared_private":
        kw.setdefault("shared_fraction", 0.1)
    return ArmSpec(arm, tuple(order), rho=kw.pop("rho", "0.8"), **kw)


def one_slot(n=4, name="model.layers.0.mlp.up_proj.weight"):
    return Footprint([(name, (n,))])


# -- capacity ------------------------------------------------------------------------------

@pytest.mark.parametrize("rho,counts", [("0.8", [20, 16, 13, 10]), ("0.9", [10, 9, 8, 7]),
                                        ("0.95", [5, 5, 5, 4]), ("0.975", [3, 2, 2, 2])])
def test_capacity_schedule(rho, counts):
    assert [b.budget for b in budgets(100, ORDER, rho)] == counts


@pytest.mark.parametrize("n,rho,expected", [(1, "0.999", 1), (10, "0.95", 1), (3, "0.5", 2), (100, "0.8", 20)])
def test_budget_rounding(n, rho, expected):
    assert budgets(n, ("R",), rho)[0].budget == expected


def test_budget_rounding_empty_pool_refuses():
    """Deliberate difference: CLARE (and e320ba33) give an EMPTY mask when no capacity is left;
    here a skill with nothing to learn with is a registration error, so the schedule refuses."""
    with pytest.raises(ValueError):
        budgets(0, ("R",), "0.8")
    with pytest.raises(ValueError):
        budgets(2, ("R", "U", "S"), "0.5")  # third skill would find the pool empty


# -- configuration ------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{"arm": "F"}, {"rho": "0"}, {"rho": "1"}, {"rho": float("nan")},
                                {"order": ("R", "R")}, {"order": ("X",)}, {"seed": True},
                                {"l1_lambda": -1.0}, {"l1_lambda": float("inf")},
                                {"lambda_fallback": (-0.1,)}, {"discovery_fraction": 0.0}])
def test_bad_arm_spec(kw):
    base = {"arm": "E_learned_disjoint", "order": ORDER, "rho": "0.8"}
    with pytest.raises(ValueError):
        ArmSpec(**{**base, **kw})


@pytest.mark.parametrize("kw", [{"peak_lr": 0.0}, {"peak_lr": float("nan")}, {"weight_decay": -1.0},
                                {"weight_decay": 0.2}])
def test_bad_recipe(kw):
    with pytest.raises(ValueError):
        Recipe(**kw)


@pytest.mark.parametrize("mutate", ["order_X", "order_repeat", "eval_role_test", "train_role_dev",
                                    "footprint_total", "dataset_pin", "signatures"])
def test_bad_study_config(tmp_path, mutate):
    from clare.config import DEFAULT_CONFIG, StudyConfig

    raw = json.loads(DEFAULT_CONFIG.read_bytes())
    if mutate == "order_X":
        raw["orders"]["canonical"] = ["R", "U", "S", "X"]
    if mutate == "order_repeat":
        raw["orders"]["canonical"] = ["R", "U", "S", "S"]
    if mutate == "eval_role_test":
        raw["evaluation"]["role"] = "test"
    if mutate == "train_role_dev":
        raw["train"]["role"] = "dev"
    if mutate == "footprint_total":
        raw["footprint"]["expected_total_coordinates"] += 1
    if mutate == "dataset_pin":
        raw["dataset"]["manifest_sha256"] = "0" * 64
    if mutate == "signatures":
        raw["evaluation"]["signatures"] = raw["evaluation"]["signatures"][:-1]
    p = tmp_path / "cfg.json"
    p.write_bytes(json.dumps(raw).encode())
    with pytest.raises(ValueError):
        StudyConfig.load(p)


def test_config_roundtrip(tmp_path):
    from clare.config import DEFAULT_CONFIG, StudyConfig

    ref = StudyConfig.load()
    raw = json.loads(DEFAULT_CONFIG.read_bytes())
    p = tmp_path / "same.json"
    p.write_bytes(json.dumps(raw, indent=4).encode())  # formatting does not change the identity
    assert StudyConfig.load(p).sha256 == ref.sha256
    raw["orders"]["canonical"] = ["P", "S", "U", "R"]
    p2 = tmp_path / "other.json"
    p2.write_bytes(json.dumps(raw).encode())
    assert StudyConfig.load(p2).sha256 != ref.sha256
    s = spec(order=("P", "S", "U", "R"), rho="0.9")
    assert ArmSpec(**asdict(s)) == s


@pytest.mark.parametrize("order", [(1,), ([],), ("RU",), (), ["R", "U"]])
def test_arm_spec_rejects_malformed_order(order):
    with pytest.raises(ValueError):
        ArmSpec("E_learned_disjoint", order)


# -- selection -------------------------------------------------------------------------------

def test_exact_ties_and_unavailable_scores():
    """Deliberate difference: e320ba33 breaks importance ties by ascending index; here ties are
    broken by a seeded hash (then index) so a budget larger than the non-zero support cannot favour
    early layers. The invariants kept: exact budget, free coordinates only, determinism."""
    fp = one_slot()
    free = {fp.slots[0].name: np.array([False, True, True, True])}
    a, _ = select_top_importance(fp, {fp.slots[0].name: np.ones(4, np.float32)}, free, 2, seed_key("t"))
    b, _ = select_top_importance(fp, {fp.slots[0].name: np.ones(4, np.float32)}, free, 2, seed_key("t"))
    sel = a[fp.slots[0].name]
    assert sel.sum() == 2 and not sel[0] and (sel == b[fp.slots[0].name]).all()
    sel, _ = select_top_importance(fp, {fp.slots[0].name: np.array([100., 1, 9, 0], np.float32)}, free, 1, seed_key("t"))
    assert sel[fp.slots[0].name].tolist() == [False, False, True, False]


def test_random_determinism():
    fp = one_slot(100)
    free = {fp.slots[0].name: np.ones(100, dtype=bool)}
    a, _ = select_random(fp, free, 12, seed_key(4))
    b, _ = select_random(fp, free, 12, seed_key(4))
    c, _ = select_random(fp, free, 12, seed_key(5))
    n = fp.slots[0].name
    assert (a[n] == b[n]).all() and (a[n] != c[n]).any() and a[n].sum() == 12


@pytest.mark.parametrize("case", ["floatmask", "shape", "nan", "budget", "negative"])
def test_selection_refusals(case):
    fp = one_slot()
    n = fp.slots[0].name
    free, scores, k = np.ones(4, dtype=bool), np.ones(4, np.float32), 1
    if case == "floatmask":
        free = free.astype(np.float32)
    if case == "shape":
        scores = np.ones(3, np.float32)
    if case == "nan":
        scores[0] = np.nan
    if case == "budget":
        k = 5
    if case == "negative":
        scores[1] = -1.0
    with pytest.raises(ValueError):
        select_top_importance(fp, {n: scores}, {n: free}, k, seed_key("r"))
    if case == "floatmask":
        with pytest.raises(ValueError):
            select_random(fp, {n: free}, k, seed_key("r"))


def test_allocation_protected_and_defensive_copy():
    fp = one_slot()
    n = fp.slots[0].name
    st = AllocationState(fp, strict=True)
    m = np.array([True, False, False, False])
    st.claim("R", SkillMask.from_bool(fp, {n: m}), expected_budget=1)
    m[0] = False  # mutating the caller's array must not change the claim
    assert not st.free_bool(n)[0]
    with pytest.raises(ValueError):
        st.claim("U", SkillMask.from_bool(fp, {n: np.array([True, False, False, False])}))
    with pytest.raises(ValueError):
        st.claim("R", st.free(), expected_budget=3)  # a skill cannot claim twice
    with pytest.raises(ValueError):
        st.claim("U", st.free(), expected_budget=2)  # budget mismatch


# -- arms ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def b_deltas():
    model, fp, host, data, loss_fn = problem()
    return run_arm(spec("B_independent_sum"), host, data, loss_fn, RECIPE).deltas()


@pytest.mark.parametrize("arm", ARMS)
def test_arm_execution_reset_finite_and_accounting(arm, b_deltas):
    model, fp, host, data, loss_fn = problem()
    res = run_arm(spec(arm), host, data, loss_fn, RECIPE,
                  independent=b_deltas if arm == "C_magnitude_sparse" else None)
    assert [r.skill for r in res.records] == list(ORDER)
    for r in res.records:
        assert all(np.isfinite(r.delta.dense(s.name)).all() for s in fp)
        if r.mask is not None:
            assert r.delta.nonzero() <= r.mask.count() or arm == "F_shared_private"
        if r.discovery and "effective_checksum_before" in r.discovery:
            assert r.discovery["effective_checksum_before"] == r.discovery["effective_checksum_after_reset"]
        if r.train_log is not None:
            assert r.train_log.updates == len(data.groups[r.skill]) and np.isfinite(r.train_log.losses).all()
    if arm in ("D_random_disjoint", "E_learned_disjoint"):
        for g in pairwise(res.deltas()):
            assert g["support_intersection"] == 0 and g["dot"] == 0.0
    assert host.mode is None


def test_random_vs_learned_exact_capacity():
    counts = {}
    for arm in ("D_random_disjoint", "E_learned_disjoint"):
        model, fp, host, data, loss_fn = problem()
        counts[arm] = [r.mask.count() for r in run_arm(spec(arm), host, data, loss_fn, RECIPE).records]
    assert counts["D_random_disjoint"] == counts["E_learned_disjoint"]


@pytest.mark.parametrize("decay", [0.0, 0.2, 1.0])
def test_adamw_frozen_coords_momentum_and_decay(decay):
    """Coordinates outside the trainable mask stay exactly zero through AdamW momentum AND decoupled
    weight decay (decay of a zero is zero), and the base never moves. The registered recipe still
    requires weight_decay = 0 (test_bad_recipe); this checks the property the refusal protects."""
    model, fp, host, data, loss_fn = problem()
    base0 = base_checksum(model)
    mask = SkillMask.from_bool(fp, {s.name: np.arange(s.numel) % 5 == 0 for s in fp})
    params = host.start_dense(mask)
    opt = torch.optim.AdamW(params, lr=0.05, weight_decay=decay, foreach=False)
    run_stage(host, data.rows["R"], data.groups["R"], loss_fn, RECIPE, optimizer=opt)
    assert host.outside_mask_nonzero() == 0
    d = host.trainable_delta()
    assert any((d.dense(s.name)[mask.bool(s.name)] != 0).any() for s in fp)
    host.stop_trainable()
    assert base_checksum(model) == base0


def test_reset_equivalent_to_fresh_masked_retrain():
    """After discovery + RESET, E's retraining equals a fresh sparse retrain on the same mask from
    the same state, bit for bit: no probe value, optimiser state or RNG leaks into Stage 2."""
    model, fp, host, data, loss_fn = problem()
    rec = run_arm(spec(order=("R",)), host, data, loss_fn, RECIPE).records[0]
    model, fp, host, data, loss_fn = problem()
    host.start_sparse(rec.mask)
    run_stage(host, data.rows["R"], data.groups["R"], loss_fn, RECIPE)
    assert host.trainable_delta().checksum() == rec.delta.checksum()
    host.stop_trainable()


def test_previous_delta_is_unchanged_after_later_skills():
    model, fp, host, data, loss_fn = problem()
    res = run_arm(spec(order=("R", "U")), host, data, loss_fn, RECIPE)
    r0, total = res.records[0], res.prefix_sum(2)
    for s in fp:
        m = r0.mask.bool(s.name)
        assert total.dense(s.name)[m].tobytes() == r0.delta.dense(s.name)[m].tobytes()


@pytest.mark.parametrize("bad", ["role", "signature", "empty_rows", "empty_group", "bad_index"])
def test_training_boundary_refuses_without_mutation(bad):
    model, fp, host, data, loss_fn = problem()
    rows = {k: [dict(r) for r in v] for k, v in data.rows.items()}
    groups = {k: [list(g) for g in v] for k, v in data.groups.items()}
    if bad == "role":
        rows["R"][0]["role"] = "test"
    if bad == "signature":
        rows["R"][0]["signature"] = "U"
    if bad == "empty_rows":
        rows["R"], groups["R"] = [], []
    if bad == "empty_group":
        groups["R"].append([])
    if bad == "bad_index":
        groups["R"][0].append(999)
    before = (host.effective_checksum(), base_checksum(model))
    with pytest.raises(ValueError):
        run_arm(spec(), host, SkillData(rows, groups), loss_fn, RECIPE)
    assert (host.effective_checksum(), base_checksum(model)) == before and host.mode is None


def test_training_boundary_refuses_assistant_leak_and_eval_roles():
    """e320ba33's 'assistant' and 'digest' cases: the vendored row compiler refuses any prompt that
    is not exactly system+user (an assistant turn would leak an answer), and the corpus refuses
    every non-TRAIN role for training (see also test_corpus_refusal)."""
    leak = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "leak"}]
    with pytest.raises(ValueError):
        compile_row(leak, '{"answer":[0]}', tokenizer=None)
    with pytest.raises(ValueError):
        compile_row([{"role": "system", "content": "s"}, {"role": "user", "content": "<|im_start|>x"}],
                    '{"answer":[0]}', tokenizer=None)


def test_failure_leaves_host_clean_and_rerun_matches():
    """e320ba33 restores RNG/modes/requires_grad after a failure. Here training uses no RNG, never
    toggles modes, and DeltaHost freezes the base permanently by design; the property kept is that a
    failure mid-skill leaves no open stage, no frozen delta, an unchanged base, and a host whose next
    run equals a fresh run."""
    model, fp, host, data, loss_fn = problem()
    base0 = base_checksum(model)
    calls = {"n": 0}

    def broken(row):
        calls["n"] += 1
        if calls["n"] == 30:  # mid-skill: inside skill R's masked retraining (8 probe + 24 train calls)
            raise RuntimeError("injected")
        return loss_fn(row)
    with pytest.raises(RuntimeError):
        run_arm(spec(), host, data, broken, RECIPE)
    assert host.mode is None and host.frozen_checksum() == DenseDelta.zeros(fp).checksum()
    assert base_checksum(model) == base0
    again = run_arm(spec(), host, data, loss_fn, RECIPE)
    model2, fp2, host2, data2, loss2 = problem()
    fresh = run_arm(spec(), host2, data2, loss2, RECIPE)
    assert [r.delta.checksum() for r in again.records] == [r.delta.checksum() for r in fresh.records]


def test_nonfinite_loss_and_guard_refused():
    model, fp, host, data, loss_fn = problem()
    nan_loss = lambda row: (loss_fn(row)[0] * float("nan"), loss_fn(row)[1])
    with pytest.raises(FloatingPointError):
        run_arm(spec("D_random_disjoint"), host, data, nan_loss, RECIPE)
    assert host.mode is None

    def guard():
        raise RuntimeError("RAM")
    with pytest.raises(RuntimeError):
        run_arm(spec("A_dense_sequential"), host, data, loss_fn, RECIPE, guard=guard)
    assert host.mode is None and host.frozen_checksum() == DenseDelta.zeros(fp).checksum()


@pytest.mark.parametrize("order", [("R", "U", "S", "P"), ("P", "S", "U", "R"), ("R", "S")])
def test_arbitrary_order_and_reproducibility(order):
    out = []
    for _ in range(2):
        model, fp, host, data, loss_fn = problem()
        res = run_arm(spec(order=order), host, data, loss_fn, RECIPE)
        out.append(res.prefix_sum(len(order)).checksum())
        assert res.spec.order == order and [r.skill for r in res.records] == list(order)
    assert out[0] == out[1]
    with pytest.raises(ValueError):
        res.allocation.claim(order[0], res.records[0].mask)


@pytest.mark.parametrize("arm", TRAINABLE)
def test_resume_matches_uninterrupted(arm, tmp_path):
    model, fp, host, data, loss_fn = problem()
    full = run_arm(spec(arm), host, data, loss_fn, RECIPE)
    # persist the first skill through the on-disk formats a real resume reads back
    r0 = full.records[0]
    stem = tmp_path / "00_R"
    write_dense_stream(stem, fp, r0.delta.iter_dense())
    mask = None
    if r0.mask is not None:
        r0.mask.save(tmp_path / "mask_00_R")
        mask = SkillMask.load(tmp_path / "mask_00_R", fp)
    done = {r0.skill: SkillRecord(r0.skill, 0, DiskDelta(stem, fp), mask, None, None)}
    model, fp, host, data, loss_fn = problem()
    resumed = run_arm(spec(arm), host, data, loss_fn, RECIPE, resume=done)
    assert [r.delta.checksum() for r in resumed.records] == [r.delta.checksum() for r in full.records]
    with pytest.raises(FileExistsError):
        write_dense_stream(stem, fp, r0.delta.iter_dense())


@pytest.mark.parametrize("tamper", ["payload_bytes", "mask_sidecar", "stage_binding", "run_spec"])
def test_artifacts_reject_drift(tmp_path, tamper):
    model, fp, host, data, loss_fn = problem()
    res = run_arm(spec(order=("R",)), host, data, loss_fn, RECIPE)
    rec = res.records[0]
    if tamper == "payload_bytes":
        write_dense_stream(tmp_path / "d", fp, rec.delta.iter_dense())
        with open(tmp_path / "d.dense.f32", "ab") as f:
            f.write(b"drift")
        with pytest.raises(ValueError):
            DiskDelta(tmp_path / "d", fp)
    if tamper == "mask_sidecar":
        rec.mask.save(tmp_path / "m")
        side = json.loads((tmp_path / "m.json").read_bytes())
        side["mask_checksum"] = "0" * 64
        (tmp_path / "m.json").write_bytes(json.dumps(side).encode())
        with pytest.raises(ValueError):
            SkillMask.load(tmp_path / "m", fp)
    if tamper == "stage_binding":
        host.start_dense(None)
        _, opt = run_stage(host, data.rows["U"], data.groups["U"][:1], loss_fn, RECIPE)
        save_stage_checkpoint(tmp_path / "ck.pt", host, opt, 1, {"stage": "U"})
        before = host.trainable_delta().checksum()
        with pytest.raises(ValueError):
            load_stage_checkpoint(tmp_path / "ck.pt", host, make_optimizer(host.trainable_parameters(), RECIPE),
                                  {"stage": "U", "model": "other"})
        assert host.trainable_delta().checksum() == before  # a refused load changes nothing
        host.stop_trainable()
    if tamper == "run_spec":
        import cli
        run = tmp_path / "run"
        run.mkdir()
        (run / "RUN_SPEC.json").write_bytes(b'{"arm": "something else"}')
        with pytest.raises(SystemExit):
            cli.main(["train", "--arm", "E_learned_disjoint", "--seed", "0", "--out", str(run)])


def test_coordinate_safety():
    m = Tiny()
    with pytest.raises(ValueError):
        Footprint([])
    with pytest.raises(ValueError):
        Footprint([("model.layers.0.mlp.up_proj.weight", (12, 6))] * 2)
    with pytest.raises(ValueError):
        DeltaHost(m, Footprint([("model.layers.9.mlp.up_proj.weight", (12, 6))]))
    with pytest.raises(ValueError):
        DeltaHost(m, Footprint([("model.layers.0.mlp.up_proj.weight", (6, 12))]))
    with pytest.raises(ValueError):
        DeltaHost(Tiny().double(), Footprint([("model.layers.0.mlp.up_proj.weight", (12, 6))]))


# -- geometry and functional interaction -------------------------------------------------------

def test_geometry_zero_norm_and_overlap():
    fp = one_slot()
    n = fp.slots[0].name
    z = DenseDelta(fp, {n: np.zeros(4, np.float32)})
    a = DenseDelta(fp, {n: np.array([1., -2, 0, 0], np.float32)})
    b = DenseDelta(fp, {n: np.array([-1., -2, 3, 0], np.float32)})
    g0 = pair_geometry(z, z)
    assert g0["cosine"] is None and g0["jaccard"] is None and g0["sign_agreement_on_overlap"] is None
    g = pair_geometry(a, b)
    assert g["support_intersection"] == 2 and g["sign_agreement_on_overlap"] == 0.5
    assert g["jaccard"] == pytest.approx(2 / 3) and g["dot"] == 3.0
    json.dumps(g0, allow_nan=False)  # undefined values are JSON-safe


def test_disjoint_geometry_does_not_imply_functional_additivity():
    # e320ba33's analytic case: disjoint coordinates, a product nonlinearity
    f = lambda x: x[0] * x[1]
    a, b = np.array([1., 0]), np.array([0., 1])
    assert float(a @ b) == 0.0
    Z = {"0": [f(np.zeros(2))], "A": [f(a)], "B": [f(b)], "AB": [f(a + b)]}
    assert interaction_rows(Z)[0] == 1.0
    # and a REAL one: two strictly disjoint trained skill deltas on the tiny network
    model, fp, host, data, loss_fn = problem()
    res = run_arm(spec("D_random_disjoint", order=("R", "U")), host, data, loss_fn, RECIPE)
    dA, dB = res.records[0].delta, res.records[1].delta
    assert pair_geometry(dA, dB)["dot"] == 0.0
    x = data.rows["S"][0]["x"]
    outs = {}
    with torch.no_grad():
        for name, parts in (("0", None), ("A", [dA]), ("B", [dB]), ("AB", [dA, dB])):
            host.set_frozen(parts)
            outs[name] = model(x).double().numpy()
    host.set_frozen(None)
    I = outs["AB"] - outs["A"] - outs["B"] + outs["0"]
    assert np.abs(I).max() > 1e-6


def test_logit_constant_shift_diagnostic():
    """Raw-logit interaction includes a meaningless per-position constant; the canonical estimator
    works on log-softmax, which removes it (e320ba33's 'vocab-centered' readout)."""
    z = torch.zeros(3, 4)
    ls = lambda t: torch.log_softmax(t, dim=-1)
    vi = vocab_interaction({"0": ls(z), "A": ls(z), "B": ls(z), "AB": ls(z + 3)}, topk=2)
    assert np.allclose(vi["I_full"], 0.0)
    raw = (z + 3) - z - z + z
    assert float(raw.pow(2).mean().sqrt()) == 3.0
    with pytest.raises(ValueError):
        vocab_interaction({"0": ls(z), "A": ls(z[:1]), "B": ls(z), "AB": ls(z)}, topk=2)


def test_rank_and_merges():
    fp = one_slot(2)
    n = fp.slots[0].name
    a = DenseDelta(fp, {n: np.array([1., 2], np.float32)})
    b = DenseDelta(fp, {n: np.array([3., 4], np.float32)})
    assert compose([a, b], [0.5, 0.5]).dense(n).tolist() == [2.0, 3.0]
    with pytest.raises(ValueError):
        compose([a], [math.nan])
    with pytest.raises(ValueError):
        compose([a, DenseDelta(one_slot(3), {one_slot(3).slots[0].name: np.zeros(3, np.float32)})])
    sq = Footprint([("model.layers.0.mlp.down_proj.weight", (2, 2))])
    m = sq.slots[0].name
    r = rank_energy(DenseDelta(sq, {m: np.diag([3., 1.]).astype(np.float32).reshape(-1)}), ks=(1,))
    assert r[0]["top1_energy_fraction"] == pytest.approx(0.9)
    r = rank_energy(DenseDelta(sq, {m: np.zeros(4, np.float32)}), ks=(1,))
    assert r[0]["top1_energy_fraction"] is None


def test_delta_checksum_binds_values_and_shape():
    """The delta dtype is fixed (FP32), so the checksum binds values, footprint shapes/order and
    names; it is representation-independent (dense == sparse) by design."""
    fp = one_slot(2)
    n = fp.slots[0].name
    a = DenseDelta(fp, {n: np.array([1., 2], np.float32)})
    assert a.checksum() != DenseDelta(fp, {n: np.array([1., 3], np.float32)}).checksum()
    fp2 = Footprint([(n, (1, 2))])
    assert a.checksum() != DenseDelta(fp2, {n: np.array([1., 2], np.float32)}).checksum()
    mask = SkillMask.from_bool(fp, {n: np.array([True, True])})
    assert a.restrict(mask).checksum() == a.checksum()


def test_optimizer_sees_zero_frozen_gradients(monkeypatch):
    model, fp, host, data, loss_fn = problem()
    mask = SkillMask.from_bool(fp, {s.name: np.arange(s.numel) < 4 for s in fp})
    seen = []
    original = torch.optim.AdamW.step

    def check(self, *args, **kwargs):
        for group in self.param_groups:
            for p, s in zip(group["params"], fp):
                seen.append(int(torch.count_nonzero(p.grad.reshape(-1)[~torch.from_numpy(mask.bool(s.name))])))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", check)
    host.start_dense(mask)
    run_stage(host, data.rows["R"], data.groups["R"], loss_fn, RECIPE)
    host.stop_trainable()
    assert seen and all(v == 0 for v in seen)


def test_l1_discovery_objective_has_effect():
    sums = {}
    for lam in (0.0, 0.5):
        model, fp, host, data, loss_fn = problem()
        host.start_dense(None)
        run_stage(host, data.rows["R"], data.groups["R"], loss_fn, RECIPE, l1_lambda=lam)
        d = host.trainable_delta()
        sums[lam] = sum(float(np.abs(d.dense(s.name)).sum()) for s in fp)
        host.stop_trainable()
    assert sums[0.5] < sums[0.0]


# -- corpus boundary (synthetic fixture corpus) --------------------------------------------------

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fixture_corpus(root: Path) -> str:
    data = root / "data"
    for d in ("canonical", "oracles", "manifests"):
        (data / d).mkdir(parents=True)
    prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "synthetic"}]
    row = {"id": "row-R", "parent_id": "p", "role": "train", "tag": "core", "kind": "composition",
           "skills": ["R"], "difficulty": 0, "template": 0, "prompt": prompt,
           "verifier": {"spec": {"initial": [1, 2, 3, 4, 5, 6]}}}
    (data / "canonical/train.jsonl").write_bytes((json.dumps(row) + "\n").encode())
    (data / "oracles/train.jsonl").write_bytes((json.dumps({"id": "row-R", "references": ['{"answer":[0]}']}) + "\n").encode())
    (data / "manifests/atomic_R.json").write_bytes(json.dumps({"name": "atomic_R", "ids": ["row-R"], "rows": 1}).encode())
    files = {str(p.relative_to(data)).replace("\\", "/"): {"sha256": _sha(p), "bytes": p.stat().st_size}
             for p in data.rglob("*") if p.is_file()}
    manifest = {"experiment": "MAXRL-COMP-002", "files": files,
                "curricula": {"atomic_R": {"path": "manifests/atomic_R.json", "rows": 1,
                                           "sha256": _sha(data / "manifests/atomic_R.json")}}}
    (data / "manifest.json").write_bytes(json.dumps(manifest).encode())
    return _sha(data / "manifest.json")


def test_corpus_adapter_reads_train_only(tmp_path):
    pin = fixture_corpus(tmp_path)
    (tmp_path / "data/canonical/test.jsonl").write_bytes(b"not even JSON")  # must never be read
    c = Corpus(tmp_path, manifest_sha256=pin)
    rows = c.atomic_bank("R")
    assert [(r.id, r.role, r.signature, r.target) for r in rows] == [("row-R", "train", "R", '{"answer":[0]}')]
    assert list(rows[0].messages) == [{"role": "system", "content": "sys"}, {"role": "user", "content": "synthetic"}]
    with pytest.raises(AccessRefused):
        c.rows("test", "evaluation")


@pytest.mark.parametrize("kind", ["pin", "bytes", "missing", "curriculum"])
def test_corpus_refusal(tmp_path, kind):
    pin = fixture_corpus(tmp_path)
    if kind == "pin":
        pin = "0" * 64
    if kind == "bytes":
        (tmp_path / "data/canonical/train.jsonl").write_bytes(b"drift\n")
    if kind == "missing":
        (tmp_path / "data/canonical/train.jsonl").unlink()
    if kind == "curriculum":
        (tmp_path / "data/manifests/atomic_R.json").write_bytes(b'{"name":"atomic_R","ids":["x"],"rows":1}')
    with pytest.raises((ValueError, OSError)):
        Corpus(tmp_path, manifest_sha256=pin).atomic_bank("R")


# -- evaluation ------------------------------------------------------------------------------------

def _rec(row, parent, sig, correct, arm="E", config="prefix:4", stage=4, cfg="a" * 64):
    return {"arm": arm, "seed": 0, "order": "RUSP", "config": config, "stage": stage, "row_id": row,
            "parent_id": parent, "signature": sig, "correct": correct, "terminated": True, "config_sha256": cfg}


def eval_rows(arm="D", flip=False):
    base = [("a", "p", "R", 1), ("b", "p", "R", 0), ("c", "q", "RS", 0)]
    return [_rec(r, p, s, (1 - c) if flip else c, arm=arm) for r, p, s, c in base]


def test_parent_cluster_pairing_and_undefined_retention():
    from clare.metrics import damage, paired

    x, y = eval_rows("D"), eval_rows("E", flip=True)
    p = paired(x, y, n_boot=40)
    assert p["clusters"] == 2 and p["gains"] == 2 and p["losses"] == 1
    d = damage(x, y)
    assert d["by_signature"]["R"]["losses"] == 1 and d["by_signature"]["R"]["gains"] == 1
    assert d["by_signature"]["RS"]["retention_fraction"] is None  # reference got none right: undefined
    assert paired(x, y, n_boot=40) == p


@pytest.mark.parametrize("kind", ["drop", "duplicate", "registration", "parent", "nonbinary"])
def test_evaluation_refuses_unmatched(kind):
    from clare.metrics import MetricsError, paired

    x, y = eval_rows("D"), eval_rows("E")
    if kind == "drop":
        y.pop()
    if kind == "duplicate":
        y.append(dict(y[0]))
    if kind == "registration":
        y = [dict(r, config_sha256="b" * 64) for r in y]
    if kind == "parent":
        y[0]["parent_id"] = "different"
    if kind == "nonbinary":
        y[0]["correct"] = "1"
    with pytest.raises(MetricsError):
        paired(x, y, n_boot=2)


def test_forgetting_matrix():
    """e320ba33's numbers (R: 0.8 after R, 0.6 after U). Its 'acquisition' is the gain over the
    pre-stage (0.7); here acquisition is the accuracy right after the skill's own stage and the
    maximum forgetting equals its 'forgetting_from_best'."""
    from clare.metrics import retention_matrix

    recs = []
    for sig, per_stage in {"R": {1: 8, 2: 6}, "U": {1: 2, 2: 9}}.items():
        for k, ncorrect in per_stage.items():
            for i in range(10):
                recs.append(_rec(f"{sig}{i}", f"w{i}", sig, int(i < ncorrect), config=f"prefix:{k}", stage=k)
                            | {"order": "RU"})
    m = retention_matrix(recs, "E", 0, "RU")
    assert m["acquisition"]["R"] == pytest.approx(0.8)
    assert m["forgetting_max"]["R"] == pytest.approx(0.2) and m["forgetting_final"]["R"] == pytest.approx(0.2)
    assert m["forgetting_max"]["U"] == 0.0


# -- native/CLI refusal (no model is ever loaded) -------------------------------------------------

def _blocked_preflight(*args, **kwargs):
    return {"status": "BLOCKED", "blockers": ["test: resources withheld"]}


def test_blocked_native_never_loads_a_model(tmp_path, monkeypatch):
    import cli
    from clare import resources

    monkeypatch.setattr(resources, "preflight", _blocked_preflight)
    monkeypatch.setattr(cli, "_load_stack", lambda *a, **k: pytest.fail("a model load was attempted"))
    out = tmp_path / "nq"
    assert cli.main(["native-qualify", "--out", str(out)]) == cli.EXIT_BLOCKED
    rec = json.loads((out / "NATIVE_QUALIFICATION.json").read_bytes())
    assert rec["status"] == "BLOCKED" and rec["is_research_result"] is False and "code_digest" in rec


def test_cli_refusal_is_recorded_and_never_overwritten(tmp_path, monkeypatch):
    import cli
    from clare import resources

    monkeypatch.setattr(resources, "preflight", _blocked_preflight)
    out = tmp_path / "nq"
    assert cli.main(["native-qualify", "--out", str(out)]) == cli.EXIT_BLOCKED
    original = (out / "NATIVE_QUALIFICATION.json").read_bytes()
    with pytest.raises(FileExistsError):
        cli.main(["native-qualify", "--out", str(out)])
    assert (out / "NATIVE_QUALIFICATION.json").read_bytes() == original
