"""The 14 registered mask-safety invariants (CLARE-RINEYA-001 protocol, section 5), each with a
negative control showing the check can fail. Tiny CPU model only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from clare.allocation import ArmSpec, SkillRecord, run_arm
from clare.deltas import DenseDelta, DiskDelta, SparseDelta, compose, write_dense_stream, write_sparse_stream
from clare.footprint import Footprint, qwen3_mlp_footprint
from clare.geometry import mask_pair, pair_geometry, pairwise
from clare.masks import AllocationState, SkillMask, layer_table, pairwise_overlaps
from clare.schedule import budgets
from clare.select import seed_key, select_random, select_top_importance
from clare.training import (base_checksum, load_stage_checkpoint, make_optimizer, run_stage,
                            save_stage_checkpoint)
from tinymodel import NO_CLIP, RECIPE, SKILLS, problem

ORDER = tuple(SKILLS)


@pytest.fixture(scope="module")
def arms():
    """D and E on the same problem (seed 0, rho 0.8), plus the base checksum around them."""
    model, fp, host, data, loss_fn = problem()
    base0 = base_checksum(model)
    out = {a: run_arm(ArmSpec(a, ORDER, rho="0.8"), host, data, loss_fn, RECIPE)
           for a in ("D_random_disjoint", "E_learned_disjoint")}
    return {"fp": fp, "host": host, "model": model, "base_before": base0, "base_after": base_checksum(model), **out}


# 1 ------------------------------------------------------------------------------------
def test_01_claimed_coordinates_cannot_be_reassigned(arms):
    for arm in ("D_random_disjoint", "E_learned_disjoint"):
        res = arms[arm]
        seen = SkillMask.empty(arms["fp"])
        for rec in res.records:
            assert rec.mask.overlap(seen) == 0, f"{arm}/{rec.skill} re-selected a claimed coordinate"
            seen = seen.union(rec.mask)
        st = res.allocation
        with pytest.raises(ValueError):  # negative control: the state refuses a re-claim
            st.claim("X", res.records[0].mask)


# 2 ------------------------------------------------------------------------------------
def test_02_masks_are_binary_and_payload_exact(arms, tmp_path):
    fp = arms["fp"]
    m = arms["E_learned_disjoint"].records[0].mask
    for s in fp:
        b = m.bool(s.name)
        assert b.dtype == np.bool_ and b.size == s.numel
    m.save(tmp_path / "m")
    back = SkillMask.load(tmp_path / "m", fp)
    assert all((back.bool(s.name) == m.bool(s.name)).all() for s in fp)
    # negative control: a set padding bit is rejected (7-coordinate slot -> 1 padding bit)
    odd = Footprint([("model.layers.0.mlp.up_proj.weight", (7, 1))])
    with pytest.raises(ValueError):
        SkillMask(odd, {odd.slots[0].name: np.array([0b10000000], dtype=np.uint8)})
    with pytest.raises(TypeError):
        SkillMask.from_bool(odd, {odd.slots[0].name: np.ones(7, dtype=np.int8)})


# 3 ------------------------------------------------------------------------------------
@pytest.mark.parametrize("rho", ["0.80", "0.90", "0.95", "0.975"])
def test_03_budget_is_exact(rho):
    model, fp, host, data, loss_fn = problem(hidden=8, inter=16)
    sched = budgets(fp.total, ORDER, rho)
    for arm in ("D_random_disjoint", "E_learned_disjoint"):
        res = run_arm(ArmSpec(arm, ORDER, rho=rho), host, data, loss_fn, RECIPE)
        assert [r.mask.count() for r in res.records] == [b.budget for b in sched]
    # ties at the threshold never enlarge the mask: all-zero importance, budget 5
    zeros = {s.name: np.zeros(s.numel, dtype=np.float32) for s in fp}
    sel, rep = select_top_importance(fp, zeros, None, 5, seed_key("ties"))
    assert sum(int(v.sum()) for v in sel.values()) == 5 and rep.at_threshold_available >= 1


# 4 ------------------------------------------------------------------------------------
def test_04_inaccessible_parameters_receive_zero_gradient():
    model, fp, host, data, loss_fn = problem()
    rng = np.random.default_rng(0)
    mask = SkillMask.from_bool(fp, {s.name: rng.random(s.numel) < 0.3 for s in fp})
    params = host.start_dense(mask)
    loss_fn(data.rows["R"][0])[0].backward()
    for s, p in zip(fp, params):
        g = p.grad.reshape(-1).numpy()
        assert (g[~mask.bool(s.name)] == 0).all()
        assert (g[mask.bool(s.name)] != 0).any()
    host.stop_trainable()
    # negative control: without a mask the same coordinates DO receive gradient
    params = host.start_dense(None)
    loss_fn(data.rows["R"][0])[0].backward()
    assert any((p.grad.reshape(-1).numpy()[~mask.bool(s.name)] != 0).any() for s, p in zip(fp, params))
    host.stop_trainable()
    # sparse stage: only the masked coordinates exist as parameters
    params = host.start_sparse(mask)
    assert sum(p.numel() for p in params) == mask.count()
    host.stop_trainable()


# 5 ------------------------------------------------------------------------------------
def test_05_frozen_coordinates_and_base_are_bit_identical(arms):
    host, fp = arms["host"], arms["fp"]
    assert arms["base_before"] == arms["base_after"], "base weights moved"
    res = arms["E_learned_disjoint"]
    for rec in res.records:
        assert rec.frozen_checksum_before == rec.frozen_checksum_after
    # the final composed model holds each earlier skill's values bit-exactly on its support
    total = res.prefix_sum(len(ORDER))
    for rec in res.records:
        for s in fp:
            m = rec.mask.bool(s.name)
            assert total.dense(s.name)[m].tobytes() == rec.delta.dense(s.name)[m].tobytes()
    # negative control: an in-place mutation of an installed frozen tensor is detected
    host.set_frozen(res.records[:2] and [r.delta for r in res.records[:2]])
    before = host.frozen_checksum()
    with torch.no_grad():
        next(iter(host.slots.values())).frozen.view(-1)[0] += 1e-3
    assert host.frozen_checksum() != before
    host.set_frozen(None)


# 6 ------------------------------------------------------------------------------------
def test_06_discovery_reset_restores_the_exact_state(arms):
    for rec in arms["E_learned_disjoint"].records:
        d = rec.discovery
        assert d["effective_checksum_before"] == d["effective_checksum_after_reset"]
    # negative control: keeping the probe values (no reset) changes the effective weights
    model, fp, host, data, loss_fn = problem()
    pre = host.effective_checksum()
    host.start_dense(None)
    run_stage(host, data.rows["R"], data.groups["R"][:1], loss_fn, RECIPE)
    kept = host.trainable_delta()
    host.stop_trainable()
    assert host.effective_checksum() == pre
    host.set_frozen(kept)
    assert host.effective_checksum() != pre
    host.set_frozen(None)


# 7 ------------------------------------------------------------------------------------
def test_07_random_and_learned_capacity_identical(arms):
    d = [r.mask.count() for r in arms["D_random_disjoint"].records]
    e = [r.mask.count() for r in arms["E_learned_disjoint"].records]
    assert d == e and sum(d) == sum(e)
    assert arms["D_random_disjoint"].allocation.free_count() == arms["E_learned_disjoint"].allocation.free_count()
    # ... while the supports themselves differ (the arms are not accidentally the same rule)
    assert arms["D_random_disjoint"].records[0].mask != arms["E_learned_disjoint"].records[0].mask


# 8 ------------------------------------------------------------------------------------
def test_08_checkpoint_resume(arms, tmp_path):
    st = arms["E_learned_disjoint"].allocation
    st.save(tmp_path / "alloc")
    back = AllocationState.load(tmp_path / "alloc", arms["fp"])
    assert back.checksum() == st.checksum() and back.skills == st.skills
    # (b) mid-stage checkpoint: straight vs split training are bit-identical
    def fresh():
        model, fp, host, data, loss_fn = problem(seed=3)
        host.start_dense(None)
        return host, data, loss_fn
    host, data, loss_fn = fresh()
    groups = data.groups["U"]
    run_stage(host, data.rows["U"], groups, loss_fn, RECIPE)
    straight = host.trainable_delta().checksum()
    host, data, loss_fn = fresh()
    binding = {"stage": "U", "code": "test"}
    _, opt = run_stage(host, data.rows["U"], groups[:2], loss_fn, RECIPE,
                       total_tokens=sum(len(data.rows["U"][i]["ids"]) for g in groups for i in g))
    save_stage_checkpoint(tmp_path / "ckpt.pt", host, opt, 2, binding)
    host, data, loss_fn = fresh()
    opt = make_optimizer(host.trainable_parameters(), RECIPE)
    nxt = load_stage_checkpoint(tmp_path / "ckpt.pt", host, opt, binding)
    run_stage(host, data.rows["U"], groups, loss_fn, RECIPE, start_update=nxt, optimizer=opt)
    assert host.trainable_delta().checksum() == straight
    # (c) negative control: a different binding is refused
    with pytest.raises(ValueError):
        load_stage_checkpoint(tmp_path / "ckpt.pt", host, opt, {"stage": "S", "code": "test"})
    # (d) skill-level resume inside run_arm reproduces the uninterrupted arm
    model, fp, host, data, loss_fn = problem()
    full = run_arm(ArmSpec("E_learned_disjoint", ORDER, rho="0.8"), host, data, loss_fn, RECIPE)
    model, fp, host, data, loss_fn = problem()
    done = {r.skill: SkillRecord(r.skill, r.position, r.delta, r.mask, None, None) for r in full.records[:2]}
    resumed = run_arm(ArmSpec("E_learned_disjoint", ORDER, rho="0.8"), host, data, loss_fn, RECIPE, resume=done)
    assert [r.delta.checksum() for r in resumed.records] == [r.delta.checksum() for r in full.records]


# 9 ------------------------------------------------------------------------------------
def test_09_skill_order_is_reproducible_and_recorded(arms):
    model, fp, host, data, loss_fn = problem()
    again = run_arm(ArmSpec("E_learned_disjoint", ORDER, rho="0.8"), host, data, loss_fn, RECIPE)
    ref = arms["E_learned_disjoint"]
    assert [r.mask.checksum() for r in again.records] == [r.mask.checksum() for r in ref.records]
    assert [r.delta.checksum() for r in again.records] == [r.delta.checksum() for r in ref.records]
    rev = run_arm(ArmSpec("E_learned_disjoint", tuple(reversed(ORDER)), rho="0.8"), host, data, loss_fn, RECIPE)
    assert rev.spec.order == tuple(reversed(ORDER)) and [r.skill for r in rev.records] == list(reversed(ORDER))
    assert rev.records[0].mask != ref.records[0].mask


# 10 -----------------------------------------------------------------------------------
DATA_ROOT = Path("C:/Users/clclc/BusinessProject/Rineya-Dataset")


@pytest.fixture(scope="module")
def corpus():
    if not (DATA_ROOT / "data" / "manifest.json").exists():
        pytest.skip("MAXRL-COMP-002 not materialised on this host")
    from clare.data import Corpus
    return Corpus(DATA_ROOT)


def test_10_no_heldout_data_reaches_discovery(corpus):
    from clare.data import AccessRefused, assert_parent_disjoint

    for role in ("test", "challenge", "guard_dev", "guard_test", "guard_challenge", "warmstart", "replay_train"):
        for purpose in ("train", "discovery", "calibration", "evaluation"):
            with pytest.raises(AccessRefused):
                corpus.rows(role, purpose)
    for purpose in ("train", "discovery"):
        with pytest.raises(AccessRefused):
            corpus.rows("dev", purpose)
    with pytest.raises(AccessRefused):
        corpus.rows("dev", "discovery", with_targets=True)
    dev = corpus.sample("dev", "evaluation", ["R", "RS", "RUSP"], 16, "test-key")
    for skill in ("R", "U", "S", "P"):
        bank = corpus.atomic_bank(skill, purpose="discovery")
        assert all(r.role == "train" and r.signature == skill and r.tag == "core" for r in bank)
        assert_parent_disjoint(bank, dev)
    # negative control: the disjointness check does fire on shared parents
    with pytest.raises(ValueError):
        assert_parent_disjoint(dev, dev)


# 11 -----------------------------------------------------------------------------------
def test_11_merge_is_deterministic_and_disjoint_sum_order_invariant(arms):
    host = arms["host"]
    ds = [r.delta for r in arms["E_learned_disjoint"].records]
    a = compose(ds).checksum()
    assert a == compose(ds).checksum() == compose(list(reversed(ds))).checksum()
    host.set_frozen(ds)
    via_list = host.frozen_checksum()
    host.set_frozen(compose(ds))
    assert host.frozen_checksum() == via_list == a
    host.set_frozen(None)
    # negative control: OVERLAPPING dense deltas are not guaranteed order-invariant bit-for-bit;
    # the invariance above is a property of disjointness, not of the helper.
    fp = arms["fp"]
    x = DenseDelta(fp, {s.name: np.full(s.numel, 1e8, np.float32) for s in fp})
    y = DenseDelta(fp, {s.name: np.full(s.numel, 1.0, np.float32) for s in fp})
    z = DenseDelta(fp, {s.name: np.full(s.numel, -1e8, np.float32) for s in fp})
    assert compose([x, y, z]).checksum() != compose([x, z, y]).checksum()


# 12 -----------------------------------------------------------------------------------
def test_12_support_metrics_match_brute_force():
    fp = qwen3_mlp_footprint([0, 1], hidden=5, intermediate=7)
    rng = np.random.default_rng(4)

    def rand():
        return DenseDelta(fp, {s.name: np.where(rng.random(s.numel) < 0.4, rng.standard_normal(s.numel), 0).astype(np.float32)
                               for s in fp})
    a, b = rand(), rand()
    g = pair_geometry(a, b)
    x = np.concatenate([a.dense(s.name) for s in fp]).astype(np.float64)
    y = np.concatenate([b.dense(s.name) for s in fp]).astype(np.float64)
    both = (x != 0) & (y != 0)
    assert g["dot"] == pytest.approx(x @ y, rel=1e-12)
    assert g["cosine"] == pytest.approx(x @ y / np.linalg.norm(x) / np.linalg.norm(y), rel=1e-12)
    assert g["support_intersection"] == both.sum() and g["support_union"] == ((x != 0) | (y != 0)).sum()
    assert g["jaccard"] == pytest.approx(both.sum() / ((x != 0) | (y != 0)).sum())
    assert g["sign_agreement_on_overlap"] == pytest.approx((np.sign(x[both]) == np.sign(y[both])).mean())
    ma = SkillMask.from_bool(fp, {s.name: a.dense(s.name) != 0 for s in fp})
    mb = SkillMask.from_bool(fp, {s.name: b.dense(s.name) != 0 for s in fp})
    assert mask_pair(ma, mb)["allocated_intersection"] == both.sum()


# 13 -----------------------------------------------------------------------------------
def test_13_strict_disjoint_deltas_have_zero_overlap_and_zero_dot(arms):
    for arm in ("D_random_disjoint", "E_learned_disjoint"):
        res = arms[arm]
        for row in pairwise(res.deltas()):
            assert row["support_intersection"] == 0 and row["dot"] == 0.0
        assert all(v["intersection"] == 0 for v in pairwise_overlaps({r.skill: r.mask for r in res.records}).values())


# 14 -----------------------------------------------------------------------------------
def test_14_parameter_accounting_is_exact(arms):
    fp = arms["fp"]
    summ = fp.summary()
    assert fp.total == sum(s.numel for s in fp) == sum(summ["coordinates_per_module"].values())
    res = arms["E_learned_disjoint"]
    rows = layer_table(res.allocation)
    assert sum(r["selected"] for r in rows) == sum(r.mask.count() for r in res.records)
    for rec in res.records:
        assert rec.delta.nonzero() <= rec.mask.count()  # active non-zeros reported separately
    assert qwen3_mlp_footprint(range(8, 28), 1024, 3072).total == 188_743_680


# extras -------------------------------------------------------------------------------
def test_dense_masked_and_sparse_training_agree_to_float32_rounding():
    """Same algorithm, different tensor layouts. MEASURED: identical losses and gradients after the
    first update, but grad-norm reductions and elementwise kernels round differently for the dense
    and compact layouts, so the deltas agree to float32 rounding, not bit-for-bit. This is why no
    arm mixes the two paths (D and E both retrain through the sparse path)."""
    out = {}
    for mode in ("dense", "sparse"):
        model, fp, host, data, loss_fn = problem(seed=5)
        rng = np.random.default_rng(1)
        mask = SkillMask.from_bool(fp, {s.name: rng.random(s.numel) < 0.25 for s in fp})
        host.start_dense(mask) if mode == "dense" else host.start_sparse(mask)
        run_stage(host, data.rows["S"], data.groups["S"], loss_fn, NO_CLIP)
        d = host.trainable_delta()
        out[mode] = np.concatenate([d.dense(s.name) for s in fp])
        if mode == "dense":
            assert host.outside_mask_nonzero() == 0
        host.stop_trainable()
    a, b = out["dense"], out["sparse"]
    assert ((a != 0) == (b != 0)).all(), "supports must match exactly"
    assert np.abs(a - b).max() <= 1e-5 * max(1e-12, np.abs(a).max())


def test_disk_deltas_round_trip_with_the_same_content_checksum(arms, tmp_path):
    host = arms["host"]
    rec = arms["E_learned_disjoint"].records[1]
    write_sparse_stream(tmp_path / "sp", rec.mask, ((s.name, rec.delta.values(s.name)) for s in arms["fp"]))
    assert DiskDelta(tmp_path / "sp").checksum() == rec.delta.checksum()
    dense = compose([r.delta for r in arms["E_learned_disjoint"].records])
    write_dense_stream(tmp_path / "dn", arms["fp"], dense.iter_dense())
    assert DiskDelta(tmp_path / "dn").checksum() == dense.checksum()
    with pytest.raises(FileExistsError):
        write_dense_stream(tmp_path / "dn", arms["fp"], dense.iter_dense())
    # save_trainable streams the open stage and round-trips
    host.start_sparse(rec.mask)
    saved = host.save_trainable(tmp_path / "st")
    assert saved.checksum() == host.trainable_delta().checksum()
    host.stop_trainable()
    # negative control: a flipped payload byte is caught on open
    p = tmp_path / "dn.dense.f32"
    raw = bytearray(p.read_bytes()); raw[3] ^= 0x40; p.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        DiskDelta(tmp_path / "dn")
