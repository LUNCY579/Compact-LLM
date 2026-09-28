"""Torch-free tests of the footprint, exact-budget selection, schedule and allocation state."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from clare.footprint import Footprint, qwen3_mlp_footprint
from clare.masks import AllocationState, SkillMask, layer_table, pairwise_overlaps
from clare.schedule import budgets, schedule_table
from clare.select import (importance_keys, random_keys, seed_key, select_random,
                          select_smallest, select_top_importance, splitmix64)


def small_fp() -> Footprint:
    return Footprint([("model.layers.0.mlp.up_proj.weight", (6, 5)),
                      ("model.layers.0.mlp.down_proj.weight", (5, 6)),
                      ("model.layers.1.mlp.up_proj.weight", (3, 7))])


def all_true(fp):
    return {s.name: np.ones(s.numel, dtype=bool) for s in fp}


# -- footprint ---------------------------------------------------------------------------

def test_footprint_offsets_digest_and_locate():
    fp = small_fp()
    assert fp.total == 30 + 30 + 21
    assert [s.offset for s in fp] == [0, 30, 60]
    assert fp.slot("model.layers.1.mlp.up_proj.weight").layer == 1
    assert fp.slot("model.layers.0.mlp.down_proj.weight").module == "mlp.down_proj"
    for g in range(fp.total):
        slot, local = fp.locate(g)
        assert slot.offset + local == g
    assert Footprint.from_spec(fp.spec()).digest() == fp.digest()
    reordered = Footprint(list(reversed([(s.name, s.shape) for s in fp])))
    assert reordered.digest() != fp.digest(), "order is part of the identity"
    with pytest.raises(ValueError):
        Footprint([("a", (2,)), ("a", (3,))])


def test_qwen3_footprint_counts_match_native_arithmetic():
    fp = qwen3_mlp_footprint(range(8, 28), hidden=1024, intermediate=3072)
    assert len(fp) == 60
    assert fp.total == 60 * 1024 * 3072 == 188_743_680


# -- schedule ----------------------------------------------------------------------------

def test_budgets_exact_and_rho_semantics():
    total = 188_743_680
    bs = budgets(total, ["R", "U", "S", "P"], "0.9")
    assert [b.budget for b in bs] == [18_874_368, 16_986_931, 15_288_238, 13_759_414]
    assert bs[-1].free_after == total - sum(b.budget for b in bs)
    for b in bs:  # CLARE Eq. 6-7: round-half-up of (1 - rho) * |A_t|
        exact = Fraction(1, 10) * b.free_before + Fraction(1, 2)
        assert b.budget == exact.numerator // exact.denominator
    # Rounding is half-up, not floor: 15 * 0.1 = 1.5 -> 2; 14 * 0.1 = 1.4 -> 1; tiny pools get >= 1.
    assert [b.budget for b in budgets(15, "R", "0.9")] == [2]
    assert [b.budget for b in budgets(14, "R", "0.9")] == [1]
    assert [b.budget for b in budgets(3, "RU", "0.9")] == [1, 1]
    table = {r["rho"]: r for r in schedule_table(total, "RUSP", ["0.80", "0.90", "0.95", "0.975"])}
    assert abs(table["0.80"]["claimed_fraction"] - 0.5904) < 1e-6
    assert abs(table["0.90"]["claimed_fraction"] - 0.3439) < 1e-6
    assert abs(table["0.95"]["claimed_fraction"] - 0.18549375) < 1e-6
    assert abs(table["0.975"]["claimed_fraction"] - 0.09631) < 1e-4
    assert budgets(total, "RUSP", 0.9) == bs, "float 0.9 must mean 9/10"
    with pytest.raises(ValueError):
        budgets(total, "RUSP", "1.0")
    with pytest.raises(ValueError):
        budgets(total, "RRSP", "0.9")


# -- selection ---------------------------------------------------------------------------

def brute_force(fp, keyfn, eligible, budget):
    rows = []
    for s in fp:
        k, e = keyfn(s), eligible[s.name]
        for i in range(s.numel):
            if e[i]:
                rows.append((int(k[i]), s.offset + i))
    rows.sort()
    chosen = {g for _, g in rows[:budget]}
    return {s.name: np.array([s.offset + i in chosen for i in range(s.numel)]) for s in fp}


@pytest.mark.parametrize("budget", [0, 1, 7, 40, 81])
def test_select_smallest_matches_brute_force_with_ties(budget):
    fp = small_fp()
    rng = np.random.default_rng(3)
    # Heavy ties: only 4 distinct keys, including values that differ only in low digits.
    table = {s.name: rng.choice(np.array([5, 5 + (1 << 20), 1 << 50, 7], dtype=np.uint64), s.numel)
             for s in fp}
    elig = all_true(fp)
    sel, rep = select_smallest(fp, lambda s: (table[s.name], elig[s.name]), budget)
    ref = brute_force(fp, lambda s: table[s.name], elig, budget)
    for s in fp:
        assert (sel[s.name] == ref[s.name]).all()
    assert sum(int(v.sum()) for v in sel.values()) == budget
    assert rep.taken_below_threshold + rep.taken_at_threshold == budget


def test_select_respects_eligibility_and_budget_bounds():
    fp = small_fp()
    rng = np.random.default_rng(0)
    elig = {s.name: rng.random(s.numel) < 0.5 for s in fp}
    n_elig = sum(int(v.sum()) for v in elig.values())
    sel, _ = select_random(fp, elig, n_elig, seed_key("t", 1))
    for s in fp:
        assert not (sel[s.name] & ~elig[s.name]).any()
        assert (sel[s.name] == elig[s.name]).all()
    with pytest.raises(ValueError):
        select_random(fp, elig, n_elig + 1, seed_key("t", 1))


def test_random_selection_is_seeded_reproducible_and_distinct_keys():
    fp = small_fp()
    a, _ = select_random(fp, all_true(fp), 20, seed_key("arm-D", 0, "R"))
    b, _ = select_random(fp, all_true(fp), 20, seed_key("arm-D", 0, "R"))
    c, _ = select_random(fp, all_true(fp), 20, seed_key("arm-D", 1, "R"))
    assert all((a[k] == b[k]).all() for k in a)
    assert any((a[k] != c[k]).any() for k in a)
    keys = np.concatenate([random_keys(s, seed_key("x")) for s in fp])
    assert np.unique(keys).size == keys.size, "splitmix64 of distinct inputs must not collide"
    x = np.arange(1000, dtype=np.uint64)
    assert np.unique(splitmix64(x)).size == 1000


def test_importance_selection_prefers_large_and_breaks_zero_ties_by_hash_not_position():
    fp = qwen3_mlp_footprint([0, 1], hidden=4, intermediate=8)
    q = {s.name: np.zeros(s.numel, dtype=np.float32) for s in fp}
    last = fp.slots[-1].name
    q[last][:3] = [3.0, 2.0, 1.0]
    sel, _ = select_top_importance(fp, q, None, 3, seed_key("tb"))
    assert sel[last][:3].all() and sum(int(v.sum()) for v in sel.values()) == 3
    # Budget beyond the non-zero support: the zero-importance picks are spread by the hash.
    sel, _ = select_top_importance(fp, q, None, 3 + 60, seed_key("tb"))
    first = fp.slots[0].name
    assert sel[last][:3].all()
    assert 0 < int(sel[first].sum()) < fp.slots[0].numel
    with pytest.raises(ValueError):
        importance_keys(fp.slots[0], -np.ones(fp.slots[0].numel, dtype=np.float32), seed_key("x"))
    with pytest.raises(ValueError):
        importance_keys(fp.slots[0], np.full(fp.slots[0].numel, np.nan, dtype=np.float32), seed_key("x"))


# -- masks and allocation ----------------------------------------------------------------

def test_mask_pack_roundtrip_checksum_and_binary(tmp_path):
    fp = small_fp()
    rng = np.random.default_rng(1)
    bools = {s.name: rng.random(s.numel) < 0.3 for s in fp}
    m = SkillMask.from_bool(fp, bools)
    for s in fp:
        assert m.bool(s.name).dtype == np.bool_
        assert (m.bool(s.name) == bools[s.name]).all()
    side = m.save(tmp_path / "m")
    m2 = SkillMask.load(tmp_path / "m", fp)
    assert m2 == m and m2.checksum() == side["mask_checksum"]
    with pytest.raises(FileExistsError):
        m.save(tmp_path / "m")
    with pytest.raises(TypeError):
        SkillMask.from_bool(fp, {s.name: bools[s.name].astype(np.uint8) for s in fp})
    raw = bytearray((tmp_path / "m.bin").read_bytes())
    raw[0] ^= 1
    (tmp_path / "m.bin").write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        SkillMask.load(tmp_path / "m", fp)


def test_invert_clears_padding_and_union_overlap():
    fp = small_fp()
    m = SkillMask.empty(fp)
    inv = m.invert()
    assert inv.count() == fp.total
    assert inv.overlap(m) == 0 and inv.union(m).count() == fp.total


def test_strict_allocation_refuses_reclaim_and_tracks_free(tmp_path):
    fp = small_fp()
    st = AllocationState(fp, strict=True)
    bs = budgets(fp.total, "RUSP", "0.8")
    for b in bs:
        free = {s.name: st.free_bool(s.name) for s in fp}
        sel, _ = select_random(fp, free, b.budget, seed_key("D", 0, b.skill))
        st.claim(b.skill, SkillMask.from_bool(fp, sel), expected_budget=b.budget)
        assert st.free_count() == b.free_after
    ov = pairwise_overlaps({k: st.mask(k) for k in st.skills})
    assert all(v["intersection"] == 0 and v["jaccard"] == 0.0 for v in ov.values())
    with pytest.raises(ValueError):  # a previously claimed coordinate
        st.claim("X", st.mask("R"))
    with pytest.raises(ValueError):  # a repeated skill name
        st.claim("R", SkillMask.empty(fp))
    # Resume preserves the allocation exactly.
    st.save(tmp_path / "state")
    back = AllocationState.load(tmp_path / "state", fp)
    assert back.checksum() == st.checksum() and back.skills == st.skills
    for s in fp:
        assert (back.owner(s.name) == st.owner(s.name)).all()
    rows = layer_table(st)
    assert sum(r["selected"] for r in rows) == sum(b.budget for b in bs)


def test_loose_allocation_allows_overlap_only_when_not_strict():
    fp = small_fp()
    st = AllocationState(fp, strict=False)
    full = SkillMask.full(fp)
    st.claim("A", full)
    st.claim("B", full)
    assert st.mask("A").overlap(st.mask("B")) == fp.total
