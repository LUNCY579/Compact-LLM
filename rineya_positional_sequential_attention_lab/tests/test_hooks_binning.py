"""Binning correctness and the chunk memory contract (SPEC 5.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rineya_psa.hooks.binning import (
    CHUNK_BUDGET_BYTES,
    AbsoluteBinning,
    DistanceBinning,
    binning_from_describe,
)

# --------------------------------------------------------------------------
# AbsoluteBinning
# --------------------------------------------------------------------------


def test_absolute_matches_floor_definition():
    for n_bins in (1, 2, 3, 7, 16):
        for T in (1, 2, 5, 17, 64, 1000):
            got = AbsoluteBinning(n_bins).bin_ids(T).numpy()
            want = np.clip(np.floor(np.arange(T) / T * n_bins).astype(np.int64), 0, n_bins - 1)
            assert np.array_equal(got, want), (n_bins, T)


def test_absolute_t_equals_one():
    for n_bins in (1, 4, 32):
        ids = AbsoluteBinning(n_bins).bin_ids(1)
        assert ids.tolist() == [0]
        assert AbsoluteBinning(n_bins).bin_ids_numpy(1).tolist() == [0]


def test_absolute_exact_bin_boundaries():
    # 1000 tokens over 10 bins: bin k starts at exactly 100k and nowhere else.
    ids = AbsoluteBinning(10).bin_ids(1000).numpy()
    starts = [int(np.argmax(ids == k)) for k in range(10)]
    assert starts == [100 * k for k in range(10)]
    assert ids[0] == 0 and ids[-1] == 9
    # The boundary token itself belongs to the *upper* bin (half-open [k/n, (k+1)/n)).
    assert ids[99] == 0 and ids[100] == 1


def test_absolute_never_exceeds_range_when_bins_exceed_tokens():
    ids = AbsoluteBinning(64).bin_ids(3)
    assert ids.tolist() == [0, 21, 42]
    assert int(ids.max()) < 64


def test_absolute_numpy_and_torch_agree():
    b = AbsoluteBinning(13)
    for T in (1, 13, 14, 257):
        assert np.array_equal(b.bin_ids(T).numpy(), b.bin_ids_numpy(T))


def test_absolute_rejects_zero_bins():
    with pytest.raises(ValueError):
        AbsoluteBinning(0)


# --------------------------------------------------------------------------
# gather / valid_mask / one_hot
# --------------------------------------------------------------------------


def test_gather_source_expands_table_over_positions():
    b = AbsoluteBinning(4)
    table = torch.tensor([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])  # [2 heads, 4 bins]
    out = b.gather_source(table, 8)
    assert out.shape == (2, 8)
    assert out[0].tolist() == [10, 10, 20, 20, 30, 30, 40, 40]
    assert out[1].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_valid_mask_excludes_padding_with_a_neutral_fill():
    b = AbsoluteBinning(2)
    table = torch.tensor([[5.0, 7.0]])
    valid = torch.tensor([False, False, True, True])
    # additive bias -> padding gets 0.0
    assert b.gather_source(table, 4, valid_mask=valid, fill=0.0)[0].tolist() == [0, 0, 7, 7]
    # multiplicative scale -> padding gets 1.0
    assert b.gather_source(table, 4, valid_mask=valid, fill=1.0)[0].tolist() == [1, 1, 7, 7]


def test_valid_mask_may_be_batched():
    b = AbsoluteBinning(2)
    table = torch.tensor([[5.0, 7.0]])
    valid = torch.tensor([[True, True, True, True], [False, False, True, True]])
    out = b.gather_source(table, 4, valid_mask=valid, fill=0.0)
    assert out.shape == (2, 1, 4)
    assert out[0, 0].tolist() == [5, 5, 7, 7]
    assert out[1, 0].tolist() == [0, 0, 7, 7]


def test_gather_rejects_a_table_with_the_wrong_bin_count():
    with pytest.raises(ValueError, match="n_bins"):
        AbsoluteBinning(4).gather_source(torch.zeros(2, 3), 8)


def test_one_hot_is_the_indicator_of_bin_ids():
    b = AbsoluteBinning(5)
    oh = b.one_hot(37)
    ids = b.bin_ids(37)
    assert oh.shape == (37, 5)
    assert torch.equal(oh.sum(dim=1), torch.ones(37))
    assert torch.equal(oh.argmax(dim=1), ids)


# --------------------------------------------------------------------------
# DistanceBinning
# --------------------------------------------------------------------------


def test_distance_uniform_matches_floor_on_the_causal_triangle():
    n = 4
    d = DistanceBinning.uniform(n)
    T = 16
    ids = d.chunk_bin_ids(T, 0, T).numpy()
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    dist = (i - j) / T
    want = np.floor(np.clip(dist, 0.0, None) * n).astype(np.int64)
    causal = i >= j
    assert np.array_equal(ids[causal], want[causal])


def test_distance_sends_the_masked_upper_triangle_to_bin_zero():
    d = DistanceBinning.uniform(8)
    T = 32
    ids = d.chunk_bin_ids(T, 0, T).numpy()
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    assert (ids[i < j] == 0).all()
    assert ids.min() >= 0 and ids.max() < d.n_bins


def test_distance_q_offset_tracks_absolute_query_position():
    d = DistanceBinning.uniform(4)
    T = 16
    # Query row 0 of a chunk that starts at absolute position 8.
    with_offset = d.chunk_bin_ids(T, 0, 1, q_offset=8)
    direct = d.chunk_bin_ids(T, 8, 9)
    assert torch.equal(with_offset, direct)


def test_distance_edges_must_increase():
    with pytest.raises(ValueError, match="increasing"):
        DistanceBinning((0.5, 0.5))
    with pytest.raises(ValueError, match="increasing"):
        DistanceBinning((0.9, 0.1))
    with pytest.raises(ValueError):
        DistanceBinning(())


def test_distance_has_no_source_only_view():
    d = DistanceBinning.uniform(4)
    assert d.is_query_dependent
    for call in (lambda: d.bin_ids(8), lambda: d.bin_ids_numpy(8), lambda: d.one_hot(8)):
        with pytest.raises(ValueError, match="query-dependent"):
            call()
    with pytest.raises(ValueError, match="query-dependent"):
        d.gather_source(torch.zeros(2, 4), 8)


def test_distance_numpy_and_torch_agree():
    d = DistanceBinning((-0.1, 0.0, 0.25, 0.6))
    a = d.chunk_bin_ids(20, 3, 11).numpy()
    b = d.chunk_bin_ids_numpy(20, 3, 11)
    assert np.array_equal(a, b)


def test_distance_t_equals_one():
    d = DistanceBinning.uniform(4)
    ids = d.chunk_bin_ids(1, 0, 1)
    assert ids.shape == (1, 1)
    assert 0 <= int(ids[0, 0]) < d.n_bins


# --------------------------------------------------------------------------
# memory contract
# --------------------------------------------------------------------------


def test_absolute_chunk_is_a_broadcast_row_costing_nothing_per_query():
    b = AbsoluteBinning(16)
    T = 8192
    ids = b.chunk_bin_ids(T, 0, 4096)
    assert ids.shape == (1, T)
    # Storage is O(T), not O(rows * T): the query axis is a broadcast of size 1.
    assert ids.untyped_storage().nbytes() == T * 8
    assert b.bytes_per_entry() == 0


def test_distance_default_chunk_stays_far_under_the_budget():
    d = DistanceBinning.uniform(16)
    for T in (4096, 8192, 32768):
        ids = d.chunk_bin_ids(T, 0, 256)
        assert ids.shape == (256, T)
        assert ids.untyped_storage().nbytes() == 256 * T * 8
        # Declared peak (int64 result + transient float32 grid) is the honest number.
        assert d.bytes_per_entry() * 256 * T <= CHUNK_BUDGET_BYTES


def test_distance_refuses_a_chunk_that_would_blow_the_budget():
    d = DistanceBinning.uniform(16)
    T = 4096
    limit = d.max_chunk_rows(T)
    assert limit * d.bytes_per_entry() * T <= CHUNK_BUDGET_BYTES
    d.assert_chunk_within_budget(T, limit)
    with pytest.raises(ValueError, match="budget"):
        d.assert_chunk_within_budget(T, limit + 1)
    # The full T x T grid at T=8192 is 768 MB; asking for it must raise, not swap.
    with pytest.raises(ValueError, match="budget"):
        d.chunk_bin_ids(8192, 0, 8192)


def test_max_chunk_rows_is_monotone_in_length():
    d = DistanceBinning.uniform(8)
    assert d.max_chunk_rows(4096) > d.max_chunk_rows(8192) >= 1


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_describe_round_trips():
    for b in (AbsoluteBinning(1), AbsoluteBinning(32), DistanceBinning.uniform(6),
              DistanceBinning((-0.5, 0.0, 0.5))):
        record = b.describe()
        assert record["n_bins"] == b.n_bins
        assert record["name"] == b.name
        rebuilt = binning_from_describe(record)
        assert rebuilt == b
        assert rebuilt.describe() == record


def test_describe_is_json_serialisable():
    import json

    for b in (AbsoluteBinning(4), DistanceBinning.uniform(4)):
        assert json.loads(json.dumps(b.describe())) == b.describe()
