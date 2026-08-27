from __future__ import annotations

import numpy as np
import pytest

from rineya_pcsm.controls import (
    bitwise_frozen_audit,
    deterministic_random_mask,
    deterministic_topk_mask,
    normalize_update_on_support,
    require_arm,
    support_digest,
    support_size,
)


def test_topk_is_exact_and_ties_break_by_flat_index() -> None:
    scores = np.asarray([[3.0, 3.0, 1.0], [3.0, 2.0, 0.0]])
    mask = deterministic_topk_mask(scores, 2)
    expected = np.asarray([[True, True, False], [False, False, False]])
    assert np.array_equal(mask, expected)


def test_seeded_random_mask_and_digest_are_reproducible() -> None:
    first = deterministic_random_mask((4, 5), 7, np.random.default_rng(91))
    second = deterministic_random_mask((4, 5), 7, np.random.default_rng(91))
    assert np.array_equal(first, second)
    assert int(np.count_nonzero(first)) == 7
    assert support_digest(first) == support_digest(second)


def test_update_normalization_enforces_l0_and_l2_budget() -> None:
    update = np.arange(1.0, 7.0).reshape(2, 3)
    support = np.asarray([[True, False, True], [False, True, False]])
    matched = normalize_update_on_support(update, support, target_l2=2.5)
    assert np.all(matched[~support] == 0.0)
    assert np.linalg.vector_norm(matched[support]) == pytest.approx(2.5)


def test_frozen_audit_is_bitwise_not_only_numeric() -> None:
    before = np.zeros((2, 2), dtype=np.float64)
    after = before.copy()
    support = np.asarray([[True, False], [False, False]])
    after[1, 1] = -0.0
    audit = bitwise_frozen_audit(before, after, support)
    assert not audit.exact
    assert audit.changed_frozen_coordinate_count == 1


def test_support_budget_and_arm_validation_reject_invalid_values() -> None:
    assert support_size(100, 0.01) == 1
    assert support_size(101, 0.01) == 2
    assert require_arm("real_peer_pcsm").uses_aligned_peers
    with pytest.raises(ValueError, match="unknown control arm"):
        require_arm("oracle")
    with pytest.raises(ValueError, match="fraction"):
        support_size(10, 0.0)
