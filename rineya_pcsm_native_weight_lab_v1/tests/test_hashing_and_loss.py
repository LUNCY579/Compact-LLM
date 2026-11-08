from __future__ import annotations

import numpy as np
import pytest
import torch

from rineya_pcsm.hashing import canonical_json_hash, fingerprint_tensors, mask_hash
from rineya_pcsm.losses import assistant_only_nll


def test_canonical_json_hash_ignores_mapping_order() -> None:
    assert canonical_json_hash({"a": 1, "b": 2}) == canonical_json_hash(
        {"b": 2, "a": 1}
    )


def test_mask_hash_covers_shape_and_binary_contents() -> None:
    mask = torch.tensor([[True, False], [False, True]])
    assert mask_hash(mask) == mask_hash(mask.numpy())
    assert mask_hash(mask) != mask_hash(mask.reshape(4))
    with pytest.raises(ValueError):
        mask_hash(np.array([0, 2]))


def test_assistant_nll_rejects_zero_supervised_tokens() -> None:
    logits = torch.zeros(1, 3, 5)
    labels = torch.full((1, 3), -100)
    with pytest.raises(RuntimeError, match="zero supervised"):
        assistant_only_nll(logits, labels)


def test_assistant_nll_counts_shifted_tokens() -> None:
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2]])
    loss, count = assistant_only_nll(logits, labels)
    assert count == 2
    assert torch.isclose(loss, torch.log(torch.tensor(3.0)))


def test_tensor_fingerprint_covers_bytes_and_respects_exclusion() -> None:
    tensors = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    original = fingerprint_tensors(tensors)
    tensors["b"][0] = 4.0
    assert fingerprint_tensors(tensors) != original
    assert fingerprint_tensors(tensors, exclude=["b"]) == fingerprint_tensors(
        {"a": tensors["a"]}
    )
