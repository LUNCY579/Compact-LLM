from __future__ import annotations

import pytest
import torch

from rineya_pcsm.descriptors import describe_tensors, tensor_descriptor


@pytest.mark.parametrize("dimension", [16, 24, 32])
def test_descriptor_has_requested_dimension_and_is_finite_float32(
    dimension: int,
) -> None:
    tensor = torch.tensor([[1.0, float("nan")], [float("inf"), -2.0]])
    descriptor = tensor_descriptor(tensor, dimension=dimension)
    assert descriptor.shape == (dimension,)
    assert descriptor.dtype == torch.float32
    assert torch.isfinite(descriptor).all()
    assert descriptor[11].item() == pytest.approx(0.5)


def test_descriptor_is_deterministic_and_scale_normalized() -> None:
    tensor = torch.arange(24, dtype=torch.float64).reshape(4, 6) - 8
    first = tensor_descriptor(tensor)
    second = tensor_descriptor(tensor)
    scaled = tensor_descriptor(tensor * 3)
    assert torch.equal(first, second)
    assert torch.allclose(scaled, first, atol=1e-6, rtol=1e-6)


def test_descriptor_includes_optional_calibration_depth_and_role() -> None:
    tensor = torch.tensor([[-2.0, 1.0], [4.0, -1.0]])
    activations = torch.tensor([-3.0, 1.0, 2.0])
    gradient = torch.tensor([[-1.0, 0.5], [2.0, -4.0]])
    descriptor = tensor_descriptor(
        tensor,
        activations=activations,
        calibration_gradient=gradient,
        normalized_depth=0.75,
        role="mlp.down_proj",
    )
    rescaled = tensor_descriptor(
        tensor * 7,
        activations=activations * 11,
        calibration_gradient=gradient * 5,
        normalized_depth=0.75,
        role="mlp.down_proj",
    )
    other_role = tensor_descriptor(
        tensor,
        normalized_depth=0.75,
        role="self_attn.q_proj",
    )
    assert descriptor[14].item() == pytest.approx(0.75)
    assert descriptor[15] != other_role[15]
    assert torch.count_nonzero(descriptor[16:22]) > 0
    assert torch.allclose(descriptor, rescaled, atol=1e-6, rtol=1e-6)


def test_describe_tensors_returns_canonical_key_order() -> None:
    result = describe_tensors(
        {"z": torch.ones(2, 2), "a": torch.zeros(2, 2)},
        normalized_depths={"z": 1.0, "a": 0.0},
        roles={"z": "mlp.up_proj", "a": "mlp.down_proj"},
    )
    assert list(result) == ["a", "z"]
    assert result["z"][14] == 1.0
    assert result["a"][15] != result["z"][15]


@pytest.mark.parametrize("dimension", [15, 33])
def test_descriptor_rejects_dimensions_outside_qualified_range(dimension: int) -> None:
    with pytest.raises(ValueError, match="descriptor dimension"):
        tensor_descriptor(torch.ones(2, 2), dimension=dimension)


@pytest.mark.parametrize("depth", [-0.1, 1.1, float("nan")])
def test_descriptor_rejects_invalid_normalized_depth(depth: float) -> None:
    with pytest.raises(ValueError, match="normalized_depth"):
        tensor_descriptor(torch.ones(2, 2), normalized_depth=depth)
