from __future__ import annotations

import pytest
import torch

from rineya_pcsm.property_maps import (
    PropertyMaps,
    aggregate_property_maps,
    build_property_maps,
)


def _assert_valid(maps: PropertyMaps, shape: tuple[int, int]) -> None:
    assert maps.shape == shape
    assert maps.element.dtype == torch.float32
    assert maps.row.shape == (shape[0],)
    assert maps.column.shape == (shape[1],)
    assert maps.joint.shape == shape
    for values in (maps.element, maps.row, maps.column, maps.joint):
        assert torch.isfinite(values).all()
        assert (values >= 0).all()


def test_all_four_property_maps_are_nonnegative_and_bounded() -> None:
    maps = build_property_maps(torch.tensor([[-4.0, 0.0, 2.0], [1.0, -8.0, 3.0]]))
    _assert_valid(maps, (2, 3))
    assert maps.channels().shape == (2, 3, 4)
    assert maps.element.square().mean().sqrt().item() == pytest.approx(1.0)
    assert maps.row.mean().item() == pytest.approx(1.0)
    assert maps.column.mean().item() == pytest.approx(1.0)
    assert maps.joint.mean().item() == pytest.approx(1.0)


def test_element_map_uses_centered_rms_standardized_magnitude() -> None:
    weight = torch.tensor([[1.0, 2.0], [5.0, 8.0]])
    first = build_property_maps(weight)
    shifted_scaled = build_property_maps((weight + 100.0) * 3.0)
    assert torch.allclose(first.element, shifted_scaled.element, atol=1e-6)
    centered = weight - weight.mean()
    expected = centered.abs() / centered.square().mean().sqrt()
    assert torch.allclose(first.element, expected)


def test_projection_is_deterministic_and_preserves_coordinate_endpoints() -> None:
    maps = build_property_maps(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    first = maps.project((5, 2))
    second = maps.project((5, 2))
    assert torch.equal(first.element, second.element)
    assert first.element[0, 0] == maps.element[0, 0]
    assert first.element[-1, -1] == maps.element[-1, -1]
    _assert_valid(first, (5, 2))


def test_equal_shape_projection_is_bitwise_identity() -> None:
    maps = build_property_maps(torch.tensor([[1.0, 7.0, 2.0], [9.0, 3.0, 5.0]]))
    projected = maps.project(maps.shape)
    for name in ("element", "row", "column", "joint"):
        assert torch.equal(getattr(projected, name), getattr(maps, name))


def test_different_shape_projection_uses_declared_nearest_coordinates() -> None:
    maps = build_property_maps(torch.tensor([[1.0, 7.0, 2.0], [9.0, 3.0, 5.0]]))
    projected = maps.project((4, 2))
    row_indices = torch.tensor([0, 0, 1, 1])
    column_indices = torch.tensor([0, 2])
    assert torch.equal(
        projected.element,
        maps.element.index_select(0, row_indices).index_select(1, column_indices),
    )
    assert torch.equal(projected.row, maps.row.index_select(0, row_indices))
    assert torch.equal(projected.column, maps.column.index_select(0, column_indices))
    assert torch.equal(
        projected.joint,
        maps.joint.index_select(0, row_indices).index_select(1, column_indices),
    )


def test_alpha_aggregation_projects_different_peer_shapes() -> None:
    first = build_property_maps(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    second = build_property_maps(torch.arange(20, dtype=torch.float32).reshape(4, 5))
    combined = aggregate_property_maps(
        (first, second), torch.tensor([0.25, 0.75]), (3, 4)
    )
    _assert_valid(combined, (3, 4))


def test_alpha_aggregation_is_direct_weighted_sum_and_differentiable() -> None:
    first = build_property_maps(torch.tensor([[1.0, 2.0], [3.0, 8.0]]))
    second = build_property_maps(torch.tensor([[8.0, 1.0], [2.0, 3.0]]))
    alphas = torch.tensor([0.25, 0.75], requires_grad=True)
    combined = aggregate_property_maps(
        (first, second), alphas, (2, 2), differentiable_attention=True
    )
    expected = 0.25 * first.element + 0.75 * second.element
    assert torch.allclose(combined.element, expected)
    combined.element[0, 0].backward()
    assert alphas.grad is not None
    assert torch.count_nonzero(alphas.grad) > 0


def test_property_maps_require_rank_two_weight() -> None:
    with pytest.raises(ValueError, match="rank-two"):
        build_property_maps(torch.ones(4))


def test_property_maps_sanitize_nonfinite_and_degenerate_weights() -> None:
    nonfinite = build_property_maps(
        torch.tensor([[float("nan"), float("inf")], [-2.0, 3.0]])
    )
    constant = build_property_maps(torch.full((2, 3), 7.0))
    _assert_valid(nonfinite, (2, 2))
    _assert_valid(constant, (2, 3))
    assert torch.equal(constant.element, torch.zeros(2, 3))
    assert torch.allclose(constant.row, torch.ones(2))
    assert torch.allclose(constant.column, torch.ones(3))
