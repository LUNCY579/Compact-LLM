from __future__ import annotations

import torch
from torch import nn

from rineya_pcsm.application import (
    NativeUpdatePlan,
    TensorUpdatePlan,
    apply_update_plan,
)
from rineya_pcsm.calibration import first_order_loss_change
from rineya_pcsm.generator import (
    GLOBAL_FEATURE_BLOCKS,
    LOCAL_FEATURE_NAMES,
    PCSMGenerator,
)
from rineya_pcsm.optimizer import ExactSupportAdamW
from rineya_pcsm.property_maps import build_property_maps
from rineya_pcsm.topk import stable_topk_mask, topk_ste


def test_generator_uses_target_gradient_and_peer_features_with_bounded_a() -> None:
    torch.manual_seed(7)
    dimension = 16
    generator = PCSMGenerator(dimension, hidden_dimension=12)
    target = torch.arange(12, dtype=torch.float32).reshape(3, 4) - 4
    gradient = torch.flip(target, dims=(1,)) / 5
    target_descriptor = torch.arange(dimension, dtype=torch.float32)
    gradient_descriptor = target_descriptor + 100
    peer_descriptor = target_descriptor + 200
    output = generator(
        target,
        gradient,
        target_descriptor,
        gradient_descriptor,
        peer_descriptor,
        build_property_maps(target.abs() + 1),
    )
    assert output.scores.shape == target.shape
    assert output.stabilizer.shape == target.shape
    assert output.local_features.shape == (*target.shape, len(LOCAL_FEATURE_NAMES))
    assert output.global_features.shape == (dimension * len(GLOBAL_FEATURE_BLOCKS),)
    assert torch.equal(output.global_features[:dimension], target_descriptor)
    assert torch.equal(
        output.global_features[dimension : 2 * dimension], gradient_descriptor
    )
    assert torch.equal(output.global_features[2 * dimension :], peer_descriptor)
    assert torch.isfinite(output.scores).all()
    assert torch.isfinite(output.local_features).all()
    assert (output.feature_weights >= 0).all()
    assert output.feature_weights.sum().item() == 1.0
    assert output.record()["feature_names"] == list(LOCAL_FEATURE_NAMES)
    assert output.record()["feature_weights"] == output.feature_weights.tolist()
    assert generator.configuration()["feature_names"] == list(LOCAL_FEATURE_NAMES)
    assert (output.stabilizer >= 0.25).all()
    assert (output.stabilizer <= 1.0).all()


def test_generator_feature_schema_matches_pcsm_x1_through_x8() -> None:
    assert LOCAL_FEATURE_NAMES == (
        "task_gradient_magnitude",
        "peer_element_magnitude",
        "peer_row_energy",
        "peer_column_energy",
        "peer_joint_leverage",
        "target_element_magnitude",
        "peer_target_structural_difference",
        "gradient_peer_interaction",
    )


def test_shuffling_peer_coordinates_changes_structural_features_and_scores() -> None:
    torch.manual_seed(11)
    generator = PCSMGenerator(16, hidden_dimension=10)
    target = torch.tensor([[1.0, 7.0, 2.0], [9.0, 3.0, 5.0]])
    gradient = torch.tensor([[0.5, -4.0, 1.0], [-2.0, 3.0, -0.25]])
    descriptor = torch.linspace(-1.0, 1.0, 16)
    real_peer = build_property_maps(torch.tensor([[8.0, 1.0, 3.0], [2.0, 9.0, 4.0]]))
    shuffled_peer = build_property_maps(
        torch.tensor([[9.0, 4.0, 1.0], [8.0, 3.0, 2.0]])
    )
    real = generator(
        target,
        gradient,
        descriptor,
        descriptor,
        descriptor,
        real_peer,
    )
    shuffled = generator(
        target,
        gradient,
        descriptor,
        descriptor,
        descriptor,
        shuffled_peer,
    )
    assert not torch.equal(
        real.local_features[..., 1:], shuffled.local_features[..., 1:]
    )
    assert not torch.equal(real.scores, shuffled.scores)


def test_generator_is_meta_differentiable() -> None:
    generator = PCSMGenerator(16, hidden_dimension=8)
    weight = torch.randn(3, 5)
    gradient = torch.randn(3, 5)
    descriptor = torch.randn(16)
    output = generator(
        weight,
        gradient,
        descriptor,
        descriptor + 1,
        descriptor - 1,
        build_property_maps(weight),
    )
    (output.scores.mean() + output.stabilizer.mean()).backward()
    assert all(parameter.grad is not None for parameter in generator.parameters())


def test_generator_receives_genuinely_nonzero_meta_gradient() -> None:
    torch.manual_seed(123)
    generator = PCSMGenerator(16, hidden_dimension=8)
    weight = torch.tensor([[1.0, 7.0, 2.0], [9.0, 3.0, 5.0]])
    calibration = torch.tensor([[0.5, -4.0, 1.0], [-2.0, 3.0, -0.25]])
    meta = torch.tensor([[-1.0, 2.0, -0.5], [4.0, -2.0, 1.0]])
    descriptor = torch.linspace(-1.0, 1.0, 16)
    output = generator(
        weight,
        calibration,
        descriptor,
        descriptor + 0.2,
        descriptor - 0.3,
        build_property_maps(torch.tensor([[8.0, 1.0, 3.0], [2.0, 9.0, 4.0]])),
    )
    objective = first_order_loss_change(
        meta,
        calibration,
        topk_ste(output.scores, 3),
        output.stabilizer,
        learning_rate=1.0,
    )
    objective.backward()
    gradients = [parameter.grad for parameter in generator.parameters()]
    assert all(gradient is not None for gradient in gradients)
    gradient_l1 = sum(
        float(gradient.abs().sum()) for gradient in gradients if gradient is not None
    )
    assert gradient_l1 > 1e-6


class _GeneratedUpdateWeights(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())


def test_generated_selected_update_cannot_reverse_task_gradient_direction() -> None:
    torch.manual_seed(19)
    generator = PCSMGenerator(16, hidden_dimension=8)
    weight = torch.tensor([[1.0, 7.0, 2.0], [9.0, 3.0, 5.0]])
    task_gradient = torch.tensor([[0.5, -4.0, 1.0], [-2.0, 3.0, -0.25]])
    descriptor = torch.linspace(-1.0, 1.0, 16)
    output = generator(
        weight,
        task_gradient,
        descriptor,
        descriptor + 0.2,
        descriptor - 0.3,
        build_property_maps(torch.tensor([[8.0, 1.0, 3.0], [2.0, 9.0, 4.0]])),
    )
    support = stable_topk_mask(output.scores, 3).reshape(-1)
    indices = torch.nonzero(support, as_tuple=False).reshape(-1)
    model = _GeneratedUpdateWeights(weight)
    plan = NativeUpdatePlan(
        (
            TensorUpdatePlan(
                name="weight",
                numel=weight.numel(),
                shape=tuple(weight.shape),
                flat_indices=indices,
                stabilizer=(
                    output.stabilizer.detach().reshape(-1).index_select(0, indices)
                ),
            ),
        )
    )
    optimizer = ExactSupportAdamW(
        apply_update_plan(model, plan),
        lr=0.05,
        weight_decay=0.0,
    )
    before = model.weight.detach().clone()
    model.weight.grad = task_gradient.clone()
    optimizer.step()
    realized = (model.weight.detach() - before).reshape(-1).index_select(0, indices)
    selected_gradient = task_gradient.reshape(-1).index_select(0, indices)
    assert torch.all(realized * selected_gradient < 0)
