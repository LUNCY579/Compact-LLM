from dataclasses import replace
import pytest
import torch
from experiments.rineya_self_distillation_axes_v1.controlled import ControlledTask, Geometry, occupancy
from experiments.rineya_self_distillation_axes_v1.plan import arms

SMALL = Geometry(depth=3, vocabulary=8, width=16)


@pytest.mark.parametrize('rho', [0., .5, 1.])
@pytest.mark.parametrize('strength', [0., .6, 1.])
def test_geometry_and_calibration(rho, strength):
    task = ControlledTask(replace(SMALL, feature_overlap=rho, context_strength=strength))
    cos = (task.old * task.new).sum(-1) / SMALL.width
    torch.testing.assert_close(cos, torch.full_like(cos, rho), atol=1e-12, rtol=1e-12)
    assert task.calibration_error < 1e-10
    assert abs(task.initial_metrics['old_similarity'] - 1) < 1e-12
    torch.testing.assert_close(task.target, task.policy(task.wref, task.new))


@pytest.mark.parametrize('compatibility', [-1., 0., .5, 1.])
def test_auxiliary_readout_compatibility(compatibility):
    task = ControlledTask(replace(SMALL, readout_compatibility=compatibility))
    cos = (task.w0 * task.task_readout).sum() / (task.w0.norm() * task.task_readout.norm())
    assert abs(float(cos) - compatibility) < 1e-12


def test_exact_occupancies_and_stop_gradient():
    p = torch.tensor([[.2, .8], [.4, .6], [.7, .3]], dtype=torch.float64, requires_grad=True)
    nu = occupancy(p, 2)
    torch.testing.assert_close(nu, torch.tensor([1., .2, .8], dtype=torch.float64))
    assert not nu.requires_grad and nu.sum() == 2
    with pytest.raises(ValueError): occupancy(p * 2, 2)


@pytest.mark.parametrize('source', ['student', 'teacher'])
@pytest.mark.parametrize('direction', ['forward', 'reverse'])
@pytest.mark.parametrize('rate', [0., .02, .25, 1.])
def test_all_independent_axes_run(source, direction, rate):
    task = ControlledTask(SMALL)
    displacement = task.displacement.clone()
    result = task.run(source, direction, rate, steps=3)
    assert result['evidence'] == 'CONTROLLED_MODEL_ONLY' and not result['native_rineya_executed']
    assert result['retention_pp'] <= 1e-9
    assert torch.equal(displacement, task.displacement)  # not recalibrated after EMA updates
    assert -100 <= result['acquisition_pp'] <= 100


def test_no_context_control_does_not_learn():
    task = ControlledTask(replace(SMALL, context_strength=0.))
    result = task.run(steps=3)
    assert abs(result['acquisition_pp']) < 1e-5
    assert abs(result['retention_pp']) < 1e-5


def test_seed_pairing_reproducible_and_varies_tasks():
    a, b, c = ControlledTask(SMALL, 17), ControlledTask(SMALL, 17), ControlledTask(SMALL, 29)
    assert torch.equal(a.w0, b.w0) and not torch.equal(a.w0, c.w0)
    assert a.run(steps=2)['acquisition_pp'] == b.run(steps=2)['acquisition_pp']


@pytest.mark.parametrize('kwargs', [{'depth': 0}, {'width': 2}, {'context_strength': 1.1}, {'concentration': 0}, {'feature_overlap': -1}])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError): ControlledTask(replace(SMALL, **kwargs))


def test_reference_tree_size():
    task = ControlledTask()
    assert task.new.shape == (585, 64)
    assert abs(float(occupancy(task.target, 4).sum()) - 4) < 1e-12



def test_controlled_and_llm_rate_grids_are_not_conflated():
    from experiments.rineya_self_distillation_axes_v1.controlled import CONTROLLED_RATES
    from experiments.rineya_self_distillation_axes_v1.plan import PAPER_RATES
    assert len(CONTROLLED_RATES) == 18 and len(PAPER_RATES) == 5
    assert CONTROLLED_RATES[0] == 0 and CONTROLLED_RATES[-1] == .16
    assert .0025 in CONTROLLED_RATES and .0025 not in PAPER_RATES
