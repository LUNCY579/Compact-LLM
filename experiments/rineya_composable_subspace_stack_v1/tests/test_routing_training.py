import numpy as np
import pytest
import torch
from experiments.rineya_composable_subspace_stack_v1.protocol import digest
from experiments.rineya_composable_subspace_stack_v1.routing import (
    PromptFeature, fit_router, route_and_predict, routing_regret, LinearSkillBank)
from experiments.rineya_composable_subspace_stack_v1.training import (
    FunctionalSkill, TrainingConfig, distill_skill, ordinary_skill, train_composition_study)
from experiments.rineya_composable_subspace_stack_v1.core import Arm


def router():
    features = [PromptFeature('c1', 'p1', digest('a'), (1., 0.)), PromptFeature('c2', 'p2', digest('b'), (0., 1.))]
    return fit_router(features, ['a', 'b'], {'a':[1.,0.], 'b':[0.,1.]}, split='CAL', feature_receipt=digest('prompt-only-encoder'))


def test_router_good_and_ambiguous_and_ood():
    r = router()
    good = r.route(PromptFeature('q', 'pq', digest('query'), (1., 0.)))
    assert good['weights'] == [1., 0.] and not good['fallback_base']
    for z in [(1., 1.), (-1., -.2)]:
        assert r.route(PromptFeature('q', 'pq', digest('query'), z))['fallback_base']


@pytest.mark.parametrize('field,value', [('row_id','c1'), ('parent_id','p2'), ('prompt_sha256',digest('a'))])
def test_router_leakage(field, value):
    kw = dict(row_id='q', parent_id='pq', prompt_sha256=digest('q'), embedding=(1.,0.)); kw[field] = value
    with pytest.raises(ValueError, match='leakage'):
        router().route(PromptFeature(**kw))


def test_routing_never_accepts_answer_field():
    with pytest.raises(TypeError):
        PromptFeature('q','pq',digest('q'),(1.,0.), answer='secret')
    with pytest.raises(ValueError, match='CAL-only'):
        fit_router([], [], {}, split='CHECK', feature_receipt=digest('f'))


def test_route_binds_prompt_and_invokes_composition_callback():
    f = PromptFeature('q','pq',digest('q'),(0.,1.))
    out = route_and_predict(router(), [f], ['q'], lambda prompt, weights: weights[1])
    assert out[0]['prediction'] == 1
    with pytest.raises(ValueError, match='bound'):
        route_and_predict(router(), [f], ['wrong'], lambda *args: 0)


def test_router_topk_and_oracle_headroom():
    r = router(); r.top_k = 2; r.margin = 0
    out = r.route(PromptFeature('q','pq',digest('q'),(1.,.5)))
    assert sum(out['weights']) == pytest.approx(1)
    report = routing_regret([[0,1],[1,0]], [0,0])
    assert report['oracle_headroom'] == .5 and report['routing_regret'] == 0


@pytest.mark.parametrize('z', [(0.,0.), (float('nan'),1.)])
def test_invalid_features(z):
    with pytest.raises(ValueError):
        router().route(PromptFeature('q','pq',digest('q'),z))


def test_linear_bank_matches_explicit_composition_and_cleans_hooks():
    layer = torch.nn.Linear(3, 2, bias=False).double().eval()
    module = torch.nn.Sequential(layer).eval()
    original = layer.weight.detach().clone()
    a, b = torch.ones_like(original)*.1, torch.ones_like(original)*.2
    x = torch.ones(1, 3, dtype=torch.float64)
    with LinearSkillBank(module, {'0':[a,b]}) as bank:
        with pytest.raises(ValueError, match='route'):
            module(x)
        bank.select([.25,.75])
        assert torch.allclose(module(x), x@(original+.25*a+.75*b).T)
        bank.select([0.,0.])
        assert torch.allclose(module(x), x@original.T)
    assert torch.equal(layer.weight, original)
    assert torch.allclose(module(x), x@original.T)
    with pytest.raises(ValueError, match='closed'):
        bank.select([1.,0.])


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, bias=False)
    def forward(self, x):
        h = torch.tanh(self.linear(x))
        return h, h


def toy_batch():
    torch.manual_seed(12)
    model = Toy().double().eval()
    x = torch.randn(6, 3, dtype=torch.float64)
    return model, {'x':x,'y':.2*torch.tanh(x),'split':'TRAIN'}


def test_jvp_central_difference_and_zero_tangent():
    model, b = toy_batch()
    f = FunctionalSkill(model, ['linear.weight'])
    report = f.qualify(b['x'])
    assert max(report['jvp_relative_errors']) < .03
    assert torch.allclose(f.tangent(f.zero(), b['x'])[0], model(b['x'])[0])
    assert report['native_rineya_qualified'] is False


def test_teacher_student_and_ordinary_train_without_mutating_base():
    model, b = toy_batch(); original = model.linear.weight.detach().clone()
    config = TrainingConfig(teacher_steps=3, student_steps=3)
    result = distill_skill(model, ['linear.weight'], [b], torch.nn.functional.mse_loss, config)
    plain = ordinary_skill(model, ['linear.weight'], [b], torch.nn.functional.mse_loss, config)
    assert result['optimizer_steps'] == 6 and plain['optimizer_steps'] == 3
    assert torch.equal(model.linear.weight, original) and model.linear.weight.grad is None
    assert torch.linalg.vector_norm(result['student_delta']['linear.weight']) > 0
    assert np.isfinite(result['history']['teacher']).all()


@pytest.mark.parametrize('case', ['check', 'unknown', 'budget', 'training_mode'])
def test_training_refusals(case):
    model, b = toy_batch(); names = ['linear.weight']; config = TrainingConfig(teacher_steps=1,student_steps=1)
    if case == 'check': b['split'] = 'CHECK'
    if case == 'unknown': names = ['wrong']
    if case == 'budget': config = TrainingConfig(max_trainable_elements=1)
    if case == 'training_mode': model.train()
    with pytest.raises(ValueError):
        distill_skill(model, names, [b], torch.nn.functional.mse_loss, config)


def test_stage2_all_variants_feed_identical_merger():
    model, b = toy_batch(); c = dict(b); c['y'] = -.1*b['y']
    result = train_composition_study(model, ['linear.weight'], {'a':[b], 'b':[c]},
        torch.nn.functional.mse_loss, [.5,.5], Arm('composed', compression='ctm'),
        training_config=TrainingConfig(teacher_steps=1,student_steps=1),
        compose_settings=dict(rank=2, protected_rank=0, shared_rank=1))
    assert set(result['compositions']) == {'ordinary','tangent','distilled'}
    assert result['student_updates_matched'] and not result['total_compute_matched']
    for matrices in result['compositions'].values():
        delta, report = matrices['linear.weight']
        assert np.linalg.matrix_rank(delta, tol=1e-8) <= 2
