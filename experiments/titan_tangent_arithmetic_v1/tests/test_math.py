import pytest
import torch
from torch import nn
from experiments.titan_tangent_arithmetic_v1.tangent import FrozenAnchor, CallBudget, combine, tensor_digest
from experiments.titan_tangent_arithmetic_v1.controls import head_tail, norm_match, composition_controls, rank_truncate, effective_export_delta
from experiments.titan_tangent_arithmetic_v1.metrics import normalized_accuracy, disentanglement_error, response_metrics, paired, interaction


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2., 3.], dtype=torch.float64))
    def forward(self, x):
        return x * self.w.square()


def test_analytic_jvp_and_trainable_delta():
    a = FrozenAnchor(Quadratic())
    d = {'w': torch.tensor([.2, -.1], dtype=torch.float64, requires_grad=True)}
    x = torch.tensor([3., 4.], dtype=torch.float64)
    out = a.evaluate(d, x, mode='linear')
    assert torch.allclose(out, x * (a.params['w'].square() + 2*a.params['w']*d['w']))
    out.sum().backward()
    assert torch.equal(d['w'].grad, 2*x*a.params['w'])
    actual = a.evaluate(d, x)
    assert torch.allclose(actual-out, x*d['w'].square())
    assert a.model.w.grad is None
    a.assert_unchanged()


def test_affine_is_exact_and_anchor_stays_frozen():
    m = nn.Linear(3, 2).double()
    a = FrozenAnchor(m); d = {k: torch.full_like(v, .1) for k,v in a.params.items()}
    x = torch.ones(1, 3, dtype=torch.float64)
    assert torch.allclose(a.evaluate(d, x), a.evaluate(d, x, mode='linear'))
    assert tensor_digest(dict(m.named_parameters())) == a.initial_digest
    a.assert_unchanged()


@pytest.mark.parametrize('bad', ['keys', 'dtype', 'shape', 'nan'])
def test_delta_contract(bad):
    a = FrozenAnchor(Quadratic()); d = a.zero()
    if bad == 'keys': d = {'other': d['w']}
    if bad == 'dtype': d['w'] = d['w'].float()
    if bad == 'shape': d['w'] = d['w'][:1]
    if bad == 'nan': d['w'][0] = float('nan')
    with pytest.raises(ValueError): a.evaluate(d, torch.ones(2))


@pytest.mark.parametrize('scope', [[], ['absent'], ['w', 'w']])
def test_scope_refusal(scope):
    with pytest.raises(ValueError): FrozenAnchor(Quadratic(), scope)


def test_no_low_precision_or_train_mode():
    with pytest.raises(ValueError): FrozenAnchor(Quadratic().bfloat16())
    a = FrozenAnchor(Quadratic()); a.model.train()
    with pytest.raises(RuntimeError): a.evaluate(a.zero(), torch.ones(2))


def test_budget_and_mutation():
    a = FrozenAnchor(Quadratic(), budget=CallBudget(1))
    a.evaluate(a.zero(), torch.ones(2))
    with pytest.raises(RuntimeError): a.evaluate(a.zero(), torch.ones(2))
    a.params['w'][0] += 1
    with pytest.raises(RuntimeError): a.assert_unchanged()


@pytest.mark.parametrize('keep', [0., .1, .25, .5, 1.])
def test_head_tail_partition_and_norm(keep):
    d = {'x': torch.tensor([1., -1., 2., 4.], dtype=torch.float64)}
    h,t = head_tail(d, keep)
    assert torch.equal(h['x'] + t['x'], d['x'])
    assert (h['x']*t['x']).sum() == 0
    assert h['x'].count_nonzero() == int(keep*4)
    n = norm_match(d,h)
    assert torch.allclose(n['x'].norm(),h['x'].norm())


def test_deterministic_tie_break():
    h,_ = head_tail({'x':torch.tensor([1.,-1.,1.,-1.])},.5)
    assert torch.equal(h['x'],torch.tensor([1.,-1.,0.,0.]))


@pytest.mark.parametrize('keep', [-1., 1.1, float('nan')])
def test_bad_pruning(keep):
    with pytest.raises(ValueError): head_tail({'x':torch.ones(4)},keep)


def test_exact_component_attenuation_control():
    a={'x':torch.tensor([3.,0.]),'y':torch.tensor([1.,1.])}
    b={'x':torch.tensor([0.,4.]),'y':torch.tensor([2.,0.])}
    c=composition_controls(a,b)
    for k in a:
        s=a[k].double().norm()/((a[k]+b[k]).double().norm())
        assert torch.allclose(c['attenuated_a'][k],a[k]*s)
        assert torch.allclose(c['matched_sum'][k],c['attenuated_a'][k]+c['attenuated_b'][k])
        assert torch.allclose(c['matched_sum'][k].norm(),a[k].norm())
    assert not torch.equal(c['attenuated_a']['x'], a['x'])


def test_no_cancellation_fallback():
    a={'x':torch.ones(2)}
    with pytest.raises(ValueError): composition_controls(a,{'x':-a['x']})
    with pytest.raises(ValueError): norm_match({'x':torch.zeros(2)},a)


def test_rank_and_export():
    d={'m':torch.diag(torch.tensor([3.,2.,1.])), 'v':torch.tensor([.1,.2])}
    r=rank_truncate(d,1)
    assert torch.linalg.matrix_rank(r['m']) == 1
    assert torch.equal(r['v'],d['v'])
    b={'x':torch.ones(4)}; tiny={'x':torch.full((4,),1e-5)}
    assert effective_export_delta(b,tiny)['x'].count_nonzero() == 0
    assert tiny['x'].count_nonzero() == 4


def test_equation_six_is_sum_not_pooled():
    assert disentanglement_error({'A':[0], 'B':[0]*3}, {'A':[1], 'B':[0]*3}) == 1
    assert disentanglement_error({'A':[0], 'B':[0]*3}, {'A':[1], 'B':[1]*3}) == 2


def test_normalization_is_method_specific_and_undefined_is_not_zero():
    r=normalized_accuracy({'A':.5,'B':.8},{'A':.25,'B':.8})
    assert r['normalized_macro_accuracy'] == 1.5
    assert normalized_accuracy({'A':.5},{'A':0})['normalized_macro_accuracy'] is None
    with pytest.raises(ValueError): normalized_accuracy({'A':float('nan')},{'A':1})


def test_paired_churn_not_net_only():
    r=paired([True,False,True],[False,True,True])
    assert r['gains']==r['losses']==1 and r['exact_mcnemar_p']==1
    assert paired([True]*15,[False]*15)['exact_mcnemar_p'] == pytest.approx(2/2**15)
    with pytest.raises(ValueError): paired([],[])


def test_orthogonal_parameter_updates_can_leak_in_linear_model():
    # f(theta)=theta_1+theta_2: orthogonal weight updates have identical output effects.
    a=torch.tensor([1.,0.]); b=torch.tensor([0.,1.]); J=torch.tensor([1.,1.])
    assert a@b == 0 and J@a == J@b == 1
    z=torch.zeros(1); x=torch.ones(1)
    assert interaction(z,x,x,2*x)==0
    assert disentanglement_error({'A':[0]}, {'A':[1]}) == 1
    assert response_metrics(z,z,z)['relative_linearization_error'] is None
