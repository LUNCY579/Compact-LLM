import copy
import torch
from torch import nn
import pytest
from idr.control import ControlConfig
from idr.moe import SparseLoRALinear, MoEConfig, inject
from idr.experiment import tiny_moe


def layer(method="id"):
    torch.manual_seed(7)
    return SparseLoRALinear(nn.Linear(8,6), control=ControlConfig(method=method, min_expected=0), identity="test")


def test_zero_initialized_adapters_exact_identity():
    m=layer().eval()
    x=torch.randn(2,5,8)
    assert torch.equal(m(x),m.base(x))


def test_base_gradients_disabled_adapter_gradients_real():
    m=layer()
    x=torch.randn(32,8)
    m.begin_step()
    m(x).square().mean().backward()
    assert m.base.weight.grad is None and m.base.bias.grad is None
    assert m.B.grad.abs().sum()>0
    m.end_step()
    with torch.no_grad(): m.B.normal_()
    m.zero_grad(set_to_none=True)
    m.begin_step()
    m(x).square().mean().backward()
    assert m.A.grad.abs().sum()>0 and m.router.weight.grad.abs().sum()>0
    m.end_step()


def test_selection_bias_does_not_enter_combination_weights():
    m=layer().eval()
    with torch.no_grad():
        m.router.weight.zero_(); m.B.normal_()
    x=torch.randn(2,8)
    m.controller.bias[:]=[.08,.06,-.06,-.08]
    a=m(x)
    m.controller.bias[:]=[.07,.05,-.05,-.07]
    b=m(x)
    assert torch.equal(a,b)
    assert m.last_routes.tolist()==[[0,1],[0,1]]


def test_training_requires_explicit_window():
    with pytest.raises(RuntimeError): layer()(torch.randn(2,8))


def test_padding_not_routed_or_counted():
    m=layer()
    x=torch.randn(2,4,8)
    m.token_mask=torch.tensor([[True,True,False,False],[True,False,False,False]])
    m.begin_step()
    y=m(x)
    assert torch.equal(y[~m.token_mask],m.base(x)[~m.token_mask])
    log=m.end_step()
    assert sum(log["counts"])==6


def test_empty_padding_is_noop():
    m=layer()
    m.token_mask=torch.zeros(2,3,dtype=torch.bool)
    x=torch.randn(2,3,8)
    m.begin_step()
    assert torch.equal(m(x),m.base(x))
    assert m.end_step()["reason"]=="empty"


def test_mask_shape_rejected():
    m=layer().eval(); m.token_mask=torch.ones(2,dtype=torch.bool)
    with pytest.raises(ValueError): m(torch.randn(2,3,8))


def test_prefix_invariance_no_future_batch_statistics():
    m=layer().eval()
    with torch.no_grad(): m.B.normal_()
    x=torch.randn(1,4,8)
    whole=m(x)
    # Counts never feed back during the forward pass.
    prefix=m(x[:,:2])
    torch.testing.assert_close(whole[:,:2],prefix,rtol=1e-6,atol=1e-6)


def test_microbatch_count_aggregation_matches_whole_batch():
    a,b=layer(),layer()
    x=torch.randn(8,8)
    a.begin_step(); a(x); ra=a.end_step()
    b.begin_step(); b(x[:4]); b(x[4:]); rb=b.end_step()
    assert ra==rb


def test_abort_has_no_controller_update():
    m=layer(); m.begin_step(); m(torch.randn(8,8)); m.end_step(False)
    assert m.controller.updates==0


def test_checkpoint_roundtrip_outputs_and_controller(tmp_path):
    a=layer().eval()
    with torch.no_grad(): a.B.normal_()
    a.controller.begin(); a.controller.observe([0,4,4,8],[4]*4); a.controller.commit()
    state=a.adapter_state()
    assert all(not k.startswith("base") for k in state)
    torch.save(state,tmp_path/"a.pt")
    b=layer().eval()
    b.load_adapter_state(torch.load(tmp_path/"a.pt",weights_only=True))
    x=torch.randn(3,8)
    assert torch.equal(a(x),b(x))
    assert a.controller.state_dict()==b.controller.state_dict()


@pytest.mark.parametrize("bad",["shape","strength","identity","base","nan"])
def test_checkpoint_failure_before_parameter_mutation(bad):
    m=layer().eval()
    saved=copy.deepcopy(m.adapter_state())
    if bad=="shape": saved["A"]=torch.zeros(1)
    if bad=="strength": saved["strengths"][0]=.25
    if bad=="identity": saved["_extra_state"]["sha256"]="0"*64
    if bad=="base": saved["base.weight"]=m.base.weight
    if bad=="nan": saved["A"][0,0,0]=float("nan")
    before=m.A.detach().clone()
    with pytest.raises(ValueError): m.load_adapter_state(saved)
    assert torch.equal(before,m.A)


@pytest.mark.parametrize("kwargs",[{"top_k":1},{"top_k":5},{"rank":0},{"alpha":0},{"auxiliary_weight":-1}])
def test_bad_moe_config(kwargs):
    with pytest.raises(ValueError): MoEConfig(**kwargs)


def test_auxiliary_and_bias_arms_must_be_separate():
    with pytest.raises(ValueError):
        SparseLoRALinear(nn.Linear(4,3),MoEConfig(auxiliary_weight=.1))


def test_bfloat16_not_silently_used():
    with pytest.raises(ValueError): SparseLoRALinear(nn.Linear(4,3).to(torch.bfloat16))


def test_inject_only_explicit_targets_and_preserves_identity():
    model=nn.Sequential(nn.Linear(8,8),nn.Tanh(),nn.Linear(8,6)).eval()
    x=torch.randn(2,8)
    old=model(x).detach()
    original=model[2]
    bank=inject(model,["2"],identity="basehash")
    model.eval()
    assert bank["2"].base is original
    assert torch.equal(old,model(x))
    assert not model[0].weight.requires_grad
    assert model[2].router.weight.requires_grad


def test_inject_invalid_path_set_is_atomic():
    model=nn.Sequential(nn.Linear(8,8),nn.Tanh())
    with pytest.raises(ValueError): inject(model,["0","1"])
    assert type(model[0]) is nn.Linear and model[0].weight.requires_grad


@pytest.mark.parametrize("method",["none","sign","integral","id","auxiliary"])
def test_real_tiny_optimizer_steps_with_frozen_base(method):
    result=tiny_moe(method,auxiliary=method=="auxiliary",steps=5)
    assert result["frozen_base_unchanged"] and not result["native_executed"]
    assert result["last_training_loss"]<result["initial_loss"]


def test_auxiliary_loss_has_router_gradient():
    m=SparseLoRALinear(nn.Linear(8,6),MoEConfig(auxiliary_weight=.1),ControlConfig(method="none"))
    m.begin_step(); m(torch.randn(32,8))
    m.last_auxiliary.backward()
    assert m.router.weight.grad is not None and torch.isfinite(m.router.weight.grad).all()
    m.end_step()
