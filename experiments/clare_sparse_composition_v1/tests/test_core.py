import copy
from dataclasses import replace
import json
import math
import random
import pytest
import torch
from experiments.clare_sparse_composition_v1 import PARENT_COMMIT
from experiments.clare_sparse_composition_v1.allocation import Allocation, budget, select
from experiments.clare_sparse_composition_v1.artifacts import load_checkpoint, save_checkpoint, tensor_hash
from experiments.clare_sparse_composition_v1.config import ARMS, Config
from experiments.clare_sparse_composition_v1.data import TrainingRow, digest_json
from experiments.clare_sparse_composition_v1.metrics import compose, interaction, pair_geometry, rank_energy, shared_private
from experiments.clare_sparse_composition_v1.training import Coordinates, SkillRunner, stage

DIGEST = digest_json("synthetic fixtures only")
IDENTITY = {"model_id": "toy", "tokenizer_id": "none", "model_sha256": "1"*64,
            "tokenizer_sha256": "2"*64, "git_commit": PARENT_COMMIT, "runtime": {"kind": "synthetic"}}


def row(skill="R", role="train", digest=DIGEST):
    return TrainingRow("id-"+skill, "parent-"+skill, skill, role, digest,
                       json.dumps([{"role": "user", "content": "fixture"}]), "{}")


def make(config=None):
    torch.manual_seed(55)
    model = torch.nn.Linear(8, 2, bias=False)
    return SkillRunner(model, ["weight"], config or Config(lr=0.05), data_digest=DIGEST)


def loss(model, r):
    target = torch.arange(16, dtype=torch.float32).reshape(2, 8)/20 + "RUSP".index(r.skill)/10
    return (model.weight-target).square().mean()


@pytest.mark.parametrize("rho,counts", [(0.8,[20,16,13,10]), (0.9,[10,9,8,7]), (0.95,[5,5,5,4]), (0.975,[3,2,2,2])])
def test_capacity_schedule(rho, counts):
    left = 100
    for count in counts:
        assert budget(left, rho) == count
        left -= count


@pytest.mark.parametrize("n,rho,expected", [(0,.8,0),(1,.999,1),(10,.95,1),(3,.5,2),(100,.8,20)])
def test_budget_rounding(n,rho,expected):
    assert budget(n,rho) == expected


@pytest.mark.parametrize("kwargs", [{"arm":"F"},{"rho":0},{"rho":1},{"rho":math.nan},{"order":("R","R")},{"order":("X",)},{"seed":True},{"lr":0},{"train_steps":0},{"discovery_l1":-1},{"weight_decay":-1},{"max_coordinates":False},{"optimizer":"SGD"}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError): Config(**kwargs)


def test_config_roundtrip():
    c=Config(order=("P","S","U","R"),rho=.9)
    assert Config.from_dict(json.loads(json.dumps(c.payload()))).digest==c.digest


def test_exact_ties_and_unavailable_scores():
    free=torch.tensor([False,True,True,True])
    assert select(free,2,scores=torch.ones(4)).tolist()==[False,True,True,False]
    assert select(free,1,scores=torch.tensor([100.,1,9,0])).tolist()==[False,False,True,False]


def test_random_determinism():
    free=torch.ones(100,dtype=torch.bool)
    assert torch.equal(select(free,12,seed=4),select(free,12,seed=4))
    assert not torch.equal(select(free,12,seed=4),select(free,12,seed=5))


@pytest.mark.parametrize("case", ["floatmask","shape","nan","budget"])
def test_selection_refusals(case):
    free=torch.ones(4,dtype=torch.bool);scores=torch.ones(4);k=1
    if case=="floatmask":free=free.float()
    if case=="shape":scores=torch.ones(3)
    if case=="nan":scores[0]=math.nan
    if case=="budget":k=5
    with pytest.raises(ValueError):select(free,k,scores=scores)


def test_allocation_protected_and_defensive_copy():
    a=Allocation(4,("R","U"));m=torch.tensor([True,False,False,False]);a.claim("R",m,1);m[0]=False
    assert not a.free[0]
    with pytest.raises(ValueError):a.claim("U",torch.tensor([True,False,False,False]),1)
    with pytest.raises(ValueError):a.claim("R",a.free,3)
    with pytest.raises(ValueError):a.claim("U",a.free,2)


@pytest.mark.parametrize("arm",ARMS)
def test_arm_execution_reset_finite_and_accounting(arm):
    r=make(Config(arm=arm,lr=.05))
    for s in r.config.order:
        rec=r.train_skill(s,[row(s)],loss)
        assert rec["reset_exact"] and len(rec["probe_trace"])==4 and len(rec["train_trace"])==8
        assert rec["selected_coordinates"]==int(rec["mask"].sum())
        assert rec["active_nonzero"]==int((r.deltas[s]!=0).sum())
        assert torch.isfinite(r.current).all()
    if "disjoint" in arm:
        for i,a in enumerate(r.config.order):
            for b in r.config.order[i+1:]:
                assert pair_geometry(r.deltas[a],r.deltas[b])["overlap"]==0
                assert pair_geometry(r.deltas[a],r.deltas[b])["dot"]==0


def test_random_vs_learned_exact_capacity():
    rs=[make(Config(arm=a,lr=.05)) for a in ("random_disjoint","learned_disjoint")]
    for s in "RUSP":
        x,y=[r.train_skill(s,[row(s)],loss) for r in rs]
        assert x["selected_coordinates"]==y["selected_coordinates"]


@pytest.mark.parametrize("decay",[0,.2,1.])
def test_adamw_frozen_coords_momentum_and_decay(decay):
    r=make(Config(weight_decay=decay,lr=.1))
    m=torch.zeros(16,dtype=torch.bool);m[:3]=True
    observed=[]
    handle=r.coords.params[0].register_hook(lambda g: observed.append(g.clone()))
    result,_=stage(r.coords,r.base,m,[row()],8,loss,r.config)
    handle.remove()
    assert torch.equal(result[~m],r.base[~m])
    assert not torch.equal(result[m],r.base[m])
    assert observed and r.coords.params[0].grad is None


def test_reset_equivalent_to_fresh_masked_retrain():
    r=make();rec=r.train_skill("R",[row()],loss)
    control=make();expected,_=stage(control.coords,control.base,rec["mask"],[row()],8,loss,control.config,seed=rec["stage_seed"])
    assert torch.equal(r.current,control.base+(expected-control.base))
    assert rec["fp32_recomposition_max_abs"] <= torch.finfo(torch.float32).eps


def test_previous_delta_is_unchanged_after_later_skills():
    r=make();r.train_skill("R",[row()],loss);m=r.records[0]["mask"];saved=r.current[m].clone()
    r.train_skill("U",[row("U")],loss)
    assert torch.equal(saved,r.current[m])


@pytest.mark.parametrize("bad", ["role","skill","digest","assistant","empty"])
def test_training_boundary_refuses_without_mutation(bad):
    r=make();rr=row()
    if bad=="role":rr=replace(rr,role="test")
    if bad=="skill":rr=replace(rr,skill="U")
    if bad=="digest":rr=replace(rr,manifest_sha256="0"*64)
    if bad=="assistant":rr=replace(rr,prompt_json=json.dumps([{"role":"assistant","content":"leak"}]))
    with pytest.raises(ValueError):r.train_skill("R",[] if bad=="empty" else [rr],loss)
    assert torch.equal(r.coords.vector(),r.base) and not r.records


def test_failure_rolls_back_rng_modes_and_flags():
    r=make();m=r.coords.model;m.train();state=torch.random.get_rng_state();py=random.getstate()
    called=0
    def broken(model,row):
        nonlocal called
        called+=1
        if called==3:raise RuntimeError("injected")
        return loss(model,row)
    with pytest.raises(RuntimeError):r.train_skill("R",[row()],broken)
    assert torch.equal(r.base,r.coords.vector()) and not r.records and r.allocation.free.all()
    assert torch.equal(torch.random.get_rng_state(),state) and random.getstate()==py and m.training
    assert all(p.requires_grad for p in m.parameters())


def test_nonfinite_loss_and_guard_refused():
    r=make()
    with pytest.raises(ValueError):r.train_skill("R",[row()],lambda m,r:m.weight.sum()*float("nan"))
    with pytest.raises(RuntimeError):r.train_skill("R",[row()],loss,guard=lambda:(_ for _ in ()).throw(RuntimeError("RAM")))
    assert torch.equal(r.coords.vector(),r.base)


@pytest.mark.parametrize("order",[("R","U","S","P"),("P","S","U","R"),("R","S")])
def test_arbitrary_order_and_reproducibility(order):
    rs=[make(Config(order=order,lr=.03)) for _ in range(2)]
    for r in rs:
        for s in order:r.train_skill(s,[row(s)],loss)
    assert torch.equal(rs[0].current,rs[1].current)
    with pytest.raises(ValueError):rs[0].train_skill(order[0],[row(order[0])],loss)


@pytest.mark.parametrize("arm",ARMS)
def test_checkpoint_resume_matches_uninterrupted(tmp_path,arm):
    c=Config(arm=arm,lr=.03);r=make(c);r.train_skill("R",[row()],loss)
    save_checkpoint(r,tmp_path/"ck",IDENTITY)
    resumed=make(c);load_checkpoint(resumed,tmp_path/"ck",IDENTITY)
    for s in "USP":
        r.train_skill(s,[row(s)],loss);resumed.train_skill(s,[row(s)],loss)
    assert torch.equal(r.current,resumed.current)
    with pytest.raises(FileExistsError):save_checkpoint(r,tmp_path/"ck",IDENTITY)


@pytest.mark.parametrize("tamper",["bytes","identity","config","base","mask"])
def test_checkpoint_rejects_drift(tmp_path,tamper):
    r=make();r.train_skill("R",[row()],loss);p=tmp_path/"ck";save_checkpoint(r,p,IDENTITY)
    target=make();identity=copy.deepcopy(IDENTITY)
    if tamper=="bytes":
        with (p/"state.safetensors").open("ab") as f:f.write(b"drift")
    if tamper=="identity":identity["model_id"]="other"
    if tamper=="config":target=make(Config(rho=.9))
    if tamper=="base":target.base+=1
    if tamper=="mask":
        m=json.loads((p/"manifest.json").read_text());m["records"][0]["mask_sha256"]="0"*64;(p/"manifest.json").write_text(json.dumps(m))
    before=target.coords.vector()
    with pytest.raises(ValueError):load_checkpoint(target,p,identity)
    assert torch.equal(before,target.coords.vector())


def test_coordinate_safety():
    m=torch.nn.Linear(2,2)
    for names in ([],["absent"],["weight","weight"]):
        with pytest.raises(ValueError):Coordinates(m,names,10)
    with pytest.raises(ValueError):Coordinates(m,["weight"],1)
    with pytest.raises(ValueError):Coordinates(m.double(),["weight"],100)


def test_geometry_zero_norm_and_overlap():
    z=torch.zeros(4);a=torch.tensor([1.,-2,0,0]);b=torch.tensor([-1.,-2,3,0])
    assert pair_geometry(z,z)["cosine"] is None
    assert pair_geometry(z,z)["jaccard"] is None
    g=pair_geometry(a,b);assert g["overlap"]==2 and g["sign_agreement"]==.5 and g["jaccard"]==2/3
    assert g["dot"]==3


def test_disjoint_geometry_does_not_imply_functional_additivity():
    a=torch.tensor([1.,0]);b=torch.tensor([0.,1]);assert pair_geometry(a,b)["dot"]==0
    f=lambda x:torch.tensor([x[0]*x[1],0.])
    result=interaction(f(torch.zeros(2)),f(a),f(b),f(a+b))
    assert result["raw_rms"]>0 and result["vocab_centered_rms"]>0


def test_logit_constant_shift_diagnostic():
    z=torch.zeros(3,4);i=interaction(z,z,z,z+3)
    assert i["raw_rms"]==3 and i["vocab_centered_rms"]==0
    with pytest.raises(ValueError):interaction(z,z[:1],z,z)


def test_rank_and_merges():
    a=torch.tensor([1.,2]);b=torch.tensor([3.,4])
    assert torch.equal(compose([a,b],[.5,.5]),torch.tensor([2.,3]))
    assert torch.equal(shared_private(a,[a,b],[1.,1.]),2*a+b)
    assert rank_energy(torch.diag(torch.tensor([3.,1.])),1)["energy_fraction"]==.9
    assert rank_energy(torch.zeros(2,2))["energy_fraction"] is None
    with pytest.raises(ValueError):rank_energy(torch.ones(100,100),max_elements=10)
    with pytest.raises(ValueError):compose([a],[math.nan])
    with pytest.raises(ValueError):compose([a,b[:1]],[1.,1.])


def test_tensor_hash_binds_dtype_shape_values():
    a=torch.tensor([1.,2.]);assert tensor_hash(a)!=tensor_hash(a.double())
    assert tensor_hash(a)!=tensor_hash(a.reshape(1,2))
    assert tensor_hash(a)!=tensor_hash(a+1)


def test_optimizer_sees_zero_frozen_gradients(monkeypatch):
    r=make();mask=torch.zeros(16,dtype=torch.bool);mask[:4]=True
    original=torch.optim.AdamW.step;seen=[]
    def check(self,*args,**kwargs):
        for group in self.param_groups:
            for p in group["params"]:
                seen.append(torch.count_nonzero(p.grad.reshape(-1)[~mask]).item())
        return original(self,*args,**kwargs)
    monkeypatch.setattr(torch.optim.AdamW,"step",check)
    stage(r.coords,r.base,mask,[row()],4,loss,r.config)
    assert seen==[0,0,0,0]


def test_l1_discovery_objective_has_effect():
    r=make();mask=torch.ones(16,dtype=torch.bool)
    plain,_=stage(r.coords,r.base,mask,[row()],8,loss,r.config,l1=0)
    sparse,_=stage(r.coords,r.base,mask,[row()],8,loss,r.config,l1=1)
    assert (sparse-r.base).abs().sum() < (plain-r.base).abs().sum()


def test_shared_private_refuses_broadcast():
    with pytest.raises(ValueError):shared_private(torch.zeros(1),[torch.ones(3)],[1.])


@pytest.mark.parametrize("order", [(1,), ([],), ("RU",), ()])
def test_config_rejects_malformed_order(order):
    with pytest.raises(ValueError): Config(order=order)
