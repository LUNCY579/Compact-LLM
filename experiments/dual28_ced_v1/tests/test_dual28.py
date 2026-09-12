from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
import torch
from safetensors.torch import load_file, save_file
from experiments.dual28_ced_v1 import guards as g
from experiments.dual28_ced_v1.checks import activate, active_path, close, grid, train_pair
from experiments.dual28_ced_v1.model import AddedRead, DualStack, DenseReference, controls, dual_budget, matrix, presets
from experiments.dual28_ced_v1.run import main, tiny_source, verify_inheritance
from experiments.full_width_transition_v1.source import state_digest


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def source():
    return tiny_source()


@pytest.mark.parametrize("mode", ["H", "P", "C"])
def test_complete_independent_donor_copies(source, mode):
    before = state_digest(source)
    m = DualStack.from_source(source, mode)
    verify_inheritance(m, source)
    assert len(m.encoder) == len(m.decoder) == source.config.total_layers
    assert m.config.hidden_size == source.config.hidden_size
    assert m.config.intermediate_size == source.config.intermediate_size
    assert sum(p.numel() for p in m.parameters()) == dual_budget(source.config, mode)["parameters"]
    with torch.no_grad():
        m.encoder[0].q_proj.weight.add_(1)
    assert state_digest(source) == before
    assert not torch.equal(m.encoder[0].q_proj.weight, m.decoder[0].q_proj.weight)


@pytest.mark.parametrize("mode", ["H", "P", "C"])
def test_official_shape_meta_only(mode):
    c = presets()["B28_1024_official_shape"]
    with torch.device("meta"):
        m = DualStack(c, mode)
    assert len(m.encoder) == len(m.decoder) == 28
    assert m.encoder[0].q_proj.weight.shape == (2048, 1024)
    assert m.decoder[27].mlp.down_proj.weight.shape == (1024, 3072)
    assert m.memory.k_proj.weight.shape == (1024, 1024)
    assert sum(p.numel() for p in m.parameters()) == matrix()[mode+"56"]["parameters"]
    assert all(p.is_meta for p in m.parameters())
    assert sum(n == "embed_tokens.weight" for n, p in m.named_parameters()) == 1


@pytest.mark.parametrize("arm,n,arch,e,d", [("B28",28,"plain",28,0), ("D56",56,"plain",56,0), ("C28",28,"ced",14,14)])
def test_control_meta_budget(arm,n,arch,e,d):
    c = replace(presets()["B28_1024_official_shape"], architecture=arch, encoder_layers=e, decoder_layers=d)
    with torch.device("meta"):
        m = DenseReference(c)
    assert c.total_layers == n
    assert sum(p.numel() for p in m.parameters()) == matrix()[arm]["parameters"]


@pytest.mark.parametrize("mode", ["H", "P"])
@pytest.mark.parametrize("batch", [1, 2])
def test_hybrid_identity_batched(source, mode, batch):
    ids = torch.arange(1, 9).repeat(batch, 1)
    m = DualStack.from_source(source, mode)
    with torch.inference_mode():
        assert torch.equal(m(ids).logits, source(ids).logits)
        pre = m(ids[:, :3], use_cache=True)
        tail = m(ids[:, 3:], past=pre.cache, use_cache=True)
        close(tail.logits, m(ids).logits[:, 3:], "batched_cache")
    trainable = m.freeze_backbones()
    assert sum(p.numel() for p in trainable) == dual_budget(source.config, mode)["engineering_trainable_parameters"]
    assert all(p.requires_grad == n.startswith(("memory.", "reads.")) for n,p in m.named_parameters())


@pytest.mark.parametrize("mode", ["H", "P", "C", "D56", "C28"])
def test_active_mechanical_grid(source, mode):
    m = DualStack.from_source(source, mode) if len(mode)==1 else controls(source, mode)
    result = active_path(m, source) if mode in {"H", "P"} else grid(m)
    assert set(result) == {"ordinary", "repeated_positions", "large_positions", "left_padding", "all_padding", "packed"}


def test_depth_identity_and_zero_new_outputs(source):
    m = controls(source, "D56")
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        assert torch.equal(m(x).logits, source(x).logits)
    for b in m.encoder[len(source.encoder):]:
        assert not (b.o_proj.weight != 0).any()
        assert not (b.mlp.down_proj.weight != 0).any()
        assert (b.q_proj.weight != 0).any()


@pytest.mark.parametrize("mode", ["H", "P", "C"])
def test_checkpoint_roundtrip_and_create_only(source, mode, tmp_path):
    m = DualStack.from_source(source, mode)
    path = tmp_path / "checkpoint"
    m.save(path)
    loaded = DualStack.load(path)
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        assert torch.equal(m(x).logits, loaded(x).logits)
    assert loaded.mode == mode
    with pytest.raises(FileExistsError):
        m.save(path)


@pytest.mark.parametrize("damage", ["missing", "nan", "shape", "dtype", "kind"])
def test_bad_checkpoint_rejected(source, tmp_path, damage):
    path = tmp_path / "ckpt"
    DualStack.from_source(source,"H").save(path)
    if damage == "kind":
        f=path/"config.json"; meta=json.loads(f.read_text());meta["kind"]="FWT-001";f.write_text(json.dumps(meta))
    else:
        f=path/"model.safetensors"; state=load_file(str(f));key="reads.0.o_proj.weight"
        if damage=="missing": del state[key]
        elif damage=="nan": state[key][0,0]=float("nan")
        elif damage=="shape": state[key]=state[key][:1]
        else: state[key]=state[key].double()
        save_file(state,str(f))
    with pytest.raises(ValueError): DualStack.load(path)


@pytest.mark.parametrize("mode", ["H", "P", "C"])
def test_checkpointed_gradient_parity(source,mode):
    a,b=DualStack.from_source(source,mode),DualStack.from_source(source,mode)
    if mode!="C":
        for m in (a,b):
            m.freeze_backbones()
            with torch.no_grad():
                for r,d in zip(m.reads,source.encoder):r.o_proj.weight.copy_(d.o_proj.weight*0.001)
    a.train();b.train();b.gradient_checkpointing=True
    ids=torch.tensor([[1,2,3,4,5]])
    la=a(ids).logits.square().mean();lb=b(ids).logits.square().mean()
    la.backward();lb.backward()
    close(la[None],lb[None],"checkpoint_loss",exact=True)
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb and (pa.grad is None)==(pb.grad is None)
        if pa.grad is not None:close(pa.grad,pb.grad,na)


@pytest.mark.parametrize("mode", ["H", "P"])
def test_first_gradient_reaches_only_zero_outputs(source,mode):
    m=DualStack.from_source(source,mode);m.freeze_backbones();m.train()
    m(torch.tensor([[1,2,3,4]])).logits.square().mean().backward()
    for name,p in m.named_parameters():
        if not p.requires_grad:assert p.grad is None
        elif name.endswith("o_proj.weight"):assert p.grad is not None and (p.grad!=0).any()
        else:assert p.grad is None or not (p.grad!=0).any()


def test_paired_25_step_engineering(source):
    before=state_digest(source)
    r=train_pair(source,[[1,2,3,4,5,6],[7,8,9,10,11,12],[2,4,6,8,10,12]])
    assert state_digest(source)==before
    assert r["H"]["order"]==r["P"]["order"]
    assert r["H"]["trainable_parameters"]==r["P"]["trainable_parameters"]
    for v in r.values():
        assert v["steps"]==25 and len(v["losses"])==25 and len(v["nonzero_gradient_tensors"])==12
        assert v["frozen_backbones"]


def test_same_token_control_not_added_cross_token_read(source):
    m=DualStack.from_source(source,"H")
    r=m.reads[0]
    with torch.no_grad():r.o_proj.weight.copy_(source.encoder[0].o_proj.weight)
    x=torch.randn(1,1,16);mem=(torch.randn(1,2,4,8),torch.randn(1,2,4,8))
    other=(mem[0].clone(),mem[1].clone());other[1][:,:,:3]+=4
    args=(torch.tensor([[3]]),torch.tensor([[0,1,2,3]]),torch.ones(1,1,dtype=torch.bool),
          torch.ones(1,4,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long),torch.zeros(1,4,dtype=torch.long))
    assert torch.equal(r(x,mem,*args,"P"),r(x,other,*args,"P"))
    assert not torch.equal(r(x,mem,*args,"H"),r(x,other,*args,"H"))


@pytest.mark.parametrize("mode", ["H", "P", "C"])
@pytest.mark.parametrize("damage", ["foreign", "parameter", "banks", "position_dtype", "position_range", "nan"])
def test_bad_cache_rejected(source,mode,damage):
    m=DualStack.from_source(source,mode);ids=torch.tensor([[1,2,3,4]])
    with torch.inference_mode():cache=m(ids[:,:2],use_cache=True).cache
    if damage=="foreign":cache=replace(cache,owner=-1)
    elif damage=="parameter":
        with torch.no_grad():m.final_norm.weight.add_(0.01)
    elif damage=="banks":cache=replace(cache,decoder=cache.decoder+cache.encoder[:1])
    elif damage=="position_dtype":cache=replace(cache,positions=cache.positions.float())
    elif damage=="position_range":cache=replace(cache,positions=cache.positions+500)
    else:
        k,v=(z.clone() for z in cache.memory);k[0,0,0,0]=float("nan");cache=replace(cache,memory=(k,v))
    with torch.inference_mode(),pytest.raises(ValueError):m(ids[:,2:],past=cache,use_cache=True)


@pytest.mark.parametrize("case", ["dtype", "negative", "vocab", "empty", "mask", "segment_dtype", "position_negative", "position_limit", "mask_shape", "arithmetic", "tail_training", "cache_grad"])
def test_invalid_inputs(source,case):
    m=DualStack.from_source(source,"H");ids=torch.tensor([[1,2,3,4]]);kwargs={}
    if case=="dtype":ids=ids.float()
    elif case=="negative":ids[0,0]=-1
    elif case=="vocab":ids[0,0]=41
    elif case=="empty":ids=ids[:,:0]
    elif case=="mask":kwargs["attention_mask"]=torch.full_like(ids,2)
    elif case=="segment_dtype":kwargs["segment_ids"]=torch.zeros_like(ids).float()
    elif case=="position_negative":kwargs["position_ids"]=torch.full_like(ids,-1)
    elif case=="position_limit":kwargs["position_ids"]=torch.full_like(ids,256)
    elif case=="mask_shape":kwargs["attention_mask"]=torch.ones(1,2)
    elif case=="arithmetic":m.memory_accumulation_dtype=torch.float64
    elif case=="tail_training":m.train();kwargs["last_position_only"]=True
    else:kwargs["use_cache"]=True
    with pytest.raises(ValueError):m(ids,**kwargs)


def test_cannot_relabel_hybrid_as_pure_or_smoke_pure(source):
    m=DualStack.from_source(source,"H")
    with pytest.raises(AttributeError):m.mode="C"
    c=DualStack.from_source(source,"C")
    with pytest.raises(ValueError):c.freeze_backbones()
    c.read_enabled=False
    with pytest.raises(ValueError):c(torch.tensor([[1,2]]))
    with pytest.raises(ValueError):DualStack.from_source(c,"H")
    with torch.inference_mode(),pytest.raises(ValueError):DualStack.from_source(source,"H")


def parent_record():
    bound={"profile":"official_supa","implementation":"fixture-code","source":"fixture-source","runtime":{},"checkout":"fixture-head","fixtures":"fixture-data"}
    checks={k:True for k in g.protocol()["required_checks"]["P1"]}
    measurements={"fixture_only":True}
    for arm in ("H56","P56","C56"):
        measurements[arm]={"checkpoint_weights_sha256":hashlib.sha256((arm+"weights").encode()).hexdigest(),
                           "checkpoint_config_sha256":hashlib.sha256((arm+"config").encode()).hexdigest()}
    return bound,g.receipt("P1",bound,checks,measurements)


def write_parent(tmp_path,body):
    for arm in ("H56","P56","C56"):
        folder=tmp_path/(arm+"_initial");folder.mkdir(exist_ok=True)
        (folder/"model.safetensors").write_bytes((arm+"weights").encode())
        (folder/"config.json").write_bytes((arm+"config").encode())
    path=tmp_path/"parent.json";path.write_text(json.dumps(body));return path


def resign(body):
    body.pop("sha256",None);body["sha256"]=hashlib.sha256(g.canonical(body)).hexdigest()


def test_exact_parent_and_terminal_p2(tmp_path):
    b,r=parent_record();digest=g.validate_parent(write_parent(tmp_path,r),b)
    p2=g.receipt("P2",b,{k:True for k in g.protocol()["required_checks"]["P2"]},{},digest)
    assert p2["status"]=="PASS" and p2["licenses_engineering_smoke"] is False
    assert all(v is False for v in p2["permissions"].values())
    with pytest.raises(ValueError):g.validate_parent(write_parent(tmp_path,p2),b)


@pytest.mark.parametrize("damage",["experiment","stage","status","binding","permissions","checks_missing","checks_false","checks_integer","license","integrity","schema","parent"])
def test_foreign_stale_or_weak_parent_rejected(tmp_path,damage):
    b,r=parent_record()
    if damage=="experiment":r["experiment"]="FWT-001"
    elif damage=="stage":r["stage"]="P2"
    elif damage=="status":r["status"]="FAIL"
    elif damage=="binding":r["binding"]={**b,"checkout":"different-head"}
    elif damage=="permissions":r["permissions"]={**r["permissions"],"recovery_training":True}
    elif damage=="checks_missing":r["checks"].pop(next(iter(r["checks"])))
    elif damage=="checks_false":r["checks"][next(iter(r["checks"]))]=False
    elif damage=="checks_integer":r["checks"][next(iter(r["checks"]))]=1
    elif damage=="license":r["licenses_engineering_smoke"]=False
    elif damage=="schema":r["extra"]=True
    elif damage=="parent":r["parent_sha256"]="x"*64
    else:r["sha256"]="bad"
    if damage!="integrity":resign(r)
    with pytest.raises(ValueError):g.validate_parent(write_parent(tmp_path,r),b)


def test_no_tiny_receipt_or_closed_stage():
    b,r=parent_record()
    with pytest.raises(ValueError):g.receipt("P1",{"profile":"tiny_random_only"},r["checks"],{})
    with pytest.raises(ValueError):g.receipt("P3",b,{}, {})
    bad=g.receipt("P1",b,{},{});assert bad["status"]=="FAIL" and not bad["licenses_engineering_smoke"]


def test_output_scope_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(g,"ROOT",tmp_path/"experiment")
    with pytest.raises(ValueError):g.new_output(tmp_path/"elsewhere")
    with pytest.raises(ValueError):g.new_output(g.ROOT/"outputs")
    p=g.new_output(g.ROOT/"outputs"/"attempt")
    with pytest.raises(FileExistsError):g.new_output(p)
    target=tmp_path/"target";target.mkdir();link=g.ROOT/"outputs"/"link";link.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError):g.new_output(link/"attempt")


def test_implementation_hash_excludes_evidence_not_code(tmp_path,monkeypatch):
    monkeypatch.setattr(g,"ROOT",tmp_path);monkeypatch.setattr(g,"dependencies",lambda:{"fixture":True})
    (tmp_path/"model.py").write_text("a=1\n");before=g.implementation()
    (tmp_path/"verification").mkdir();(tmp_path/"verification"/"result.json").write_text("{}")
    assert g.implementation()==before
    (tmp_path/"model.py").write_text("a=2\n");assert g.implementation()!=before


def test_pinned_dependency_inventory():
    assert len(g.dependencies()["git_blob_sha1_lf"])==9


def test_source_cannot_be_substituted(tmp_path):
    with pytest.raises(ValueError,match="Supa"):g.binding(tmp_path)


@pytest.mark.parametrize("command",["recover","train","evaluate","deploy"])
def test_no_campaign_command(command):
    with pytest.raises(SystemExit) as e:main([command])
    assert e.value.code==2


def test_smoke_requires_explicit_acknowledgment():
    with pytest.raises(SystemExit) as e:main(["smoke","--source","Supa","--parent","p.json","--out","x"])
    assert e.value.code==2


def test_reload_and_controls_reject_inference_mode_construction(source,tmp_path):
    path=tmp_path/"ckpt";DualStack.from_source(source,"H").save(path)
    with torch.inference_mode(),pytest.raises(ValueError):DualStack.load(path)
    with torch.inference_mode(),pytest.raises(ValueError):controls(source,"D56")


@pytest.mark.parametrize("mode",["H","P","C"])
def test_full_28_plus_28_depth_at_small_random_width(source,mode):
    torch.manual_seed(17)
    donor=DenseReference(replace(source.config,encoder_layers=28)).eval()
    m=DualStack.from_source(donor,mode)
    verify_inheritance(m,donor)
    assert len(m.encoder)==len(m.decoder)==28
    x=torch.tensor([[1,2,3,4]])
    with torch.inference_mode():
        out=m(x,use_cache=True)
        assert len(out.cache.encoder)==28
        assert len(out.cache.decoder)==(0 if mode=="C" else 28)
        if mode!="C":assert torch.equal(out.logits,donor(x).logits)
    result=active_path(m,donor) if mode!="C" else grid(m)
    assert len(result)==6


def test_full_depth_small_width_gradient_coverage(source):
    torch.manual_seed(17)
    donor=DenseReference(replace(source.config,encoder_layers=28)).eval()
    r=train_pair(donor,[[1,2,3,4,5,6],[7,8,9,10,11,12],[2,4,6,8,10,12]])
    assert all(v["steps"]==25 and len(v["nonzero_gradient_tensors"])==116 for v in r.values())


def test_changed_parent_checkpoint_rejected(tmp_path):
    b,r=parent_record();path=write_parent(tmp_path,r)
    (tmp_path/"H56_initial"/"model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed P1 artifact"):g.validate_parent(path,b)
