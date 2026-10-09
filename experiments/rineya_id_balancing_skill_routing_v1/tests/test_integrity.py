import copy
import json
from pathlib import Path
import pytest
from idr.artifacts import write_json, load_json, tree_manifest, verify_tree, freeze, verify_freeze, validate_rows
from idr.control import digest
from idr.experiment import synthetic_rows, synthetic_spec, calibrate, evaluate
from idr.evaluation import paired_interval, compare
from idr.native import preflight, load_base, train_native
from idr.quality import fit_quality
from idr.__main__ import main


def test_json_write_once_and_no_nan(tmp_path):
    p=tmp_path/"report.json"
    write_json(p,{"a":1})
    assert load_json(p)=={"a":1}
    with pytest.raises(FileExistsError): write_json(p,{"a":2})
    with pytest.raises(ValueError): write_json(tmp_path/"bad.json",{"x":float("nan")})
    assert not (tmp_path/"bad.json").exists()


@pytest.mark.parametrize("text",['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json(tmp_path,text):
    p=tmp_path/"bad.json"; p.write_text(text)
    with pytest.raises(ValueError): load_json(p)


def test_manifest_integrity_and_extra_file_detection(tmp_path):
    root=tmp_path/"model"; root.mkdir()
    (root/"a.txt").write_text("weights")
    expected=tree_manifest(root)
    assert verify_tree(root,expected)==expected["tree_sha256"]
    (root/"extra.txt").write_text("unexpected")
    with pytest.raises(ValueError): verify_tree(root,expected)


def test_symlink_not_admitted(tmp_path):
    root=tmp_path/"model"; root.mkdir()
    original=tmp_path/"original"; original.write_text("x")
    (root/"link").symlink_to(original)
    with pytest.raises(ValueError): tree_manifest(root)


def test_freeze_rejects_changed_input(tmp_path):
    data=tmp_path/"data.json"; write_json(data,[1,2,3])
    freeze([data],tmp_path/"freeze.json")
    verify_freeze(tmp_path/"freeze.json")
    data.write_text("[4]")
    with pytest.raises(ValueError): verify_freeze(tmp_path/"freeze.json")


@pytest.mark.parametrize("kind",["parent","payload","id","split"])
def test_split_custody(kind):
    a=synthetic_rows("train",4)
    b=synthetic_rows("dev",4)
    if kind=="parent": b[0]["parent_id"]=a[0]["parent_id"]
    if kind=="payload": b[0]["payload"]=a[0]["payload"]
    if kind=="id": b[0]["id"]=a[0]["id"]
    if kind=="split": b[0]["split"]="other"
    with pytest.raises(ValueError): validate_rows(a+b)


def test_parent_cluster_bootstrap_counts_parents_not_goals():
    result=paired_interval([0]*256+[1]*256,["ledger"]*512,["a"]*256+["b"]*256,200,0)
    assert result["parents"]==2 and result["rows"]==512
    assert result["low"]==0 and result["high"]==1


def test_zero_variance_not_false_tight_retention():
    result=paired_interval([0]*10,["ledger"]*10,list(range(10)),200)
    assert result["low"]<-.02 and result["high"]>.02


def test_bootstrap_requires_independent_parents():
    with pytest.raises(ValueError): paired_interval([0,1],["ledger"]*2,["same"]*2,200)


def reports():
    spec=synthetic_spec()
    state=calibrate(spec,synthetic_rows("calibration"))
    a=evaluate(spec,synthetic_rows("confirm"),state,split="confirm")
    b=copy.deepcopy(a)
    for r in a["records"]: r["success"]=1.0
    for r in b["records"]: r["success"]=0.0
    return a,b


def test_synthetic_cannot_promote_even_perfect_gain():
    a,b=reports()
    r=compare(a,b,repetitions=200)
    assert r["numeric_criterion_pass"]
    assert r["status"]=="NO_PROMOTION" and not r["native_qualified"]


def test_replay_not_an_end_to_end_efficiency_benchmark():
    a,b=reports()
    for r in a["records"]: r["elapsed_seconds"]*=.01
    r=compare(a,b,"efficiency",200)
    assert not r["efficiency"]["eligible"] and not r["numeric_criterion_pass"]


@pytest.mark.parametrize("field",["data_sha256","pool_sha256","budget","evaluation_split"])
def test_unmatched_comparison_rejected(field):
    a,b=reports(); b[field]="wrong"
    with pytest.raises(ValueError): compare(a,b,repetitions=200)


def test_missing_pair_rejected():
    a,b=reports(); b["records"].pop()
    with pytest.raises(ValueError): compare(a,b,repetitions=200)


def test_no_native_bypass(tmp_path):
    config={"allow_native":False,"dtype":"bfloat16","device":"cpu","base":{"path":""}}
    r=preflight(config)
    assert not r["ready"] and not r["native_executed"]
    with pytest.raises(RuntimeError): load_base(config)
    with pytest.raises(ValueError): train_native(config,synthetic_rows("confirm"),tmp_path/"out")


def test_native_registry_safetensors_contract(tmp_path):
    model=tmp_path/"base"; model.mkdir()
    write_json(model/"config.json",{"model_type":"gpt2"})
    (model/"weights.bin").write_text("fixture only")
    config={"allow_native":True,"dtype":"float32","device":"cpu",
            "base":{"path":str(model),"manifest":tree_manifest(model)}}
    r=preflight(config)
    assert not r["ready"]
    assert any("dense, small Qwen3" in v for v in r["blockers"])


def test_quality_head_real_training_but_unqualified():
    rows=synthetic_rows("train",count=32)
    for i,r in enumerate(rows):
        r["features"]=[i%2,1-i%2]
        r["outcomes"]=[i%2,1-i%2]
        r["observed"]=[True,True]
    head,report=fit_quality(rows,epochs=50)
    import torch
    predictions=head(torch.tensor([[0.,1.],[1.,0.]]))
    assert predictions.argmax(1).tolist()==[1,0]
    assert not report["qualified"]


def test_quality_head_rejects_confirm_and_unobserved_expert():
    with pytest.raises(ValueError): fit_quality(synthetic_rows("confirm",4))
    rows=synthetic_rows("train",4)
    for row in rows:
        row["features"]=[1,0]; row["outcomes"]=[1,0]; row["observed"]=[True,False]
    with pytest.raises(ValueError): fit_quality(rows)


def test_cli_calibrate_replay_and_confirmation_claim(tmp_path):
    spec=tmp_path/"spec.json"; cal=tmp_path/"cal.json"; dev=tmp_path/"dev.json"
    state=tmp_path/"state.json"; out=tmp_path/"result.json"
    write_json(spec,synthetic_spec()); write_json(cal,synthetic_rows("calibration",32))
    write_json(dev,synthetic_rows("confirm",32))
    assert main(["calibrate","--spec",str(spec),"--rows",str(cal),"--out",str(state)])==0
    args=["replay","--spec",str(spec),"--rows",str(dev),"--state",str(state),"--split","confirm"]
    assert main(args+["--out",str(out)])==2
    frozen=tmp_path/"freeze.json"
    freeze([spec,dev,state],frozen)
    assert main(args+["--freeze",str(frozen),"--out",str(out)])==0
    assert load_json(out)["evaluation_split"]=="confirm"
    # Even changing output names cannot rerun a consumed freeze.
    assert main(args+["--freeze",str(frozen),"--out",str(tmp_path/"again.json")])==2
    assert not (tmp_path/"again.json").exists()


def test_existing_output_prevents_reexecution(tmp_path):
    out=tmp_path/"already.json"; write_json(out,{})
    assert main(["preflight","--config","missing.json","--out",str(out)])==2


def test_confirmation_freeze_must_bind_every_input(tmp_path):
    spec=tmp_path/"spec.json"; rows=tmp_path/"rows.json"; state=tmp_path/"state.json"
    s=synthetic_spec(); write_json(spec,s); write_json(rows,synthetic_rows("confirm",4))
    write_json(state,calibrate(s,synthetic_rows("calibration",32)))
    frozen=tmp_path/"freeze.json"; freeze([rows],frozen)
    assert main(["replay","--spec",str(spec),"--rows",str(rows),"--state",str(state),
                 "--split","confirm","--freeze",str(frozen),"--out",str(tmp_path/"out.json")])==2
