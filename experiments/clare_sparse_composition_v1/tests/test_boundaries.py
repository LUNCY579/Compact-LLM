import json
from dataclasses import replace
import pytest
from experiments.clare_sparse_composition_v1 import PARENT_COMMIT
from experiments.clare_sparse_composition_v1.data import DATA_MANIFEST,digest_json,file_hash,load_atomic
from experiments.clare_sparse_composition_v1.evaluation import forgetting_matrix,paired
from experiments.clare_sparse_composition_v1.native import preflight,smoke
from experiments.clare_sparse_composition_v1.config import Config
from experiments.clare_sparse_composition_v1.cli import main


def fixture_data(tmp_path):
    (tmp_path/"canonical").mkdir();(tmp_path/"manifests").mkdir()
    prompt=[{"role":"user","content":"synthetic"}]
    rows=[{"id":"row-R","parent_id":"p","role":"train","tag":"core","kind":"composition",
           "skills":["R"],"prompt":prompt,"prompt_hash":digest_json(prompt),"verifier":{}}]
    (tmp_path/"canonical/train.jsonl").write_text(json.dumps(rows[0])+"\n")
    (tmp_path/"manifests/atomic_R.json").write_text(json.dumps({"name":"atomic_R","ids":["row-R"],"rows":1}))
    files={str(p.relative_to(tmp_path)):{"sha256":file_hash(p),"bytes":p.stat().st_size} for p in tmp_path.rglob("*") if p.is_file()}
    (tmp_path/"manifest.json").write_text(json.dumps({"experiment":"MAXRL-COMP-002","files":files}))
    return file_hash(tmp_path/"manifest.json")


def test_actual_corpus_schema_adapter_no_eval_access(tmp_path):
    pin=fixture_data(tmp_path);(tmp_path/"canonical/test.jsonl").write_text("not even JSON")
    rows=load_atomic(tmp_path,"R",pin);assert rows[0].role=="train" and rows[0].row_id=="row-R"
    assert rows[0].messages==[{"role":"user","content":"synthetic"}]


@pytest.mark.parametrize("kind",["pin","bytes","missing"])
def test_corpus_refusal(tmp_path,kind):
    pin=fixture_data(tmp_path)
    if kind=="pin":pin="0"*64
    if kind=="bytes":(tmp_path/"canonical/train.jsonl").write_text("drift")
    if kind=="missing":(tmp_path/"canonical/train.jsonl").unlink()
    with pytest.raises((ValueError,OSError)):load_atomic(tmp_path,"R",pin)


def eval_rows():
    return [{"id":"a","parent_id":"p","domain":"R","prompt_sha256":"a"*64,"correct":True},
            {"id":"b","parent_id":"p","domain":"R","prompt_sha256":"a"*64,"correct":False},
            {"id":"c","parent_id":"q","domain":"RS","prompt_sha256":"a"*64,"correct":False}]


def meta():
    from experiments.clare_sparse_composition_v1.evaluation import MATCH
    return {k:("a"*64 if k.endswith("sha256") else "same") for k in MATCH}


def test_parent_cluster_pairing_and_undefined_retention():
    a=eval_rows();b=[dict(x,correct=not x["correct"]) for x in a]
    result=paired(a,b,meta(),meta(),draws=40)
    assert result["independent_parents"]==2
    assert result["domains"]["R"]["damage"]==1 and result["domains"]["R"]["gains"]==1
    assert result["domains"]["RS"]["retention_fraction"] is None
    assert result==paired(a,b,meta(),meta(),draws=40)


@pytest.mark.parametrize("kind",["drop","duplicate","metadata","parent","nonbool"])
def test_evaluation_refuses_unmatched(kind):
    a=eval_rows();b=eval_rows();m=meta()
    if kind=="drop":b.pop()
    if kind=="duplicate":b.append(b[0])
    if kind=="metadata":m["tokenizer_sha256"]="different"
    if kind=="parent":b[0]["parent_id"]="different"
    if kind=="nonbool":b[0]["correct"]=1
    with pytest.raises(ValueError):paired(a,b,meta(),m,draws=2)


def test_forgetting_matrix():
    r=forgetting_matrix(("R","U"),[{"R":.1,"U":.1},{"R":.8,"U":.2},{"R":.6,"U":.9}])
    assert r["R"]["acquisition"]==pytest.approx(.7)
    assert r["R"]["forgetting_from_best"]==pytest.approx(.2)
    with pytest.raises(ValueError):forgetting_matrix(("R",),[{"R":.2}])


def test_unbound_native_is_blocked_without_model_load():
    binding={"parent_commit":PARENT_COMMIT,"dataset_sha256":DATA_MANIFEST}
    r=preflight(binding);assert r["status"]=="BLOCKED" and r["native_model_loads"]==0
    assert smoke(binding,Config())["status"]=="BLOCKED"


def test_cli_preflight_writes_refusal_and_never_overwrites(tmp_path):
    b=tmp_path/"binding.json";b.write_text("{}")
    out=tmp_path/"report.json"
    assert main(["preflight","--binding",str(b),"--out",str(out)])==2
    original=out.read_bytes()
    assert main(["preflight","--binding",str(b),"--out",str(out)])==2
    assert out.read_bytes()==original
