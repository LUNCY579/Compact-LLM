import copy
import json
import pytest
from idr.artifacts import validate_rows
from idr.control import digest
from idr.worlds import transition, ledger_rows, self_rollout, input_features
from idr.experiment import synthetic_spec, calibrate, evaluate
from idr.dataflow import capture, save_quality, predict_quality


@pytest.mark.parametrize("status",["void","posted ","POSTED","unposted","not posted","nonce","cancelled"])
def test_rule_is_exact_posted_not_lexical_void(status):
    assert transition({"a":10,"b":2},{"status":status})=={"a":10,"b":2}


@pytest.mark.parametrize("op,expected",[("add",{"a":13,"b":2}),("scale",{"a":30,"b":2}),
                                       ("transfer",{"a":7,"b":5})])
def test_ledger_oracle(op,expected):
    assert transition({"a":10,"b":2},{"status":"posted","source":"a","destination":"b","op":op,"amount":3})==expected


def test_generated_banks_reproducible_and_disjoint():
    banks=[ledger_rows(s,20,23) for s in ("train","calibration","dev","confirm")]
    assert banks[0]==ledger_rows("train",20,23)
    validate_rows(sum(banks,[]))
    assert len(banks[2])==120
    assert all(r["target"]==json.dumps(r["expected_state"],sort_keys=True) for r in banks[0])


def test_one_factor_challenges_keep_other_factors_fixed():
    rows=ledger_rows("dev",2)
    baseline, status, magnitude, renamed, reordered, combined=rows[:6]
    assert baseline["world_input"]["state"]==status["world_input"]["state"]
    assert baseline["world_input"]["event"]["amount"]==status["world_input"]["event"]["amount"]
    assert baseline["world_input"]==reordered["world_input"]
    assert baseline["payload"]!=reordered["payload"]
    assert status["expected_state"]==status["world_input"]["state"]
    assert baseline["world_input"]["event"]["status"]=="posted"


def test_self_rollout_does_not_teacher_force():
    calls=[]
    def bad_predict(state,event):
        calls.append(dict(state))
        return {k:v+10 for k,v in state.items()}
    result=self_rollout({"a":0},[{},{}],bad_predict)
    assert calls==[{"a":0},{"a":10}]
    assert result["final_state"]=={"a":20}


def test_bad_rollout_schema_fails_closed():
    assert not self_rollout({"a":0},[{}],lambda s,e:{"wrong":1})["valid"]


def ledger_spec():
    spec=synthetic_spec()
    for expert in spec["experts"]: expert["operation"]="ledger_transition"
    return spec


def fixture_backend(name,strength,request):
    # Deliberately synthetic test callback, NOT native model evidence.
    state=json.loads(request["prompt"].split("State: ")[1].split("\nEvent: ")[0])
    event=json.loads(request["prompt"].split("\nEvent: ")[1].split("\nNext state:")[0])
    result=transition(state,event) if name in {"expert_0","expert_2"} else state
    return {"text":json.dumps(result),"elapsed_seconds":.1,"evidence_kind":"synthetic"}


def test_capture_accounts_all_counterfactual_experts():
    data=capture(ledger_spec(),ledger_rows("train",4),fixture_backend)
    assert all(r["counterfactual_calls"]==4 for r in data)
    assert all(r["counterfactual_seconds"]==pytest.approx(.4) for r in data)
    assert all(r["observed"]==[True]*4 for r in data)


def test_counterfactual_confirm_capture_rejected():
    with pytest.raises(ValueError): capture(ledger_spec(),ledger_rows("confirm",4),fixture_backend)


def test_train_predict_calibrate_evaluate_end_to_end_fixture(tmp_path):
    spec=ledger_spec()
    training=capture(spec,ledger_rows("train",20),fixture_backend)
    path=tmp_path/"quality.pt"
    report=save_quality(spec,training,path,epochs=8)
    cal=predict_quality(spec,ledger_rows("calibration",20),path,report["checkpoint_sha256"])
    state=calibrate(spec,cal)
    dev=capture(spec,ledger_rows("dev",20),fixture_backend)
    predicted=predict_quality(spec,dev,path,report["checkpoint_sha256"])
    a=evaluate(spec,predicted,state)
    altered=copy.deepcopy(dev)
    for row in altered: row["outcomes"]=[1-v for v in row["outcomes"]]
    prediction_b=predict_quality(spec,altered,path,report["checkpoint_sha256"])
    assert [r["quality"] for r in predicted]==[r["quality"] for r in prediction_b]
    assert a["evidence_kind"]=="synthetic" and not a["native_qualified"]
    with pytest.raises(ValueError): predict_quality(spec,ledger_rows("dev",2),path,"0"*64)
    with pytest.raises(ValueError): predict_quality(spec,ledger_rows("train",2),path,report["checkpoint_sha256"])


def test_quality_source_cannot_change_after_calibration():
    spec=ledger_spec()
    cal=ledger_rows("calibration",4)
    for r in cal:
        r.update(quality=[.8]*4,predicted_cost=[1]*4,quality_source={"head":"v1"})
    state=calibrate(spec,cal)
    dev=ledger_rows("dev",4)
    for r in dev:
        r.update(quality=[.8]*4,predicted_cost=[1]*4,quality_source={"head":"v2"},outcomes=[1]*4,observed_cost=[1]*4)
    with pytest.raises(ValueError): evaluate(spec,dev,state)
