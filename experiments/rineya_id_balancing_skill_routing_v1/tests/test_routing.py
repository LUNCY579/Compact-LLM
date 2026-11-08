import copy
import numpy as np
import pytest
from idr.control import ControlConfig, digest
from idr.routing import Expert, PoolRouter, RoutingConfig, diagnostic, CallExecutor
from idr.experiment import synthetic_spec, synthetic_rows, calibrate, evaluate, exact_state


def experts():
    return [Expert("a", "ledger", "a"*64, True, .8, .25),
            Expert("b", "ledger", "b"*64, True, .2),
            Expert("c", "ledger", "c"*64, False)]


def test_ineligible_expert_never_selected_even_with_huge_bias():
    p = PoolRouter(experts())
    p.controller.bias[:] = [-100, -100, 1000]
    assert p.choose("ledger", [.8,.79,1], [0]*3)["expert"] != "c"


def test_guard_precedes_cost_and_balancing():
    p = PoolRouter(experts(), RoutingConfig(quality_slack=.05, cost_weight=100))
    p.controller.bias[:] = [-100,100,0]
    assert p.choose("ledger", [.9,.7,.8], [100,0,0])["expert"] == "a"


def test_strength_preserved_and_predicted_gap_bounded():
    p = PoolRouter(experts())
    r = p.choose("ledger", [.9,.8,.1],[0]*3)
    assert r["expert"] == "a" and r["strength"] == .25 and r["predicted_quality_gap"] == 0


def test_target_conditioned_on_safe_eligibility():
    p = PoolRouter(experts())
    r = p.choose("ledger", [.8,.8,.8],[0]*3)
    np.testing.assert_allclose(r["target"], [.8,.2,0])
    r = p.choose("ledger", [.9,.1,.8],[0]*3)
    assert r["target"] == [1,0,0]


def test_single_eligible_fixed_point():
    p = PoolRouter(experts(), control=ControlConfig(min_expected=0))
    p.controller.begin()
    for _ in range(12):
        p.choose("ledger", [.8]*3, [0]*3, [True,False,False], record=True)
    p.controller.commit()
    np.testing.assert_array_equal(p.controller.bias, [0]*3)


def test_zero_prior_excluded():
    p = PoolRouter([Expert("a","ledger","a"*64,True,0), Expert("b","ledger","b"*64,True)])
    assert p.choose("ledger", [1,.5],[0,0])["expert"] == "b"


def test_no_valid_alternative_abstains_by_exception():
    with pytest.raises(ValueError):
        PoolRouter(experts()).choose("ledger", [.8]*3,[0]*3, [False]*3)


def test_cross_operation_rejected():
    with pytest.raises(ValueError):
        PoolRouter(experts()).choose("blocks", [.8]*3,[0]*3)
    with pytest.raises(ValueError):
        PoolRouter(experts() + [Expert("d","blocks","d"*64,True)])


@pytest.mark.parametrize("q,c,mask", [([2,.8,.8],[0]*3,None), ([float("nan")]*3,[0]*3,None),
    ([.8]*3,[-1]*3,None), ([.8]*3,[0]*3,[1,1,1]), ([.8]*3,[0]*3,[True])])
def test_invalid_prediction_input(q,c,mask):
    with pytest.raises(ValueError):
        PoolRouter(experts()).choose("ledger", q,c,mask)


def test_stable_tie_breaking():
    assert PoolRouter(experts()).choose("ledger", [.8]*3,[0]*3)["expert"] == "a"


def test_pool_state_identity_includes_strength_and_target_policy():
    a = PoolRouter(experts())
    b = PoolRouter(experts(), RoutingConfig(target_policy="uniform"))
    with pytest.raises(ValueError): b.controller.load_state_dict(a.state_dict())


def test_executor_cache_namespace_and_no_hidden_calls():
    calls=[]
    p = PoolRouter(experts())
    backend=lambda name, strength, req: calls.append((name,strength,req)) or {"text":"ok"}
    execute=CallExecutor(p,backend)
    a=execute("ledger",[.9,.7,.8],[0]*3,{"prompt":"state action"})
    b=execute("ledger",[.7,.9,.8],[0]*3,{"prompt":"state action"})
    assert len(calls)==2 and a["cache_namespace"] != b["cache_namespace"]
    assert calls[0][1] == .25


def test_diagnostic_separates_missing_skill_from_missed_route():
    d=diagnostic([[0,0],[0,1],[1,0]], [0,0,0], [[True,True]]*3)
    assert d["competence_gap_fraction"] == pytest.approx(1/3)
    assert d["missed_available_success_fraction"] == pytest.approx(1/3)
    assert d["routing_regret"] == pytest.approx(1/3)


def test_gold_outcomes_cannot_change_routing_decisions():
    spec=synthetic_spec()
    cal=calibrate(spec,synthetic_rows("calibration",seed=1))
    rows=synthetic_rows("dev",seed=2)
    a=evaluate(spec,rows,cal)
    for row in rows: row["outcomes"]=[1-v for v in row["outcomes"]]
    b=evaluate(spec,rows,cal)
    assert [r["choice"] for r in a["records"]] == [r["choice"] for r in b["records"]]


def test_replay_frozen_and_deterministic():
    spec=synthetic_spec()
    cal=calibrate(spec,synthetic_rows("calibration",seed=1))
    before=copy.deepcopy(cal)
    rows=synthetic_rows("dev",seed=2)
    assert evaluate(spec,rows,cal) == evaluate(spec,rows,cal)
    assert cal == before


@pytest.mark.parametrize("field",["parent_id","payload"])
def test_calibration_overlap_detected_across_separate_files(field):
    spec=synthetic_spec()
    training=synthetic_rows("calibration",seed=1)
    cal=calibrate(spec,training)
    rows=synthetic_rows("dev",seed=2)
    rows[0][field]=training[0][field]
    with pytest.raises(ValueError): evaluate(spec,rows,cal)


def test_calibrate_rejects_sealed_split():
    with pytest.raises(ValueError): calibrate(synthetic_spec(),synthetic_rows("confirm"))


def test_replay_cannot_claim_native():
    spec=synthetic_spec()
    state=calibrate(spec,synthetic_rows("calibration"))
    with pytest.raises(ValueError):
        evaluate(spec,synthetic_rows("dev"),state,evidence_kind="native_execution")


@pytest.mark.parametrize("text,expected,score", [(' {"a":1} ',{"a":1},1),
    ('{"a":2}',{"a":1},0), ('not JSON',{},0), ('true',1,0)])
def test_exact_state(text,expected,score):
    assert exact_state(text,expected)==score
