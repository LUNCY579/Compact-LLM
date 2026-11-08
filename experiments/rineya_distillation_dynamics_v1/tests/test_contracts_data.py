import json
from dataclasses import asdict, replace

import pytest

from experiments.rineya_distillation_dynamics_v1.contracts import (
    ContractError, Plan, digest, matrix, read_json, sealed, unseal, write_new)
from experiments.rineya_distillation_dynamics_v1.data import Task, audit, cross_audit, load_catalog, normalized, verify
from experiments.rineya_distillation_dynamics_v1.fixtures import fixture_catalog, fixture_tasks


@pytest.mark.parametrize("kwargs", [
    {"seeds": (1, 1)}, {"seeds": (True,)}, {"seeds": ()}, {"steps": True}, {"steps": 0},
    {"learning_rates": (.01, .01)}, {"learning_rates": (float("nan"), .01)},
    {"learning_rates": (.01,)}, {"replay_tasks": 0}, {"replay_weight": 0},
    {"sampling_temperature": float("inf")}, {"max_grad_norm": 0}, {"schema": 2},
    {"min_acquisition_gain": -.1}, {"eval_samples": 100}, {"max_new_tokens": 1024}])
def test_invalid_plan(kwargs):
    with pytest.raises(ContractError): Plan(**kwargs)


def test_factorial_matrix():
    p = Plan()
    full = matrix(p)
    assert len(full) == 30 and len({a.name for a in full}) == 30
    assert len(matrix(p, "screen")) == 18
    for seed in p.seeds:
        assert {(a.source, a.objective, a.learning_rate) for a in full if a.seed == seed and a.objective != "ce"} == {
            (source, loss, lr) for source in ("student", "teacher") for loss in ("forward", "reverse") for lr in p.learning_rates}


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_strict_json(tmp_path, text):
    f = tmp_path / "bad.json"
    f.write_text(text)
    with pytest.raises(ContractError): read_json(f)


def test_exclusive_evidence(tmp_path):
    f = tmp_path / "a.json"
    write_new(f, {"a": 1})
    with pytest.raises(FileExistsError): write_new(f, {"a": 2})
    assert read_json(f) == {"a": 1}


def test_evidence_checksum():
    s = sealed({"a": 1})
    assert unseal(s) == {"a": 1}
    s["payload"]["a"] = 2
    with pytest.raises(ContractError): unseal(s)


def test_catalog_roundtrip(tmp_path):
    tasks = fixture_tasks()
    f = tmp_path / "tasks.json"
    write_new(f, fixture_catalog(tasks))
    loaded, provenance, result = load_catalog(f)
    assert loaded == tasks and provenance == "mechanical_fixture"
    assert not result["semantic_decontamination"]


@pytest.mark.parametrize("kind", ["parent", "prompt", "duplicate_key", "sealed", "wrong_solution", "heldout_solution", "unknown_grader"])
def test_catalog_contamination(kind):
    ts = fixture_tasks()
    if kind == "parent": ts[-1] = replace(ts[-1], parent=ts[0].parent)
    if kind == "prompt": ts[-1] = replace(ts[-1], prompt=ts[0].prompt.upper() + "  ")
    if kind == "duplicate_key": ts[-1] = replace(ts[-1], key=ts[0].key)
    if kind == "sealed": ts[0] = replace(ts[0], role="sealed")
    if kind == "wrong_solution": ts[-1] = replace(ts[-1], solution="wrong")
    if kind == "heldout_solution": ts[0] = replace(ts[0], solution="1")
    if kind == "unknown_grader": ts[0] = replace(ts[0], verifier="LLM")
    with pytest.raises(ContractError): audit(ts)


@pytest.mark.parametrize("kind", ["parent", "prompt"])
def test_exclusions(kind):
    ts = fixture_tasks()
    e = {"parents": [], "prompt_hashes": []}
    e["parents" if kind == "parent" else "prompt_hashes"] = [ts[0].parent if kind == "parent" else digest(normalized(ts[0].prompt))]
    with pytest.raises(ContractError): audit(ts, e)


def test_confirmation_collision():
    ts = fixture_tasks()
    c = [replace(ts[0], key="new", role="confirm_acquisition")]
    with pytest.raises(ContractError): cross_audit(ts, c)


@pytest.mark.parametrize("text,expected", [
    ("2+3=5. <final>5</final>", True), ("<final>05</final>", False),
    ("<final>5</final> done", False), ("<final>5</final><final>5</final>", False),
    ("<final>4</final>", False), ("<final>5</final>\n", True)])
def test_final_integer(text, expected):
    t = Task("k", "p", "s", "v", "train", "2+3?", "5", "<final>5</final>", "final-int-v1")
    assert verify(t, text) is expected


def test_public_problem_has_no_gold():
    public = fixture_tasks()[0].public()
    assert set(asdict(public)) == {"key", "prompt"}
