from dataclasses import asdict, replace
import math
import random

import pytest

from experiments.rineya_tropical_verified_composition_v1.archive import Archive, top
from experiments.rineya_tropical_verified_composition_v1.domain import (
    ACTIONS, InvalidAction, State, Task, audit_splits, prompt, root, step, success, terminal)
from experiments.rineya_tropical_verified_composition_v1.runtime import (
    BudgetExceeded, Ledger, Sample, ScriptedPolicy, replay, rollout)
from experiments.rineya_tropical_verified_composition_v1.study import mechanism_demo, preflight, make_tasks, write_new


def task(**kwargs):
    return replace(Task("x", "p", "train", (0, 0), (2, 1), 4), **kwargs)


def collect(archive, actions, identity, prefix=(), complete=True):
    return archive.collect(ScriptedPolicy(tuple(actions)), random.Random(0), Ledger(),
                           identity, tuple(prefix), complete)


def rescore(archive):
    archive.rescore("v1", lambda *_: -math.log(6), Ledger())


@pytest.mark.parametrize("action,expected", [
    ("inc_x", (2, 2)), ("inc_y", (1, 3)), ("double_x", (2, 2)),
    ("double_y", (1, 4)), ("swap", (2, 1))])
def test_exact_operations(action, expected):
    t = task(initial=(1, 2), goal=(8, 8))
    state = step(t, root(t), action)
    assert (state.x, state.y) == expected and state.depth == 1


@pytest.mark.parametrize("kwargs", [
    {"horizon": 0}, {"horizon": True}, {"horizon": 13}, {"bound": 0},
    {"role": "sealed_train"}, {"task_id": ""}, {"parent_id": ""},
    {"initial": [0, 0]}, {"initial": (True, 0)}, {"goal": (33, 0)},
    {"goal": (0, 0)}, {"rules": "different"}])
def test_invalid_task(kwargs):
    with pytest.raises(ValueError):
        task(**kwargs)


def test_json_roundtrip():
    t = task()
    assert Task.from_dict(asdict(t)) == t


def test_markov_diamond_and_depth():
    t = task()
    left = step(t, step(t, root(t), "inc_x"), "inc_y")
    right = step(t, step(t, root(t), "inc_y"), "inc_x")
    assert left == right and prompt(t, left) == prompt(t, right)
    assert left != replace(left, depth=3)


@pytest.mark.parametrize("other", [{"goal": (3, 3)}, {"horizon": 5}, {"task_id": "other"}, {"role": "dev"}])
def test_foreign_interface(other):
    t, u = task(), task(**other)
    with pytest.raises(ValueError):
        step(t, root(u), "inc_x")


def test_forged_root():
    t = task()
    with pytest.raises(ValueError):
        step(t, State(t.key, 0, 1, 0), "inc_x")


@pytest.mark.parametrize("action", ["INVALID", "__import__('os')", "inc_x; inc_y", "", "INC_X"])
def test_action_parser_is_closed(action):
    with pytest.raises(InvalidAction):
        step(task(), root(task()), action)


def test_bound_and_terminal():
    t = task(initial=(2, 2), goal=(0, 0), bound=2)
    with pytest.raises(InvalidAction):
        step(t, root(t), "double_x")
    t = task(goal=(1, 0))
    state = step(t, root(t), "inc_x")
    assert terminal(t, state) and success(t, state)
    with pytest.raises(InvalidAction):
        step(t, state, "inc_y")


@pytest.mark.parametrize("kwargs", [{"role": "test", "parent_id": "q"}, {"role": "guard", "initial": (1, 0)}])
def test_split_leakage(kwargs):
    with pytest.raises(ValueError):
        audit_splits([task(), task(task_id="second", **kwargs)])


def test_empty_duplicate_splits():
    for tasks in ([], [task(), task()]):
        with pytest.raises(ValueError):
            audit_splits(tasks)


@pytest.mark.parametrize("role", ["test", "guard", "dev"])
def test_evaluation_cannot_enter_archive(role):
    with pytest.raises(ValueError):
        Archive(task(role=role))


def test_partial_failure_reused_with_restart():
    result = mechanism_demo()
    assert result["failed_only_candidates"] == 0
    assert result["certified_solutions"] == 1
    assert result["first_origin"] == "restart_completion"
    assert result["root_sampled_successes"] == 0
    candidate = result["archive"]["candidates"][0]
    assert set(candidate["support_attempts"]) == {"failed-root", "successful-frontier"}
    assert result["ledger"]["used"]["generated_tokens"] == 0


def test_real_graph_recombination_after_root_success():
    a = Archive(task())
    collect(a, ("inc_x", "inc_y", "inc_x"), "success")
    collect(a, ("inc_y", "inc_x", "INVALID"), "failed")
    rescore(a)
    additions = a.compose("v1", Ledger())
    assert any(c.actions == ("inc_y", "inc_x", "inc_x") for c in additions)
    assert all(c.origin == "graph_composition" for c in additions)
    assert list(a.candidates.values())[0].origin == "root_sample"
    assert a.compose("v1", Ledger()) == []


@pytest.mark.parametrize("seed", range(8))
def test_root_only_no_terminal_no_first_solve(seed):
    t = task(goal=(3, 3), horizon=2)
    a, rng = Archive(t), random.Random(seed)
    for index in range(8):
        episode = collect(a, tuple(rng.choices(ACTIONS, k=2)), str(index))
        assert not episode.accepted
    rescore(a)
    assert not a.accepting_states and not a.compose("v1", Ledger())


def test_deferred_restart_has_observed_successful_suffix():
    a = Archive(task(goal=(1, 1)))
    collect(a, ("inc_x", "INVALID"), "failed")
    collect(a, ("inc_y",), "suffix", prefix=("inc_x",), complete=False)
    assert not a.candidates and a.accepting_states
    rescore(a)
    assert len(a.compose("v1", Ledger())) == 1
    assert a.episodes["suffix"].accepted


def test_no_unobserved_restart_or_admission():
    a = Archive(task())
    with pytest.raises(ValueError, match="unobserved"):
        collect(a, ("inc_y",), "fake", prefix=("inc_x",))
    with pytest.raises(ValueError):
        a._admit(("inc_x", "inc_y", "inc_x"), "fake", Ledger())


def test_duplicate_attempt_and_edge_idempotence():
    a = Archive(task())
    collect(a, ("inc_x", "INVALID"), "one")
    collect(a, ("inc_x", "INVALID"), "two")
    assert len(a.edges) == 1 and len(next(iter(a.edges.values())).sources) == 2
    with pytest.raises(ValueError):
        collect(a, (), "one")


def test_graph_capacity_fails_without_eviction():
    a = Archive(task(), max_edges=1)
    collect(a, ("inc_x", "INVALID"), "one")
    retained = set(a.edges)
    with pytest.raises(RuntimeError, match="capacity"):
        collect(a, ("inc_y", "INVALID"), "two")
    assert set(a.edges) == retained and "two" in a.episodes and a.capacity_events == 1


def test_solution_capacity_fails_without_eviction():
    a = Archive(task(goal=(1, 1)), max_paths=1)
    collect(a, ("inc_x", "inc_y"), "one")
    retained = set(a.candidates)
    with pytest.raises(RuntimeError, match="capacity"):
        collect(a, ("inc_y", "inc_x"), "two")
    assert set(a.candidates) == retained


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), 0.1])
def test_invalid_scores_invalidate_revision(score):
    a = Archive(task())
    collect(a, ("inc_x",), "one")
    rescore(a)
    with pytest.raises(ValueError):
        a.rescore("v2", lambda *_: score, Ledger())
    with pytest.raises(ValueError, match="Stale"):
        a.compose("v1", Ledger())


def test_new_edges_require_rescore():
    a = Archive(task())
    rescore(a)
    collect(a, ("inc_x",), "one")
    with pytest.raises(ValueError):
        a.dynamic_program("v1")


def test_topk_dp_and_coverage():
    a = Archive(task(goal=(1, 1)))
    for i, path in enumerate((("inc_x", "inc_y"), ("inc_y", "inc_x"))):
        collect(a, path, str(i))
    rescore(a)
    f, b = a.dynamic_program("v1")
    assert len(b[root(a.task)]) == 2
    assert all(math.isclose(v, -2 * math.log(6)) for v, _ in b[root(a.task)])
    selected = a.coverage_selection("v1", 2)
    a.mark_trained(selected)
    assert all(e.training_uses == 1 for e in a.edges.values())
    assert len(selected) == 2
    assert top([(-1.0, ("b",)), (-1.0, ("a",)), (-1.0, ("a",))], 2) == [(-1.0, ("a",)), (-1.0, ("b",))]


def test_actual_overspend_is_recorded():
    ledger = Ledger({"generated_tokens": 1})
    with pytest.raises(BudgetExceeded):
        ledger.spend(generated_tokens=2)
    assert ledger.used["generated_tokens"] == 2 and ledger.violations
    with pytest.raises(BudgetExceeded):
        ledger.require(policy_calls=1)


def test_precheck_stops_before_policy_call():
    ledger = Ledger({"policy_calls": 0})
    with pytest.raises(BudgetExceeded):
        rollout(task(), ScriptedPolicy(("inc_x",)), random.Random(0), ledger, "one")
    assert ledger.used["policy_calls"] == ledger.used["env_steps"] == 0
    assert ledger.used["root_attempts"] == 1


@pytest.mark.parametrize("kwargs", [{"bogus": 1}, {"env_steps": -1}, {"env_steps": True}])
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        Ledger(kwargs)


def test_invalid_actions_count_but_do_not_enter_graph():
    a, ledger = Archive(task()), Ledger()
    episode = a.collect(ScriptedPolicy(("INVALID",)), random.Random(0), ledger, "bad")
    assert not a.edges and not episode.accepted
    assert ledger.used["policy_calls"] == ledger.used["env_steps"] == 1


def test_replay_accounts_verifier_and_actions():
    ledger = Ledger()
    replay(task(goal=(1, 1)), ("inc_x", "inc_y"), ledger)
    assert ledger.used["verification_calls"] == 1 and ledger.used["replay_steps"] == 2
    with pytest.raises(ValueError):
        replay(task(), ("inc_x",), Ledger())


def test_native_preflight_never_accepts_declarations_as_measurements():
    result = preflight({})
    assert result["status"] == "BLOCKED_NATIVE_UNQUALIFIED"
    assert len(result["reasons"]) == 10 and result["model_loads"] == 0


def test_corpus_is_deterministic_and_disjoint():
    a = make_tasks(7, {"train": 8, "dev": 8, "guard": 8})
    b = make_tasks(7, {"train": 8, "dev": 8, "guard": 8})
    assert a == b and len({t.semantic_key for t in a}) == 24
    assert audit_splits(a)["tasks"] == 24


def test_output_refuses_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    write_new(path, {"ok": True})
    with pytest.raises(FileExistsError):
        write_new(path, {"bad": True})


def test_external_generator_cap_violation_keeps_actual_cost():
    class BadProvider:
        def sample(self, task, state, rng, max_new_tokens):
            assert max_new_tokens == 32
            return Sample("inc_x", generated_tokens=33)
    ledger = Ledger()
    with pytest.raises(BudgetExceeded):
        rollout(task(), BadProvider(), random.Random(0), ledger, "bad-provider")
    assert ledger.used["generated_tokens"] == 33
    assert ledger.used["env_steps"] == 0 and ledger.violations
