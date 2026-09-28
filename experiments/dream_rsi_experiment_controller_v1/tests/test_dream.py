"""Mechanics tests; all measurements and historical-shaped receipts here are synthetic."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from experiments.dream_rsi_experiment_controller_v1 import schema, study
from experiments.dream_rsi_experiment_controller_v1.__main__ import main
from experiments.dream_rsi_experiment_controller_v1.fixtures import create_bundle, make_world, outcome
from experiments.dream_rsi_experiment_controller_v1.replay import (
    Controller, Decision, Plan, Replay, Spec, evaluate, frontier, gate, hypervolume)
from experiments.dream_rsi_experiment_controller_v1.schema import (
    Invalid, Outcome, canonical, digest, load_partition, parse_world, pointer,
    read_json, safe_path, validate_catalog, write_new)


def tiny_world():
    raw = make_world()
    raw["nodes"] = [{"id": "a", "parent": "root", "order": 1, "context": ["root"],
                     "outcome": outcome(0.8, 0.8)}]
    return raw


def changed(raw, key, value):
    result = copy.deepcopy(raw)
    result[key] = value
    return result


def test_world_is_immutable():
    world = parse_world(make_world())
    with pytest.raises(FrozenInstanceError):
        world.key = "changed"
    assert isinstance(world.nodes, tuple)
    assert isinstance(world.nodes[0].context, tuple)


@pytest.mark.parametrize("key,value", [
    ("closed", False), ("sealed", True), ("kind", "native"), ("schema", "v0"),
    ("id", ""), ("nodes", []), ("leakage_keys", []),
    ("leakage_keys", ["x", "x"]), ("admission", {}),
])
def test_world_envelope_rejected(key, value):
    with pytest.raises(Invalid):
        parse_world(changed(make_world(), key, value))


@pytest.mark.parametrize("key,value", [
    ("capability", float("nan")), ("retention", float("inf")), ("capability", -0.1),
    ("retention", 1.1), ("gpu_seconds", -1), ("gpu_seconds", True),
    ("qualified", "yes"), ("protocol_valid", 1), ("status", "PASS"),
    ("failure", "made_up"), ("parameter_delta", 0.5),
    ("latency_ratio", -1), ("peak_vram_gib", -1),
])
def test_outcome_scalar_validation(key, value):
    with pytest.raises(Invalid):
        Outcome.parse(changed(outcome(), key, value))


def test_failed_metrics_never_imputed():
    for status in ("failed", "blocked", "not_run"):
        raw = outcome(0, 0, status, "resource")
        with pytest.raises(Invalid):
            Outcome.parse(raw)
        assert Outcome.parse(outcome(None, None, status, "resource")).capability is None


def test_not_run_is_not_a_replay_attempt():
    raw = tiny_world()
    raw["nodes"][0]["outcome"] = outcome(None, None, "not_run", "resource")
    with pytest.raises(Invalid, match="unexecuted"):
        parse_world(raw)


def test_measured_invalid_is_not_an_execution_error():
    raw = outcome()
    raw["qualified"] = False
    value = Outcome.parse(raw)
    assert value.failure == "none"
    assert not gate(value, Plan())


@pytest.mark.parametrize("change", ["cycle", "duplicate", "order", "missing_parent", "branch", "context"])
def test_invalid_topologies(change):
    raw = make_world()
    if change == "cycle":
        raw["nodes"][0]["parent"] = raw["nodes"][0]["id"]
    elif change == "duplicate":
        raw["nodes"][1]["id"] = raw["nodes"][0]["id"]
    elif change == "order":
        raw["nodes"][1]["order"] = raw["nodes"][0]["order"]
    elif change == "missing_parent":
        raw["nodes"][0]["parent"] = "unknown"
    elif change == "branch":
        raw["nodes"][2]["parent"] = raw["nodes"][0]["id"]
    else:
        raw["nodes"][4]["context"].append(raw["nodes"][0]["id"])
    with pytest.raises(Invalid):
        parse_world(raw)


@pytest.mark.parametrize("key,value", [
    ("budget", 0), ("workers", True), ("max_rounds", -1), ("retention_floor", 1.01),
    ("seeds", []), ("seeds", [17, 17]), ("search_seed", -1),
    ("paper_beta_work", -1), ("generations", 0), ("mutations", 0),
])
def test_plan_rejects_bad_values(key, value):
    with pytest.raises(Invalid):
        Plan.parse(changed(asdict(Plan()), key, value))


@pytest.mark.parametrize("key,value", [
    ("method", "exec"), ("width", 0), ("explore", 3), ("trend", float("inf")),
    ("repair", -1), ("patience", False), ("min_depth", 0),
])
def test_policy_dsl_rejects_bad_values(key, value):
    with pytest.raises(Invalid):
        Spec.parse(changed(asdict(Spec()), key, value))


@pytest.mark.parametrize("key,value", [
    ("retention", 0.49), ("parameter_delta", 1), ("latency_ratio", 1.26),
    ("peak_host_gib", 8.01), ("peak_vram_gib", 8.01),
    ("latency_ratio", None), ("parameter_delta", None), ("protocol_valid", False),
])
def test_gates_fail_closed(key, value):
    assert not gate(Outcome.parse(changed(outcome(), key, value)), Plan())


def test_gate_boundaries():
    value = Outcome.parse(outcome(0.4, 0.5))
    value = replace(value, latency_ratio=1.25, peak_host_gib=8, peak_vram_gib=8)
    assert gate(value, Plan())
    with pytest.raises(Invalid, match="baseline"):
        Replay(parse_world(make_world()), replace(Plan(), retention_floor=0.9))


def test_hypervolume_union_and_dominance():
    points = [(0.5, 0.8), (0.8, 0.5), (0.4, 0.4), (0.5, 0.8)]
    assert frontier(points) == ((0.5, 0.8), (0.8, 0.5))
    assert hypervolume(points) == pytest.approx(0.55)
    assert hypervolume([]) == 0
    assert hypervolume([(1, 1)]) == 1


def test_hypervolume_matches_grid_union():
    import random
    rng = random.Random(2)
    for _ in range(100):
        points = [(rng.randint(0, 10)/10, rng.randint(0, 10)/10) for _ in range(6)]
        exact = sum(any(x <= a and y <= b for a, b in points)
                    for x in [i/10 for i in range(1, 11)]
                    for y in [i/10 for i in range(1, 11)]) / 100
        assert hypervolume(points) == pytest.approx(exact)


def test_hidden_suffix_and_source_ids_do_not_leak():
    raw = make_world()
    alt = copy.deepcopy(raw)
    alt["id"] = "secret-win-999"
    alt["nodes"][-1]["outcome"] = outcome(0.01, 0.01)
    mapping = {n["id"]: "private-" + n["id"] for n in alt["nodes"]} | {"root": "root"}
    for node in alt["nodes"]:
        node["id"] = mapping[node["id"]]
        node["parent"] = mapping[node["parent"]]
        node["context"] = [mapping[c] for c in node["context"]]
    one, two = Replay(parse_world(raw), Plan()), Replay(parse_world(alt), Plan())
    assert one.view() == two.view()
    for _ in range(3):
        decision = Controller(Spec(method="fixed"), 17).choose(one.view())
        one.step(decision)
        two.step(decision)
        assert one.view() == two.view()
    assert "private-" not in json.dumps(asdict(two.view()))
    assert "secret-win" not in json.dumps(asdict(two.view()))


def test_future_cost_and_length_do_not_leak():
    raw = tiny_world()
    alt = copy.deepcopy(raw)
    alt["nodes"][0]["outcome"]["gpu_seconds"] = 999
    alt["nodes"].append({"id": "extra", "parent": "a", "order": 2, "context": ["root", "a"],
                         "outcome": outcome(1, 1)})
    assert Replay(parse_world(raw), Plan()).view() == Replay(parse_world(alt), Plan()).view()


def test_view_is_immutable_and_has_no_world_reference():
    view = Replay(parse_world(make_world()), Plan()).view()
    assert set(asdict(view)) == {"observations", "legal", "remaining", "workers", "rounds", "stagnation"}
    with pytest.raises(FrozenInstanceError):
        view.remaining = 100


@pytest.mark.parametrize("batch", [("root", "root"), ("hidden",), ("root", "hidden")])
def test_invalid_batch_is_atomic(batch):
    replay = Replay(parse_world(make_world()), Plan())
    before = replay.view()
    with pytest.raises(Invalid):
        replay.step(Decision(batch, "test"))
    assert replay.view() == before
    assert replay.spent == 0


def test_non_leaf_and_oversized_batch_rejected():
    replay = Replay(parse_world(make_world()), replace(Plan(), workers=2))
    replay.step(Decision(("root",), "start"))
    replay.step(Decision(("n00001",), "continue"))
    with pytest.raises(Invalid):
        replay.step(Decision(("n00001",), "nonleaf"))
    single = Replay(parse_world(make_world()), Plan())
    single.step(Decision(("root",), "start"))
    with pytest.raises(Invalid):
        single.step(Decision(("root", "n00001"), "oversized"))


def test_batch_is_synchronous_and_costs_are_accounted():
    replay = Replay(parse_world(make_world()), replace(Plan(), workers=2))
    replay.step(Decision(("root",), "start"))
    replay.step(Decision(("root", "n00001"), "parallel"))
    assert replay.spent == 3 and replay.rounds == 2
    assert len(replay.trace[-1]["prefix"]["observations"]) == 2
    assert len(replay.trace[-1]["revealed"]) == 2
    assert replay.gpu == 6 and replay.wall == 6


def test_support_misses_count_and_are_not_native_failures():
    replay = Replay(parse_world(tiny_world()), Plan())
    replay.step(Decision(("root",), "start"))
    replay.step(Decision(("n00001",), "no recorded continuation"))
    assert replay.spent == 2 and replay.misses == 1
    assert "n00001" not in replay.view().legal
    assert len(replay.observed) == 2
    assert not replay.gpu_complete
    result = replay.run(Controller(Spec(method="fixed"), 17))
    assert result["attempts"] == 3 and result["revealed"] == 1
    assert result["support_misses"] == 2
    assert result["gpu_seconds"] is None
    assert result["paper_score_diagnostic"] == pytest.approx(0.8 - 0.01)


def test_unknown_cost_stays_unknown():
    raw = tiny_world()
    raw["nodes"][0]["outcome"]["gpu_seconds"] = None
    result = evaluate(parse_world(raw), Spec(), replace(Plan(), budget=1), 17)
    assert result["gpu_seconds"] is None
    assert result["wall_seconds"] == 3


def test_batch_auc_uses_no_early_hindsight_and_carries_stop():
    class Once:
        def choose(self, view):
            return Decision(("root",) if view.rounds == 0 else (), "stop after one")
    result = Replay(parse_world(tiny_world()), replace(Plan(), budget=3)).run(Once())
    assert result["auc"] == pytest.approx((0.4 + 2 * 0.64) / 3)
    assert result["stop"] == "policy_stop"
    final = evaluate(parse_world(tiny_world()), Spec(), replace(Plan(), budget=1), 17)
    assert final["auc"] == pytest.approx(0.4)
    assert final["hypervolume"] == pytest.approx(0.64)


def test_round_budget_is_hard():
    result = evaluate(parse_world(make_world()), Spec(), replace(Plan(), max_rounds=1), 17)
    assert result["rounds"] == 1 and result["stop"] == "round_budget"


@pytest.mark.parametrize("method", ["fixed", "greedy", "random", "adaptive"])
def test_deterministic_reset_and_monotone_frontier(method):
    world = parse_world(make_world())
    one = evaluate(world, Spec(method=method), Plan(), 17)
    assert one == evaluate(world, Spec(method=method), Plan(), 17)
    assert one["attempts"] <= Plan().budget
    values = [p["hypervolume"] for p in one["curve"]]
    assert values == sorted(values)
    assert len(world.nodes) == 16


def test_fixed_schedule_is_outcome_blind():
    raw = make_world()
    other = copy.deepcopy(raw)
    for node in other["nodes"]:
        node["outcome"] = outcome(0.99, 0.99)
    def actions(world):
        return [r["decision"]["batch"] for r in evaluate(parse_world(world), Spec(method="fixed"), Plan(), 1)["trace"]]
    assert actions(raw) == actions(other)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/drive", "a\\b"])
def test_unsafe_paths_rejected(tmp_path, path):
    with pytest.raises(Invalid):
        safe_path(tmp_path, path)


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "outside")
    outside.mkdir()
    try:
        (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not supported on this host")
    with pytest.raises(Invalid):
        safe_path(tmp_path, "link/x")


def test_strict_json_and_exclusive_writes(tmp_path):
    file = tmp_path / "test.json"
    for content in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
        file.write_text(content)
        with pytest.raises(Invalid):
            read_json(file)
    with pytest.raises(FileExistsError):
        write_new(file, {})


def test_json_pointer_escaping_and_rejection():
    assert pointer({"a/b": {"~c": [4]}}, "/a~1b/~0c/0") == 4
    for location in ("bad", "/missing", "/a~9b", "/x/-1", "/x/01"):
        with pytest.raises(Invalid):
            pointer({"x": [1]}, location)


def test_catalog_split_leakage_and_duplicates(tmp_path):
    path = create_bundle(tmp_path / "bundle")
    raw = read_json(path)
    for mutation in ("shared", "duplicate", "missing_split", "same_hash"):
        altered = copy.deepcopy(raw)
        if mutation == "shared":
            altered["worlds"][-1]["leakage_keys"] = altered["worlds"][0]["leakage_keys"]
        elif mutation == "duplicate":
            altered["worlds"][-1]["id"] = altered["worlds"][0]["id"]
        elif mutation == "missing_split":
            altered["worlds"] = [w for w in altered["worlds"] if w["split"] != "test"]
        else:
            altered["worlds"][-1]["sha256"] = altered["worlds"][0]["sha256"]
        with pytest.raises(Invalid):
            validate_catalog(altered)


def test_grouping_is_transitive():
    entries = [{"id": "a", "leakage_keys": ["x"]},
               {"id": "b", "leakage_keys": ["x", "y"]},
               {"id": "c", "leakage_keys": ["y"]},
               {"id": "d", "leakage_keys": ["z"]}]
    grouped = study.groups(entries)
    assert grouped["a"] == grouped["b"] == grouped["c"] != grouped["d"]


def test_world_hash_and_metadata_checked(tmp_path):
    path = create_bundle(tmp_path / "bundle")
    cat = read_json(path)
    target = path.parent / cat["worlds"][0]["path"]
    raw = read_json(target)
    raw["nodes"][0]["outcome"]["wall_seconds"] += 1
    target.write_bytes(canonical(raw))
    with pytest.raises(Invalid, match="commitment"):
        load_partition(path.parent, cat, "train")
    cat["worlds"][0]["sha256"] = digest(raw)
    cat["worlds"][0]["id"] = "wrong"
    with pytest.raises(Invalid, match="metadata"):
        load_partition(path.parent, cat, "train")


def historical_shaped_fixture(tmp_path):
    # Fabricated receipts only to exercise byte/field binding, NEVER real evidence.
    raw = tiny_world()
    raw["kind"] = "historical"
    source = {"identity": raw["identity"], "baseline": raw["baseline"], "nodes": raw["nodes"]}
    data = canonical(source)
    (tmp_path / "receipt.json").write_bytes(data)
    bindings = [{"target": "/identity/"+k, "source": 0, "pointer": "/identity/"+k} for k in raw["identity"]]
    for key in raw["baseline"]:
        bindings.append({"target": "/baseline/"+key, "source": 0, "pointer": "/baseline/"+key})
    for key in raw["nodes"][0]:
        names = [f"outcome/{k}" for k in raw["nodes"][0]["outcome"]] if key == "outcome" else [key]
        for name in names:
            loc = "/nodes/0/" + name
            bindings.append({"target": loc, "source": 0, "pointer": loc})
    raw["admission"] = {"curator": "TEST FIXTURE NOT REAL", "review_id": "fixture", "context_review": True,
                        "complete_failure_log": True,
                        "sources": [{"path": "receipt.json", "sha256": hashlib.sha256(data).hexdigest(),
                                     "git_blob": hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest(),
                                     "git_commit": "a"*40, "repository_path": "fixtures/not-real.json"}],
                        "bindings": bindings}
    return raw


def test_historical_source_field_binding(tmp_path):
    raw = historical_shaped_fixture(tmp_path)
    assert parse_world(raw, tmp_path).kind == "historical"
    altered = copy.deepcopy(raw)
    altered["nodes"][0]["outcome"]["capability"] = 0.9
    with pytest.raises(Invalid, match="receipt field"):
        parse_world(altered, tmp_path)


@pytest.mark.parametrize("change", ["source_hash", "blob", "binding", "review", "empty", "no_root"])
def test_historical_admission_fails_closed(tmp_path, change):
    raw = historical_shaped_fixture(tmp_path)
    root = tmp_path
    if change == "source_hash":
        raw["admission"]["sources"][0]["sha256"] = "0"*64
    elif change == "blob":
        raw["admission"]["sources"][0]["git_blob"] = "0"*40
    elif change == "binding":
        raw["admission"]["bindings"].pop()
    elif change == "review":
        raw["admission"]["context_review"] = False
    elif change == "empty":
        raw["admission"]["sources"] = []
    else:
        root = None
    with pytest.raises(Invalid):
        parse_world(raw, root)


def test_fit_never_reads_test_and_incumbent_is_retained(tmp_path, monkeypatch):
    cat_path = create_bundle(tmp_path / "bundle")
    calls = []
    original = schema.read_json
    def audited(path):
        calls.append(path.name)
        assert not path.name.startswith("synthetic-test"), "test world opened during fit"
        return original(path)
    monkeypatch.setattr(schema, "read_json", audited)
    run = tmp_path / "run"
    study.fit(cat_path, Plan(), run)
    assert calls and all(not name.startswith("synthetic-test") for name in calls)
    search = read_json(run / "search.json")
    for generation in search["trajectory"]:
        assert generation["previous"] in generation["candidates"]
        assert tuple(generation["selected_rank"]) >= tuple(generation["previous_rank"])
    candidates = search["candidates"]
    selected = read_json(run / "freeze.json")["payload"]["selected_id"]
    assert next(c for c in candidates if c["id"] == selected)["validation_rank"] == max(c["validation_rank"] for c in candidates)


def test_freeze_reproducible_and_holdout_one_shot(tmp_path):
    cat = create_bundle(tmp_path / "bundle")
    first = study.fit(cat, Plan(), tmp_path / "run1")
    second = study.fit(cat, Plan(), tmp_path / "run2")
    assert first == second
    report = study.evaluate_frozen(cat, tmp_path / "run1")
    assert report["episodes"] == 60 and report["kind"] == "synthetic"
    assert report["native_model_loads"] == report["llm_calls"] == 0
    assert not report["live_efficiency_improvement_established"]
    with pytest.raises(FileExistsError):
        study.evaluate_frozen(cat, tmp_path / "run1")
    with pytest.raises(FileExistsError):
        study.fit(cat, Plan(), tmp_path / "run2")


def test_freeze_and_source_tampering_rejected(tmp_path, monkeypatch):
    cat = create_bundle(tmp_path / "bundle")
    run = tmp_path / "run"
    study.fit(cat, Plan(), run)
    monkeypatch.setattr(study, "engine_hash", lambda: "0"*64)
    with pytest.raises(Invalid, match="engine"):
        study.evaluate_frozen(cat, run)
    assert not (run / "holdout_opened.json").exists()
    envelope = read_json(run / "freeze.json")
    envelope["payload"]["selected_spec"]["explore"] = 1.999
    (run / "freeze.json").write_bytes(canonical(envelope))
    with pytest.raises(Invalid, match="freeze content"):
        study.evaluate_frozen(cat, run)


def test_failed_holdout_read_still_consumes_marker(tmp_path):
    cat = create_bundle(tmp_path / "bundle")
    run = tmp_path / "run"
    study.fit(cat, Plan(), run)
    entry = next(e for e in read_json(cat)["worlds"] if e["split"] == "test")
    (cat.parent / entry["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        study.evaluate_frozen(cat, run)
    assert (run / "holdout_opened.json").exists()
    with pytest.raises(FileExistsError):
        study.evaluate_frozen(cat, run)


def test_too_few_declared_groups_block_fit(tmp_path):
    cat = create_bundle(tmp_path / "bundle")
    raw = read_json(cat)
    raw["worlds"] = [e for e in raw["worlds"] if e["id"] != "synthetic-test-4"]
    cat.write_bytes(canonical(raw))
    with pytest.raises(Invalid, match="5 independent"):
        study.fit(cat, Plan(), tmp_path / "run")


def test_cluster_average_does_not_overweight_seed_count():
    rows = [{"world_id": "a", "auc": 1}, {"world_id": "a", "auc": 1},
            {"world_id": "b", "auc": 0}, {"world_id": "c", "auc": 0}]
    assert study.cluster_mean(rows, {"a": "x", "b": "x", "c": "y"}, "auc") == pytest.approx(0.25)


def test_small_cluster_bootstrap_is_suppressed():
    rows = [{"world_id": "a", "seed": 17, "arm": arm, "auc": value}
            for arm, value in (("adaptive", 0.7), ("fixed", 0.5))]
    result = study.paired_comparison(rows, {"a": "x"}, Plan())
    assert result["mean_delta"] == pytest.approx(0.2)
    assert result["bootstrap_95_percentile"] is None


def test_cli_native_blocked_and_history_not_admitted(capsys):
    assert main(["native"]) == 2
    assert "BLOCKED" in capsys.readouterr().err
    assert main(["history-status"]) == 0
    assert '"admitted_historical_worlds": 0' in capsys.readouterr().out


@pytest.mark.parametrize("key", ["checkpoint_sha256", "protocol_sha256", "dataset_sha256", "study_family", "task_family", "lineage_family"])
def test_identity_is_required_and_cannot_hide_leakage(key):
    raw = make_world()
    raw["identity"].pop(key)
    with pytest.raises(Invalid):
        parse_world(raw)
    raw = make_world()
    raw["leakage_keys"] = ["arbitrary-key-only"]
    with pytest.raises(Invalid, match="identity leakage"):
        parse_world(raw)


@pytest.mark.parametrize("overrides", [{"workers": 65}, {"generations": 100, "mutations": 100}, {"seeds": list(range(33))}])
def test_resource_limits_on_offline_search(overrides):
    with pytest.raises(Invalid):
        Plan.parse(asdict(Plan()) | overrides)
