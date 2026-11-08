"""Actual rendered-prompt checks; these fixtures are not native evidence."""
from collections import Counter, defaultdict
from copy import deepcopy
import inspect
import json
import random

import pytest

from titan_joint_quality_v1.common import DOMAIN_ROLE, TRAIN_COUNTS, digest
from titan_joint_quality_v1 import data


@pytest.fixture(scope="module")
def banks():
    return {split: data.generate_bank(split) for split in ("train", "pilot")}


@pytest.mark.parametrize("task,expected", [
    ({"kind": "program", "initial": {"x": 3, "y": 4}, "instructions": [
        {"dst": "x", "op": "add_from", "arg": "y"}, {"dst": "x", "op": "multiply", "arg": 2},
        {"dst": "x", "op": "if_even_add", "arg": 3}], "query": "x"}, "17"),
    ({"kind": "dependencies", "leaves": {"a": 2, "b": 7}, "expressions": {
        "d": ["c", "-", "a"], "c": ["a", "*", "b"]}, "query": "d"}, "12"),
    ({"kind": "item_events", "initial": {"key": "north", "other": "west"}, "events": [
        {"item": "key", "bin": "east"}, {"item": "other", "bin": "north"}], "query": "key"}, "east"),
    ({"kind": "fsm", "initial": "s0", "transitions": {"s0": {"a": "s1"}, "s1": {"a": "s0"}},
      "symbols": ["a", "a", "a"], "query": {"after_symbols": 2}}, "s0"),
    ({"kind": "baskets", "baskets": {"left": [{"quantity": 3, "price": 7}, {"quantity": 2, "price": 5}],
      "right": [{"quantity": 8, "price": 9}]}, "query": "left"}, "31"),
    ({"kind": "sharing", "groups": {"alpha": {"count": 43, "people": 6}},
      "query": {"group": "alpha", "component": "remainder"}}, "1"),
    ({"kind": "authority", "authoritative_source": "registry", "facts": [
        {"entity": "bird", "attribute": "color", "source": "rumor", "value": "blue"},
        {"entity": "bird", "attribute": "color", "source": "registry", "value": "red"}],
      "query": {"entity": "bird", "attribute": "color"}}, "red"),
    ({"kind": "horn", "facts": ["root", "badge"], "rules": [
        {"premises": ["root"], "conclusion": "ready"}, {"premises": ["ready", "badge"], "conclusion": "allowed"}],
      "query": "allowed"}, "SUPPORTED"),
    ({"kind": "latest_record", "records": [
        {"entity": "ring", "timestamp": 8, "value": "new"}, {"entity": "ring", "timestamp": 1, "value": "old"},
        {"entity": "wrong", "timestamp": 9, "value": "unrelated"}], "query": {"entity": "ring", "field": "value"}}, "new"),
    ({"kind": "alias_join", "aliases": {"sun": "k7", "moon": "k8"}, "records": [
        {"code": "k8", "city": "north"}, {"code": "k7", "city": "west"}], "query": {"alias": "sun", "field": "city"}}, "west"),
    ({"kind": "field_transform", "fields": {"first": "Abc", "second": "Def", "noise": "ghi"},
      "query": {"keys": ["second", "first"], "case": "upper", "separator": "|"}}, "DEF|ABC"),
    ({"kind": "filtered_roster", "records": [{"label": "zoe", "tier": "A", "score": 8},
      {"label": "amy", "tier": "A", "score": 7}, {"label": "bob", "tier": "B", "score": 9}],
      "query": {"tier": "A", "minimum": 7, "order": "asc"}}, "amy,zoe"),
])
def test_teachers_solve_actual_rendered_tasks(task, expected):
    messages = data.render_messages(task, "json")
    assert data.solve_prompt_a(messages) == expected
    assert data.solve_prompt_b(messages) == expected
    assert data.format_target(expected, "json") == json.dumps({"answer": expected}, separators=(",", ":"), sort_keys=True)


def test_corpus_counts_roles_teachers_pairs_and_input_gold_separation(banks):
    for split, rows in banks.items():
        expected = TRAIN_COUNTS if split == "train" else {domain: 64 for domain in DOMAIN_ROLE}
        assert Counter(r["domain"] for r in rows) == Counter(expected)
        assert len(rows) == (2000 if split == "train" else 384)
        parents = defaultdict(list)
        for row in rows:
            assert data.solve_prompt_a(row["messages"]) == row["canonical_answer"] == data.solve_prompt_b(row["messages"])
            assert row["target"] == data.format_target(row["canonical_answer"], row["requested_format"])
            text = json.dumps(row["messages"])
            for field in ("case_id", "parent_id", "family_id"):
                assert row[field] not in text
            assert "canonical_answer" not in text and "solver_trace" not in text
            parents[row["parent_id"]].append(row)
        assert len(parents) == len(rows) // 2
        for pair in parents.values():
            assert len(pair) == 2 and {r["variant_id"] for r in pair} == {0, 1}
            assert len({r["requested_format"] for r in pair}) == 1
            assert (pair[0]["canonical_answer"] != pair[1]["canonical_answer"]) == pair[0]["pair_kind"].startswith("changed")
    assert Counter(r["role"] for r in banks["train"]) == Counter(reasoning=1000, preservation=500, grounding=300, format=200)


def test_train_pilot_use_different_task_algorithms_and_no_content_duplicates(banks):
    audit = data.partition_audit(banks["train"], banks["pilot"])
    assert all(audit[key] == 0 for key in ("case_id_overlap_count", "parent_id_overlap_count", "family_id_overlap_count", "rendered_content_overlap_count"))
    for domain in DOMAIN_ROLE:
        kinds = {split: {data.parse_prompt(r["messages"])["kind"] for r in banks[split] if r["domain"] == domain}
                 for split in banks}
        assert kinds["train"].isdisjoint(kinds["pilot"])
    for rows in banks.values():
        assert len({digest(row["messages"]) for row in rows}) == len(rows)


def test_pilot_long_input_padding_is_legitimate_irrelevant_and_pair_matched(banks):
    rows = [row for row in banks["pilot"] if row["domain"] == "long_context"]
    parents = defaultdict(list)
    query_positions = []
    for row in rows:
        task = data.parse_prompt(row["messages"])
        index = int(row["parent_id"].rsplit(".", 1)[1])
        assert len(task["records"]) == data.PILOT_JOIN_RECORD_COUNTS[index]
        assert len(task["records"]) in range(50, 113, 2)
        assert len({record["code"] for record in task["records"]}) == len(task["records"])
        assert set(task["aliases"].values()) <= {record["code"] for record in task["records"]}
        selected_code = task["aliases"][task["query"]["alias"]]
        position = next(i for i, record in enumerate(task["records"]) if record["code"] == selected_code)
        query_positions.append(position / len(task["records"]))
        minimal = {**task, "aliases": {task["query"]["alias"]: selected_code},
                   "records": [task["records"][position]]}
        assert data.solve_prompt_a(data.render_messages(minimal)) == row["canonical_answer"]
        assert data.solve_prompt_b(data.render_messages(minimal)) == row["canonical_answer"]
        parents[row["parent_id"]].append(row)
    assert {len(data.parse_prompt(row["messages"])["records"]) for row in rows} == set(data.PILOT_JOIN_RECORD_COUNTS)
    # Answer-bearing records are not always at the short leading part of input.
    assert min(query_positions) < .25 and max(query_positions) > .75
    for pair in parents.values():
        tasks = [data.parse_prompt(row["messages"]) for row in sorted(pair, key=lambda row: row["variant_id"])]
        assert tasks[0]["query"] != tasks[1]["query"]
        assert {key: value for key, value in tasks[0].items() if key != "query"} == {key: value for key, value in tasks[1].items() if key != "query"}
        assert [record["code"] for record in tasks[0]["records"]] == [record["code"] for record in tasks[1]["records"]]
    # The bounded TRAIN retrieval generator was not expanded by this repair.
    train_tasks = [data.parse_prompt(row["messages"]) for row in banks["train"] if row["domain"] == "long_context"]
    assert {task["kind"] for task in train_tasks} == {"latest_record"}
    assert {len(task["records"]) for task in train_tasks} == {24}
    for task in train_tasks:
        entities = {record["entity"] for record in task["records"]}
        assert len(entities) == 6
        for entity in entities:
            times = [record["timestamp"] for record in task["records"] if record["entity"] == entity]
            assert len(times) == len(set(times)) == 4


def test_support_removal_requires_unknown_despite_visible_distractors(banks):
    # Support intervention remains a mechanical test, separate from bank pairs.
    authority = {"kind": "authority", "authoritative_source": "charter", "facts": [
        {"entity": "bird", "attribute": "color", "value": "red", "source": "charter"},
        {"entity": "other", "attribute": "color", "value": "blue", "source": "charter"},
        {"entity": "bird", "attribute": "color", "value": "blue", "source": "rumor"}],
        "query": {"entity": "bird", "attribute": "color"}}
    horn = {"kind": "horn", "facts": ["root", "neutral"], "rules": [
        {"premises": ["root"], "conclusion": "middle"}, {"premises": ["middle"], "conclusion": "goal"}], "query": "goal"}
    for task in (authority, horn):
        assert data.solve_prompt_a(data.render_messages(task)) != "UNKNOWN"
        removed = deepcopy(task)
        if task["kind"] == "authority": removed["facts"] = removed["facts"][1:]
        else: removed["facts"][0] = "unrelated"
        assert removed["facts"]
        assert data.solve_prompt_a(data.render_messages(removed)) == data.solve_prompt_b(data.render_messages(removed)) == "UNKNOWN"


def test_v2_program_repairs_full_queryblind_dst_leak_without_first_last_or_lex_replacement(banks):
    rows = [row for row in banks["train"] if row["domain"] == "reasoning"]
    parents = defaultdict(list)
    for row in rows:
        task = data.parse_prompt(row["messages"])
        counts = Counter(instruction["dst"] for instruction in task["instructions"])
        assert counts["x"] == counts["y"] == 5
        assert counts["z"] in (0, 1)
        assert row["pair_kind"] == "changed_answer_query_switch"
        parents[row["parent_id"]].append(row)
    assert len(parents) == 500
    for pair in parents.values():
        tasks = [data.parse_prompt(row["messages"]) for row in pair]
        assert {task["query"] for task in tasks} == {"x", "y"}
        assert {key: value for key, value in tasks[0].items() if key != "query"} == {key: value for key, value in tasks[1].items() if key != "query"}
        assert pair[0]["canonical_answer"] != pair[1]["canonical_answer"]
        assert pair[0]["requested_format"] == pair[1]["requested_format"]
    audit = data.counterdirection_audit(rows)["program"]
    assert audit["query_counts"] == {"x": 500, "y": 500}
    assert audit["unique_most_written_dst_rows"] == 0
    assert audit["same_world_different_answer_pairs"] == 500
    assert audit["query_blind_pair_accuracy_upper_bound"] == .5
    assert audit["shortcut_correct_counts"]["most_written_lex_tie"] == 500
    assert audit["shortcut_correct_counts"]["lex_first_register"] == 500
    assert all(correct <= 500 for correct in audit["shortcut_correct_counts"].values())


def test_v2_horn_repairs_full_cardinality_leak_and_crosses_both_answer_directions(banks):
    rows = [row for row in banks["pilot"] if row["domain"] == "grounding"]
    parents = defaultdict(list)
    for row in rows:
        task = data.parse_prompt(row["messages"])
        assert len(task["facts"]) == 2 and len(task["rules"]) == 4
        assert len(task["query"]) == 7 and task["query"].startswith("p")
        assert data.solve_prompt_a(row["messages"]) == data.solve_prompt_b(row["messages"]) == row["canonical_answer"]
        parents[row["parent_id"]].append(row)
    query_pairs = 0
    directions = Counter()
    for pair in parents.values():
        pair = sorted(pair, key=lambda row: row["variant_id"])
        tasks = [data.parse_prompt(row["messages"]) for row in pair]
        assert {row["canonical_answer"] for row in pair} == {"SUPPORTED", "UNKNOWN"}
        directions[(pair[0]["pair_kind"], pair[0]["canonical_answer"])] += 1
        assert pair[0]["pair_kind"] == "changed_answer_query_switch"
        assert {key: value for key, value in tasks[0].items() if key != "query"} == {key: value for key, value in tasks[1].items() if key != "query"}
        assert tasks[0]["query"] != tasks[1]["query"]
        query_pairs += 1
    assert query_pairs == 32
    assert set(directions.values()) == {16}
    audit = data.counterdirection_audit(rows)["horn"]
    assert audit["fact_counts"] == {2: 64}
    assert audit["shortcut_correct_counts"]["fact_count_two"] == 32
    assert audit["same_query_support_swap_pairs"] == 0
    assert 0 <= audit["shortcut_correct_counts"]["question_name_hash_parity"] <= 64


def test_every_domain_has_full_same_world_query_crossovers_and_structural_controls_cannot_exceed_half(banks):
    for split, rows in banks.items():
        audit = data.counterdirection_audit(rows)["all_domains"]
        for domain, result in audit.items():
            assert result["certified_same_world_query_pairs"] == result["parents"] == result["rows"] // 2
            assert result["deterministic_queryblind_pair_accuracy_upper_bound"] == .5
            assert all(correct <= result["rows"] // 2 for correct in result["structural_control_correct_counts"].values())
        parents = defaultdict(list)
        for row in rows: parents[row["parent_id"]].append(row)
        for pair in parents.values():
            assert {row["pair_kind"] for row in pair} == {"changed_answer_query_switch"}
            tasks = [data.parse_prompt(row["messages"]) for row in pair]
            assert {key: value for key, value in tasks[0].items() if key != "query"} == {key: value for key, value in tasks[1].items() if key != "query"}
            assert tasks[0]["query"] != tasks[1]["query"]
            assert pair[0]["canonical_answer"] != pair[1]["canonical_answer"]
            for task in tasks:
                world = {key: value for key, value in task.items() if key != "query"}
                assert task["query"] in data.enumerate_valid_queries(world)


def test_finite_public_queries_are_independently_verified_and_generation_retries_are_logged(banks):
    kinds = {}
    for rows in banks.values():
        for row in rows:
            task = data.parse_prompt(row["messages"])
            kinds.setdefault(task["kind"], {key: value for key, value in task.items() if key != "query"})
    assert len(kinds) == 12
    for world in kinds.values():
        queries = data.enumerate_valid_queries(world)
        assert len({digest(query) for query in queries}) == len(queries) >= 2
        truths = set()
        for query in queries:
            messages = data.render_messages({**world, "query": query})
            assert data.solve_prompt_a(messages) == data.solve_prompt_b(messages)
            truths.add(data.solve_prompt_a(messages))
        assert len(truths) >= 2
    for rows in banks.values():
        audit = data.generation_audit(rows)
        assert not audit["eligible_only_subsetting"]
        for result in audit["domains"].values():
            assert result["world_attempts"] == result["parents"] + result["degenerate_worlds"] + result["exact_content_rejections"]


def test_degenerate_world_is_prospectively_regenerated_with_receipt_not_silent_filtering(monkeypatch):
    calls = []
    def world(split, domain, rng, index):
        calls.append(index)
        return {"kind": "baskets", "baskets": {"cedar": [{"quantity": 1, "price": 2}],
            "maple": [{"quantity": 1, "price": 2 if len(calls) == 1 else 3}]}}
    monkeypatch.setattr(data, "_generate_world", world)
    first, second, kind, audit = data._pair_tasks("train", "arithmetic", random.Random(17), 0)
    assert audit["world_attempts"] == 2 and audit["degenerate_worlds"] == 1
    assert audit["valid_query_count"] == audit["distinct_truth_count"] == 2
    assert kind == "changed_answer_query_switch"
    assert data.solve_prompt_a(data.render_messages(first)) != data.solve_prompt_a(data.render_messages(second))


def test_visible_wrapper_request_binding_refuses_poisoned_instruction_and_metadata(banks):
    rows = deepcopy(banks["pilot"])
    index = next(i for i, row in enumerate(rows) if row["requested_format"] == "json")
    row = rows[index]
    task = data.parse_prompt(row["messages"])
    row["messages"] = data.render_messages(task, "plain")
    assert data.solve_prompt_a(row["messages"]) == row["canonical_answer"]
    with pytest.raises(ValueError, match="visible wrapper"):
        data.validate_bank(rows, "pilot")
    rows = deepcopy(banks["pilot"])
    row = rows[index]
    row["requested_format"] = "plain"
    row["target"] = data.format_target(row["canonical_answer"], "plain")
    with pytest.raises(ValueError, match="visible wrapper"):
        data.validate_bank(rows, "pilot")
    for form in data.FORMATS:
        assert data.requested_format_from_messages(data.render_messages(task, form)) == form


@pytest.mark.parametrize("kind,task", [
    ("authority", {"authoritative_source": "registry", "facts": [
        {"entity": "bird", "attribute": "color", "source": "rumor", "value": "blue"}],
        "query": {"entity": "bird", "attribute": "color"}}),
    ("horn", {"facts": ["irrelevant"], "rules": [{"premises": ["missing"], "conclusion": "goal"}], "query": "goal"}),
    ("latest_record", {"records": [{"entity": "other", "timestamp": 3, "value": "blue"}], "query": {"entity": "missing", "field": "value"}}),
    ("alias_join", {"aliases": {"sun": "k7"}, "records": [{"code": "k8", "city": "north"}], "query": {"alias": "sun", "field": "city"}}),
])
def test_missing_support_cannot_be_filled_from_non_authoritative_or_unrelated_facts(kind, task):
    messages = data.render_messages({"kind": kind, **task})
    assert data.solve_prompt_a(messages) == data.solve_prompt_b(messages) == "UNKNOWN"


def test_teacher_b_does_not_delegate_evaluation_to_teacher_a(monkeypatch, banks):
    assert "solve_prompt_a" not in inspect.getsource(data.solve_prompt_b)
    monkeypatch.setattr(data, "solve_prompt_a", lambda messages: (_ for _ in ()).throw(AssertionError("teacher A unavailable")))
    for rows in banks.values():
        samples = {data.parse_prompt(row["messages"])["kind"]: row for row in rows}
        for row in samples.values():
            assert data.solve_prompt_b(row["messages"]) == row["canonical_answer"]


def test_teachers_reject_hidden_row_and_ignore_poisoned_reference_values(banks):
    row = deepcopy(banks["train"][0])
    expected = data.solve_prompt_a(row["messages"])
    row["canonical_answer"] = "POISONED_GOLD"
    row["target"] = "POISONED_TARGET"
    row["solver_trace"] = {"teacher_a": "POISONED_TRACE"}
    assert data.solve_prompt_a(row["messages"]) == expected == data.solve_prompt_b(row["messages"])
    with pytest.raises(TypeError): data.solve_prompt_a(row)
    with pytest.raises(TypeError): data.solve_prompt_b(row)


def test_format_counterbalance_and_seed_reproducibility(banks):
    assert data.generate_bank("pilot", 17) == banks["pilot"]
    assert data.generate_bank("pilot", 23) != banks["pilot"]
    for rows in banks.values():
        for domain in DOMAIN_ROLE:
            counts = Counter(r["requested_format"] for r in rows if r["domain"] == domain)
            assert set(counts) == set(data.FORMATS)
            assert max(counts.values()) - min(counts.values()) <= 2


def test_confirmation_refuses_before_generation_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_pair_tasks", lambda *args: (_ for _ in ()).throw(AssertionError("materialized a closed example")))
    with pytest.raises(PermissionError, match="confirmation"):
        data.generate_bank("confirmation")
    with pytest.raises(PermissionError, match="confirmation"):
        data.validate_bank([], "confirmation")
    assert list(tmp_path.iterdir()) == []


def test_exclusive_artifacts_bind_projection_and_do_not_claim_native_qualification(tmp_path):
    out = tmp_path / "compiled"
    manifest = data.build_dataset(out)
    assert {p.name for p in out.iterdir()} == {"train.jsonl", "pilot.prompts.jsonl", "pilot.references.jsonl", "DATA_MANIFEST.json"}
    prompts = [json.loads(line) for line in (out / "pilot.prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    refs = [json.loads(line) for line in (out / "pilot.references.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(prompts) == len(refs) == 384
    assert all(set(row) == {"case_id", "messages"} for row in prompts)
    assert all("messages" not in row for row in refs)
    assert {r["case_id"] for r in prompts} == {r["case_id"] for r in refs}
    assert manifest["train"]["rows"] == 2000 and manifest["pilot"]["parents"] == 192
    assert manifest["teacher"]["model_teacher_calls"] == manifest["native_model_calls"] == 0
    assert not manifest["native_authorized"] and not manifest["training_authorized"]
    assert not manifest["confirmation_materialized"] and not manifest["frozen"]
    assert manifest["overlap_review"] == "UNREVIEWED" and not manifest["native_token_caps_verified"]
    assert len(manifest["teacher"]["source_digest"]) == 64
    for name, sha in manifest["artifact_sha256"].items():
        assert __import__("hashlib").sha256((out / name).read_bytes()).hexdigest() == sha
    with pytest.raises(FileExistsError): data.build_dataset(out)


def test_floor_audit_is_explicit_and_gold_does_not_enter_query_blind_prediction(banks):
    audit = data.floor_audit(banks["pilot"])
    assert audit["status"] == "UNREVIEWED_NO_ACCEPTANCE_GATE"
    assert "NOT_NATIVE" in audit["evidence_kind"]
    assert set(audit["domains"]) == set(DOMAIN_ROLE)
    for domain, record in audit["domains"].items():
        assert record["rows"] == 64
        for field in ("constant_accuracy", "copy_accuracy", "query_blind_accuracy"):
            assert 0 <= record[field] <= 1
