"""Bounded public-rule source-sensitivity mechanics; no native or corpus writer."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import pytest

from titan_joint_quality_v1 import data
from titan_joint_quality_v1.common import safe_output


def public(task, source):
    return data.render_messages({**task, "query": {"entity": "parcel", "attribute": "zone", "authoritative_source": source}})


def test_explicit_selected_source_rule_missing_conflict_and_duplicate_equivalence():
    task = {"kind": "authority_query_source_v3", "sources": ["charter", "rumor"], "facts": [
        {"entity": "parcel", "attribute": "zone", "source": "charter", "value": "vamber"},
        {"entity": "parcel", "attribute": "zone", "source": "rumor", "value": "vblue"},
        {"entity": "parcel", "attribute": "zone", "source": "rumor", "value": "vgreen"},
        {"entity": "other", "attribute": "zone", "source": "charter", "value": "vblack"}]}
    for teacher in (data.solve_prompt_a, data.solve_prompt_b):
        assert teacher(public(task, "charter")) == "vamber"
        assert teacher(public(task, "rumor")) == "UNKNOWN"
        assert teacher(public(task, "missing_source")) == "UNKNOWN"
        duplicate = deepcopy(task)
        duplicate["facts"][2]["value"] = "vblue"
        assert teacher(public(duplicate, "rumor")) == "vblue"
        assert teacher(public({**task, "facts": list(reversed(task["facts"]))}, "rumor")) == "UNKNOWN"
    assert "query.authoritative_source" in public(task, "charter")[0]["content"]
    assert "two or more distinct" in public(task, "charter")[0]["content"]


def test_sixteen_proposed_parent_shapes_cross_sources_and_defeat_value_prefix_control():
    base = {"kind": "authority", "authoritative_source": "charter", "facts": [
        {"entity": entity, "attribute": attribute, "source": source, "value": "oldvalue"}
        for entity in ("parcel", "carton") for attribute in ("zone", "color") for source in ("charter", "rumor")]}
    known, unknown, kinds = Counter(), Counter(), Counter()
    prefix_correct = 0
    for index in range(16):
        world, first, second, trace = data._repair_source_world(base, 17, index)
        kinds[trace["unknown_kind"]] += 1
        messages = [data.render_messages({**world, "query": query}, "json") for query in (first, second)]
        answers = [data.solve_prompt_a(message) for message in messages]
        assert answers == [data.solve_prompt_b(message) for message in messages]
        assert (first["entity"], first["attribute"]) == (second["entity"], second["attribute"])
        assert first["authoritative_source"] != second["authoritative_source"]
        assert len(set(answers)) == 2 and "UNKNOWN" in answers
        assert data.source_blind_projection(messages[0]) == data.source_blind_projection(messages[1])
        assert all(fact["value"].startswith("v") and len(fact["value"]) == 7 for fact in world["facts"])
        for query, answer in zip((first, second), answers):
            (unknown if answer == "UNKNOWN" else known)[query["authoritative_source"]] += 1
            # Reads entity/attribute/value morphology but neither source field.
            values = {fact["value"] for fact in world["facts"] if fact["entity"] == query["entity"] and fact["attribute"] == query["attribute"] and fact["value"].startswith("v")}
            prediction = next(iter(values)) if len(values) == 1 else "UNKNOWN"
            prefix_correct += prediction == answer
    assert kinds == {"missing": 8, "conflict": 8}
    assert known == unknown == {"charter": 8, "rumor": 8}
    assert prefix_correct == 16


def test_source_label_and_value_renaming_are_equivariant_not_lexical_trust():
    task = {"kind": "authority_query_source_v3", "sources": ["charter", "rumor"], "facts": [
        {"entity": "parcel", "attribute": "zone", "source": "charter", "value": "vamber"},
        {"entity": "parcel", "attribute": "zone", "source": "rumor", "value": "vblue"},
        {"entity": "parcel", "attribute": "zone", "source": "rumor", "value": "vgreen"}]}
    labels = {"charter": "archive", "rumor": "registry"}
    values = {"vamber": "rgreen", "vblue": "ramber", "vgreen": "rblue"}
    renamed = {"kind": task["kind"], "sources": list(labels.values()), "facts": [
        {**fact, "source": labels[fact["source"]], "value": values[fact["value"]]} for fact in task["facts"]]}
    for teacher in (data.solve_prompt_a, data.solve_prompt_b):
        assert teacher(public(renamed, "archive")) == "rgreen"
        assert teacher(public(renamed, "registry")) == "UNKNOWN"


def test_seed17_epoch_preserves_all_other_public_inputs_targets_and_pilot():
    old_train, old_pilot = data.load_bound_v2_epoch()
    transformed = data.transform_grounding_rows(old_train, seed=17)
    assert len(transformed) == 2000
    for old, new in zip(old_train, transformed):
        assert old["case_id"] == new["case_id"]
        if old["domain"] != "grounding": assert old == new
        else:
            assert new["family_id"] == "train_query_selected_authority_v3"
            assert data.parse_prompt(new["messages"])["kind"] == "authority_query_source_v3"
    assert data.generate_bank("pilot") == old_pilot
    audit = data.counterdirection_audit(transformed)["selected_authority"]
    assert audit["sourceblind_certified_pairs"] == 150
    assert audit["old_value_prefix_control_correct"] == 150
    assert audit["known_source_counts"] == audit["UNKNOWN_source_counts"] == {"charter": 75, "rumor": 75}


def test_actual_base_world_and_seed_affect_repair_and_unbound_seeds_refuse():
    base = {"kind": "authority", "facts": [
        {"entity": entity, "attribute": attribute, "source": "charter", "value": "vold"}
        for entity in ("parcel", "carton") for attribute in ("zone", "color")]}
    changed = deepcopy(base)
    changed["facts"][0]["value"] = "vfresh"
    assert data._repair_source_world(base, 17, 0)[0] != data._repair_source_world(changed, 17, 0)[0]
    assert data._repair_source_world(base, 17, 0)[0] != data._repair_source_world(base, 23, 0)[0]
    with pytest.raises(ValueError, match="independent-draw"):
        data.generate_bank("train", 23)
    with pytest.raises(PermissionError): data.generate_bank("confirmation")


def test_v3_output_policy_refuses_all_source_epochs():
    for version in (1, 2, 3):
        with pytest.raises(ValueError):
            safe_output(Path("C:/Users/clclc/BusinessProject") / ("Titan-Joint-Quality-Pilot-v" + str(version)) / "generated.json")


@pytest.mark.parametrize("mutation", ["assistant_hint", "extra_user_hint", "suffix_hint", "system_hint"])
def test_source_blind_projection_rejects_different_full_public_hints_on_existing_row(mutation):
    # Existing open epoch001 is immutable input; do not regenerate a bank.
    path = Path("C:/Users/clclc/BusinessProject/Titan-Joint-Quality-Pilot-v3-runs/data-001/train.jsonl")
    with path.open(encoding="utf-8") as handle:
        row = next(value for value in map(json.loads, handle) if value["domain"] == "grounding")
    original = row["messages"]
    assert data.source_blind_projection(original)
    for hint in ("The readable answer hint is VALPHA.", "The readable answer hint is VBETA."):
        changed = deepcopy(original)
        if mutation == "assistant_hint": changed.append({"role": "assistant", "content": hint})
        elif mutation == "extra_user_hint": changed.append({"role": "user", "content": hint})
        elif mutation == "suffix_hint": changed[1]["content"] += "\n" + hint
        else: changed[0]["content"] += "\n" + hint
        # The shared parser still sees the same task; full policy input differs.
        assert data.parse_prompt(changed) == data.parse_prompt(original)
        with pytest.raises(ValueError, match="full exact canonical"):
            data.source_blind_projection(changed)
