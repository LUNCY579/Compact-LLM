"""Synthetic encoding/selection guards; these tests contain no Titan task bank."""
from copy import deepcopy

from titan_joint_quality_v1.common import CONTRACT, PARENT_SHA, digest, file_sha
from titan_joint_quality_v1.exposure import build_exposure_plan, encode_rows


class FakeTokenizerAdapter:
    """Explicit opaque fixture IDs, supervised EOS=5, masked newline=6."""
    def encode_case(self, messages, target):
        length = int(messages[-1]["content"].split("|")[-1])
        start = length - 4
        return {"ids": [1] * start + [2, 3, 5, 6], "response_start": start,
                "response_end": start + 3, "eos_position": start + 2,
                "eos_token_id": 5, "truncated": False, "padded": False}


def fixture_rows(length=20):
    roles = ["reasoning"] * 10 + ["preservation"] * 5 + ["grounding"] * 3 + ["format"] * 2
    domain = {"reasoning": "reasoning", "preservation": "transition", "grounding": "grounding", "format": "format"}
    return [{"case_id": f"fixture-{index}", "parent_id": f"fixture-parent-{index}",
             "family_id": f"fixture-{role}", "domain": domain[role], "role": role,
             "messages": [{"role": "user", "content": f"opaque fixture|{length}"}],
             "canonical_answer": 7, "target": "7", "requested_format": "integer"}
            for index, role in enumerate(roles)]


def bindings():
    return {"parent_sha256": PARENT_SHA, "data_sha256": digest("synthetic data only"),
            "tokenizer_sha256": digest("fake tokenizer; not native")}


def test_shift_mask_labels_response_eos_once_and_excludes_prompt_newline():
    receipt = encode_rows(fixture_rows(), FakeTokenizerAdapter())
    assert receipt["status"] == "ENCODED_OFFLINE_ONLY"
    for row in receipt["encoded_rows"]:
        assert row["label_mask"] == [False] * 16 + [True, True, True, False]
        assert row["response_eos_tokens"] == 3
        assert row["label_mask"][row["eos_position"]]
        assert not row["label_mask"][-1]


def test_missing_adapter_or_one_bad_row_blocks_entire_encoded_population():
    receipt = encode_rows(fixture_rows(), object())
    assert receipt["status"] == "BLOCKED_TOKENIZATION"
    assert not receipt["encoded_rows"]
    rows = fixture_rows()
    rows[-1]["messages"][-1]["content"] = "opaque fixture|1025"
    receipt = encode_rows(rows, FakeTokenizerAdapter())
    assert receipt["summary"]["validated_rows"] == 19
    assert receipt["summary"]["rejected_rows"] == 1
    assert not receipt["encoded_rows"]
    assert "truncation" in receipt["rejections"][0]["reason"]


def test_strict_integer_boundaries_and_eos_refuse_tampered_encodings():
    for field, value in (("response_start", True), ("response_end", 20),
                         ("eos_position", 16), ("eos_token_id", False)):
        class BadAdapter(FakeTokenizerAdapter):
            def encode_case(self, messages, target):
                result = super().encode_case(messages, target)
                result[field] = value
                return result
        result = encode_rows(fixture_rows(), BadAdapter())
        assert result["status"] == "BLOCKED_TOKENIZATION"
        assert result["encoded_rows"] == []
    class BadIDs(FakeTokenizerAdapter):
        def encode_case(self, messages, target):
            result = super().encode_case(messages, target)
            result["ids"][0] = True
            return result
    assert encode_rows(fixture_rows(), BadIDs())["status"] == "BLOCKED_TOKENIZATION"


def test_plan_is_deterministic_and_control_has_identical_exposure_without_extra_r():
    encoding = encode_rows(fixture_rows(256), FakeTokenizerAdapter())
    first = build_exposure_plan(encoding, bindings())
    second = build_exposure_plan(encoding, bindings())
    assert first == second
    assert first["status"] == "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN"
    assert first["processed_tokens"] == 5120
    assert first["same_sequences_order_groups"] and first["same_reasoning_targets"]
    assert first["control_extra_reasoning_rows"] == 0
    assert first["processed_token_shares_by_role"] == {"reasoning": .5, "preservation": .25,
                                                       "grounding": .15, "format": .1}
    assert len(set(first["selected_case_ids"])) == len(first["selected_case_ids"])
    assert all(group["processed_tokens"] <= 2048 and group["reasoning_label_tokens"] > 0
               for group in first["groups"])


def test_selected_two_long_rows_and_two_r_rows_can_cross_group_boundaries():
    rows = fixture_rows(1024)
    for index in (0, 1):
        rows[index]["parent_id"] = "two-row-reasoning-parent"
    for index in (10, 11):
        rows[index]["parent_id"] = "two-row-long-parent"
        rows[index]["domain"] = "long_context"
    plan = build_exposure_plan(encode_rows(rows, FakeTokenizerAdapter()), bindings())
    assert plan["status"] == "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN"
    assert len(plan["selected_case_ids"]) == 20
    long_groups = [group for group in plan["groups"]
                   if "two-row-long-parent" in group["parent_ids"]]
    assert len(long_groups) == 2
    assert all(group["reasoning_label_tokens"] > 0 and group["processed_tokens"] <= 2048
               for group in long_groups)


def test_four_balanced_1024_r_j_rows_pack_when_parent_units_put_r_buddies_first():
    from titan_joint_quality_v1.exposure import _pack_groups
    rows = fixture_rows(1024)
    chosen = [deepcopy(rows[index]) for index in (0, 1, 2, 3, 10, 11, 12, 13)]
    parents = ("r-pair-1", "r-pair-1", "r-pair-2", "r-pair-2",
               "j-pair-1", "j-pair-1", "j-pair-2", "j-pair-2")
    for row, parent in zip(chosen, parents):
        row["parent_id"] = parent
    encoded = encode_rows(chosen, FakeTokenizerAdapter())["encoded_rows"]
    # Preserve the adversarial unit ordering: Rpair, Rpair, Jpair, Jpair.
    units = [{"parent_id": encoded[index]["parent_id"], "role": encoded[index]["role"],
              "processed_tokens": 2048, "rows": encoded[index:index + 2]}
             for index in (0, 2, 4, 6)]
    groups = _pack_groups(units)
    assert len(groups) == 4
    assert all(group["processed_tokens"] == 2048 and len(group["rows"]) == 2 for group in groups)
    assert all([row["role"] for row in group["rows"]] == ["reasoning", "preservation"]
               for group in groups)
    consumed = [case for group in groups for case in group["case_ids"]]
    assert len(consumed) == len(set(consumed)) == 8
    assert set(consumed) == {row["case_id"] for row in chosen}


def test_plan_binds_actual_exposure_and_common_producer_bytes():
    from pathlib import Path
    import titan_joint_quality_v1.exposure as exposure
    plan = build_exposure_plan(encode_rows(fixture_rows(), FakeTokenizerAdapter()), bindings())
    assert plan["status"] == "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN"
    source = Path(exposure.__file__).resolve()
    assert plan["exposure_source_sha256"] == file_sha(source)
    assert plan["common_source_sha256"] == file_sha(source.with_name("common.py"))
    unsigned = dict(plan)
    del unsigned["plan_sha256"]
    assert plan["plan_sha256"] == digest(unsigned)


def test_missing_source_or_changed_masks_blocks_without_partial_population():
    encoding = encode_rows(fixture_rows(), FakeTokenizerAdapter())
    assert build_exposure_plan(encoding, {})["status"] == "BLOCKED_EXPOSURE_PLAN"
    tampered = deepcopy(encoding)
    tampered["encoded_rows"][0]["label_mask"][-1] = True
    result = build_exposure_plan(tampered, bindings())
    assert result["status"] == "BLOCKED_EXPOSURE_PLAN"
    assert result["groups"] == []
    assert not result["selected_case_ids"]


def test_budget_nonselection_is_explicit_and_all_parent_rows_remain_accounted():
    rows = fixture_rows(512)
    rows = [{**deepcopy(row), "case_id": f"copy-{copy}-{row['case_id']}",
             "parent_id": f"copy-{copy}-{row['parent_id']}"}
            for copy in range(6) for row in rows]
    plan = build_exposure_plan(encode_rows(rows, FakeTokenizerAdapter()), bindings())
    assert plan["status"] == "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN"
    assert plan["processed_tokens"] <= CONTRACT["processed_tokens_cap_per_arm"]
    accounted = plan["selected_case_ids"] + [case for unit in plan["unselected"] for case in unit["case_ids"]]
    assert len(accounted) == len(rows) == len(set(accounted))
    assert plan["unselected"]


def test_budget_selection_that_cannot_supply_one_r_per_group_remains_blocked(monkeypatch):
    import titan_joint_quality_v1.exposure as exposure
    rows = [{**deepcopy(row), "case_id": f"copy-{copy}-{row['case_id']}",
             "parent_id": f"copy-{copy}-{row['parent_id']}"}
            for copy in range(5) for row in fixture_rows(1024)]
    original = deepcopy(rows)
    observed = {}
    actual_pack = exposure._pack_groups
    def inspect_selected(units):
        selected = [row for unit in units for row in unit["rows"]]
        observed["tokens"] = sum(len(row["ids"]) for row in selected)
        observed["reasoning_rows"] = sum(row["role"] == "reasoning" for row in selected)
        return actual_pack(units)
    monkeypatch.setattr(exposure, "_pack_groups", inspect_selected)
    result = exposure.build_exposure_plan(encode_rows(rows, FakeTokenizerAdapter()), bindings())
    # 52 complete 1024-token rows need at least 26 groups, but only 25
    # distinct R rows were selected. Duplicating R or dropping a selected row
    # would change the registered exposure, so this fixture is infeasible.
    assert observed == {"tokens": 53248, "reasoning_rows": 25}
    assert observed["reasoning_rows"] < (observed["tokens"] + 2047) // 2048
    assert result["status"] == "BLOCKED_EXPOSURE_PLAN"
    assert result["groups"] == [] and result["selected_case_ids"] == []
    assert "cannot form a group with R labels" in result["rejections"][0]["reason"]
    assert rows == original


def test_missing_role_cannot_be_silently_promoted_to_different_mixture():
    rows = [row for row in fixture_rows() if row["role"] != "grounding"]
    result = build_exposure_plan(encode_rows(rows, FakeTokenizerAdapter()), bindings())
    assert result["status"] == "BLOCKED_EXPOSURE_PLAN"
    assert "mixture" in result["rejections"][0]["reason"]


def test_current_train_generator_full_metadata_survives_encoding_and_planning():
    from titan_joint_quality_v1.data import generate_bank
    rows = generate_bank("train")
    class MetadataFixtureTokenizer:
        def encode_case(self, messages, target):
            # Opaque fixture tokenization only: no tokenizer, model or source assets.
            return {"ids": [1] * 16 + [2, 3, 5, 6], "response_start": 16,
                    "response_end": 19, "eos_position": 18, "eos_token_id": 5}
    encoding = encode_rows(rows, MetadataFixtureTokenizer())
    assert encoding["status"] == "ENCODED_OFFLINE_ONLY"
    assert len(encoding["encoded_rows"]) == len(rows) == 2000
    originals = {row["case_id"]: row for row in rows}
    for encoded in encoding["encoded_rows"]:
        original = originals[encoded["case_id"]]
        assert encoded["source_record"] == original
        assert encoded["row_sha256"] == digest(original)
        assert digest({key: encoded[key] for key in original}) == digest(original)
        assert {"variant_id", "pair_kind", "solver_trace"}.issubset(encoded["source_record"])
    plan = build_exposure_plan(encoding, {**bindings(), "data_sha256": digest(rows)})
    assert plan["status"] == "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN"
    assert plan["selected_rows"] == 2000
    assert plan["processed_tokens"] == 40000


def test_full_source_metadata_projection_and_trace_mutations_are_rejected():
    rows = fixture_rows()
    for row in rows:
        row.update({"variant_id": 1, "pair_kind": "fixture_pair",
                    "solver_trace": {"source": "fixture only", "teacher_a": 7},
                    "future_source_provenance": {"revision": "fixture-v1"}})
    encoding = encode_rows(rows, FakeTokenizerAdapter())
    assert build_exposure_plan(encoding, bindings())["status"] == "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN"
    changed = deepcopy(encoding)
    changed["encoded_rows"][0]["variant_id"] = True  # Python True == 1 must not evade binding.
    result = build_exposure_plan(changed, bindings())
    assert result["status"] == "BLOCKED_EXPOSURE_PLAN"
    assert "projection" in result["rejections"][0]["reason"]
    changed = deepcopy(encoding)
    changed["encoded_rows"][0]["source_record"]["solver_trace"]["teacher_a"] = 8
    result = build_exposure_plan(changed, bindings())
    assert result["status"] == "BLOCKED_EXPOSURE_PLAN"
    assert "provenance binding" in result["rejections"][0]["reason"]
    changed = deepcopy(encoding)
    del changed["encoded_rows"][0]["future_source_provenance"]
    assert build_exposure_plan(changed, bindings())["status"] == "BLOCKED_EXPOSURE_PLAN"
    changed = deepcopy(encoding)
    changed["encoded_rows"][0]["unbound_metadata"] = "must not be silently accepted"
    assert build_exposure_plan(changed, bindings())["status"] == "BLOCKED_EXPOSURE_PLAN"


def test_original_metadata_is_copied_without_mutating_or_aliasing_the_source():
    rows = fixture_rows()
    rows[0]["solver_trace"] = {"teacher_a": 7}
    before = deepcopy(rows)
    encoding = encode_rows(rows, FakeTokenizerAdapter())
    encoding["encoded_rows"][0]["solver_trace"]["teacher_a"] = 99
    assert rows == before
    assert encoding["encoded_rows"][0]["source_record"]["solver_trace"]["teacher_a"] == 7
