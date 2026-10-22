import copy
import hashlib
import json

import pytest

from experiments.rineya_self_distillation_axes_v1.data import FIELDS, score, validate_rows
from experiments.rineya_self_distillation_axes_v1.proposal_data import (
    COUNTS, MODULUS, STATUS, TASK, build_candidate_bank, sequential_states,
    solve_composed_affine, validate_disjoint_cohorts,
)


@pytest.mark.parametrize("initial,operations,expected", [
    (3, ((2, 1), (-1, 4)), (3, 7, 14)),
    (-1, ((-3, -8), (2, 8)), (16, 12, 15)),
    (17, ((1, 0), (-1, -1)), (0, 0, 16)),
    (2, ((1, 1),) * 7, tuple(range(2, 10))),
])
def test_two_independent_affine_oracles_known_cases(initial, operations, expected):
    assert sequential_states(initial, operations) == expected
    assert solve_composed_affine(initial, operations) == expected[-1]


def _cohorts(bank):
    by_id = {row["id"]: row for row in bank["screen_rows"] + bank["teacher_gate_rows"]}
    return {name: [by_id[row_id] for row_id in item["ids"]]
            for name, item in bank["cohort_manifest"]["cohorts"].items()}


def test_fixed_candidate_counts_schema_and_no_admission_claim():
    bank = build_candidate_bank()
    assert bank == build_candidate_bank()
    assert bank["status"] == STATUS
    assert len(bank["screen_rows"]) == 56
    assert len(bank["teacher_gate_rows"]) == 8
    assert {name: len(rows) for name, rows in _cohorts(bank).items()} == COUNTS
    assert bank["cohort_manifest"]["paired_row_order_seeds"] == [17, 29]
    assert bank["cohort_manifest"]["admission_granted"] is False
    assert bank["cohort_manifest"]["launch_authorized"] is False
    assert bank["audit"]["native_token_budgets_verified"] is False
    for row in bank["screen_rows"] + bank["teacher_gate_rows"]:
        assert set(row) == FIELDS
        assert score(row["reference"], row["answer"], row["scorer"])
    # Existing loader retains both acquisition strata because they share TASK.
    assert len(validate_rows(bank["screen_rows"], TASK)) == 56


def test_gate_and_all_underlying_parents_are_disjoint():
    bank = build_candidate_bank()
    audit = validate_disjoint_cohorts(_cohorts(bank), bank["audit"]["case_records"])
    assert audit == {"unique_rows": 64, "unique_parents": 64,
                     "unique_underlying_inputs": 64, "unique_affine_input_map_pairs": 48,
                     "all_disjoint": True}
    gate_ids = {r["id"] for r in bank["teacher_gate_rows"]}
    assert not gate_ids & {r["id"] for r in bank["screen_rows"]}
    assert all(r["split"] == "train" and r["purpose"] == "acquisition"
               for r in bank["teacher_gate_rows"])
    assert all(r["split"] == "dev" for r in bank["screen_rows"] if r["purpose"] == "retention")


def test_relabeling_parent_across_acquisition_retention_is_rejected():
    bank = build_candidate_bank()
    cohorts = copy.deepcopy(_cohorts(bank))
    cohorts["retention_arithmetic"][0]["parent_id"] = cohorts["train"][0]["parent_id"]
    with pytest.raises(ValueError, match="parent reused"):
        validate_disjoint_cohorts(cohorts, bank["audit"]["case_records"])


def test_relabeling_underlying_input_cannot_create_independence():
    bank = build_candidate_bank()
    records = copy.deepcopy(bank["audit"]["case_records"])
    records[1]["canonical_input"] = copy.deepcopy(records[0]["canonical_input"])
    with pytest.raises(ValueError, match="underlying input"):
        validate_disjoint_cohorts(_cohorts(bank), records)


def test_different_sequences_with_equivalent_affine_map_are_rejected():
    bank = build_candidate_bank()
    cohorts, records = copy.deepcopy(_cohorts(bank)), copy.deepcopy(bank["audit"]["case_records"])
    source = records[0]
    target_row = cohorts["harder_transfer"][0]
    target = next(record for record in records if record["id"] == target_row["id"])
    target["canonical_input"] = copy.deepcopy(source["canonical_input"])
    target["canonical_input"]["operations"] += [[1, 0]] * 3
    signature = hashlib.sha256(json.dumps(target["canonical_input"], sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    target["input_sha256"] = target["parent_id"] = target_row["parent_id"] = signature
    target["affine_semantic_sha256"] = source["affine_semantic_sha256"]
    with pytest.raises(ValueError, match="Equivalent normalized input/affine map"):
        validate_disjoint_cohorts(cohorts, records)


def test_harder_changes_only_operation_count_rule_and_support():
    bank = build_candidate_bank()
    for record in bank["audit"]["case_records"]:
        if record["cohort"] not in {"train", "iid_acquisition", "harder_transfer", "teacher_gate"}:
            continue
        case = record["canonical_input"]
        assert case["kind"] == "signed_modular_affine" and case["modulus"] == MODULUS
        assert 0 <= case["initial"] < MODULUS
        assert all(a in {-3, -2, -1, 1, 2, 3} and -8 <= b <= 8 for a, b in case["operations"])
        assert len(case["operations"]) in ({5, 6, 7} if record["cohort"] == "harder_transfer" else {2})
        assert sequential_states(case["initial"], case["operations"])[-1] == solve_composed_affine(
            case["initial"], case["operations"])
    cohort_map = _cohorts(bank)
    for name in ("train", "iid_acquisition", "harder_transfer", "teacher_gate"):
        for row in cohort_map[name]:
            assert row["task"] == TASK
            assert "s = (a*s + b) mod 17" in row["prompt"]
            assert "unique residue from 0 through 16" in row["prompt"]
    assert bank["cohort_manifest"]["legacy_pooled_acquisition_endpoint_is_primary"] is False
