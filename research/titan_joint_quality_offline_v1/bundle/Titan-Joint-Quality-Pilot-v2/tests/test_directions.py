from copy import deepcopy
import pytest
from titan_joint_quality_v1.data import generate_bank
from titan_joint_quality_v1.directions import audit_query_crossover, public_without_query

def test_full_train_and_pilot_queryblind_bound():
    for split, count in (("train", 2000), ("pilot", 384)):
        report = audit_query_crossover(generate_bank(split))
        assert report["rows"] == count
        assert report["status"] == "PASS_QUERYBLIND_PAIR_BOUND_MECHANICS_ONLY", report["errors"][:3]
        assert report["deterministic_queryblind_pair_accuracy_upper_bound"] == .5
        assert not report["model_quality_gain_established"]

@pytest.mark.parametrize("defect", ["fact", "truth", "visible_format", "drop_variant"])
def test_no_metadata_only_format_or_favorable_pair_subset(defect):
    rows = generate_bank("pilot"); parent = rows[0]["parent_id"]
    pair = [r for r in rows if r["parent_id"] == parent]
    target = pair[1]
    if defect == "fact":
        target["messages"][0]["content"] += " Different facts."
    elif defect == "truth": target["canonical_answer"] = pair[0]["canonical_answer"]
    elif defect == "visible_format": target["requested_format"] = next(f for f in ("plain", "json", "tag") if f != pair[0]["requested_format"])
    else: rows.remove(target)
    result = audit_query_crossover(rows)
    assert result["status"] == "BLOCKED_QUERY_CROSSOVER_INSTRUMENT"
    assert result["deterministic_queryblind_pair_accuracy_upper_bound"] is None
