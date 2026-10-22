"""Prospective synthetic endpoint/cost checks; no native outcome reads."""
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import open_bank
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.open_analysis import aggregate, raw_diagnostics
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_open_analysis import fixture


def test_validation_primary_comparator_stays_dev_frozen_when_envelope_changes():
    args = fixture()
    by_id = {row["id"]: row for row in args[2]}
    for row in args[0]:
        if row["split"] == "OPEN_VALIDATION" and row["arm"] in ("A2", "A2_frozen") and row["fixed_alpha"] == .8:
            raw = by_id[row["case_id"]]
            row.update(response="", score=0, format_score=0, **raw_diagnostics("", raw))
    for row in args[0]:
        if row["split"] == "OPEN_VALIDATION" and row["arm"] == "A4_example":
            source = next(item for item in args[0] if item["physical_run"] and item["case_id"] == row["case_id"]
                          and item["seed"] == row["seed"] and item["arm"] == "A2" and item["fixed_alpha"] == .9)
            row.update(score=source["score"], fixed_alpha=.9, token_ids=source["token_ids"])
    report = aggregate(*args)
    validation = report["splits"]["OPEN_VALIDATION"]
    assert report["complete"]
    assert validation["DEV_frozen_comparator"] == {"alpha": .8, "accuracy": 0}
    assert validation["pairs"]["DEV_frozen_fixed"]["comparator"] == "A2:0.8"
    assert validation["pairs"]["DEV_frozen_fixed"]["score_delta"] == 1
    assert validation["retrospective_fixed_envelope"] == {"alpha": .9, "accuracy": 1, "diagnostic_only": True}
    assert validation["constant_oracle"]["headroom"] == 0
    assert validation["constant_oracle"]["capture"] is None


def test_absent_physical_measurements_are_null_with_known_worker_cost_retained():
    args = fixture()
    args[0] = [row for row in args[0] if row["physical_run"] is False]
    args[1], args[10] = [], []
    report = aggregate(*args)
    assert not report["complete"] and report["physical_rows"] == 0
    assert report["costs"]["worker_wall_seconds"] == 50
    for value in report["costs"]["physical_row_measurements"].values():
        assert value["complete_value"] is None and value["known_subtotal"] is None and value["measured_rows"] == 0
    for split in report["splits"].values():
        assert not split["pairs"] and all(row["accuracy"] is None for row in split["tables"])
