"""Independent synthetic parent/selection/stochastic/cost tests; no native data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_analysis import aggregate
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_bulk_analysis import fixture

C = Path(__file__).resolve().parent.parent


def main():
    results = []
    def run(name, mutate, validate):
        args = fixture(); mutate(args)
        report = aggregate(*args)
        try: validate(report); passed = True
        except AssertionError: passed = False
        results.append({"probe": name, "passed": passed, "complete": report["complete"], "errors": report["errors"]})
    def freeze_vs_envelope(args):
        raw = {case["id"]: case for case in args[2]}
        for row in args[0]:
            if row["split"] == "OPEN_VALIDATION" and row["arm"] in ("A2", "A2_frozen") and row["fixed_alpha"] == .8:
                case = raw[row["case_id"]]
                row.update(response="NO" if case["gold"] == "YES" else "YES" if case["answer_kind"] == "choice" else "99999999", score=0, token_ids=[2])
        for row in args[0]:
            if row["split"] == "OPEN_VALIDATION" and row["arm"] == "A4_example":
                row.update(score=1, fixed_alpha=.9)
    def check_freeze(report):
        v = report["reports"]["OPEN_VALIDATION"]
        assert report["complete"] and v["frozen_fixed_primary"] == {"alpha": .8, "accuracy": 0}
        assert v["retrospective_envelope"]["alpha"] == .9 and v["retrospective_envelope"]["diagnostic_only"]
        assert v["contrasts"]["frozen_fixed"]["score_delta"] == 1 and v["constant_oracle"]["capture"] is None
    run("DEV_frozen_primary_distinct_from_validation_envelope", freeze_vs_envelope, check_freeze)
    def masked_harm(args):
        pair = sorted([case for case in args[2] if case["split"] == "OPEN_VALIDATION" and case["pair_kind"] == "EVIDENCE_TRANSFER"], key=lambda r: r["id"])[:2]
        answerable, unknown = pair
        assert answerable["parent_id"] == unknown["parent_id"]
        for row in args[0]:
            if row["case_id"] == answerable["id"] and row["arm"] in ("A2", "A2_frozen") and row["fixed_alpha"] == .8:
                row.update(response="UNKNOWN", score=0, token_ids=[2])
            elif row["case_id"] == unknown["id"] and row["arm"] == "A3":
                row.update(response="99999999", score=0, token_ids=[2])
            elif row["case_id"] == answerable["id"] and row["arm"] == "A4_example":
                row.update(score=1, fixed_alpha=.9)
        for event in args[1]:
            if event["case_id"] == unknown["id"] and event["arm"] == "A3" and event["iteration"] == 1:
                event["alpha"] = .95
    def check_harm(report):
        v = report["reports"]["OPEN_VALIDATION"]["contrasts"]["frozen_fixed"]
        assert report["complete"] and v["score_delta"] == 0
        assert not v["observed_preservation_screen"] and len(v["variant_losses"]) == 1 and len(v["correct_case_retention_losses"]) == 1
    run("parent_average_cannot_mask_evidence_variant_harm", masked_harm, check_harm)
    def no_measurements(args):
        args[0] = [row for row in args[0] if row["physical_run"] is False]
        args[1], args[10] = [], []
    def check_nulls(report):
        assert not report["complete"] and report["costs"]["worker_wall_seconds"] == 50
        assert all(value["complete_value"] is None and value["known_subtotal"] is None and value["measured_rows"] == 0 for value in report["costs"]["physical_only"].values())
        assert not report["reports"]["OPEN_VALIDATION"]["contrasts"]
    run("absent_metrics_NULL_known_worker_cost_retained", no_measurements, check_nulls)
    def impossible_random_variance(args):
        row = next(row for row in args[0] if row["physical_run"] and row["split"] == "OPEN_VALIDATION" and row["arm"] == "A5" and row["seed"] == 29)
        raw = next(case for case in args[2] if case["id"] == row["case_id"])
        row.update(token_ids=[2], response="NO" if raw["gold"] == "YES" else "YES" if raw["answer_kind"] == "choice" else "99999999", score=0)
    def check_impossible(report): assert not report["complete"]
    run("same_input_and_alpha_cannot_produce_A5_seed_token_divergence", impossible_random_variance, check_impossible)
    name = sys.argv[1] if len(sys.argv) > 1 else "bulk_analysis_probes_initial_001.json"
    if Path(name).name != name: raise ValueError("review output must be review-local")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "native_model_loads": 0, "analyzer_sha256": hashlib.sha256((C / "bulk_analysis.py").read_bytes()).hexdigest(),
              "tests": results, "unresolved": [row["probe"] for row in results if not row["passed"]]}
    with (C / "reviews" / name).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps({"probes": len(results), "unresolved": report["unresolved"]}))


if __name__ == "__main__": main()
