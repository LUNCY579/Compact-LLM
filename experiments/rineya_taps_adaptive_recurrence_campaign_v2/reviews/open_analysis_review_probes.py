"""Independent adversarial synthetic OPEN analyzer probes; no native data."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.open_analysis import aggregate
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_open_analysis import fixture

C = Path(__file__).resolve().parent.parent


def main():
    checks = []
    def probe(name, mutation, expected):
        args = fixture()
        mutation(args)
        report = aggregate(*args)
        checks.append({"probe": name, "expected_complete": expected, "observed_complete": report["complete"],
                       "passed": report["complete"] is expected, "errors": report["errors"]})
    probe("complete_fixture", lambda a: None, True)
    probe("missing_cell", lambda a: a[0].pop(0), False)
    probe("duplicate_cell", lambda a: a[0].append(deepcopy(a[0][0])), False)
    probe("pool_mutation", lambda a: a[5]["pool"].append(1.1), False)
    def altered_unit(a):
        next(e for e in a[1] if e["arm"] == "A1" and e["iteration"] == 1)["alpha"] = 1.1
    probe("unit_trace_is_scaled", altered_unit, False)
    def rounded_fixed(a):
        for row in a[0]:
            if row["arm"] == "A2" and row["fixed_alpha"] == 1.1473425965417514:
                row["fixed_alpha"] = 1.14734
        for event in a[1]:
            if event["arm"] == "A2" and event["iteration"] == 1 and abs(event["alpha"] - 1.1473425965417514) < 1e-6:
                event["alpha"] = 1.14734
    probe("sixth_fixed_scalar_substituted_after_g_format_rounding", rounded_fixed, False)
    probe("recorded_parent_RSS_breach", lambda a: a[7].update(peak_rss_bytes=7*1024**3), False)
    probe("recorded_GPU_peak_breach", lambda a: a[6].update(after_load_resources={"gpu_peak_allocated_bytes": 6*1024**3}), False)
    probe("recorded_host_reserve_breach", lambda a: a[6].update(final_resources={"host_available_bytes": 0}), False)
    probe("unlogged_initial_block_count", lambda a: a[0][0].update(initial_block_applications=99), False)
    probe("nonbinary_SMOKE_score", lambda a: a[0][0].update(score=2), False)
    def over_cap(a):
        case_id = next(row["case_id"] for row in a[0] if row["split"] == "DEVELOPMENT" and row["arm"] == "A0")
        for row in a[0]:
            if row["case_id"] == case_id and row["arm"] == "A0":
                row.update(token_ids=[1]*9, full_model_calls_including_gold_nll=9,
                           processed_token_positions=63, initial_block_applications=9)
        trace = []
        for event in a[1]:
            if event["case_id"] == case_id and event["arm"] == "A0":
                trace.extend({**event, "forward_index": index} for index in range(1, 10))
            else: trace.append(event)
        a[1] = trace
        maxima = [event for event in a[10] if not event["event"].startswith("FORWARD_")]
        a[10] = [{"event": event, "case_id": row["case_id"], "split": row["split"], "seed": row["seed"],
                  "arm": row["arm"], "extra_loops": row["extra_loops"], "full_model_call": index,
                  "prefix_tokens": 3+index-1, "forward_seconds": .1 if event == "FORWARD_COMPLETED" else None}
                 for row in a[0] if row["physical_run"] for index in range(1, len(row["token_ids"])+1)
                 for event in ("FORWARD_STARTED", "FORWARD_COMPLETED")] + maxima
    probe("OPEN_generation_exceeds_registered8_cap", over_cap, False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "native_model_loads": 0,
              "analyzer_sha256": hashlib.sha256((C / "open_analysis.py").read_bytes()).hexdigest(),
              "checks": checks, "unexpected_acceptances": [row["probe"] for row in checks if not row["passed"]]}
    name = sys.argv[1] if len(sys.argv) > 1 else "open_analysis_review_probes_initial.json"
    if Path(name).name != name:
        raise ValueError("review receipt must remain in review ownership")
    with (C / "reviews" / name).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps({"probes": len(checks), "unexpected_acceptances": report["unexpected_acceptances"]}))


if __name__ == "__main__":
    main()
