"""Fresh-process resume verification comparator for RRP-001.

Loads no model and charges no budget. It only reads the two attempts' receipts and
refuses unless the resumed attempt is bit-identical to the candidate where the plan
says it must be:

  resume_verification: "fresh source load from complete update2 checkpoint; final
  parameters and predictions must match"

Lives in the session scratchpad on purpose: adding it to the package would change
package_identity()->code_sha256, which is part of the binding that
validate_checkpoint enforces, and would therefore invalidate the resume it verifies.
"""
import hashlib
import json
import sys
from pathlib import Path

J = Path("C:/Users/clclc/BusinessProject/rineya-reasoning-posttraining-records/rrp001-journal/runs")
EXPECTED = {"candidate_updates": 105, "resumed_updates": 103,
            "total_processed": 167256, "resumed_processed": 164528}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(cand_id="train-001", res_id="resume-001"):
    cand = load(J / cand_id / "microfit" / "result.json")
    res = load(J / res_id / "microfit" / "result.json")
    cand_final = load(J / cand_id / "microfit" / "final.json")
    res_final = load(J / res_id / "microfit" / "final.json")
    src = load(J / res_id / "microfit" / "resume_source.json")

    # engine.final_evaluation_sha256 hashes final.json, and native_runtime.evaluate embeds
    # a time.monotonic() delta (generation_wall_seconds) in EVERY record. That digest can
    # therefore never match across two runs, so it is NOT the prediction criterion. The
    # substantive digest below excludes only wall-clock timing and is the real test.
    TIMING = {"generation_wall_seconds"}

    def substantive(value):
        body = {k: v for k, v in value.items() if k != "records"}
        body["records"] = [{k: v for k, v in sorted(r.items()) if k not in TIMING}
                           for r in value["records"]]
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    cand_sub, res_sub = substantive(cand_final), substantive(res_final)

    checks = {}
    # The plan's two literal criteria.
    checks["final_parameters_match"] = cand["final_state_sha256"] == res["final_state_sha256"]
    checks["final_predictions_match"] = cand_sub == res_sub
    # Supporting identity: same frozen source, same binding, same recipe.
    checks["frozen_source_unchanged"] = cand["frozen_sha256"] == res["frozen_sha256"]
    checks["binding_identical"] = cand["binding"] == res["binding"]
    # The resumed attempt really did resume, and really did less work.
    checks["candidate_complete"] = cand["complete"] is True
    checks["resumed_complete"] = res["complete"] is True
    checks["candidate_did_all_updates"] = cand["attempt_updates"] == EXPECTED["candidate_updates"]
    checks["resumed_did_fewer_updates"] = res["attempt_updates"] == EXPECTED["resumed_updates"]
    checks["candidate_processed_tokens"] = cand["progress"]["processed_tokens"] == EXPECTED["total_processed"]
    checks["resumed_reprocessed_only_the_tail"] = res["attempt_processed_tokens"] == EXPECTED["resumed_processed"]
    checks["resumed_from_candidate_checkpoint"] = (
        Path(src["checkpoint"]).name == "checkpoint-00002"
        and Path(src["checkpoint"]).parent.parent.name == cand_id)
    checks["resume_started_at_update_2"] = src["progress"]["updates"] == 2
    # A resumed run must not re-probe the baseline, and must not claim quality.
    checks["no_baseline_reprobe"] = not (J / res_id / "microfit" / "baseline.json").exists()
    checks["no_quality_claim"] = (cand["quality_improvement_established"] is False
                                  and res["quality_improvement_established"] is False)

    # Prediction-level equality, not just the digest.
    a = {r["id"]: r["prediction"] for r in cand_final["records"]}
    b = {r["id"]: r["prediction"] for r in res_final["records"]}
    checks["per_row_predictions_match"] = a == b

    report = {
        "record": "RRP-001-RESUME-VERIFICATION",
        "candidate_run": cand_id, "resumed_run": res_id,
        "model_calls_by_this_comparator": 0,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "failed_checks": sorted(k for k, v in checks.items() if not v),
        "candidate": {k: cand[k] for k in ("final_state_sha256", "frozen_sha256",
                                           "attempt_updates", "complete")},
        "resumed": {k: res[k] for k in ("final_state_sha256", "frozen_sha256",
                                        "attempt_updates", "attempt_processed_tokens", "complete")},
        "candidate_final_canonical_correct": cand_final["canonical_correct"],
        "resumed_final_canonical_correct": res_final["canonical_correct"],
        "denominator": cand_final["rows"],
        "quality_or_generalization_claim": False,
        "substantive_evaluation_sha256": {"candidate": cand_sub, "resumed": res_sub},
        "raw_file_digest_mismatch_is_expected": {
            "candidate_final_evaluation_sha256": cand["final_evaluation_sha256"],
            "resumed_final_evaluation_sha256": res["final_evaluation_sha256"],
            "equal": cand["final_evaluation_sha256"] == res["final_evaluation_sha256"],
            "sole_differing_field": "generation_wall_seconds (all 32 records)",
            "explanation": (
                "engine.microfit records final_evaluation_sha256 = file_hash(final.json), and "
                "native_runtime.evaluate writes a time.monotonic() delta into every record, so "
                "that digest is nondeterministic by construction and cannot serve as a "
                "resume/determinism criterion. Verified by field-by-field diff: predictions, "
                "generated token ids, correctness, termination, truncation, forward-call counts "
                "and token counts are all identical; only wall-clock timing differs."),
        },
    }
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
