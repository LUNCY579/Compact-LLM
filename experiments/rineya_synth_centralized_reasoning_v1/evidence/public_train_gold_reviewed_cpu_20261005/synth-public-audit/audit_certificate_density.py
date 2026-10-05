"""Exact actual-row and frozen-schedule uniform-k-subset baseline; no resampling."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import time

from audit_student_visible import parse, require, digest, read_bound, git_identity, SOURCE, OUT, BUNDLE


def rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator, "decimal": float(value),
            "percent": float(value * 100)}


def aggregate(value):
    """A sum across cases is an expected-success count, never a percentage."""
    return {"numerator": value.numerator, "denominator": value.denominator, "decimal": float(value),
            "unit": "expected_successes_for_one_uniform_k_subset_draw_per_case"}


def main():
    start, cpu = time.monotonic(), time.process_time()
    original = OUT / "SYNTH_ACTUAL640_PUBLIC_GOLD_AUDIT_001.json"
    original_raw = original.read_bytes()
    require(hashlib.sha256(original_raw).hexdigest() == "9fec2144b09e9dc71d1a0cec5c44a40d38e20dcc1a3671401c941227bca57657", "independent public-oracle report drift")
    audit = parse(original_raw)
    require(audit["status"] == "PASS_ALL640_ACTUAL_PUBLIC_TARGETS_CPU_ONLY" and audit["roster"]["completed_rows"] == 640,
            "complete independent actual640 certificate population")
    rows = {row["row_id"]: row for row in audit["rows"]}
    require(len(rows) == 640, "complete unique source roster")
    source_before = git_identity()
    raw, schedule_ref = read_bound(BUNDLE + "schedule_result.json", "2664eabfa4cfb9c4e28742cb58f4c6649b24eb80a16df3d5f2cf1a195a058744")
    schedule = parse(raw)
    raw, occurrences_ref = read_bound(BUNDLE + "training_occurrences.json", "9fb20d227e4b72d30dfe44be47e0764f9939544f7de34992c6ebce7a0b5d39dc")
    occurrences = parse(raw)
    require(schedule["schedules"] == occurrences["row_ids"] and occurrences["schedule_sha256"] == schedule["sha256"], "frozen schedule/occurrence join")
    summaries, probability_rows = {}, []
    means = {}
    for arm in ("M1", "M2"):
        candidates = [row for row in audit["rows"] if row["arm"] == arm and row["public_skill"] == "S"]
        require(len(candidates) == 160 and len(schedule["schedules"][arm]) == 540, "whole160 S candidate/540 sequence population")
        ids = schedule["schedules"][arm]
        require(all(row_id in rows and rows[row_id]["arm"] == arm for row_id in ids), "scheduled source row IDs")
        exposure = Counter(ids)
        require(sum(exposure.values()) == 540, "actual consumed occurrences, no invented resampling")
        chance = {}
        for row in candidates:
            n, k = row["difficulty"]["n"], row["difficulty"]["k"]
            solutions = row["all_certificate_count"]
            denominator = comb(n, k)
            require(0 < solutions <= denominator, "independent actual certificate count bound")
            chance[row["row_id"]] = Fraction(solutions, denominator)
            probability_rows.append({"row_id": row["row_id"], "arm": arm, "parent_id": row["parent_id"], "condition": row["condition"],
                "public_messages_sha256": row["public_messages_sha256"], "independent_certificate_set_sha256": row["certificate_set_sha256"],
                "n": n, "k": k, "valid_unordered_certificates": solutions, "all_unordered_k_subsets": denominator,
                "uniform_k_subset_probability": rational(chance[row["row_id"]]), "scheduled_occurrences": exposure[row["row_id"]]})
        S_ids = [row_id for row_id in ids if rows[row_id]["public_skill"] == "S"]
        R_ids = [row_id for row_id in ids if rows[row_id]["public_skill"] == "R"]
        require(len(S_ids) == 348 and len(R_ids) == 192 and len(S_ids) + len(R_ids) == 540,
                "whole348 S/192 R training exposure counts")
        pool_sum = sum(chance.values(), Fraction())
        schedule_sum = sum((chance[row_id] for row_id in S_ids), Fraction())
        pool_mean, schedule_mean = pool_sum / 160, schedule_sum / 348
        means[arm] = (pool_mean, schedule_mean)
        summaries[arm] = {"candidate_S_rows": 160, "candidate_S_probability_sum": aggregate(pool_sum), "candidate_S_mean_chance": rational(pool_mean),
            "scheduled_S_occurrences": 348, "scheduled_R_occurrences": 192, "scheduled_total_occurrences": 540,
            "scheduled_S_probability_sum": aggregate(schedule_sum), "scheduled_S_mean_chance": rational(schedule_mean),
            "actual_schedule_row_order_sha256": digest(ids),
            "n_k_joint_candidate_histogram": dict(sorted(Counter(str((row["difficulty"]["n"], row["difficulty"]["k"])) for row in candidates).items()))}
    require(len(probability_rows) == 320, "all actual S candidate rows retained")
    for ref in (schedule_ref, occurrences_ref):
        _, after = read_bound(ref["relative_path"], ref["sha256"])
        require(after == ref, "stable scheduled-source bytes")
    source_after = git_identity()
    report = {"schema": "synth-actual-certificate-density-baseline-v1", "status": "PASS_EXACT_ACTUAL320_S_AND_SCHEDULED696_S_BASELINE",
        "scope": "ACTUAL_APPROVED_PUBLIC_TRAIN_AND_FROZEN_OCCURRENCES_NO_MODEL_DIFFICULTY_OR_CAPABILITY_MEASUREMENT",
        "source_before": source_before, "source_after": source_after,
        "source_refs": {"independent_actual640_oracle_report": {"path": str(original), "sha256": hashlib.sha256(original_raw).hexdigest()},
                        "schedule": schedule_ref, "training_occurrences": occurrences_ref,
                        "implementation": {"path": str(Path(__file__)), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}},
        "definition": "Draw one unordered k-subset uniformly from all binomial(n,k) subsets of the stated n items. Success probability equals independently enumerated valid certificates divided by binomial(n,k).",
        "rows": probability_rows, "arm_summaries": summaries,
        "M2_minus_M1": {"candidate_S_mean_probability": rational(means["M2"][0] - means["M1"][0]),
                        "scheduled_S_mean_probability": rational(means["M2"][1] - means["M1"][1])},
        "interpretation": ["M2 has a higher exact uniform-random certificate success baseline in both actual candidate pool and scheduled S exposures.",
            "This is a certificate-density residual; more solutions can make finding any valid certificate easier. It does not measure model difficulty or capability.",
            "Do not subtract this training-corpus chance delta from DEV gains or reinterpret the frozen study endpoint. The source, dose, schedules, targets and enrollment remain intact.",
            "Exact arithmetic uses all160 candidates and348 actual S occurrences per arm. No fake data, resampling, tokenization or model calls.",
            "Matched whole-run n/k marginals and input tokens do not isolate condition diversity from joint difficulty, certificate density or ordering differences."],
        "resource_measurement": {"input_bound_per_selected_file": 32 * 1024 * 1024,
            "newly_authorized_schedule_input_bytes": schedule_ref["bytes"] + occurrences_ref["bytes"],
            "CPU_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - start,
            "budget_seconds": 300, "hard_containment_or_native_resource_admission_claim": False},
        "frozen_study_or_data_modified": False, "native_model_loads": 0, "native_forward_calls": 0,
        "native_or_HF_tokenizer_instantiated": False, "optimizer_updates": 0, "protected_or_DEV_or_confirmation_oracle_reads": 0,
        "training_data_or_derivatives_admitted": False, "native_quality_or_retention_established": False}
    predecessor = OUT / "SYNTH_CERTIFICATE_DENSITY_BASELINE_002.json"
    report["supersedes_units_display_only"] = {"path": str(predecessor), "sha256": hashlib.sha256(predecessor.read_bytes()).hexdigest(),
        "correction": "Probability sums are expected-success counts; misleading percentage display removed from sum fields. All exact fractions, row probabilities and arm means unchanged."}
    report["sha256"] = digest(report)
    path = OUT / "SYNTH_CERTIFICATE_DENSITY_BASELINE_003.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "summaries": summaries, "M2_minus_M1": report["M2_minus_M1"]}, indent=2))


if __name__ == "__main__":
    main()
