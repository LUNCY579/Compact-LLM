"""Recompute CPU H3 counts and exploratory direction checks from raw journals."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank(values):
    return [1+sum(other < value for other in values)+(sum(other == value for other in values)-1)/2
            for value in values]


def correlation(x, y):
    if len(x) < 2:
        return None
    a, b = sum(x)/len(x), sum(y)/len(y)
    denominator = math.sqrt(sum((v-a)**2 for v in x)*sum((v-b)**2 for v in y))
    return sum((u-a)*(v-b) for u, v in zip(x, y))/denominator if denominator else None


def analyze(run):
    result = run / "result"
    receipt = json.loads((result / "receipt.json").read_text(encoding="utf-8"))
    terminal = json.loads((run / "terminal_receipt.json").read_text(encoding="utf-8"))
    events = rows(result / "attempts.jsonl") if (result / "attempts.jsonl").exists() else []
    outcomes = rows(result / "outcomes.jsonl") if (result / "outcomes.jsonl").exists() else []
    population = json.loads((result / "population.json").read_text(encoding="utf-8")) if (result / "population.json").exists() else []
    completed = [e for e in events if e["event"] in ("VALUE_FORWARD_COMPLETED", "DIAGNOSTIC_FORWARD_COMPLETED")]
    starts = [e for e in events if e["event"] in ("VALUE_FORWARD_STARTED", "DIAGNOSTIC_FORWARD_STARTED", "CACHE_REJECTION_STARTED")]
    failed = [e for e in events if e["event"].endswith("FAILED")]
    qualification = json.loads((result / "qualification.json").read_text(encoding="utf-8")) if (result / "qualification.json").exists() else {}
    forecast = json.loads((result / "matrix_forecast.json").read_text(encoding="utf-8")) if (result / "matrix_forecast.json").exists() else {}
    measured = [r for r in outcomes if r.get("physical_run")]
    eligible = [r["case_id"] for r in population if r["status"] == "ELIGIBLE"]
    complete = (receipt["status"] == "CPU_H3_COMPLETE_NOT_SEMANTIC_EFFICACY"
                and receipt.get("lane") == "CPU_H3_STATE_SENSITIVITY_FP32"
                and receipt.get("loaded_state_sha256") == "b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551"
                and terminal["status"] == "WORKER_TERMINAL" and terminal.get("exit_code") == 0
                and len(eligible) == 7 and len(measured) == 7
                and sorted(eligible) == sorted(r["case_id"] for r in measured)
                and len(outcomes) == 16
                and len(starts) == 47 and len(completed) == 44
                and sum(e["prefix_tokens"] for e in completed) == 4818
                and sum(e.get("state_gradient_evaluations", 0) for e in completed) == 9
                and sum(e["event"] == "CACHE_REJECTED" for e in events) == 2
                and len(failed) == 1 and failed[0].get("error") == "RuntimeError: EXPECTED_DIAGNOSTIC_SUFFIX_EXCEPTION"
                and qualification.get("status") == "PASS_CPU_H3_NATIVE_MECHANICS"
                and qualification.get("lane") == "CPU_H3_STATE_SENSITIVITY_FP32"
                and qualification.get("device") == "cpu"
                and qualification.get("before_state_sha256") == qualification.get("after_state_sha256") == receipt.get("loaded_state_sha256")
                and qualification.get("source_unchanged") is True
                and qualification.get("parameter_gradients") == "NONE"
                and forecast.get("admitted") is True and forecast.get("device") == "cpu"
                and forecast.get("GPU_timing_used") is False
                and receipt.get("source_unchanged") is True
                and receipt.get("checkpoint_files_unchanged") is True
                and receipt.get("parameter_gradients") == "NONE"
                and receipt.get("before_state_sha256") == receipt.get("after_state_sha256") == receipt.get("loaded_state_sha256")
                and receipt.get("optimizer_steps") == 0
                and receipt.get("training") is False and receipt.get("native_test") is False
                and receipt.get("generation_pool_access") is False)
    report = {
        "status": "COMPLETE_EXPOSED_CPU_SURROGATE_ONLY" if complete else "INCOMPLETE_NO_PRIMARY_MECHANISM_COMPARISON",
        "primary_semantic_efficacy": None, "bulk_validation_primary": None,
        "native_worker_seconds": terminal["native_worker_seconds"],
        "model_API_attempts": len(starts), "completed_full_forwards": len(completed),
        "completed_prefix_positions": sum(e["prefix_tokens"] for e in completed),
        "state_adjoints": sum(e.get("state_gradient_evaluations", 0) for e in completed),
        "failed_or_expected_partial": failed,
        "cache_rejections": sum(e["event"] == "CACHE_REJECTED" for e in events),
        "eligible_cases": len(eligible), "measured_cases": len(measured),
        "unmeasured_case_ids": sorted(set(eligible)-{r["case_id"] for r in measured}),
        "mechanism": None,
        "scope": "CPU FP32 actual native suffix/head; exposed first-token CE; no generation, fitting or GPU pooling",
        "source_sha256": {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (run / "terminal_receipt.json", result / "receipt.json",
                                    result / "attempts.jsonl", result / "outcomes.jsonl", result / "population.json",
                                    result / "qualification.json", result / "matrix_forecast.json") if p.exists()},
    }
    if complete:
        checks = []
        for row in measured:
            nominal = row["nominal"]
            derivative = nominal["sensitivity"]
            fd = (row["plus"]["loss"]-row["minus"]["loss"])/.02
            balance = nominal["balance"]
            if not all(math.isfinite(x) for x in (derivative, fd, balance)):
                raise ValueError("nonfinite purported complete native row")
            if abs(derivative-fd) > .001+.03*abs(derivative):
                raise ValueError("purported complete row fails fixed numerical gate")
            useful_direction = 0 if abs(derivative) <= .001 else (-1 if derivative > 0 else 1)
            proposed_direction = 0 if balance == 0 else (1 if balance > 0 else -1)
            checks.append({"case_id": row["case_id"], "scenario_id": row["scenario_id"],
                           "first_token_loss_at_unit": nominal["loss"], "sensitivity": derivative,
                           "finite_difference": fd, "absolute_discrepancy": abs(derivative-fd),
                           "balance": balance, "closed_form_implied_alpha": min(1.2, max(.8, 1+.275*balance)),
                           "locally_useful_direction": useful_direction,
                           "coherence_direction": proposed_direction,
                           "direction_agrees": proposed_direction == useful_direction if useful_direction else None,
                           "loss_delta_099_vs_unit": row["minus"]["loss"]-nominal["loss"],
                           "loss_delta_101_vs_unit": row["plus"]["loss"]-nominal["loss"]})
        nonneutral = [r for r in checks if r["locally_useful_direction"]]
        report["mechanism"] = {
            "cases": checks, "scenario_clusters": len({r["scenario_id"] for r in checks}),
            "nonneutral_cases": len(nonneutral), "neutral_cases": len(checks)-len(nonneutral),
            "coherence_sign_agreement_count": sum(r["direction_agrees"] for r in nonneutral),
            "constant_increase_sign_agreement_count": sum(r["locally_useful_direction"] == 1 for r in nonneutral),
            "constant_decrease_sign_agreement_count": sum(r["locally_useful_direction"] == -1 for r in nonneutral),
            "spearman_balance_vs_negative_derivative": correlation(rank([r["balance"] for r in checks]), rank([-r["sensitivity"] for r in checks])),
            "max_FD_absolute_discrepancy": max(r["absolute_discrepancy"] for r in checks),
            "feature_implementation": "offline closed-form two-observation FP32 energies; symbolic TAPS mapping; incremental v1 rounding parity is not measured",
            "inference": "counts/ranks exploratory; related exposed scenarios, no broad prediction, semantic gain, equivalence or fitted rule",
        }
    return report


def main(run, out):
    report = analyze(run)
    out.mkdir(parents=True, exist_ok=False)
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    text = ["# CPU H3 evidence", "", report["status"], "",
            f"Measured {report['measured_cases']}/{report['eligible_cases']} eligible prefixes. "
            f"Actual {report['model_API_attempts']} API attempts, {report['completed_full_forwards']} complete forwards, "
            f"{report['state_adjoints']} state adjoints and {report['completed_prefix_positions']} prefix positions.", ""]
    if report["mechanism"]:
        m = report["mechanism"]
        text.extend([f"Coherence direction agrees with the local CE direction on {m['coherence_sign_agreement_count']}/{m['nonneutral_cases']} nonneutral cases "
                     f"({m['neutral_cases']} neutral; {m['scenario_clusters']} related scenario clusters). "
                     f"Constant increase agrees on {m['constant_increase_sign_agreement_count']}; constant decrease on {m['constant_decrease_sign_agreement_count']}.", "",
                     f"Exploratory Spearman(balance, -derivative)={m['spearman_balance_vs_negative_derivative']}. "
                     f"Maximum absolute adjoint/FD discrepancy={m['max_FD_absolute_discrepancy']}; every row meets the fixed tolerance.", ""])
    text.append("First-token surrogate evidence does not establish complete-answer correctness or a controller solution. The larger-bank primary validation remains null; no CPU/GPU pooling, fitting or speed claim.")
    (out / "REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
