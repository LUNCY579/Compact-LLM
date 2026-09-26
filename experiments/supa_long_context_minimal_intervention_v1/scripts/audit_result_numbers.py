#!/usr/bin/env python3
"""CPU-only audit of cached Phase 0 observations, never a model or training run.

Run from the repository root with ../Chat-Bubble/.venv/Scripts/python.exe.
Every score, contrast, and measured token statistic starts from per_example.jsonl.
Cached conditions provide accepted-answer metadata only; summaries are comparison
targets and never inputs to recomputed estimates. Original files are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
SEED = 20260905
RESAMPLES = 10_000
UNKNOWN = {"unknown", "not determinable", "cannot be determined"}
CONTRASTS = {
    "B_minus_A": ("A_full", "B_query"),
    "B_minus_C": ("C_random", "B_query"),
    "C_minus_A": ("A_full", "C_random"),
    "D_minus_A": ("A_full", "D_oracle"),
    "D_minus_B": ("B_query", "D_oracle"),
}
STAGES = {
    "phase0_eval": ("phase0_conditions", (45, 125, 170)),
    "phase0_eval_a11_ungated": ("phase0_conditions", (41, 121, 162)),
    "phase0_eval_a1_gated": ("phase0_conditions", (43, 125, 168)),
    "phase0_eval_a1_ungated": ("phase0_conditions", None),
    "phase0_eval_lcr_dev": ("phase0_conditions_lcr_dev", None),
    "phase0_eval_smoke": ("phase0_conditions_smoke", None),
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_normalize():
    # Import the suite's authoritative scorer without importing model libraries.
    path = REPO / "Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py"
    spec = importlib.util.spec_from_file_location("audit_suite_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize, path


def describe(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values), "sum": sum(values)}


def summarize(correct):
    n, k = len(correct), sum(correct)
    return {"n": n, "correct": k, "accuracy": round(k / n, 6),
            "accuracy_points": round(100 * k / n, 2)}


@lru_cache(maxsize=None)
def bootstrap(a, b):
    n = len(a)
    assert n == len(b) and n
    rng = random.Random(SEED)
    differences = [int(y) - int(x) for x, y in zip(a, b)]
    samples = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(RESAMPLES))
    delta = sum(b) / n - sum(a) / n
    lo, hi = samples[250], samples[9750]
    return {"n": n, "delta": round(delta, 6), "delta_points": round(100 * delta, 3),
            "ci_low_points": round(100 * lo, 3), "ci_high_points": round(100 * hi, 3),
            "ci_excludes_zero": lo > 0 or hi < 0, "resamples": RESAMPLES, "seed": SEED}


def mcnemar(a, b, rounded=True):
    b01 = sum(not x and y for x, y in zip(a, b))
    b10 = sum(x and not y for x, y in zip(a, b))
    n = b01 + b10
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2**n) if n else 1.0
    return {"b_only_correct": b01, "a_only_correct": b10, "discordant": n,
            "p_value": round(p, 6) if rounded else p}


def frame(by, ids):
    present = [c for c in by if all(i in by[c] for i in ids)]
    scores = {c: tuple(by[c][i]["correct"] for i in ids) for c in present}
    result = {
        "n_examples": len(ids),
        "headline": {c: summarize(scores[c]) for c in present},
        "contrasts": {}, "by_task_type": {}, "by_variant": {},
        "malformed_rate": {}, "mean_prompt_tokens": {}, "mean_inference_seconds": {},
        "abstention_rate": {}, "abstention_counts": {},
    }
    for label, (base, arm) in CONTRASTS.items():
        if base in present and arm in present:
            result["contrasts"][label] = {**bootstrap(scores[base], scores[arm]),
                                           "mcnemar": mcnemar(scores[base], scores[arm]),
                                           "delta_points_from_integer_difference": 100 * (sum(scores[arm]) - sum(scores[base])) / len(ids),
                                           "mcnemar_unrounded_p": mcnemar(scores[base], scores[arm], rounded=False)["p_value"]}
    for condition in present:
        rows = [by[condition][i] for i in ids]
        for field, key in [("task_type", "by_task_type"), ("variant", "by_variant")]:
            groups = defaultdict(list)
            for row in rows:
                groups[row[field]].append(row["correct"])
            result[key][condition] = {k: summarize(v) for k, v in sorted(groups.items())}
        result["malformed_rate"][condition] = round(sum(r["malformed"] for r in rows) / len(rows), 4)
        result["mean_prompt_tokens"][condition] = round(statistics.mean(r["prompt_tokens"] for r in rows), 1)
        result["mean_inference_seconds"][condition] = round(statistics.mean(r["inference_seconds"] for r in rows), 4)
        abstentions = sum(r["normalized_prediction"].strip() in UNKNOWN for r in rows)
        result["abstention_counts"][condition] = abstentions
        result["abstention_rate"][condition] = round(abstentions / len(rows), 4)
    return result


def compare_subset(stored, recomputed, prefix=""):
    """Compare only keys supplied by an existing summary to their recomputation."""
    discrepancies = []
    if isinstance(stored, dict):
        for key, value in stored.items():
            p = f"{prefix}.{key}" if prefix else key
            if key not in recomputed:
                discrepancies.append({"path": p, "stored": value, "recomputed": "MISSING"})
            else:
                discrepancies.extend(compare_subset(value, recomputed[key], p))
    elif stored != recomputed:
        discrepancies.append({"path": prefix, "stored": stored, "recomputed": recomputed})
    return discrepancies


def cluster_sensitivity(by, ids, base, arm):
    """Unregistered sensitivity: sample whole scenarios, preserving variants."""
    packs = defaultdict(list)
    for i in ids:
        packs[by[base][i]["scenario_id"]].append(int(by[arm][i]["correct"]) - int(by[base][i]["correct"]))
    values = [(sum(packs[k]), len(packs[k])) for k in sorted(packs)]
    rng = random.Random(SEED)
    samples = []
    for _ in range(RESAMPLES):
        sampled = [values[rng.randrange(len(values))] for _ in values]
        samples.append(100 * sum(s for s, n in sampled) / sum(n for s, n in sampled))
    samples.sort()
    return {"status": "unregistered descriptive sensitivity, not confirmatory",
            "unit": "scenario_id", "n_scenarios": len(values), "n_rows": len(ids),
            "ci_low_points": round(samples[250], 3), "ci_high_points": round(samples[9750], 3),
            "resamples": RESAMPLES, "seed": SEED}


def stage_audit(stage, condition_stage, expected, normalize):
    source = LAB / "results" / stage / "per_example.jsonl"
    rows = read_jsonl(source)
    conditions_path = LAB / "results" / condition_stage / "conditions.jsonl"
    metadata = {(r["condition"], r["example_id"]): r for r in read_jsonl(conditions_path)}
    by = defaultdict(dict)
    score_mismatches, normalization_mismatches, malformed_mismatches = [], [], []
    metadata_mismatches = []
    for row in rows:
        condition, row_id = row["condition"], row["example_id"]
        if row_id in by[condition]:
            raise ValueError(f"Duplicate observation in {source}: {condition}/{row_id}")
        gold = metadata[condition, row_id]
        for key in ("gold_answer", "scenario_id", "answerable", "task_type", "variant", "split",
                    "original_context_tokens", "selected_context_tokens", "retention_ratio"):
            if row[key] != gold[key]:
                metadata_mismatches.append({"id": row_id, "condition": condition, "key": key})
        norm = normalize(row["raw_prediction"])
        correct = norm in {normalize(answer) for answer in gold["accepted_answers"]}
        malformed = (not row["raw_prediction"].strip()
                     or "<think>" in row["raw_prediction"] or "</think>" in row["raw_prediction"]
                     or len(re.findall(r"\b[\w'-]+\b", row["raw_prediction"].strip())) > 12)
        for actual, stored, target in [(correct, row["correct"], score_mismatches),
                                       (norm, row["normalized_prediction"], normalization_mismatches),
                                       (malformed, row["malformed"], malformed_mismatches)]:
            if actual != stored:
                target.append({"id": row_id, "condition": condition, "stored": stored, "recomputed": actual})
        by[condition][row_id] = {**row, "correct": correct, "normalized_prediction": norm, "malformed": malformed}
    all_ids = sorted(by["A_full"])
    assert all(set(by[c]) == set(all_ids) for c in ("B_query", "C_random"))
    assert set(by["D_oracle"]) <= set(all_ids)
    ids_by_frame = {
        "all": all_ids,
        "answerable_only": [i for i in all_ids if by["A_full"][i]["answerable"]],
        "unanswerable_only": [i for i in all_ids if not by["A_full"][i]["answerable"]],
        "oracle_eligible": sorted(by["D_oracle"]),
    }
    frames = {name: frame(by, ids) for name, ids in ids_by_frame.items() if ids}
    split_scores = {}
    for split in sorted({r["split"] for r in by["A_full"].values()}):
        split_scores[split] = summarize([r["correct"] for r in by["A_full"].values() if r["split"] == split])
    reproduced = {"showcase": split_scores.get("showcase", {}).get("correct", 0),
                  "stability_test": split_scores.get("stability_test", {}).get("correct", 0),
                  "all": sum(r["correct"] for r in by["A_full"].values())}
    tokens = {}
    for condition, observations in by.items():
        values = list(observations.values())
        tokens[condition] = {field: describe([r[field] for r in values]) for field in
                             ("original_context_tokens", "selected_context_tokens", "prompt_tokens", "retention_ratio", "inference_seconds")}
        tokens[condition]["ratio_total_selected_to_original"] = sum(r["selected_context_tokens"] for r in values) / sum(r["original_context_tokens"] for r in values)
    strict_lenient = {}
    for condition, observations in by.items():
        answerable = [r for r in observations.values() if r["answerable"]]
        strict = sum(r["correct"] for r in answerable)
        lenient = sum(normalize(r["gold_answer"]) in r["normalized_prediction"] for r in answerable)
        strict_lenient[condition] = {"n": len(answerable), "strict_correct": strict,
                                    "lenient_correct": lenient, "strict_points": 100 * strict / len(answerable),
                                    "lenient_points": 100 * lenient / len(answerable),
                                    "lenient_minus_strict_points": 100 * (lenient - strict) / len(answerable)}
    stored = json.loads((source.parent / "summary.json").read_text(encoding="utf-8"))
    diffs = compare_subset(stored["frames"], frames, "frames")
    if "reproduced" in stored["baseline_validity"]:
        diffs.extend(compare_subset(stored["baseline_validity"]["reproduced"], reproduced, "baseline_validity.reproduced"))
    analysis_diffs = []
    analysis_path = source.parent / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        mapping = {**frames, "oracle_eligible_answerable": frames["oracle_eligible"]}
        for name, prior in analysis["question_1_abstention_shift"]["frames"].items():
            analysis_diffs.extend(compare_subset(prior, mapping[name], f"analysis.frames.{name}"))
        for task, prior in analysis["question_2_task_families"].items():
            task_ids = [i for i in all_ids if by["A_full"][i]["task_type"] == task]
            task_frame = frame(by, task_ids)
            fresh = {"n": len(task_ids), "accuracy_points": {c: v["accuracy_points"] for c, v in task_frame["headline"].items()},
                     "B_minus_A": {k: v for k, v in task_frame["contrasts"]["B_minus_A"].items() if k != "mcnemar"}}
            analysis_diffs.extend(compare_subset(prior, fresh, f"analysis.tasks.{task}"))
        fresh_retention = {c: round(v["retention_ratio"]["mean"], 4) for c, v in tokens.items()}
        analysis_diffs.extend(compare_subset(analysis["retention_ratio_mean"], fresh_retention, "analysis.retention_ratio_mean"))
    sensitivity = {}
    if stage in ("phase0_eval", "phase0_eval_lcr_dev"):
        for name in ("all", "answerable_only", "oracle_eligible"):
            ids = ids_by_frame[name]
            sensitivity[name] = {label: cluster_sensitivity(by, ids, base, arm)
                                 for label, (base, arm) in CONTRASTS.items()
                                 if label != "C_minus_A" and all(i in by[base] and i in by[arm] for i in ids)}
    task_deltas = {task: {"n": stat["n"], "A_correct": stat["correct"],
                          "B_correct": frames["all"]["by_task_type"]["B_query"][task]["correct"],
                          "B_minus_A_points": 100 * (frames["all"]["by_task_type"]["B_query"][task]["correct"] - stat["correct"]) / stat["n"]}
                   for task, stat in frames["all"]["by_task_type"]["A_full"].items()}
    return {
        "source": str(source.relative_to(REPO)), "source_sha256": digest(source),
        "accepted_answers_source": str(conditions_path.relative_to(REPO)),
        "accepted_answers_source_sha256": digest(conditions_path),
        "n_observations": len(rows), "n_scenarios": len({r["scenario_id"] for r in by["A_full"].values()}),
        "score_mismatches": score_mismatches, "normalization_mismatches": normalization_mismatches,
        "malformed_mismatches": malformed_mismatches, "metadata_mismatches": metadata_mismatches,
        "condition_counts": dict(Counter(r["condition"] for r in rows)),
        "baseline_split_scores": split_scores, "baseline_reproduced": reproduced,
        "expected_document_claim": list(expected) if expected else None,
        "published_claim_count_match": tuple(reproduced.values()) == expected if expected else None,
        "frames": frames, "tokens": tokens,
        "B_to_C_mean_selected_tokens_ratio": tokens["B_query"]["selected_context_tokens"]["mean"] / tokens["C_random"]["selected_context_tokens"]["mean"],
        "strict_vs_lenient_answerable": strict_lenient, "task_deltas": task_deltas,
        "summary_discrepancies": diffs, "analysis_discrepancies": analysis_diffs,
        "cluster_bootstrap_sensitivity": sensitivity,
    }, by


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LAB / "results/audit_20260905/numbers.json")
    args = parser.parse_args()
    normalize, scorer_path = load_normalize()
    reports, data = {}, {}
    for stage, (condition_stage, expected) in STAGES.items():
        print(f"Recomputing {stage}", flush=True)
        reports[stage], data[stage] = stage_audit(stage, condition_stage, expected, normalize)
    amendment = json.loads((LAB / "results/amendment_01_core_gate_comparison.json").read_text(encoding="utf-8"))
    amendment_diffs = []
    for cell in amendment["cells"].values():
        for frame_name, prior in cell["frames"].items():
            fresh = reports[cell["stage"]]["frames"][frame_name]
            projected = {"n": fresh["n_examples"],
                         "accuracy_points": {c: h["accuracy_points"] for c, h in fresh["headline"].items()},
                         "abstention_rate": fresh["abstention_rate"], "contrasts": fresh["contrasts"]}
            amendment_diffs.extend(compare_subset(prior, projected, f"{cell['stage']}.{frame_name}"))
    pairwise = {}
    for first, second, label in [
        ("phase0_eval", "phase0_eval_a1_gated", "checkpoint_gated_a1_minus_a11"),
        ("phase0_eval_a11_ungated", "phase0_eval_a1_ungated", "checkpoint_ungated_a1_minus_a11"),
        ("phase0_eval_a11_ungated", "phase0_eval", "a11_gate_on_minus_off"),
    ]:
        pairwise[label] = {}
        for fname in ("all", "answerable_only", "unanswerable_only", "oracle_eligible"):
            f, s = reports[first]["frames"][fname], reports[second]["frames"][fname]
            pairwise[label][fname] = {
                "accuracy_difference_points": {c: 100 * (s["headline"][c]["correct"] - h["correct"]) / h["n"] for c, h in f["headline"].items()},
                "contrast_difference_points": {c: s["contrasts"][c]["delta_points_from_integer_difference"] - v["delta_points_from_integer_difference"] for c, v in f["contrasts"].items()},
            }
    # Exact interaction from row-level differences (not rounded component deltas).
    means = {}
    for stage in list(STAGES)[:4]:
        a, b = data[stage]["A_full"], data[stage]["B_query"]
        ids = sorted(i for i in a if a[i]["answerable"])
        means[stage] = sum(int(b[i]["correct"]) - int(a[i]["correct"]) for i in ids) / len(ids)
    interactions = {
        "a11_gate_off_minus_on_B_minus_A_answerable_points": 100 * (means["phase0_eval_a11_ungated"] - means["phase0_eval"]),
        "core_by_gate_B_minus_A_answerable_points_a11_minus_a1_gate_on_minus_off": 100 * ((means["phase0_eval"] - means["phase0_eval_a11_ungated"]) - (means["phase0_eval_a1_gated"] - means["phase0_eval_a1_ungated"])),
        "ungated_to_gated_B_minus_A_loss_ratio": means["phase0_eval_a11_ungated"] / means["phase0_eval"],
    }
    output = {
        "audit": "recompute cached Phase 0 observations; no inference, training, CUDA, or summary-derived estimates",
        "command": "../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/audit_result_numbers.py",
        "scorer": str(scorer_path.relative_to(REPO)), "scorer_sha256": digest(scorer_path),
        "statistics_reference": "experiments/supa_long_context_minimal_intervention_v1/src/slcmi/stats.py",
        "statistics_reference_sha256": digest(LAB / "src/slcmi/stats.py"),
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "resampling_unit": "example_id", "order": "sorted example_id"},
        "stages": reports, "amendment_discrepancies": amendment_diffs,
        "pairwise_core_gate": pairwise, "interactions": interactions,
        "limitations": [
            "Recorded token counts are re-aggregated from per-example observations; this script does not retokenize contexts.",
            "Memory peaks, whole-stage wall-clock, model loads and checkpoint preservation cannot be reconstructed from per_example.jsonl; they are not endorsed by this audit.",
            "Row bootstrap preserves condition pairing but treats variants within a scenario as independent; scenario sensitivity is unregistered and descriptive.",
            "Neither repeated cells nor suites isolate length causally; suites differ in task/content/scoring and cells reuse rows.",
            "P-values are uncorrected across comparisons; no significance claims are licensed.",
            "Source summary delta_points can differ by 0.001 at half-rounding boundaries because the source subtracts floating means; integer-difference deltas are also retained.",
            "Strict-versus-lenient scoring tests output containment only and does not independently validate short-target semantic correctness.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary_discrepancies": sum(len(s["summary_discrepancies"]) for s in reports.values()),
                      "analysis_discrepancies": sum(len(s["analysis_discrepancies"]) for s in reports.values()),
                      "amendment_discrepancies": len(amendment_diffs), "interactions": interactions}, indent=2))


if __name__ == "__main__":
    main()
