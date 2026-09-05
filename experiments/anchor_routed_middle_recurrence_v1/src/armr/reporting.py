"""Frozen gates, CSV atlas, and bounded scientific interpretation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .provenance import write_json


def mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict], baseline: list[dict], config: dict) -> dict:
    ratios = [h["rms_ratio_to_normal"] for r in rows for h in r["hidden"]]
    ratios.extend(r[k] for r in rows for k in ("trajectory_min_rms_ratio", "trajectory_max_rms_ratio"))
    invalid = mean([int(r["invalid"]) for r in rows])
    base_invalid = mean([int(r["invalid"]) for r in baseline])
    result = {
        "n": len(rows), "mean_logit_cosine": mean([r["logit_cosine"] for r in rows]),
        "mean_kl": mean([r["kl_base_to_arm"] for r in rows]),
        "top1_agreement": mean([r["top1_agreement"] for r in rows]),
        "topk_overlap": mean([r["topk_overlap"] for r in rows]),
        "score": mean([r["score"] for r in rows]),
        "score_delta": mean([r["score"] for r in rows]) - mean([r["score"] for r in baseline]),
        "format_score": mean([r["format_score"] for r in rows if r["format_score"] is not None]),
        "invalid_rate": invalid, "invalid_increase": invalid - base_invalid,
        "safety_stops": sum(r["safety_stopped"] for r in rows),
        "eos_rate": mean([r["eos_emitted"] for r in rows]),
        "min_rms_ratio": min(ratios, default=1.0), "max_rms_ratio": max(ratios, default=1.0),
        "mean_wall_seconds": mean([r["wall_seconds"] for r in rows]),
    }
    lo, hi = config["hidden_rms_ratio_bounds"]
    checks = {
        "complete": bool(rows) and len(rows) == len(baseline),
        "no_safety_stops": result["safety_stops"] == 0,
        "stable_norms": result["min_rms_ratio"] >= lo and result["max_rms_ratio"] <= hi,
        "logit_preservation": result["mean_logit_cosine"] >= config["minimum_mean_logit_cosine"],
        "kl_preservation": result["mean_kl"] <= config["maximum_mean_kl"],
        "score_preservation": result["score_delta"] >= -config["maximum_score_drop"],
        "validity_preservation": result["invalid_increase"] <= config["maximum_invalid_increase"],
    }
    result.update(gate_checks=checks, go_fixed_smoke=all(checks.values()))
    return result


def csv_file(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def screen_report(out: Path, results: list[dict], history: dict, config: dict) -> dict:
    atlas = []
    hist = {r["layer"]: r for r in history["ranked"]}
    for r in results:
        layer = r["layer"]
        skip = hist[layer]["mean_logit_cosine_vs_intact"]
        removal = "removable" if skip >= .9 else "moderate"
        repeat = "safe_smoke" if r["go_fixed_smoke"] else "fails_gate"
        if r["safety_stops"]:
            repeat = "corrupt_or_looping"
        elif r["min_rms_ratio"] < .25 or r["max_rms_ratio"] > 4:
            repeat = "unstable_norms"
        atlas.append({**{k: v for k, v in r.items() if k != "gate_checks"},
                      "skip_cosine": skip, "removal_class": removal, "repeat_class": repeat,
                      "benefit_evidence": "smoke_only_not_efficacy"})
    atlas.sort(key=lambda r: (not r["go_fixed_smoke"], -r["mean_logit_cosine"], r["layer"], r["applications"]))
    passing = [r for r in atlas if r["go_fixed_smoke"] and r["applications"] == 2 and r["layer"] < 25]
    decision = {"status": "PASS" if passing else "FAIL", "fixed_recurrence": bool(passing),
                "selected_layer": passing[0]["layer"] if passing else None,
                "adaptive_routing": False, "peft_recovery": False, "sealed_evaluation": False,
                "reason": "smoke gate only; fixed development qualification required" if passing else "no middle repeat-2 arm met the frozen smoke gate"}
    write_json(out / "atlas.json", atlas)
    write_json(out / "decision.json", decision)
    csv_file(out / "summary.csv", atlas)
    lines = ["# Necessity × repeatability atlas", "", "Sorted by smoke gate then logit preservation. Benefit is not established.", "",
             "| Layer | Apps | Skip cos | Repeat cos | KL | Score Δ | Invalid | RMS range | Gate |",
             "|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in atlas:
        lines.append(f"| {r['layer']} | {r['applications']} | {r['skip_cosine']:.4f} | {r['mean_logit_cosine']:.4f} | {r['mean_kl']:.4f} | {r['score_delta']:+.3f} | {r['invalid_rate']:.3f} | {r['min_rms_ratio']:.3f}–{r['max_rms_ratio']:.3f} | {'GO' if r['go_fixed_smoke'] else 'NO-GO'} |")
    (out / "ranking.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    report = f"""# Middle reapplication screen

Status: {decision['status']} for the fixed-recurrence mechanical gate.

Question: which pretrained middle operators tolerate one/two extra applications
at their native site? Setup: Supa-A1.1, CPU FP32 eager, seed 17, 4 threads,
eight historical smoke prompts, frozen configuration; exact hashes/hardware and
commit are in run_manifest.json. Lower/upper layers and abstention head retained.

Mechanical qualification: passing matching baseline receipt required; no-cache
entry guard, exact argument reuse, finite hidden/logit checks, frozen parameter
hash and checkpoint-file comparison, exact restoration at stage end.

Results: see ranking.md, atlas.json, summary.csv, hidden_state_metrics.csv,
logit_metrics.csv and raw_results.jsonl. Selected layer: {decision['selected_layer']}.
Each row is a separately bounded case; stopped cases are recorded and not retried.

Interpretation: {decision['reason']}. Stable norms, low divergence, and task
scores are separate columns. A removable layer is not assumed reusable.

Non-claims: no reasoning/capability improvement, learned routing, training,
long-context, cached recurrence, or sealed efficacy evidence. Small prompt count
and multiple comparisons preclude efficacy selection claims; task changes are
descriptive only. RMS ratios/rho are not proofs of convergence or contraction.

Next gate: {'fixed recurrence on unsealed MERGEDEV-fast' if passing else 'STOP direct recurrence; document the negative screen'}. Adaptive routing, PEFT and sealed evaluation remain closed.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    return decision
