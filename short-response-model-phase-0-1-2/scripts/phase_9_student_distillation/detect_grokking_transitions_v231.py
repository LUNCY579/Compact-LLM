from __future__ import annotations

"""Synthesize v2.3.1 temporal/dose/factor evidence without overclaiming grokking."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import finite, progress_step, read_json, read_jsonl, write_json


def temporal_summary(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: int(r["step"]))
    train = [(int(r["step"]), float(r.get("train_answerable_correctness") or 0.0)) for r in rows]
    dev = [(int(r["step"]), float(r.get("dev_answerable_correctness") or 0.0)) for r in rows]
    ood = [(int(r["step"]), float(r.get("ood_answerable_correctness") or 0.0)) for r in rows]
    tfrac = float(cfg["train_progress_fraction"])
    ofrac = float(cfg["ood_progress_fraction"])
    train_step = progress_step(train, tfrac)
    dev_step = progress_step(dev, ofrac)
    ood_step = progress_step(ood, ofrac)
    delay_ratio = (ood_step / train_step) if train_step and ood_step and train_step > 0 else None
    interval_steps = sorted({step for step, _ in train})
    plateau_intervals = 0
    if train_step is not None and ood_step is not None:
        plateau_intervals = sum(train_step <= step < ood_step for step in interval_steps)
    train_gain = max(v for _, v in train) - train[0][1] if train else 0.0
    dev_gain = max(v for _, v in dev) - dev[0][1] if dev else 0.0
    ood_gain = max(v for _, v in ood) - ood[0][1] if ood else 0.0
    grokking_like = bool(
        train_step is not None
        and ood_step is not None
        and delay_ratio is not None
        and delay_ratio >= float(cfg["minimum_delay_ratio_for_grokking_like"])
        and plateau_intervals >= int(cfg["minimum_plateau_intervals"])
        and ood_gain >= float(cfg["minimum_observed_ood_gain_for_grokking_like"])
    )
    return {
        "train_progress_step": train_step,
        "dev_progress_step": dev_step,
        "ood_progress_step": ood_step,
        "ood_to_train_delay_ratio": delay_ratio,
        "plateau_intervals": plateau_intervals,
        "train_observed_gain": train_gain,
        "dev_observed_gain": dev_gain,
        "ood_observed_gain": ood_gain,
        "grokking_like_under_primary_thresholds": grokking_like,
        "final_train_answerable": train[-1][1] if train else None,
        "final_dev_answerable": dev[-1][1] if dev else None,
        "final_ood_answerable": ood[-1][1] if ood else None,
        "max_step": train[-1][0] if train else None,
    }


def sensitivity(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    values = []
    base = dict(cfg)
    for tf in cfg["sensitivity"]["train_progress_fraction"]:
        for of in cfg["sensitivity"]["ood_progress_fraction"]:
            for ratio in cfg["sensitivity"]["minimum_delay_ratio"]:
                local = dict(base)
                local["train_progress_fraction"] = tf
                local["ood_progress_fraction"] = of
                local["minimum_delay_ratio_for_grokking_like"] = ratio
                summary = temporal_summary(rows, local)
                values.append({"train_fraction": tf, "ood_fraction": of, "delay_ratio_threshold": ratio, "detected": summary["grokking_like_under_primary_thresholds"]})
    positives = sum(bool(v["detected"]) for v in values)
    return {"trials": len(values), "positive_trials": positives, "positive_fraction": positives / len(values) if values else None, "details": values}


def dose_summary(surface: dict[str, Any]) -> dict[str, Any]:
    rows = surface.get("rows") or []
    if not rows:
        return {}
    final_step = max(int(r["step"]) for r in rows)
    final = sorted([r for r in rows if int(r["step"]) == final_step], key=lambda r: float(r["dose"]))
    jumps = []
    for left, right in zip(final, final[1:]):
        jump = float(right.get("ood_answerable_correctness") or 0.0) - float(left.get("ood_answerable_correctness") or 0.0)
        margin_jump = (finite(right.get("ood_first_target_margin_mean")) or 0.0) - (finite(left.get("ood_first_target_margin_mean")) or 0.0)
        jumps.append({"from": left["dose"], "to": right["dose"], "accuracy_jump": jump, "margin_jump": margin_jump})
    positive = [max(0.0, float(j["accuracy_jump"])) for j in jumps]
    median = statistics.median(positive) if positive else 0.0
    largest = max(jumps, key=lambda j: j["accuracy_jump"]) if jumps else None
    margins = [finite(r.get("ood_first_target_margin_mean")) for r in final]
    valid = [v for v in margins if v is not None]
    monotonic_pairs = 0
    total_pairs = 0
    for left, right in zip(valid, valid[1:]):
        total_pairs += 1
        monotonic_pairs += right >= left
    monotonic_fraction = monotonic_pairs / total_pairs if total_pairs else None
    threshold_like = bool(
        largest
        and float(largest["to"]) >= 0.75
        and float(largest["accuracy_jump"]) > max(1e-12, 2.0 * median)
        and (monotonic_fraction is None or monotonic_fraction >= 0.7)
    )
    return {
        "final_step": final_step,
        "largest_ood_accuracy_jump": largest,
        "median_nonnegative_ood_accuracy_jump": median,
        "ood_logit_margin_monotonic_pair_fraction": monotonic_fraction,
        "readout_threshold_like": threshold_like,
    }


def factor_summaries(manifest: dict[str, Any], transition_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    records = []
    reproduced_seed_count = 0
    for record in manifest.get("records") or []:
        path = Path(str(record.get("time_curve") or ""))
        if not path.exists():
            continue
        rows = read_json(path).get("rows") or []
        summary = temporal_summary(rows, transition_cfg)
        item = {"axis": record.get("axis"), "value": record.get("value"), **summary}
        records.append(item)
        if item["axis"] == "seed" and summary["grokking_like_under_primary_thresholds"]:
            reproduced_seed_count += 1
    return records, reproduced_seed_count


def representation_summary(path: Path | None, temporal: dict[str, Any]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    records = payload.get("records") or []
    if not records:
        return {}
    by_step: dict[int, list[float]] = {}
    for row in records:
        value = finite(row.get("cross_format_nearest_centroid_reasoning_accuracy"))
        if value is not None:
            by_step.setdefault(int(row["step"]), []).append(value)
    curve = [(step, statistics.mean(values)) for step, values in sorted(by_step.items()) if values]
    rep50 = progress_step(curve, 0.5)
    ood_step = temporal.get("ood_progress_step")
    return {
        "cross_format_centroid_curve": [{"step": step, "accuracy": value} for step, value in curve],
        "representation_progress_step_50pct": rep50,
        "representation_precedes_ood_transition": bool(rep50 is not None and ood_step is not None and rep50 < ood_step),
    }


def adapter_summary(path: Path | None, temporal: dict[str, Any]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    checkpoints = read_json(path).get("checkpoints") or []
    if not checkpoints:
        return {}
    transition = temporal.get("ood_progress_step")
    nearest = None
    if transition is not None:
        nearest = min(checkpoints, key=lambda row: abs(int(row["step"]) - int(transition)))
    return {
        "checkpoint_count": len(checkpoints),
        "delta_norm_at_or_near_ood_transition": nearest.get("effective_delta_frobenius_norm") if nearest else None,
        "relative_delta_norm_at_or_near_ood_transition": nearest.get("relative_delta_norm_to_anchor_target_weights") if nearest else None,
        "transition_nearest_step": nearest.get("step") if nearest else None,
        "final_delta_norm": checkpoints[-1].get("effective_delta_frobenius_norm"),
        "final_delta_effective_rank_mean": checkpoints[-1].get("delta_effective_rank_mean"),
    }


def optimizer_summary(path: Path | None, transition_step: int | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    rows = read_jsonl(path)
    sampled = [r for r in rows if finite(r.get("slow_to_total_gradient_norm_ratio")) is not None]
    if not sampled:
        return {"rows": len(rows), "gradient_samples": 0}
    before = [float(r["slow_to_total_gradient_norm_ratio"]) for r in sampled if transition_step is None or int(r.get("step") or 0) < transition_step]
    after = [float(r["slow_to_total_gradient_norm_ratio"]) for r in sampled if transition_step is not None and int(r.get("step") or 0) >= transition_step]
    return {
        "rows": len(rows),
        "gradient_samples": len(sampled),
        "slow_to_total_ratio_before_transition_mean": statistics.mean(before) if before else None,
        "slow_to_total_ratio_after_transition_mean": statistics.mean(after) if after else None,
        "gradient_cosine_ema_final": sampled[-1].get("gradient_cosine_ema"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--time-curve", type=Path, required=True)
    ap.add_argument("--dose-surface", type=Path, required=True)
    ap.add_argument("--factor-manifest", type=Path)
    ap.add_argument("--representation", type=Path)
    ap.add_argument("--adapter-dynamics", type=Path)
    ap.add_argument("--optimizer-dynamics", type=Path)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()
    cfg = read_json(args.config)
    transition_cfg = cfg["transition_detection"]
    time_rows = read_json(args.time_curve).get("rows") or []
    temporal = temporal_summary(time_rows, transition_cfg)
    sens = sensitivity(time_rows, transition_cfg)
    dose = dose_summary(read_json(args.dose_surface))
    factors: list[dict[str, Any]] = []
    reproduced = 0
    if args.factor_manifest and args.factor_manifest.exists():
        factors, reproduced = factor_summaries(read_json(args.factor_manifest), transition_cfg)
    representation = representation_summary(args.representation, temporal)
    adapter = adapter_summary(args.adapter_dynamics, temporal)
    optimizer = optimizer_summary(args.optimizer_dynamics, temporal.get("ood_progress_step"))

    if temporal["grokking_like_under_primary_thresholds"] and reproduced >= int(transition_cfg["reproduced_seed_count"]):
        label = "DELAYED_GENERALIZATION_REPRODUCED"
    elif temporal["grokking_like_under_primary_thresholds"]:
        label = "GROKKING_LIKE_TRANSITION_DETECTED"
    elif temporal["train_observed_gain"] >= 0.1 and temporal["ood_observed_gain"] < float(transition_cfg["minimum_observed_ood_gain_for_grokking_like"]):
        label = "MEMORIZATION_WITHOUT_ABSTRACTION"
    elif dose.get("readout_threshold_like"):
        label = "READOUT_THRESHOLD_LIKELY"
    elif temporal["train_observed_gain"] > 0 and temporal["ood_observed_gain"] > 0:
        label = "ORDINARY_FEATURE_LEARNING"
    else:
        label = "INSUFFICIENT_EVIDENCE"

    payload = {
        "schema_version": "2.3.1",
        "diagnostic_label": label,
        "claim_policy": cfg["experiment"]["claim_policy"],
        "sealed_data_used": False,
        "temporal": temporal,
        "sensitivity": sens,
        "dose_surface": dose,
        "factor_phase_diagram": factors,
        "reproduced_grokking_like_seed_count": reproduced,
        "representation": representation,
        "adapter_dynamics": adapter,
        "optimizer_dynamics": optimizer,
    }
    write_json(args.output_json, payload)
    lines = [
        "# v2.3.1 Grokking / Phase-Transition Detection",
        "",
        f"**Diagnostic label:** `{label}`",
        "",
        "This is a research diagnostic, not a release gate. A dose jump alone is never called grokking.",
        "",
        "## Temporal evidence",
        "",
        f"- Train progress step: `{temporal.get('train_progress_step')}`",
        f"- OOD progress step: `{temporal.get('ood_progress_step')}`",
        f"- OOD/train delay ratio: `{temporal.get('ood_to_train_delay_ratio')}`",
        f"- Plateau intervals: `{temporal.get('plateau_intervals')}`",
        f"- Train observed gain: `{temporal.get('train_observed_gain')}`",
        f"- OOD observed gain: `{temporal.get('ood_observed_gain')}`",
        f"- Primary grokking-like criterion: `{temporal.get('grokking_like_under_primary_thresholds')}`",
        f"- Sensitivity positive fraction: `{sens.get('positive_fraction')}`",
        "",
        "## Dose/readout evidence",
        "",
        f"- Readout-threshold-like: `{dose.get('readout_threshold_like')}`",
        f"- Largest OOD accuracy jump: `{dose.get('largest_ood_accuracy_jump')}`",
        f"- OOD logit-margin monotonic pair fraction: `{dose.get('ood_logit_margin_monotonic_pair_fraction')}`",
        "",
        "## Internal dynamics",
        "",
        f"- Representation precedes OOD transition: `{representation.get('representation_precedes_ood_transition')}`",
        f"- Delta norm near transition: `{adapter.get('delta_norm_at_or_near_ood_transition')}`",
        f"- Relative delta norm near transition: `{adapter.get('relative_delta_norm_at_or_near_ood_transition')}`",
        f"- Slow/total gradient ratio before: `{optimizer.get('slow_to_total_ratio_before_transition_mean')}`",
        f"- Slow/total gradient ratio after: `{optimizer.get('slow_to_total_ratio_after_transition_mean')}`",
        "",
        "## Reproducibility",
        "",
        f"- Grokking-like seed replicates: `{reproduced}`",
        "",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
