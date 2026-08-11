from __future__ import annotations

"""Synthesize v2.3.1 temporal/dose/factor evidence without overclaiming grokking."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import (
    critical_dose_trajectory,
    finite,
    progress_step,
    read_json,
    read_jsonl,
    temporal_readout_summary,
    write_json,
)


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
                values.append(
                    {
                        "train_fraction": tf,
                        "ood_fraction": of,
                        "delay_ratio_threshold": ratio,
                        "detected": summary["grokking_like_under_primary_thresholds"],
                    }
                )
    positives = sum(bool(v["detected"]) for v in values)
    flipping = [v for v in values if bool(v["detected"]) != bool(values[0]["detected"])] if values else []
    return {
        "trials": len(values),
        "positive_trials": positives,
        "positive_fraction": positives / len(values) if values else None,
        "fragile": bool(0.0 < (positives / len(values) if values else 0.0) < 1.0),
        "flipping_cells": flipping,
        "details": values,
    }


def dose_summary(surface: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = surface.get("rows") or []
    if not rows:
        return {}
    cfg = cfg or {}
    dest_min = float(cfg.get("readout_destination_dose_minimum", 0.75))
    jump_mult = float(cfg.get("readout_jump_vs_median_multiplier", 2.0))
    mono_min = float(cfg.get("readout_margin_monotonic_pair_fraction_minimum", 0.7))
    final_step = max(int(r["step"]) for r in rows)
    final = sorted([r for r in rows if int(r["step"]) == final_step], key=lambda r: float(r["dose"]))
    jumps = []
    for left, right in zip(final, final[1:]):
        jump = float(right.get("ood_answerable_correctness") or 0.0) - float(left.get("ood_answerable_correctness") or 0.0)
        margin_jump = (finite(right.get("ood_first_target_margin_mean")) or 0.0) - (
            finite(left.get("ood_first_target_margin_mean")) or 0.0
        )
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
        and float(largest["to"]) >= dest_min
        and float(largest["accuracy_jump"]) > max(1e-12, jump_mult * median)
        and (monotonic_fraction is None or monotonic_fraction >= mono_min)
    )
    trajectory = critical_dose_trajectory(
        rows,
        progress_fraction=float(cfg.get("critical_dose_progress_fraction", 0.5)),
        absolute_floor=float(cfg.get("critical_dose_absolute_floor", 0.01)),
        leftward_shift_minimum=float(cfg.get("critical_dose_leftward_shift_minimum", 0.15)),
        fixed_high_minimum=float(cfg.get("fixed_high_critical_dose_minimum", 0.75)),
    )
    return {
        "final_step": final_step,
        "largest_ood_accuracy_jump": largest,
        "median_nonnegative_ood_accuracy_jump": median,
        "ood_logit_margin_monotonic_pair_fraction": monotonic_fraction,
        "readout_threshold_like": threshold_like,
        **trajectory,
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
        "delta_singular_max_at_or_near_ood_transition": nearest.get("delta_singular_max") if nearest else None,
        "transition_nearest_step": nearest.get("step") if nearest else None,
        "final_delta_norm": checkpoints[-1].get("effective_delta_frobenius_norm"),
        "final_delta_effective_rank_mean": checkpoints[-1].get("delta_effective_rank_mean"),
        "final_delta_singular_max": checkpoints[-1].get("delta_singular_max"),
    }


def optimizer_summary(path: Path | None, transition_step: int | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    rows = read_jsonl(path)
    sampled = [r for r in rows if finite(r.get("slow_to_total_gradient_norm_ratio")) is not None]
    if not sampled:
        return {"rows": len(rows), "gradient_samples": 0}
    before = [
        float(r["slow_to_total_gradient_norm_ratio"])
        for r in sampled
        if transition_step is None or int(r.get("step") or 0) < transition_step
    ]
    after = [
        float(r["slow_to_total_gradient_norm_ratio"])
        for r in sampled
        if transition_step is not None and int(r.get("step") or 0) >= transition_step
    ]
    return {
        "rows": len(rows),
        "gradient_samples": len(sampled),
        "slow_to_total_ratio_before_transition_mean": statistics.mean(before) if before else None,
        "slow_to_total_ratio_after_transition_mean": statistics.mean(after) if after else None,
        "gradient_cosine_ema_final": sampled[-1].get("gradient_cosine_ema"),
    }


def competing_evidence(
    *,
    label: str,
    temporal: dict[str, Any],
    dose: dict[str, Any],
    temporal_readout: dict[str, Any],
    representation: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    mem_train = float(cfg.get("memorization_minimum_train_gain", 0.1))
    ood_floor = float(cfg["minimum_observed_ood_gain_for_grokking_like"])
    flags = {
        "critical_dose_migrates_left": bool(dose.get("critical_dose_migrates_left")),
        "fixed_high_critical_dose": bool(dose.get("fixed_high_critical_dose")),
        "temporal_readout_like": bool(temporal_readout.get("smooth_logits_with_sharp_accuracy")),
        "representation_precedes_ood": bool(representation.get("representation_precedes_ood_transition")),
        "dose_readout_like": bool(dose.get("readout_threshold_like")),
        "grokking_like_temporal": bool(temporal.get("grokking_like_under_primary_thresholds")),
        "memorization_like": bool(
            float(temporal.get("train_observed_gain") or 0.0) >= mem_train
            and float(temporal.get("ood_observed_gain") or 0.0) < ood_floor
        ),
        "ordinary_learning_like": bool(
            float(temporal.get("train_observed_gain") or 0.0) > 0.0
            and float(temporal.get("ood_observed_gain") or 0.0) > 0.0
            and not temporal.get("grokking_like_under_primary_thresholds")
        ),
    }
    pattern_names = {
        "critical_dose_migrates_left": "LEFTWARD_CRITICAL_DOSE_MIGRATION",
        "fixed_high_critical_dose": "FIXED_HIGH_CRITICAL_DOSE",
        "temporal_readout_like": "TEMPORAL_READOUT_THRESHOLD",
        "representation_precedes_ood": "LATENT_BEFORE_READOUT",
        "dose_readout_like": "FINAL_STEP_DOSE_READOUT",
        "grokking_like_temporal": "GROKKING_LIKE_TEMPORAL",
        "memorization_like": "MEMORIZATION_WITHOUT_ABSTRACTION",
        "ordinary_learning_like": "ORDINARY_FEATURE_LEARNING",
    }
    label_support = {
        "DELAYED_GENERALIZATION_REPRODUCED": {
            "grokking_like_temporal",
            "representation_precedes_ood",
            "critical_dose_migrates_left",
        },
        "GROKKING_LIKE_TRANSITION_DETECTED": {
            "grokking_like_temporal",
            "representation_precedes_ood",
            "critical_dose_migrates_left",
        },
        "MEMORIZATION_WITHOUT_ABSTRACTION": {"memorization_like"},
        "READOUT_THRESHOLD_LIKELY": {"dose_readout_like", "temporal_readout_like", "fixed_high_critical_dose"},
        "ORDINARY_FEATURE_LEARNING": {"ordinary_learning_like"},
        "INSUFFICIENT_EVIDENCE": set(),
    }
    support_keys = label_support.get(label, set())
    supporting: list[str] = []
    contradicting: list[str] = []
    for key, active in flags.items():
        if not active:
            continue
        name = pattern_names[key]
        if key in support_keys:
            supporting.append(name)
        else:
            contradicting.append(name)
    return {
        "flags": flags,
        "supporting_patterns": supporting,
        "contradicting_patterns": contradicting,
        "primary_label": label,
        "note": (
            "Primary diagnostic_label follows the conservative temporal-first cascade; "
            "supporting/contradicting patterns surface competing mechanisms that the cascade may suppress."
        ),
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
    dose = dose_summary(read_json(args.dose_surface), transition_cfg)
    temporal_readout = temporal_readout_summary(
        time_rows,
        accuracy_jump_fraction_minimum=float(transition_cfg.get("temporal_readout_accuracy_jump_fraction_minimum", 0.5)),
        margin_linear_r2_minimum=float(transition_cfg.get("temporal_readout_margin_linear_r2_minimum", 0.7)),
        nll_linear_r2_minimum=float(transition_cfg.get("temporal_readout_nll_linear_r2_minimum", 0.7)),
    )
    factors: list[dict[str, Any]] = []
    reproduced = 0
    if args.factor_manifest and args.factor_manifest.exists():
        factors, reproduced = factor_summaries(read_json(args.factor_manifest), transition_cfg)
    representation = representation_summary(args.representation, temporal)
    adapter = adapter_summary(args.adapter_dynamics, temporal)
    optimizer = optimizer_summary(args.optimizer_dynamics, temporal.get("ood_progress_step"))

    mem_train = float(transition_cfg.get("memorization_minimum_train_gain", 0.1))
    if temporal["grokking_like_under_primary_thresholds"] and reproduced >= int(transition_cfg["reproduced_seed_count"]):
        label = "DELAYED_GENERALIZATION_REPRODUCED"
    elif temporal["grokking_like_under_primary_thresholds"]:
        label = "GROKKING_LIKE_TRANSITION_DETECTED"
    elif temporal["train_observed_gain"] >= mem_train and temporal["ood_observed_gain"] < float(
        transition_cfg["minimum_observed_ood_gain_for_grokking_like"]
    ):
        label = "MEMORIZATION_WITHOUT_ABSTRACTION"
    elif dose.get("readout_threshold_like") or temporal_readout.get("smooth_logits_with_sharp_accuracy"):
        label = "READOUT_THRESHOLD_LIKELY"
    elif temporal["train_observed_gain"] > 0 and temporal["ood_observed_gain"] > 0:
        label = "ORDINARY_FEATURE_LEARNING"
    else:
        label = "INSUFFICIENT_EVIDENCE"

    evidence = competing_evidence(
        label=label,
        temporal=temporal,
        dose=dose,
        temporal_readout=temporal_readout,
        representation=representation,
        cfg=transition_cfg,
    )

    payload = {
        "schema_version": "2.3.1",
        "diagnostic_label": label,
        "claim_policy": cfg["experiment"]["claim_policy"],
        "sealed_data_used": False,
        "preregistered_thresholds": {
            "train_progress_fraction": transition_cfg["train_progress_fraction"],
            "ood_progress_fraction": transition_cfg["ood_progress_fraction"],
            "minimum_plateau_intervals": transition_cfg["minimum_plateau_intervals"],
            "minimum_observed_ood_gain_for_grokking_like": transition_cfg["minimum_observed_ood_gain_for_grokking_like"],
            "minimum_delay_ratio_for_grokking_like": transition_cfg["minimum_delay_ratio_for_grokking_like"],
            "reproduced_seed_count": transition_cfg["reproduced_seed_count"],
            "memorization_minimum_train_gain": mem_train,
            "readout_destination_dose_minimum": transition_cfg.get("readout_destination_dose_minimum", 0.75),
            "critical_dose_progress_fraction": transition_cfg.get("critical_dose_progress_fraction", 0.5),
            "critical_dose_absolute_floor": transition_cfg.get("critical_dose_absolute_floor", 0.01),
            "temporal_readout_accuracy_jump_fraction_minimum": transition_cfg.get(
                "temporal_readout_accuracy_jump_fraction_minimum", 0.5
            ),
        },
        "temporal": temporal,
        "temporal_readout": temporal_readout,
        "sensitivity": sens,
        "dose_surface": dose,
        "factor_phase_diagram": factors,
        "reproduced_grokking_like_seed_count": reproduced,
        "representation": representation,
        "adapter_dynamics": adapter,
        "optimizer_dynamics": optimizer,
        "competing_evidence": evidence,
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
        "## Temporal readout evidence",
        "",
        f"- Smooth logits with sharp accuracy: `{temporal_readout.get('smooth_logits_with_sharp_accuracy')}`",
        f"- Largest OOD accuracy jump fraction of gain: `{temporal_readout.get('largest_ood_accuracy_jump_fraction_of_gain')}`",
        f"- OOD margin linear R²: `{temporal_readout.get('ood_margin_linear_r2')}`",
        f"- OOD NLL linear R²: `{temporal_readout.get('ood_nll_linear_r2')}`",
        "",
        "## Dose / critical-dose evidence",
        "",
        f"- Readout-threshold-like (final step): `{dose.get('readout_threshold_like')}`",
        f"- Critical dose start → end: `{dose.get('critical_dose_start')}` → `{dose.get('critical_dose_end')}`",
        f"- Critical-dose leftward shift: `{dose.get('critical_dose_leftward_shift')}`",
        f"- Critical dose migrates left: `{dose.get('critical_dose_migrates_left')}`",
        f"- Fixed high critical dose: `{dose.get('fixed_high_critical_dose')}`",
        f"- Largest OOD accuracy jump: `{dose.get('largest_ood_accuracy_jump')}`",
        f"- OOD logit-margin monotonic pair fraction: `{dose.get('ood_logit_margin_monotonic_pair_fraction')}`",
        "",
        "## Internal dynamics",
        "",
        f"- Representation precedes OOD transition: `{representation.get('representation_precedes_ood_transition')}`",
        f"- Delta norm near transition: `{adapter.get('delta_norm_at_or_near_ood_transition')}`",
        f"- Relative delta norm near transition: `{adapter.get('relative_delta_norm_at_or_near_ood_transition')}`",
        f"- Delta singular max near transition: `{adapter.get('delta_singular_max_at_or_near_ood_transition')}`",
        f"- Slow/total gradient ratio before: `{optimizer.get('slow_to_total_ratio_before_transition_mean')}`",
        f"- Slow/total gradient ratio after: `{optimizer.get('slow_to_total_ratio_after_transition_mean')}`",
        "",
        "## Competing evidence",
        "",
        f"- Supporting patterns: `{evidence.get('supporting_patterns')}`",
        f"- Contradicting patterns: `{evidence.get('contradicting_patterns')}`",
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
