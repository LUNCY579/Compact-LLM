from __future__ import annotations

"""Compile the complete v2.3.1 Grokking Dynamics Observatory report."""

import argparse
import json
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import read_json, write_json


def maybe(path: Path | None) -> dict[str, Any]:
    return read_json(path) if path is not None and path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-manifest", type=Path, required=True)
    ap.add_argument("--time-curve", type=Path, required=True)
    ap.add_argument("--dose-surface", type=Path, required=True)
    ap.add_argument("--transition", type=Path, required=True)
    ap.add_argument("--adapter-dynamics", type=Path)
    ap.add_argument("--representation", type=Path)
    ap.add_argument("--factor-phase", type=Path)
    ap.add_argument("--v23-mining-audit", type=Path)
    ap.add_argument("--procedure-control-curve", type=Path)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()
    cfg = read_json(a.config)
    data = read_json(a.data_manifest)
    curve = read_json(a.time_curve)
    surface = read_json(a.dose_surface)
    transition = read_json(a.transition)
    adapter = maybe(a.adapter_dynamics)
    representation = maybe(a.representation)
    factors = maybe(a.factor_phase)
    mining = maybe(a.v23_mining_audit)
    procedure_control = maybe(a.procedure_control_curve)

    payload = {
        "schema_version": "2.3.1",
        "experiment": cfg["experiment"],
        "sealed_data_used": False,
        "diagnostic_label": transition.get("diagnostic_label"),
        "data": {
            "source_counts": data.get("source_counts"),
            "variant_count": len(data.get("variants") or {}),
            "probe_manifest": data.get("probes"),
        },
        "v23_teacher_ceiling_context": mining,
        "temporal": transition.get("temporal"),
        "dose_surface": transition.get("dose_surface"),
        "sensitivity": transition.get("sensitivity"),
        "adapter_dynamics": transition.get("adapter_dynamics"),
        "optimizer_dynamics": transition.get("optimizer_dynamics"),
        "representation": transition.get("representation"),
        "factor_phase_diagram": factors,
        "procedure_long_horizon_control": procedure_control,
        "artifact_paths": {
            "training_time_curve": str(a.time_curve),
            "dose_time_surface": str(a.dose_surface),
            "transition_detection": str(a.transition),
            "adapter_dynamics": str(a.adapter_dynamics) if a.adapter_dynamics else None,
            "representation_dynamics": str(a.representation) if a.representation else None,
            "factor_phase_diagram": str(a.factor_phase) if a.factor_phase else None,
        },
    }
    write_json(a.output_json, payload)

    temporal = transition.get("temporal") or {}
    dose = transition.get("dose_surface") or {}
    rep = transition.get("representation") or {}
    opt = transition.get("optimizer_dynamics") or {}
    lines = [
        "# v2.3.1 Grokking Dynamics Observatory",
        "",
        f"**Diagnostic label:** `{transition.get('diagnostic_label')}`",
        "",
        "v2.3.1 is a dynamics experiment, not a release-candidate selection phase. It deliberately uses no consumed or fresh sealed data.",
        "",
        "## What would count as grokking here?",
        "",
        "A capability-dose jump alone is insufficient. The experiment looks for training fit occurring materially earlier than held-out/OOD generalization, a measurable plateau under continued optimization with no new information, and later OOD improvement. Reproduction across seeds strengthens the claim.",
        "",
        "## Core temporal result",
        "",
        f"- Train observed gain: `{temporal.get('train_observed_gain')}`",
        f"- DEV observed gain: `{temporal.get('dev_observed_gain')}`",
        f"- OOD observed gain: `{temporal.get('ood_observed_gain')}`",
        f"- Train progress step: `{temporal.get('train_progress_step')}`",
        f"- OOD progress step: `{temporal.get('ood_progress_step')}`",
        f"- OOD/train delay ratio: `{temporal.get('ood_to_train_delay_ratio')}`",
        f"- Plateau intervals: `{temporal.get('plateau_intervals')}`",
        f"- Primary grokking-like criterion: `{temporal.get('grokking_like_under_primary_thresholds')}`",
        f"- Threshold sensitivity positive fraction: `{(transition.get('sensitivity') or {}).get('positive_fraction')}`",
        "",
        "## Capability-dose / readout result",
        "",
        f"- Readout-threshold-like pattern: `{dose.get('readout_threshold_like')}`",
        f"- Largest OOD accuracy jump: `{dose.get('largest_ood_accuracy_jump')}`",
        f"- Logit-margin monotonicity: `{dose.get('ood_logit_margin_monotonic_pair_fraction')}`",
        "",
        "## Internal representation and optimization",
        "",
        f"- Representation progress precedes OOD transition: `{rep.get('representation_precedes_ood_transition')}`",
        f"- Cross-format representation progress step: `{rep.get('representation_progress_step_50pct')}`",
        f"- Relative residual-delta norm near OOD transition: `{(transition.get('adapter_dynamics') or {}).get('relative_delta_norm_at_or_near_ood_transition')}`",
        f"- Slow/total gradient ratio before transition: `{opt.get('slow_to_total_ratio_before_transition_mean')}`",
        f"- Slow/total gradient ratio after transition: `{opt.get('slow_to_total_ratio_after_transition_mean')}`",
        "",
        "## Factor phase diagram",
        "",
        f"One-factor runs recorded: `{len((factors.get('rows') or [])) if factors else 0}`",
        "",
        "The screened axes include training time, data quantity, unique-case count versus repeated exposure, residual/preservation ratio, weight decay, KL pressure, learning rate, LoRA rank, seed, attention-vs-MLP target modules, and measured layer scope. Extended profile additionally screens gradient accumulation, warmup ratio, and anchor temperature.",
        "",
        "## v2.3 teacher-ceiling context",
        "",
        f"- Direct oracle-union accuracy from v2.3 mining: `{((mining.get('direct') or {}).get('oracle_union_accuracy')) if mining else None}`",
        f"- Direct unresolved anchor failures: `{((mining.get('direct') or {}).get('unresolved_anchor_failures')) if mining else None}`",
        "",
        "## Interpretation policy",
        "",
        "- `READOUT_THRESHOLD_LIKELY`: dose/logit behavior is more consistent with crossing output boundaries than delayed temporal abstraction.",
        "- `ORDINARY_FEATURE_LEARNING`: train and OOD improve without a strong delayed plateau.",
        "- `MEMORIZATION_WITHOUT_ABSTRACTION`: train improves materially while structural OOD remains largely flat.",
        "- `GROKKING_LIKE_TRANSITION_DETECTED`: a delayed OOD transition satisfies the preregistered temporal criteria in the core run.",
        "- `DELAYED_GENERALIZATION_REPRODUCED`: the core delayed transition is also reproduced across the configured number of seed controls.",
        "- `INSUFFICIENT_EVIDENCE`: the run does not cleanly support the other labels.",
        "",
        "No label in this report changes the v2.2/v2.3 production anchor automatically.",
        "",
    ]
    a.output_md.parent.mkdir(parents=True, exist_ok=True)
    a.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
