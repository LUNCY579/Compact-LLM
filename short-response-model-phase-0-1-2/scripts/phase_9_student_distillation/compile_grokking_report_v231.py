from __future__ import annotations

"""Compile the complete v2.3.1 Grokking Dynamics Observatory scientific report."""

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


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return lines


PATTERN_CATALOG = [
    {
        "id": "READOUT_THRESHOLD",
        "name": "Readout / logit threshold",
        "signature": (
            "Discrete accuracy jumps sharply while continuous gold-token margins/NLL move smoothly, "
            "and/or capability appears only near high residual dose with little leftward critical-dose migration."
        ),
        "confirming": [
            "temporal_readout.smooth_logits_with_sharp_accuracy",
            "dose_surface.readout_threshold_like",
            "dose_surface.fixed_high_critical_dose",
            "dose_surface.ood_logit_margin_monotonic_pair_fraction",
        ],
        "falsifiers": [
            "dose_surface.critical_dose_migrates_left",
            "temporal.grokking_like_under_primary_thresholds with smooth joint train/OOD rise",
        ],
    },
    {
        "id": "LEFTWARD_CRITICAL_DOSE",
        "name": "Leftward critical-dose migration",
        "signature": (
            "The lowest dose needed to reach a fixed fraction of max-dose OOD accuracy decreases over training. "
            "This indicates the residual direction itself is becoming more capable, not merely crossing a fixed interpolation threshold."
        ),
        "confirming": [
            "dose_surface.critical_dose_by_step",
            "dose_surface.critical_dose_leftward_shift",
            "dose_surface.critical_dose_migrates_left",
        ],
        "falsifiers": [
            "dose_surface.fixed_high_critical_dose",
        ],
    },
    {
        "id": "GROKKING_LIKE",
        "name": "Grokking-like delayed generalization",
        "signature": (
            "Train fitting reaches a large fraction of its observed gain materially earlier than OOD; "
            "OOD remains near baseline across a plateau of retained checkpoints; continued optimization later yields measurable OOD gain."
        ),
        "confirming": [
            "temporal.train_progress_step",
            "temporal.ood_progress_step",
            "temporal.ood_to_train_delay_ratio",
            "temporal.plateau_intervals",
            "temporal.grokking_like_under_primary_thresholds",
        ],
        "falsifiers": [
            "simultaneous train/OOD rise without plateau",
            "dose-only final-step jump without temporal delay",
        ],
    },
    {
        "id": "MEMORIZATION",
        "name": "Memorization without abstraction",
        "signature": "Train answerable accuracy improves materially while structural-OOD answerable accuracy remains below the grokking-like OOD-gain floor.",
        "confirming": [
            "temporal.train_observed_gain",
            "temporal.ood_observed_gain",
        ],
        "falsifiers": [
            "temporal.ood_observed_gain above minimum_observed_ood_gain_for_grokking_like",
            "representation_precedes_ood with later OOD rise",
        ],
    },
    {
        "id": "ORDINARY_LEARNING",
        "name": "Ordinary feature learning",
        "signature": "Train and OOD improve together without a delayed plateau that satisfies the preregistered grokking-like criteria.",
        "confirming": [
            "temporal.train_observed_gain > 0",
            "temporal.ood_observed_gain > 0",
            "not temporal.grokking_like_under_primary_thresholds",
        ],
        "falsifiers": [
            "temporal.grokking_like_under_primary_thresholds",
            "memorization_like train-only gains",
        ],
    },
    {
        "id": "LATENT_BEFORE_READOUT",
        "name": "Latent representation progress before discrete OOD",
        "signature": (
            "Cross-format nearest-centroid reasoning-family organization reaches half of its observed gain "
            "before the OOD accuracy progress step."
        ),
        "confirming": [
            "representation.representation_progress_step_50pct",
            "representation.representation_precedes_ood_transition",
            "cross-format CKA / separation / activation rank trajectories",
        ],
        "falsifiers": [
            "representation progress only after OOD accuracy already rises",
        ],
    },
]


def build_methods_block(cfg: dict[str, Any], transition: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    td = cfg.get("transition_detection") or {}
    thresholds = transition.get("preregistered_thresholds") or {
        "train_progress_fraction": td.get("train_progress_fraction"),
        "ood_progress_fraction": td.get("ood_progress_fraction"),
        "minimum_plateau_intervals": td.get("minimum_plateau_intervals"),
        "minimum_observed_ood_gain_for_grokking_like": td.get("minimum_observed_ood_gain_for_grokking_like"),
        "minimum_delay_ratio_for_grokking_like": td.get("minimum_delay_ratio_for_grokking_like"),
        "reproduced_seed_count": td.get("reproduced_seed_count"),
        "memorization_minimum_train_gain": td.get("memorization_minimum_train_gain", 0.1),
        "readout_destination_dose_minimum": td.get("readout_destination_dose_minimum", 0.75),
        "critical_dose_progress_fraction": td.get("critical_dose_progress_fraction", 0.5),
        "critical_dose_absolute_floor": td.get("critical_dose_absolute_floor", 0.01),
        "temporal_readout_accuracy_jump_fraction_minimum": td.get(
            "temporal_readout_accuracy_jump_fraction_minimum", 0.5
        ),
    }
    return {
        "data_policy": {
            "optimization_anchor": "v2.2 greedy_soup",
            "source_data": "completed v2.3 residual/preservation pools + packaged DEV/OOD-DEV",
            "sealed_data_used": False,
            "sealed_policy": (cfg.get("experiment") or {}).get("sealed_policy"),
        },
        "profiles": cfg.get("profiles"),
        "core_training": cfg.get("core_training"),
        "probes": cfg.get("probes"),
        "dose_surface": cfg.get("dose_surface"),
        "representation": cfg.get("representation_dynamics"),
        "factor_sweep_axes": sorted((cfg.get("factor_sweep") or {}).keys()),
        "lora_rank_method": (
            "Non-zero singular spectrum of scaled B@A from an r×r QR core; "
            "full dense deltas are not materialized for rank analysis."
        ),
        "preregistered_thresholds": thresholds,
        "source_counts": data.get("source_counts"),
        "variant_count": data.get("variant_count"),
        "probe_manifest": data.get("probe_manifest"),
        "label_priority": [
            "DELAYED_GENERALIZATION_REPRODUCED",
            "GROKKING_LIKE_TRANSITION_DETECTED",
            "MEMORIZATION_WITHOUT_ABSTRACTION",
            "READOUT_THRESHOLD_LIKELY",
            "ORDINARY_FEATURE_LEARNING",
            "INSUFFICIENT_EVIDENCE",
        ],
    }


def compile_markdown(
    *,
    cfg: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    transition = payload
    temporal = payload.get("temporal") or {}
    dose = payload.get("dose_surface") or {}
    readout = payload.get("temporal_readout") or {}
    rep = payload.get("representation") or {}
    opt = payload.get("optimizer_dynamics") or {}
    adapter = payload.get("adapter_dynamics") or {}
    evidence = payload.get("competing_evidence") or {}
    sens = payload.get("sensitivity") or {}
    methods = payload.get("methods") or {}
    thresholds = methods.get("preregistered_thresholds") or {}
    factors = payload.get("factor_phase_diagram") or {}
    factor_rows = factors.get("rows") if isinstance(factors, dict) else factors
    if not isinstance(factor_rows, list):
        factor_rows = payload.get("factor_rows") or []
    procedure = payload.get("procedure_long_horizon_control") or {}
    mining = payload.get("v23_teacher_ceiling_context") or {}
    label = payload.get("diagnostic_label")

    lines: list[str] = [
        "# v2.3.1 Grokking Dynamics Observatory",
        "",
        "## Abstract / diagnostic verdict",
        "",
        f"**Primary diagnostic label:** `{label}`",
        "",
        str((cfg.get("experiment") or {}).get("claim_policy") or payload.get("claim_policy") or ""),
        "",
        "v2.3.1 is a dynamics experiment, not a release-candidate selection phase. "
        "It uses no consumed v2.2 sealed TEST and no fresh v2.3 sealed TEST. "
        "A capability-dose jump alone never yields a grokking label.",
        "",
        "## Experimental question and mechanism taxonomy",
        "",
        "Central question: when residual reasoning capability appears, is the evidence more consistent with "
        "a readout/logit threshold, ordinary gradual learning, memorization without abstraction, "
        "grokking-like delayed generalization, or latent representation progress before discrete OOD answers?",
        "",
        "The observatory measures the same residual capability across discrete behavior, continuous logits, "
        "step×dose surfaces, LoRA geometry, optimizer telemetry, representation probes, and one-factor screens.",
        "",
        "## Methods (preregistered)",
        "",
        "### Data policy",
        "",
        f"- Optimization anchor: `{((methods.get('data_policy') or {}).get('optimization_anchor'))}`",
        f"- Source data: `{((methods.get('data_policy') or {}).get('source_data'))}`",
        f"- Sealed data used: `{((methods.get('data_policy') or {}).get('sealed_data_used'))}`",
        f"- Sealed policy: `{((methods.get('data_policy') or {}).get('sealed_policy'))}`",
        "",
        "### Training and measurement design",
        "",
        f"- Profiles: `{json.dumps(methods.get('profiles'), sort_keys=True)}`",
        f"- Probe configuration: `{json.dumps(methods.get('probes'), sort_keys=True)}`",
        f"- Dose grid: `{json.dumps((methods.get('dose_surface') or {}).get('doses'), sort_keys=True)}`",
        f"- LoRA spectrum method: {methods.get('lora_rank_method')}",
        f"- Representation depths / metrics: `{json.dumps(methods.get('representation'), sort_keys=True)}`",
        f"- One-factor axes: `{methods.get('factor_sweep_axes')}`",
        "",
        "### Preregistered transition thresholds",
        "",
    ]
    lines.extend(
        md_table(
            ["Parameter", "Value"],
            [[key, value] for key, value in sorted(thresholds.items())],
        )
    )
    lines.extend(
        [
            "",
            "### Label priority cascade",
            "",
            "Primary labels are assigned in this conservative order (first match wins):",
            "",
        ]
    )
    for index, name in enumerate(methods.get("label_priority") or [], start=1):
        lines.append(f"{index}. `{name}`")
    lines.extend(
        [
            "",
            "Competing active patterns are always reported even when suppressed by the cascade.",
            "",
            "## Observation pattern catalog",
            "",
            "Each pattern below pairs a mechanistic signature with the techniques used to confirm or falsify it.",
            "",
        ]
    )
    for pattern in PATTERN_CATALOG:
        lines.extend(
            [
                f"### {pattern['name']} (`{pattern['id']}`)",
                "",
                f"**Signature.** {pattern['signature']}",
                "",
                "**Confirming techniques / fields.**",
                "",
            ]
        )
        for item in pattern["confirming"]:
            lines.append(f"- `{item}`")
        lines.extend(["", "**Falsifiers.**", ""])
        for item in pattern["falsifiers"]:
            lines.append(f"- `{item}`")
        lines.append("")

    lines.extend(
        [
            "## Results with technique↔evidence mapping",
            "",
            "### Core temporal curve",
            "",
        ]
    )
    lines.extend(
        md_table(
            ["Metric", "Value"],
            [
                ["Train observed gain", temporal.get("train_observed_gain")],
                ["DEV observed gain", temporal.get("dev_observed_gain")],
                ["OOD observed gain", temporal.get("ood_observed_gain")],
                ["Train progress step", temporal.get("train_progress_step")],
                ["OOD progress step", temporal.get("ood_progress_step")],
                ["OOD/train delay ratio", temporal.get("ood_to_train_delay_ratio")],
                ["Plateau intervals", temporal.get("plateau_intervals")],
                ["Grokking-like under primary thresholds", temporal.get("grokking_like_under_primary_thresholds")],
                ["Final TRAIN answerable", temporal.get("final_train_answerable")],
                ["Final DEV answerable", temporal.get("final_dev_answerable")],
                ["Final OOD answerable", temporal.get("final_ood_answerable")],
            ],
        )
    )
    lines.extend(
        [
            "",
            "### Temporal readout (sharp accuracy vs smooth logits)",
            "",
        ]
    )
    lines.extend(
        md_table(
            ["Metric", "Value"],
            [
                ["Smooth logits with sharp accuracy", readout.get("smooth_logits_with_sharp_accuracy")],
                ["Largest OOD accuracy jump fraction of gain", readout.get("largest_ood_accuracy_jump_fraction_of_gain")],
                ["Sharp OOD accuracy jump", readout.get("sharp_ood_accuracy_jump")],
                ["OOD margin linear R²", readout.get("ood_margin_linear_r2")],
                ["Smooth OOD margin trajectory", readout.get("smooth_ood_margin_trajectory")],
                ["OOD NLL linear R²", readout.get("ood_nll_linear_r2")],
                ["Smooth OOD NLL trajectory", readout.get("smooth_ood_nll_trajectory")],
            ],
        )
    )
    lines.extend(
        [
            "",
            "### Critical-dose trajectory and final-step dose readout",
            "",
        ]
    )
    lines.extend(
        md_table(
            ["Metric", "Value"],
            [
                ["Final-step readout-threshold-like", dose.get("readout_threshold_like")],
                ["Critical dose start", dose.get("critical_dose_start")],
                ["Critical dose end", dose.get("critical_dose_end")],
                ["Critical-dose leftward shift", dose.get("critical_dose_leftward_shift")],
                ["Critical dose migrates left", dose.get("critical_dose_migrates_left")],
                ["Fixed high critical dose", dose.get("fixed_high_critical_dose")],
                ["Largest final-step OOD accuracy jump", dose.get("largest_ood_accuracy_jump")],
                ["OOD margin monotonic pair fraction", dose.get("ood_logit_margin_monotonic_pair_fraction")],
            ],
        )
    )
    crit_rows = [
        [item.get("step"), item.get("critical_dose"), item.get("max_ood_accuracy"), item.get("target_accuracy")]
        for item in (dose.get("critical_dose_by_step") or [])
    ]
    if crit_rows:
        lines.extend(["", "Critical dose by step:", ""])
        lines.extend(md_table(["Step", "Critical dose", "Max OOD accuracy", "Target accuracy"], crit_rows))

    lines.extend(
        [
            "",
            "### Representation, LoRA, and optimizer dynamics",
            "",
        ]
    )
    lines.extend(
        md_table(
            ["Metric", "Value"],
            [
                ["Representation precedes OOD transition", rep.get("representation_precedes_ood_transition")],
                ["Representation 50% progress step", rep.get("representation_progress_step_50pct")],
                ["Delta norm near OOD transition", adapter.get("delta_norm_at_or_near_ood_transition")],
                ["Relative delta norm near OOD transition", adapter.get("relative_delta_norm_at_or_near_ood_transition")],
                ["Delta singular max near OOD transition", adapter.get("delta_singular_max_at_or_near_ood_transition")],
                ["Final delta effective rank mean", adapter.get("final_delta_effective_rank_mean")],
                ["Final delta singular max", adapter.get("final_delta_singular_max")],
                ["Slow/total gradient ratio before transition", opt.get("slow_to_total_ratio_before_transition_mean")],
                ["Slow/total gradient ratio after transition", opt.get("slow_to_total_ratio_after_transition_mean")],
            ],
        )
    )

    lines.extend(
        [
            "",
            "### Factor phase diagram outcomes",
            "",
            f"One-factor runs recorded: `{len(factor_rows)}`",
            "",
        ]
    )
    if factor_rows:
        factor_table = [
            [
                row.get("axis"),
                row.get("value"),
                row.get("train_progress_step"),
                row.get("ood_progress_step"),
                row.get("ood_to_train_delay_ratio"),
                row.get("grokking_like_under_primary_thresholds"),
                row.get("train_observed_gain"),
                row.get("ood_observed_gain"),
            ]
            for row in factor_rows
        ]
        lines.extend(
            md_table(
                [
                    "Axis",
                    "Value",
                    "Train progress step",
                    "OOD progress step",
                    "Delay ratio",
                    "Grokking-like",
                    "Train gain",
                    "OOD gain",
                ],
                factor_table,
            )
        )
    else:
        lines.append("No factor-phase rows were available for this compilation.")

    lines.extend(
        [
            "",
            "### Procedure long-horizon contrast control",
            "",
        ]
    )
    if procedure:
        proc_rows = procedure.get("rows") or []
        lines.append(f"Procedure curve rows: `{len(proc_rows)}`")
        if proc_rows:
            first = proc_rows[0]
            last = proc_rows[-1]
            lines.extend(
                [
                    "",
                    f"- First-step keys present: `{sorted(first.keys())[:12]}...`" if len(first) > 12 else f"- First-step keys: `{sorted(first.keys())}`",
                    f"- Final recorded step: `{last.get('step')}`",
                ]
            )
            for key in (
                "procedural_dev_correctness",
                "procedural_ood_correctness",
                "dev_correctness",
                "ood_correctness",
            ):
                if key in last:
                    lines.append(f"- Final `{key}`: `{last.get(key)}`")
        lines.append("")
        lines.append(
            "This branch is a contrast/control and does not determine the reasoning grokking label."
        )
    else:
        lines.append("Procedure control curve unavailable or skipped.")

    lines.extend(
        [
            "",
            "### v2.3 teacher-ceiling context",
            "",
            f"- Direct oracle-union accuracy: `{((mining.get('direct') or {}).get('oracle_union_accuracy')) if mining else None}`",
            f"- Direct unresolved anchor failures: `{((mining.get('direct') or {}).get('unresolved_anchor_failures')) if mining else None}`",
            "",
            "## Sensitivity analysis",
            "",
            f"- Sensitivity trials: `{sens.get('trials')}`",
            f"- Positive trials: `{sens.get('positive_trials')}`",
            f"- Positive fraction: `{sens.get('positive_fraction')}`",
            f"- Fragile under threshold grid: `{sens.get('fragile')}`",
            "",
        ]
    )
    flipping = sens.get("flipping_cells") or []
    if flipping:
        lines.append("Threshold grid cells that disagree with the first grid cell:")
        lines.append("")
        lines.extend(
            md_table(
                ["Train fraction", "OOD fraction", "Delay-ratio threshold", "Detected"],
                [
                    [
                        cell.get("train_fraction"),
                        cell.get("ood_fraction"),
                        cell.get("delay_ratio_threshold"),
                        cell.get("detected"),
                    ]
                    for cell in flipping
                ],
            )
        )
    else:
        lines.append("No flipping cells relative to the first sensitivity grid cell, or sensitivity details unavailable.")
    lines.extend(
        [
            "",
            "A fragile conclusion means the grokking-like call depends on the exact preregistered thresholds; "
            "treat such a call as provisional.",
            "",
            "## Competing explanations",
            "",
            f"- Supporting patterns for `{label}`: `{evidence.get('supporting_patterns')}`",
            f"- Contradicting / competing active patterns: `{evidence.get('contradicting_patterns')}`",
            f"- Evidence flags: `{json.dumps(evidence.get('flags') or {}, sort_keys=True)}`",
            "",
            str(evidence.get("note") or ""),
            "",
            "### Label glossary",
            "",
            "- `READOUT_THRESHOLD_LIKELY`: dose/logit behavior is more consistent with crossing output boundaries than delayed temporal abstraction.",
            "- `ORDINARY_FEATURE_LEARNING`: train and OOD improve without a strong delayed plateau.",
            "- `MEMORIZATION_WITHOUT_ABSTRACTION`: train improves materially while structural OOD remains largely flat.",
            "- `GROKKING_LIKE_TRANSITION_DETECTED`: a delayed OOD transition satisfies the preregistered temporal criteria in the core run.",
            "- `DELAYED_GENERALIZATION_REPRODUCED`: the core delayed transition is also reproduced across the configured number of seed controls.",
            "- `INSUFFICIENT_EVIDENCE`: the run does not cleanly support the other labels.",
            "",
            "## Limitations",
            "",
            "- Probe sizes are finite; continuous-logit estimates use answerable subsets only.",
            "- Factor screens are one-factor-at-a-time, not a full factorial design.",
            "- Final-step dose readout remains a secondary historical heuristic; critical-dose trajectories and temporal logit smoothness are the preferred readout tests.",
            "- Representation and LoRA/optimizer metrics are first-class evidence fields but do not override the temporal-first primary label cascade.",
            "- No sealed TEST evaluation is performed in v2.3.1; generalization claims are restricted to packaged DEV/OOD-DEV and residual-format probes.",
            "- Label priority can suppress competing mechanisms; always inspect the competing-evidence matrix.",
            "- Sensitivity fragility should be reported before any strong delayed-generalization claim.",
            "",
            "No label in this report changes the v2.2/v2.3 production anchor automatically.",
            "",
            "## Artifact index",
            "",
        ]
    )
    artifacts = payload.get("artifact_paths") or {}
    for key, path in sorted(artifacts.items()):
        lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


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

    factor_rows = factors.get("rows") if isinstance(factors, dict) else None
    if factor_rows is None and isinstance(transition.get("factor_phase_diagram"), list):
        factor_rows = transition.get("factor_phase_diagram")

    methods = build_methods_block(
        cfg,
        transition,
        {
            "source_counts": data.get("source_counts"),
            "variant_count": len(data.get("variants") or {}),
            "probe_manifest": data.get("probes"),
        },
    )

    payload = {
        "schema_version": "2.3.1",
        "experiment": cfg["experiment"],
        "sealed_data_used": False,
        "diagnostic_label": transition.get("diagnostic_label"),
        "claim_policy": transition.get("claim_policy") or (cfg.get("experiment") or {}).get("claim_policy"),
        "methods": methods,
        "observation_pattern_catalog": PATTERN_CATALOG,
        "data": {
            "source_counts": data.get("source_counts"),
            "variant_count": len(data.get("variants") or {}),
            "probe_manifest": data.get("probes"),
        },
        "v23_teacher_ceiling_context": mining,
        "temporal": transition.get("temporal"),
        "temporal_readout": transition.get("temporal_readout"),
        "dose_surface": transition.get("dose_surface"),
        "sensitivity": transition.get("sensitivity"),
        "adapter_dynamics": transition.get("adapter_dynamics") or adapter,
        "optimizer_dynamics": transition.get("optimizer_dynamics"),
        "representation": transition.get("representation") or representation,
        "factor_phase_diagram": factors if factors else {"rows": factor_rows or []},
        "factor_rows": factor_rows or [],
        "competing_evidence": transition.get("competing_evidence"),
        "preregistered_thresholds": transition.get("preregistered_thresholds") or methods.get("preregistered_thresholds"),
        "procedure_long_horizon_control": procedure_control,
        "time_curve_artifact": {
            "row_count": len(curve.get("rows") or []),
            "path": str(a.time_curve),
        },
        "dose_surface_artifact": {
            "row_count": len(surface.get("rows") or []),
            "path": str(a.dose_surface),
            "critical_dose_trajectory": surface.get("critical_dose_trajectory"),
        },
        "artifact_paths": {
            "training_time_curve": str(a.time_curve),
            "dose_time_surface": str(a.dose_surface),
            "transition_detection": str(a.transition),
            "adapter_dynamics": str(a.adapter_dynamics) if a.adapter_dynamics else None,
            "representation_dynamics": str(a.representation) if a.representation else None,
            "factor_phase_diagram": str(a.factor_phase) if a.factor_phase else None,
            "procedure_control_curve": str(a.procedure_control_curve) if a.procedure_control_curve else None,
            "report_json": str(a.output_json),
            "report_md": str(a.output_md),
        },
    }
    write_json(a.output_json, payload)
    markdown = compile_markdown(cfg=cfg, payload=payload)
    a.output_md.parent.mkdir(parents=True, exist_ok=True)
    a.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"diagnostic_label": payload["diagnostic_label"], "output_md": str(a.output_md)}, indent=2))


if __name__ == "__main__":
    main()
