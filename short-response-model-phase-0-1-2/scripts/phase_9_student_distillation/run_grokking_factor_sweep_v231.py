from __future__ import annotations

"""One-factor-at-a-time v2.3.1 grokking screen.

The goal is not a giant Cartesian hyperparameter sweep. Each factor moves one
axis away from the same anchored reasoning baseline so training-time and OOD
phase behavior remain interpretable.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import read_json, write_json


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def choose_values(values: list[Any], baseline: Any, limit: int) -> list[Any]:
    if limit <= 0 or len(values) <= limit:
        return list(values)
    picked: list[Any] = []
    if baseline in values:
        picked.append(baseline)
    # Add contrasting endpoints first, then fill deterministically.  The limit
    # counts the baseline itself, so standard profile really bounds compute.
    for value in [values[0], values[-1], *values[1:-1]]:
        if value not in picked:
            picked.append(value)
        if len(picked) >= limit:
            break
    return picked[:limit]


def layer_indices(scope: str, n_layers: int) -> str:
    if scope == "all":
        return ""
    fraction = 0.2 if scope == "middle_20" else 0.4
    count = max(1, round(n_layers * fraction))
    start = max(0, (n_layers - count) // 2)
    return ",".join(str(i) for i in range(start, min(n_layers, start + count)))


def slug(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace("/", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--variants-manifest", type=Path, required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-probe", type=Path, required=True)
    ap.add_argument("--dev-probe", type=Path, required=True)
    ap.add_argument("--ood-probe", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--profile", choices=("pilot", "standard", "extended"), default="standard")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--skip-axis", action="append", default=[])
    ap.add_argument("--max-runs", type=int)
    a = ap.parse_args()
    cfg = read_json(a.config)
    variants = read_json(a.variants_manifest)["variants"]
    core = dict(cfg["core_training"])
    sweep = cfg["factor_sweep"]
    profile = cfg["profiles"][a.profile]
    max_steps = int(profile["factor_max_steps"])
    save_steps = int(profile["factor_save_steps"])
    value_limit = int(profile.get("factor_values_per_axis") or 0)
    model_cfg = AutoConfig.from_pretrained(a.base_model, trust_remote_code=True)
    n_layers = int(getattr(model_cfg, "num_hidden_layers", getattr(model_cfg, "n_layer", 0)))
    baseline_file = str(variants["baseline_original_v23"]["path"])

    specs: list[dict[str, Any]] = []
    def add(axis: str, value: Any, **overrides: Any) -> None:
        if axis in set(a.skip_axis):
            return
        specs.append({"axis": axis, "value": value, "overrides": overrides})

    # Dataset/exposure axes use prebuilt deterministic training views.
    for value in choose_values(list(sweep["data_fraction"]), 1.0, value_limit):
        name = f"data_fraction_{float(value):.2f}".replace(".", "p")
        add("data_fraction", value, train_file=str(variants[name]["path"]))
    for value in choose_values(list(sweep["unique_residual_cases"]), "all", value_limit):
        if value == "all":
            unique = read_json(a.variants_manifest)["source_counts"]["reasoning_residual"]
        else:
            unique = min(int(value), int(read_json(a.variants_manifest)["source_counts"]["reasoning_residual"]))
        name = f"unique_residual_{unique}"
        add("unique_residual_cases", value, train_file=str(variants[name]["path"]))
    for value in choose_values(list(sweep["residual_fraction"]), core["residual_fraction"], value_limit):
        name = f"residual_fraction_{float(value):.2f}".replace(".", "p")
        add("residual_fraction", value, train_file=str(variants[name]["path"]))

    scalar_axes = [
        ("weight_decay", sweep["weight_decay"], core["weight_decay"]),
        ("anchor_kl_weight", sweep["anchor_kl_weight"], core["anchor_kl_weight"]),
        ("learning_rate", sweep["learning_rate"], core["learning_rate"]),
        ("lora_rank", sweep["lora_rank"], core["lora_rank"]),
        ("seed", sweep["seed"], core["seed"]),
        ("target_module_group", sweep["target_module_group"], core["target_module_group"]),
        ("layer_scope", sweep["layer_scope"], "all"),
    ]
    if a.profile == "extended":
        scalar_axes += [
            ("gradient_accumulation_steps", sweep["gradient_accumulation_steps_extended"], core["gradient_accumulation_steps"]),
            ("warmup_ratio", sweep["warmup_ratio_extended"], core["warmup_ratio"]),
            ("anchor_temperature", sweep["anchor_temperature_extended"], core["anchor_temperature"]),
        ]
    for axis, values, baseline in scalar_axes:
        for value in choose_values(list(values), baseline, value_limit):
            add(axis, value, **{axis: value})

    if a.max_runs is not None:
        specs = specs[: a.max_runs]
    a.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index, spec in enumerate(specs, 1):
        axis, value = spec["axis"], spec["value"]
        run_root = a.output_root / axis / slug(value)
        train_root = run_root / "training"
        train_manifest = train_root / "training_manifest.json"
        overrides = dict(spec["overrides"])
        train_file = str(overrides.pop("train_file", baseline_file))
        settings = dict(core)
        settings.update(overrides)
        if not (a.reuse_existing and train_manifest.exists()):
            cmd = [
                sys.executable,
                "scripts/phase_9_student_distillation/train_student.py",
                "--student", a.base_model,
                "--objective", "sft",
                "--train-file", train_file,
                "--output-dir", str(train_root),
                "--max-steps", str(max_steps),
                "--learning-rate", str(settings["learning_rate"]),
                "--weight-decay", str(settings["weight_decay"]),
                "--gradient-accumulation-steps", str(settings["gradient_accumulation_steps"]),
                "--warmup-ratio", str(settings["warmup_ratio"]),
                "--max-sequence-length", str(core["max_sequence_length"]),
                "--maximum-evidence-tokens", str(core["maximum_evidence_tokens"]),
                "--max-completion-tokens", str(core["max_completion_tokens"]),
                "--first-answer-token-weight", "1.5",
                "--evidence-anchor-token-weight", "1.5",
                "--abstention-token-weight", "1.5",
                "--eos-loss-weight", "1.3",
                "--seed", str(settings["seed"]),
                "--pipeline-revision", "phase9_grokking_dynamics_v2_3_1_factor_screen",
                "--lora",
                "--lora-rank", str(settings["lora_rank"]),
                "--target-module-group", str(settings["target_module_group"]),
                "--anchor-model", a.base_model,
                "--anchor-kl-weight", str(settings["anchor_kl_weight"]),
                "--anchor-temperature", str(settings["anchor_temperature"]),
                "--anchor-device", "auto",
                "--save-steps", str(save_steps),
                "--save-total-limit", "2",
                "--keep-checkpoints",
                "--no-save-only-model-checkpoints",
                "--dynamics-log-file", str(train_root / "optimizer_dynamics.jsonl"),
                "--dynamics-snapshot-dir", str(train_root / "dynamics_snapshots"),
                "--dynamics-gradient-interval", str(core["dynamics_gradient_interval"]),
                "--dynamics-ema-beta", str(core["dynamics_ema_beta"]),
            ]
            scope = str(settings.get("layer_scope", "all"))
            indices = layer_indices(scope, n_layers)
            if indices:
                cmd += ["--lora-layer-indices", indices]
            if a.require_cuda:
                cmd.append("--require-cuda")
            run(cmd)
        curve_root = run_root / "time_curve"
        curve = curve_root / "training_time_curve.json"
        if not (a.reuse_existing and curve.exists()):
            cmd = [
                sys.executable,
                "scripts/phase_9_student_distillation/evaluate_grokking_time_curve_v231.py",
                "--training-root", str(train_root),
                "--base-model", a.base_model,
                "--train-probe", str(a.train_probe),
                "--dev-probe", str(a.dev_probe),
                "--ood-probe", str(a.ood_probe),
                "--output-root", str(curve_root),
                "--dose", "1.0",
                "--logit-limit", "48",
                "--limit-per-split", "180",
                "--reuse-existing",
            ]
            if a.require_cuda:
                cmd.append("--require-cuda")
            run(cmd)
        payload = read_json(curve)
        rows = payload.get("rows") or []
        final = rows[-1] if rows else {}
        records.append(
            {
                "axis": axis,
                "value": value,
                "run_root": str(run_root),
                "train_file": train_file,
                "max_steps": max_steps,
                "final": final,
                "time_curve": str(curve),
            }
        )
        print(f"[{index}/{len(specs)}] completed {axis}={value}")

    manifest = {
        "schema_version": "2.3.1",
        "profile": a.profile,
        "sealed_data_used": False,
        "run_count": len(records),
        "records": records,
    }
    write_json(a.output_root / "factor_sweep_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
