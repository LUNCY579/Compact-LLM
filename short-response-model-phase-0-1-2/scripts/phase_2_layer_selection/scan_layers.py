from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig

from scripts.common.quality_evaluation import require_quality_v2, summary_quality_score

# Default model for the first static layer-selection experiment.
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
ROOT = Path(__file__).resolve().parents[2]
PHASE_1 = ROOT / "scripts" / "phase_1_dense_sft"


# Configure the candidate layers and keep the data, step budget, seed,
# benchmark, and training method fixed across variants.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate equivalent static single-layer adapters."
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--train-file", type=Path, default=ROOT / "data" / "train" / "train.jsonl")
    p.add_argument("--dev-file", type=Path, default=ROOT / "data" / "dev" / "dev.jsonl")
    p.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "benchmark" / "shortchatbench_pilot.jsonl",
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "layer_scan")
    p.add_argument(
        "--layers",
        default=None,
        help="Optional comma-separated layer IDs; otherwise use a coarse scan.",
    )
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--include-all-layer-reference", action="store_true")
    return p.parse_args()


# Sample positions across the full stack, with extra coverage around the
# middle. Fractions adapt automatically to architectures with other depths.
def coarse_candidates(num_layers: int) -> list[int]:
    fractions = (0.0, 0.17, 0.33, 0.42, 0.50, 0.58, 0.67, 0.83, 1.0)
    return sorted(
        {
            min(num_layers - 1, max(0, round((num_layers - 1) * fraction)))
            for fraction in fractions
        }
    )


# Launch each training/evaluation stage as a subprocess using the current
# Python environment. `check=True` stops the scan on any failed variant.
def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


# Read machine-readable outputs created by the training/evaluation scripts.
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# Train equivalent adapters, evaluate each one, rank the results, and
# create a preliminary layer-importance report.
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Query the checkpoint configuration so layer IDs are validated against
    # the model's actual transformer depth.
    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = int(config.num_hidden_layers)
    # Respect an explicit candidate list, otherwise use the reproducible
    # coarse early/middle/late scan.
    if args.layers:
        layer_ids = sorted({int(x.strip()) for x in args.layers.split(",") if x.strip()})
    else:
        layer_ids = coarse_candidates(num_layers)
    invalid = [x for x in layer_ids if not 0 <= x < num_layers]
    if invalid:
        raise ValueError(f"Invalid layer IDs for {num_layers} layers: {invalid}")

    # Reuse the project's canonical trainer and evaluator rather than
    # duplicating their logic inside the scan orchestrator.
    train_script = PHASE_1 / "train_dense_sft.py"
    eval_script = PHASE_1 / "evaluate_model.py"
    # Give every candidate a separate artifact directory. Optionally include
    # the all-layer reference under the same experimental settings.
    variants: list[tuple[str, str]] = [
        (f"layer_{layer:02d}", str(layer)) for layer in layer_ids
    ]
    if args.include_all_layer_reference:
        variants.insert(0, ("all_layers", "all"))

    results: list[dict[str, Any]] = []
    # Each variant receives the same base model, data, steps, and seed; only
    # adapter layer placement changes.
    for variant, selected_layers in variants:
        variant_dir = args.output_dir / variant
        train_dir = variant_dir / "training"
        eval_dir = variant_dir / "evaluation"
        # Construct the exact training command for this layer selection.
        train_cmd = [
            sys.executable,
            str(train_script),
            "--model-id", args.model_id,
            "--train-file", str(args.train_file),
            "--dev-file", str(args.dev_file),
            "--output-dir", str(train_dir),
            "--selected-layers", selected_layers,
            "--max-steps", str(args.max_steps),
            "--seed", str(args.seed),
        ]
        if args.qlora:
            train_cmd.append("--qlora")
        if args.gradient_checkpointing:
            train_cmd.append("--gradient-checkpointing")
        run(train_cmd)

        # Evaluate the saved adapter on the same fixed benchmark.
        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--model-id", args.model_id,
            "--adapter-path", str(train_dir / "adapter"),
            "--benchmark", str(args.benchmark),
            "--output-dir", str(eval_dir),
        ]
        run(eval_cmd)

        # Combine behavioral metrics with the trainer's measured memory and
        # target-module count.
        summary = load_json(eval_dir / "summary.json")
        require_quality_v2(summary, f"layer scan {variant}")
        manifest = load_json(train_dir / "training_manifest.json")
        results.append(
            {
                "variant": variant,
                "selected_layers": selected_layers,
                "quality_evaluation_version": summary.get("quality_evaluation_version"),
                "mean_quality_score": summary_quality_score(summary),
                "mean_heuristic_score": summary_quality_score(summary),
                "quality_pass_rate": summary.get("quality_pass_rate"),
                "intelligibility_pass_rate": summary.get("intelligibility_pass_rate"),
                "hard_failure_case_count": summary.get("hard_failure_case_count"),
                "worst_decile_mean_quality_score": summary.get("worst_decile_mean_quality_score"),
                "semantic_pass_rate": summary.get("semantic_pass_rate"),
                "critical_failure_count": summary.get("critical_failure_count"),
                "major_case_count": summary.get("major_case_count"),
                "median_reply_tokens": summary["median_reply_tokens"],
                "p90_reply_tokens": summary["p90_reply_tokens"],
                "complete_utterance_rate": summary["complete_utterance_rate"],
                "peak_eval_vram_gb": summary["peak_vram_gb"],
                "peak_train_vram_gb": manifest["peak_allocated_vram_gb"],
                "target_module_count": manifest["target_module_count"],
            }
        )

    # Rank primarily by heuristic quality; use lower peak training VRAM as
    # a tie-breaker rather than allowing efficiency to hide quality loss.
    ranked = sorted(
        results,
        key=lambda row: (
            -int(row.get("hard_failure_case_count", 0)),
            -int(row.get("critical_failure_count", 0)),
            -int(row.get("major_case_count", 0)),
            float(row.get("intelligibility_pass_rate", 0.0)),
            float(row.get("quality_pass_rate", 0.0)),
            float(row.get("worst_decile_mean_quality_score", 0.0)),
            float(row["mean_quality_score"]),
            -float(row["peak_train_vram_gb"]),
        ),
        reverse=True,
    )
    # Preserve both original execution order and ranked results in JSON.
    payload = {
        "model_id": args.model_id,
        "num_hidden_layers": num_layers,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "results": results,
        "ranked": ranked,
    }
    (args.output_dir / "layer_scan_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Produce a human-readable comparison while warning that one seed and
    # one aggregate quality score cannot establish scientific layer importance.
    rows = "\n".join(
        "| {rank} | {variant} | {layers} | {score:.3f} | {median} | {p90} | {vram:.3f} |".format(
            rank=i,
            variant=row["variant"],
            layers=row["selected_layers"],
            score=row["mean_quality_score"],
            median=row["median_reply_tokens"],
            p90=row["p90_reply_tokens"],
            vram=row["peak_train_vram_gb"],
        )
        for i, row in enumerate(ranked, 1)
    )
    report = f"""# LAYER IMPORTANCE REPORT

## Scope

Static coarse layer scan for `{args.model_id}`. Each variant uses the same
training data, step budget, seed, benchmark, and evaluation configuration.

## Inputs

- Transformer layers: {num_layers}
- Candidate layers: {layer_ids}
- Seed: {args.seed}
- Maximum steps per variant: {args.max_steps}
- QLoRA: {args.qlora}

## Ranked Results

| Rank | Variant | Layers | Mean quality score | Median tokens | P90 tokens | Peak train VRAM (GB) |
|---:|---|---|---:|---:|---:|---:|
{rows}

## Interpretation Boundary

This pilot score is not enough to declare a layer scientifically important.
Review raw category scores, human judgments, multiple seeds, and controls.
Training-layer importance is also not inference-layer importance.

## Gate Status

`HUMAN REVIEW REQUIRED`

Compare the best single layer, adjacent pair, selected top-four, random four,
low-ranked four, and all-layer reference before selecting a final subset.
"""
    (args.output_dir / "LAYER_IMPORTANCE_REPORT.md").write_text(report, encoding="utf-8")
    print(f"\nReport: {args.output_dir / 'LAYER_IMPORTANCE_REPORT.md'}")


# Standard script entry point.
if __name__ == "__main__":
    main()
