from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate equivalent static single-layer adapters."
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--train-file", type=Path, default=Path("data/train/train.jsonl"))
    p.add_argument("--dev-file", type=Path, default=Path("data/dev/dev.jsonl"))
    p.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/benchmark/shortchatbench_pilot.jsonl"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/layer_scan"))
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


def coarse_candidates(num_layers: int) -> list[int]:
    fractions = (0.0, 0.17, 0.33, 0.42, 0.50, 0.58, 0.67, 0.83, 1.0)
    return sorted(
        {
            min(num_layers - 1, max(0, round((num_layers - 1) * fraction)))
            for fraction in fractions
        }
    )


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = int(config.num_hidden_layers)
    if args.layers:
        layer_ids = sorted({int(x.strip()) for x in args.layers.split(",") if x.strip()})
    else:
        layer_ids = coarse_candidates(num_layers)
    invalid = [x for x in layer_ids if not 0 <= x < num_layers]
    if invalid:
        raise ValueError(f"Invalid layer IDs for {num_layers} layers: {invalid}")

    train_script = Path(__file__).with_name("train_dense_sft.py")
    eval_script = Path(__file__).with_name("evaluate_base.py")
    variants: list[tuple[str, str]] = [
        (f"layer_{layer:02d}", str(layer)) for layer in layer_ids
    ]
    if args.include_all_layer_reference:
        variants.insert(0, ("all_layers", "all"))

    results: list[dict[str, Any]] = []
    for variant, selected_layers in variants:
        variant_dir = args.output_dir / variant
        train_dir = variant_dir / "training"
        eval_dir = variant_dir / "evaluation"
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

        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--model-id", args.model_id,
            "--adapter-path", str(train_dir / "adapter"),
            "--benchmark", str(args.benchmark),
            "--output-dir", str(eval_dir),
        ]
        run(eval_cmd)

        summary = load_json(eval_dir / "summary.json")
        manifest = load_json(train_dir / "training_manifest.json")
        results.append(
            {
                "variant": variant,
                "selected_layers": selected_layers,
                "mean_heuristic_score": summary["mean_heuristic_score"],
                "median_reply_tokens": summary["median_reply_tokens"],
                "p90_reply_tokens": summary["p90_reply_tokens"],
                "complete_utterance_rate": summary["complete_utterance_rate"],
                "peak_eval_vram_gb": summary["peak_vram_gb"],
                "peak_train_vram_gb": manifest["peak_allocated_vram_gb"],
                "target_module_count": manifest["target_module_count"],
            }
        )

    ranked = sorted(
        results,
        key=lambda row: (
            float(row["mean_heuristic_score"]),
            -float(row["peak_train_vram_gb"]),
        ),
        reverse=True,
    )
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

    rows = "\n".join(
        "| {rank} | {variant} | {layers} | {score:.3f} | {median} | {p90} | {vram:.3f} |".format(
            rank=i,
            variant=row["variant"],
            layers=row["selected_layers"],
            score=row["mean_heuristic_score"],
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

| Rank | Variant | Layers | Mean heuristic score | Median tokens | P90 tokens | Peak train VRAM (GB) |
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


if __name__ == "__main__":
    main()
