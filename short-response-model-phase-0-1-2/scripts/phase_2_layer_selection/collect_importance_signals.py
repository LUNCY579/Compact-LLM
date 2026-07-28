from __future__ import annotations

import argparse
from pathlib import Path

from phase2_common import ROOT, contribution, load_json, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Phase 2 layer-importance signals.")
    p.add_argument("--results", type=Path, default=ROOT / "artifacts/phase_2_layer_study/phase_2_results.json")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/phase_2_layer_study/importance_signals.json")
    args = p.parse_args()
    data = load_json(args.results)
    base = float(data["base_score"])
    all_score = float(data["all_layer_score"])
    rows = []
    for row in data["ranked_single_layers"]:
        rows.append({
            "layer": int(row["layers"][0]),
            "direct_single_layer_score": float(row["mean_heuristic_score"]),
            "contribution": contribution(float(row["mean_heuristic_score"]), base, all_score),
            "peak_train_vram_gb": float(row["peak_train_vram_gb"]),
            "train_runtime_seconds": float(row.get("train_runtime_seconds", 0.0)),
            "signal_status": "measured",
        })
    write_json(args.output, {
        "measured_signals": ["single_layer_training_score", "project_contribution", "training_vram", "training_runtime"],
        "not_yet_measured": ["gradient_norm", "adapter_ablation", "fisher_style_score"],
        "warning": "Training-layer importance is not inference-layer sensitivity.",
        "layers": rows,
    })
    print(f"Importance signals: {args.output}")


if __name__ == "__main__":
    main()
