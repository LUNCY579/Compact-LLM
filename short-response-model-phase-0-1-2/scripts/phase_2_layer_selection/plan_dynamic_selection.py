from __future__ import annotations

"""Optional IST-style comparison scaffold.

This experiment intentionally does not run automatically in Phase 2. The blueprint
requires static selection first. A future implementation may update an EMA of layer
importance at intervals and mask gradients for unselected LoRA layers. Such gradient
masking tests behavioral adaptation, but does not by itself prove optimizer-state or
wall-clock memory savings because adapters may still exist in every layer.
"""

import argparse
from pathlib import Path

from phase2_common import ROOT, load_json, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Create a reproducible optional dynamic-selection experiment plan.")
    p.add_argument("--static-results", type=Path, default=ROOT / "artifacts/phase_2_layer_study/phase_2_results.json")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/phase_2_dynamic/dynamic_selection_plan.json")
    p.add_argument("--layer-budget", type=int, default=4)
    p.add_argument("--selection-interval", type=int, default=25)
    p.add_argument("--ema-beta", type=float, default=0.9)
    args = p.parse_args()
    static = load_json(args.static_results)
    top = [int(x["layers"][0]) for x in static["ranked_single_layers"][: args.layer_budget]]
    write_json(args.output, {
        "status": "PLANNED_NOT_EXECUTED",
        "purpose": "Compare interval-updated IST-style selection against the accepted static subset.",
        "initial_active_layers": top,
        "layer_budget": args.layer_budget,
        "selection_interval": args.selection_interval,
        "importance_ema_beta": args.ema_beta,
        "required_controls": ["accepted_static_subset", "all_layer_reference", "random_same_budget"],
        "required_reporting": ["quality", "category_floors", "peak_vram", "train_runtime", "optimizer_state_memory"],
        "evidence_boundary": "Dynamic selection is optional and is not required for Stop Gate 2. Gradient masking alone is not a memory-saving claim.",
    })
    print(f"Dynamic selection plan: {args.output}")


if __name__ == "__main__":
    main()
