"""Post-hoc descriptive tokenwise errors; never changes fitted maps or gates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(REPO/"experiments/anchor_routed_middle_recurrence_v1/src"),
               str(REPO/"Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]
import torch
from graft import Bridges
from armr.provenance import sha256, write_json


def metrics(predicted, target):
    error = (predicted-target).square().mean(-1)
    energy = target.square().mean(-1)
    cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
    return {"tokens": len(target), "mean_token_cosine": float(cosine.mean()),
            "median_token_cosine": float(cosine.median()),
            "mean_token_relative_mse": float((error/energy.clamp_min(1e-12)).mean()),
            "median_token_relative_mse": float((error/energy.clamp_min(1e-12)).median()),
            "global_relative_mse": float(error.mean()/energy.mean().clamp_min(1e-12)),
            "median_target_token_rms": float(energy.sqrt().median()),
            "max_target_token_rms": float(energy.sqrt().max()),
            "largest_token_fraction_of_total_energy": float(energy.max()/energy.sum())}


def main():
    run = Path(sys.argv[1]).resolve()
    if not run.is_relative_to((ROOT/"evidence").resolve()):
        raise ValueError("expected this experiment's evidence")
    local = ROOT/"local"/run.name
    fit = json.loads((run/"bridge_fit.json").read_text())
    for name, h in fit["artifacts"].items():
        if sha256(local/name) != h:
            raise ValueError("local calibration artifact changed")
    torch.set_num_threads(2)
    data = torch.load(local/"activations.pt", weights_only=True, map_location="cpu")
    weights = torch.load(local/"bridges.pt", weights_only=True, map_location="cpu")["aligned"]
    bridge = Bridges(*(weights[k] for k in ("up", "up_bias", "down", "down_bias")))
    small = torch.cat([v for v, train in zip(data["small"], data["train_prompt_mask"]) if not train])
    big = torch.cat([v for v, train in zip(data["big_input"], data["train_prompt_mask"]) if not train])
    held_ids = [v[0] for v, train in zip(data["input_ids"], data["train_prompt_mask"]) if not train]
    ids = torch.cat(held_ids)
    last = torch.cat([torch.arange(len(v)) == len(v)-1 for v in held_ids])
    # Qwen chat structural tokens have IDs >=151643. IDs and mask rule are explicit.
    masks = {"all_heldout_tokens": torch.ones(len(ids), dtype=torch.bool),
             "text_token_ids_below_151643": ids < 151643, "last_prompt_token": last}
    results = {}
    for name, mask in masks.items():
        results[name] = {"up": metrics(bridge.project(small[mask]), big[mask]),
                         "down": metrics(big[mask] @ bridge.down+bridge.down_bias, small[mask])}
    write_json(run/"posthoc_alignment_diagnostics.json", {
        "status": "DESCRIPTIVE_ONLY", "source_sha256": sha256(Path(__file__)),
        "note": "Post-hoc per-token diagnostics. No fitting, selection, or evaluation gate is changed. Global MSE weights high-norm tokens heavily.",
        "metrics": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
