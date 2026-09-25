"""Run the bounded synthetic qualification for the F-WANDA mechanics."""

from __future__ import annotations

import json

import torch

from .fwanda_delta import prune_layer


def main() -> int:
    delta = torch.arange(32, dtype=torch.float32).reshape(4, 8) - 7.0
    activations = torch.tensor([1.0, 2.0, 1.0, 0.5, 1.5, 0.75, 1.25, 0.25])
    fisher = torch.tensor([0.25, 1.0, 4.0, 9.0])
    output = {"experiment": "fwanda_delta_retention_v1", "synthetic": True, "arms": {}}
    for arm in ("F0", "F1", "F2"):
        result = prune_layer(
            delta,
            arm=arm,
            keep_fraction=0.25,
            activation_norms=activations,
            row_fisher=fisher,
            layer_name="synthetic.linear",
            min_keep=1,
        )
        output["arms"][arm] = result.record()
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
