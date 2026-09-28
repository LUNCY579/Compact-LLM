"""Bounded objective-only throughput probe; never loads a model or starts training."""
from __future__ import annotations

import argparse
import json
import platform
import time

import torch

from .objective import GFlowRLConfig, RolloutBatch, compute_gflowrl_loss


def run_probe(seconds: float = 0.25, seed: int = 0, group_size: int = 4) -> dict[str, object]:
    if seconds <= 0 or seconds > 5:
        raise ValueError("probe duration must be in (0, 5] seconds")
    torch.manual_seed(seed)
    config = GFlowRLConfig(expected_group_size=group_size)
    steps = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        current = torch.randn((2, group_size), requires_grad=True)
        old = torch.randn((2, group_size))
        ref = torch.randn((2, group_size))
        rewards = torch.randn((2, group_size))
        lengths = torch.randint(1, 32, (2, group_size)).float()
        result = compute_gflowrl_loss(
            RolloutBatch(current, old, ref, rewards, lengths), config
        )
        result.loss.backward()
        if not torch.isfinite(result.loss.detach()):
            raise FloatingPointError("probe produced nonfinite loss")
        steps += 1
    elapsed = time.perf_counter() - start
    return {
        "probe": "objective_only",
        "duration_seconds": elapsed,
        "steps": steps,
        "steps_per_second": steps / elapsed if elapsed else 0.0,
        "seed": seed,
        "group_size": group_size,
        "model_loaded": False,
        "training_occurred": False,
        "host": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "exact_rtx4070_gate": "NOT_ESTABLISHED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.seconds, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
