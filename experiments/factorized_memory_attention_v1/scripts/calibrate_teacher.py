#!/usr/bin/env python
"""Memory-only calibration against precomputed unsealed teacher residuals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, load_shape
from experiments.factorized_memory_attention_v1.fma.memory import FactorizedMemoryBank, calibration_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    packet = torch.load(args.packet, map_location="cpu", weights_only=True)
    data_class = str(packet.get("data_class", "")).lower()
    if data_class in {"sealed", "confirmation", "final"}:
        raise RuntimeError("sealed/confirmation/final data are forbidden for FMA calibration")
    if not packet.get("backbone_frozen", False):
        raise RuntimeError("teacher packet must attest that the backbone was frozen")
    if not packet.get("data_id"):
        raise RuntimeError("teacher packet must carry data_id")
    samples = packet.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("teacher packet must contain a non-empty samples list")

    shape = load_shape(args.model_config)
    layers = tuple(sorted({int(sample["layer_idx"]) for sample in samples}))
    torch.manual_seed(args.seed)
    bank = FactorizedMemoryBank(shape, FMAConfig(rank=args.rank, layers=layers, gate=0.0))
    optimizer = torch.optim.AdamW(bank.parameters(), lr=args.lr)
    last_loss = None
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for sample in samples:
            losses.append(
                calibration_loss(
                    bank,
                    sample["token_ids"].long(),
                    int(sample["layer_idx"]),
                    sample["standard_value"],
                    sample["key_raw"],
                )
            )
        total = torch.stack(losses).mean()
        total.backward()
        optimizer.step()
        last_loss = float(total.detach())

    metadata = {
        "kind": "FMA_TEACHER_CALIBRATION",
        "data_id": packet["data_id"],
        "data_class": data_class or "unspecified_unsealed",
        "source_checkpoint_sha256": packet.get("source_checkpoint_sha256"),
        "backbone_frozen": True,
        "rank": args.rank,
        "layers": list(layers),
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "last_loss": last_loss,
        "sealed_data_access": False,
    }
    torch.save({"state_dict": bank.state_dict(), "metadata": metadata}, args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
