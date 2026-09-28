#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.factorized_memory_attention_v1.fma.calibration import calibrate_qwen3_memory
from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, load_shape
from experiments.factorized_memory_attention_v1.fma.memory import FactorizedMemoryBank
from experiments.factorized_memory_attention_v1.fma.resources import require_host_ram


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-backbone FMA teacher calibration")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    guard = require_host_ram()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(f"BLOCKED: Transformers unavailable: {exc}") from exc

    shape = load_shape(args.config)
    cfg = FMAConfig(rank=args.rank, gate=0.0)
    bank = FactorizedMemoryBank(shape, cfg)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = bank.to(device=args.device, dtype=dtype)

    stats = calibrate_qwen3_memory(
        model,
        tokenizer,
        bank,
        args.dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )
    record = {
        "kind": "FMA_CALIBRATION",
        "status": "completed",
        "rank": args.rank,
        "device": args.device,
        "resource_guard": guard,
        "stats": stats.to_dict(),
        "sealed_data_access": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    torch.save(bank.state_dict(), str(output.with_suffix(".pt")))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
