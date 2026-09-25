"""Measure the exact bounded soft-token tensor path without training."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.soft_token_reasoning_v1.implementation.soft_tokens import noisy_soft_embedding


def probe(vocabulary: int, width: int, batch: int, sequence: int, seed: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    embedding = torch.randn(vocabulary, width, device=device, dtype=torch.float32, requires_grad=True)
    logits = torch.randn(batch, sequence, vocabulary, device=device, dtype=torch.float32, requires_grad=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    means = []
    states = []
    for position in range(sequence):
        mean, state, _ = noisy_soft_embedding(
            torch.softmax(logits[:, position, :], dim=-1),
            embedding,
            0.33,
            generator=generator,
        )
        means.append(mean)
        states.append(state)
    loss = torch.stack(states).square().mean()
    loss.backward()
    result = {
        "status": "MEASURED" if device.type == "cuda" else "BLOCKED_NO_CUDA",
        "device": str(device),
        "dtype": "float32",
        "vocabulary": vocabulary,
        "embedding_width": width,
        "batch": batch,
        "sequence": sequence,
        "seed": seed,
        "loss": float(loss.detach().cpu()),
        "finite": bool(torch.isfinite(loss.detach()).item()),
        "embedding_numel": embedding.numel(),
        "logits_numel": logits.numel(),
        "measured_tensor_bytes": (embedding.numel() + logits.numel()) * embedding.element_size(),
        "vram_peak_allocated_bytes": None,
        "vram_peak_reserved_bytes": None,
        "approximation": None,
        "note": ("CPU hosts cannot establish a VRAM measurement; no resource guard was weakened."
                 if device.type != "cuda" else "CUDA peak allocation/reservation measured for this exact tensor path."),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__},
    }
    if device.type == "cuda":
        result["vram_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["vram_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, default=17)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(argv)
    result = probe(args.vocabulary, args.width, args.batch, args.sequence, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
