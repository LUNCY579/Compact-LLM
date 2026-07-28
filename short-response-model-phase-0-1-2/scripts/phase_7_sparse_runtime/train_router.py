from __future__ import annotations

"""Train one small block router per decoder layer from Phase 6 oracle labels."""

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import write_json
from scripts.common.sparse_mlp import TinyBlockRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train confidence-calibrated per-layer block routers.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_7_router/routers.pt")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-ratio", type=float, default=0.65)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-weighted-recall", type=float, default=0.95)
    parser.add_argument("--budget-weight", type=float, default=0.10)
    parser.add_argument("--missed-contribution-weight", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def confidence(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    entropy = -(
        probability * torch.log(probability.clamp_min(1e-6))
        + (1 - probability) * torch.log((1 - probability).clamp_min(1e-6))
    )
    return 1.0 - entropy.mean(dim=-1) / math.log(2.0)


def predicted_mask(logits: torch.Tensor, ratio: float) -> torch.Tensor:
    blocks = logits.shape[-1]
    keep = max(1, min(blocks, int(round(blocks * ratio))))
    indices = torch.topk(logits, k=keep, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter(-1, indices, True)


def weighted_recall(mask: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    normalized = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (normalized * mask).sum(dim=-1)


def calibrate_threshold(
    logits: torch.Tensor,
    scores: torch.Tensor,
    ratio: float,
    minimum_recall: float,
) -> tuple[float, dict[str, float]]:
    conf = confidence(logits)
    mask = predicted_mask(logits, ratio)
    recall = weighted_recall(mask, scores)
    best = (1.0, {"accepted_fraction": 0.0, "accepted_weighted_recall": 1.0})
    for threshold in torch.linspace(0.0, 1.0, steps=101).tolist():
        accepted = conf >= threshold
        if not accepted.any():
            continue
        accepted_recall = float(recall[accepted].mean().item())
        accepted_fraction = float(accepted.float().mean().item())
        if accepted_recall >= minimum_recall and accepted_fraction >= best[1]["accepted_fraction"]:
            best = (
                float(threshold),
                {
                    "accepted_fraction": accepted_fraction,
                    "accepted_weighted_recall": accepted_recall,
                    "overall_weighted_recall": float(recall.mean().item()),
                },
            )
    return best


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    checkpoints: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    for layer_id, layer_data in dataset["layers"].items():
        hidden = layer_data["hidden"].float()
        labels = layer_data["oracle_mask"].float()
        scores = layer_data["block_scores"].float()
        if hidden.shape[0] < 10:
            raise RuntimeError(f"Layer {layer_id} has too few samples for a train/validation split.")
        generator = torch.Generator().manual_seed(args.seed + int(layer_id))
        permutation = torch.randperm(hidden.shape[0], generator=generator)
        validation_count = max(1, int(round(hidden.shape[0] * args.validation_fraction)))
        validation_indices = permutation[:validation_count]
        train_indices = permutation[validation_count:]

        router = TinyBlockRouter(hidden.shape[-1], args.width, labels.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate)
        router.train()
        final_loss = float("nan")
        for _ in range(args.epochs):
            epoch_indices = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
            for start in range(0, epoch_indices.numel(), args.batch_size):
                batch_indices = epoch_indices[start : start + args.batch_size]
                x = hidden.index_select(0, batch_indices).to(device)
                y = labels.index_select(0, batch_indices).to(device)
                importance = scores.index_select(0, batch_indices).to(device)
                importance = importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                logits = router(x)
                mask_loss = F.binary_cross_entropy_with_logits(logits, y)
                probabilities = torch.sigmoid(logits)
                missed_contribution = ((1 - probabilities) * importance).sum(dim=-1).mean()
                budget = (probabilities.mean() - args.target_ratio).square()
                loss = (
                    mask_loss
                    + args.missed_contribution_weight * missed_contribution
                    + args.budget_weight * budget
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu().item())

        router.eval()
        with torch.no_grad():
            validation_hidden = hidden.index_select(0, validation_indices).to(device)
            validation_scores = scores.index_select(0, validation_indices).to(device)
            validation_labels = labels.index_select(0, validation_indices).to(device)
            validation_logits = router(validation_hidden)
            validation_bce = F.binary_cross_entropy_with_logits(validation_logits, validation_labels)
            threshold, calibration = calibrate_threshold(
                validation_logits,
                validation_scores,
                args.target_ratio,
                args.minimum_weighted_recall,
            )
            predicted = predicted_mask(validation_logits, args.target_ratio)
            exact_mask = float((predicted == validation_labels.bool()).all(dim=-1).float().mean().item())

        checkpoints[layer_id] = {
            "state_dict": {key: value.detach().cpu() for key, value in router.state_dict().items()},
            "hidden_size": int(hidden.shape[-1]),
            "router_width": args.width,
            "num_blocks": int(labels.shape[-1]),
            "confidence_threshold": threshold,
        }
        metrics[layer_id] = {
            "train_samples": int(train_indices.numel()),
            "validation_samples": int(validation_indices.numel()),
            "final_train_loss": final_loss,
            "validation_bce": float(validation_bce.item()),
            "exact_mask_match_rate": exact_mask,
            **calibration,
        }

    payload = {
        "version": 2,
        "source_dataset": str(args.dataset),
        "source_model": dataset.get("model"),
        "block_size": int(dataset["block_size"]),
        "target_ratio": args.target_ratio,
        "minimum_weighted_recall": args.minimum_weighted_recall,
        "routers": checkpoints,
        "warning": (
            "These routers are validated against oracle contribution labels. Runtime quality and wall-clock behavior "
            "must still be measured with dense fallback enabled."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    write_json(
        args.out.with_suffix(".json"),
        {
            "version": payload["version"],
            "source_dataset": str(args.dataset),
            "source_model": payload["source_model"],
            "block_size": payload["block_size"],
            "target_ratio": args.target_ratio,
            "layers": metrics,
            "warning": payload["warning"],
        },
    )
    print(args.out)


if __name__ == "__main__":
    main()
