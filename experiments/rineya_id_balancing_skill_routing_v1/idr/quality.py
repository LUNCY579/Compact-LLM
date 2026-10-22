"""Small predicted-success head for frozen-call routing; never train on confirmation."""
import numpy as np
import torch
from torch import nn
from .artifacts import validate_rows


class QualityHead(nn.Module):
    def __init__(self, features, experts):
        super().__init__()
        self.linear = nn.Linear(features, experts)
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))

    def forward(self, features):
        return torch.sigmoid(self.linear((features - self.mean) / self.scale))


def fit_quality(rows, epochs=100, lr=0.03, seed=0):
    if epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("positive training budget required")
    validate_rows(rows)
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("quality training accepts TRAIN rows only")
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
    y = torch.tensor([r["outcomes"] for r in rows], dtype=torch.float32)
    mask = torch.tensor([r["observed"] for r in rows], dtype=torch.bool)
    if (x.ndim != 2 or y.ndim != 2 or mask.shape != y.shape
            or not torch.isfinite(x).all() or not torch.isfinite(y).all()
            or (y < 0).any() or (y > 1).any() or not mask.any()):
        raise ValueError("invalid feature/outcome/missing-label matrix")
    if not mask.any(dim=0).all():
        raise ValueError("every expert needs observed training outcomes")
    torch.manual_seed(seed)
    head = QualityHead(x.shape[1], y.shape[1])
    head.mean.copy_(x.mean(0))
    head.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-6))
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = head.linear((x - head.mean) / head.scale)
        loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], y[mask])
        loss.backward()
        optimizer.step()
    head.eval().requires_grad_(False)
    return head, {"kind": "quality_head_training", "rows": len(rows),
                  "epochs": epochs, "seed": seed, "loss": float(loss.detach()),
                  "qualified": False,
                  "note": "Requires independent calibration and held-out routing evaluation."}
