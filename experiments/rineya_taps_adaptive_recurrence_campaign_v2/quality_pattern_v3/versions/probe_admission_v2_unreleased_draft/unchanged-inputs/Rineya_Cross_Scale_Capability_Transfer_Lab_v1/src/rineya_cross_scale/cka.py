from __future__ import annotations

from typing import Any

import torch


def _center_columns(matrix: torch.Tensor) -> torch.Tensor:
    value = matrix.float()
    return value - value.mean(dim=0, keepdim=True)


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    """Linear CKA between sample-by-feature matrices."""
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("CKA requires rank-2 activation matrices")
    if left.shape[0] != right.shape[0]:
        raise ValueError("CKA sample counts differ")
    x = _center_columns(left)
    y = _center_columns(right)
    cross = torch.linalg.norm(x.transpose(0, 1) @ y, ord="fro") ** 2
    xx = torch.linalg.norm(x.transpose(0, 1) @ x, ord="fro") ** 2
    yy = torch.linalg.norm(y.transpose(0, 1) @ y, ord="fro") ** 2
    denominator = torch.sqrt(xx * yy)
    if float(denominator.item()) <= 0:
        return 0.0
    return float((cross / denominator).item())


def pairwise_cka(target_layers: list[torch.Tensor], donor_layers: list[torch.Tensor]) -> torch.Tensor:
    matrix = torch.zeros((len(target_layers), len(donor_layers)), dtype=torch.float32)
    for i, target in enumerate(target_layers):
        for j, donor in enumerate(donor_layers):
            matrix[i, j] = linear_cka(target, donor)
    return matrix


def monotonic_alignment(similarity: torch.Tensor) -> dict[str, Any]:
    """Dynamic-programming increasing donor→target layer map."""
    if similarity.ndim != 2:
        raise ValueError("alignment requires a 2D similarity matrix")
    rows, cols = similarity.shape
    if rows == 0 or cols == 0:
        raise ValueError("empty similarity matrix")
    score = torch.full((rows, cols), float("-inf"))
    previous = torch.full((rows, cols), -1, dtype=torch.int64)
    score[0] = similarity[0]
    for i in range(1, rows):
        best = float("-inf")
        best_j = -1
        for j in range(cols):
            if j > 0 and float(score[i - 1, j - 1].item()) > best:
                best = float(score[i - 1, j - 1].item())
                best_j = j - 1
            if best_j >= 0:
                score[i, j] = similarity[i, j] + best
                previous[i, j] = best_j
    end = int(torch.argmax(score[-1]).item())
    mapping = [0] * rows
    cursor = end
    for i in range(rows - 1, -1, -1):
        mapping[i] = cursor
        cursor = int(previous[i, cursor].item()) if i > 0 else cursor
    increasing = all(mapping[i] < mapping[i + 1] for i in range(len(mapping) - 1))
    return {
        "mapping": mapping,
        "score": float(score[-1, end].item()),
        "strictly_increasing": increasing,
        "target_layers": rows,
        "donor_layers": cols,
    }
