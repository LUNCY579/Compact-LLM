"""Transparent peer attention: cosine similarity plus a bounded residual."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class PeerAttentionResult:
    """All inspectable components of a peer-attention calculation."""

    peer_names: tuple[str, ...]
    similarities: Tensor
    residuals: Tensor
    depth_distances: Tensor
    logits: Tensor
    alphas: Tensor
    peer_summary: Tensor

    def record(self) -> dict[str, object]:
        """Return JSON-compatible diagnostics, including persisted alphas."""

        return {
            "peer_names": list(self.peer_names),
            "similarities": self.similarities.detach().cpu().tolist(),
            "residuals": self.residuals.detach().cpu().tolist(),
            "depth_distances": self.depth_distances.detach().cpu().tolist(),
            "logits": self.logits.detach().cpu().tolist(),
            "alphas": self.alphas.detach().cpu().tolist(),
            "self_alpha": 0.0,
        }


class PeerAttention(nn.Module):
    """Attend to same-role peers with an auditable score decomposition.

    ``score_i = cosine(target, peer_i) / temperature + residual_i``.
    The residual network is zero-initialized, so the initial policy is exactly
    descriptor similarity rather than an opaque learned attention mechanism.
    """

    def __init__(
        self,
        descriptor_dimension: int,
        *,
        residual_hidden: int = 16,
        residual_scale: float = 0.25,
        temperature: float = 1.0,
        depth_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        if descriptor_dimension <= 0 or residual_hidden <= 0:
            raise ValueError("attention dimensions must be positive")
        if (
            not math.isfinite(residual_scale)
            or not math.isfinite(temperature)
            or not math.isfinite(depth_penalty)
            or residual_scale < 0
            or temperature <= 0
            or depth_penalty < 0
        ):
            raise ValueError(
                "residual/depth scales must be nonnegative and temperature positive"
            )
        self.descriptor_dimension = descriptor_dimension
        self.residual_hidden = residual_hidden
        self.residual_scale = float(residual_scale)
        self.temperature = float(temperature)
        self.depth_penalty = float(depth_penalty)
        self.residual_network = nn.Sequential(
            nn.Linear(descriptor_dimension * 3, residual_hidden),
            nn.Tanh(),
            nn.Linear(residual_hidden, 1),
        )
        final = self.residual_network[-1]
        if not isinstance(final, nn.Linear):  # pragma: no cover
            raise TypeError("unexpected residual network")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def configuration(self) -> dict[str, object]:
        return {
            "version": "pcsm-peer-attention-v1",
            "descriptor_dimension": self.descriptor_dimension,
            "residual_hidden": self.residual_hidden,
            "residual_scale": self.residual_scale,
            "temperature": self.temperature,
            "depth_penalty": self.depth_penalty,
        }

    def forward(
        self,
        target_name: str,
        target_descriptor: Tensor,
        peer_descriptors: Mapping[str, Tensor],
        *,
        target_depth: float | None = None,
        peer_depths: Mapping[str, float] | None = None,
    ) -> PeerAttentionResult:
        if target_name in peer_descriptors:
            raise ValueError("self advice is forbidden: target appears in peer set")
        if not peer_descriptors:
            raise ValueError("peer attention requires at least one same-role peer")
        peer_names = tuple(sorted(peer_descriptors))
        target = target_descriptor.float().reshape(-1)
        peers = torch.stack(
            [peer_descriptors[name].float().reshape(-1) for name in peer_names]
        )
        if target.numel() != self.descriptor_dimension:
            raise ValueError("target descriptor has the wrong dimension")
        if peers.shape[1] != self.descriptor_dimension:
            raise ValueError("a peer descriptor has the wrong dimension")
        target = torch.nan_to_num(target)
        peers = torch.nan_to_num(peers)
        if (target_depth is None) != (peer_depths is None):
            raise ValueError("target and peer depths must be supplied together")
        if target_depth is None:
            depth_distances = torch.zeros(
                len(peer_names), dtype=torch.float32, device=peers.device
            )
        else:
            if not math.isfinite(target_depth) or not 0.0 <= target_depth <= 1.0:
                raise ValueError("target depth must be finite and in [0, 1]")
            if peer_depths is None or set(peer_depths) != set(peer_names):
                raise ValueError("peer depths must exactly match the peer set")
            depths = torch.tensor(
                [peer_depths[name] for name in peer_names],
                dtype=torch.float32,
                device=peers.device,
            )
            if not bool(torch.isfinite(depths).all()) or bool(
                torch.logical_or(depths < 0, depths > 1).any()
            ):
                raise ValueError("peer depths must be finite and in [0, 1]")
            depth_distances = (depths - target_depth).abs()
        expanded_target = target.unsqueeze(0).expand_as(peers)
        similarities = F.cosine_similarity(expanded_target, peers, dim=1, eps=1e-8)
        residual_features = torch.cat(
            (expanded_target, peers, expanded_target - peers), dim=1
        )
        residuals = self.residual_scale * torch.tanh(
            self.residual_network(residual_features).squeeze(1)
        )
        logits = (
            similarities / self.temperature
            + residuals
            - self.depth_penalty * depth_distances
        )
        alphas = torch.softmax(logits.float(), dim=0)
        peer_summary = torch.sum(alphas[:, None] * peers, dim=0)
        return PeerAttentionResult(
            peer_names=peer_names,
            similarities=similarities.float(),
            residuals=residuals.float(),
            depth_distances=depth_distances.float(),
            logits=logits.float(),
            alphas=alphas.float(),
            peer_summary=peer_summary.float(),
        )


__all__ = ["PeerAttention", "PeerAttentionResult"]
