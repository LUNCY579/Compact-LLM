"""Independent FP64 implementation of arXiv:2609.39494 Appendix E, Eqs. 6--11.

Exact finite-prefix occupancy, not sampled trajectories. These are distributional
mechanics, never pretrained Rineya capability measurements.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from .objectives import token_kl, validate_rate

CONTROLLED_RATES = (0.,) + tuple(.000625 * 2 ** (j / 2) for j in range(17))


@dataclass(frozen=True)
class Geometry:
    depth: int = 4
    vocabulary: int = 8
    width: int = 64
    concentration: float = 2.5
    context_strength: float = 0.6
    feature_overlap: float = 0.5
    readout_compatibility: float = 0.0

    def validate(self):
        if any(type(x) is not int for x in (self.depth, self.vocabulary, self.width)):
            raise ValueError("Integer geometry required")
        if not 1 <= self.depth <= 5 or not 4 <= self.vocabulary <= 16 or not self.vocabulary - 1 <= self.width <= 128:
            raise ValueError("Geometry outside bounded full-rank construction")
        if sum(self.vocabulary**j for j in range(self.depth)) > 5000:
            raise ValueError("Finite-tree budget exceeded")
        for x in (self.concentration, self.context_strength, self.feature_overlap, self.readout_compatibility):
            if not math.isfinite(x):
                raise ValueError("Nonfinite geometry")
        if not 0 < self.concentration <= 10 or not 0 <= self.context_strength <= 1:
            raise ValueError("Invalid concentration or context strength")
        if not 0 <= self.feature_overlap <= 1 or not -1 <= self.readout_compatibility <= 1:
            raise ValueError("Invalid overlap/compatibility")


def occupancy(policy, depth):
    """Breadth-first state order. Result has no gradient, sums to depth."""
    if policy.ndim != 2 or not torch.isfinite(policy).all() or (policy < 0).any():
        raise ValueError("Invalid probability table")
    k = policy.shape[1]
    if policy.shape[0] != sum(k**j for j in range(depth)):
        raise ValueError("Tree shape mismatch")
    if not torch.allclose(policy.sum(-1), torch.ones_like(policy[:, 0]), atol=1e-10, rtol=1e-8):
        raise ValueError("Policy is not normalized")
    with torch.no_grad():
        levels, start = [policy.new_ones(1)], 0
        for j in range(depth - 1):
            length = k**j
            levels.append((levels[-1][:, None] * policy[start:start + length]).reshape(-1))
            start += length
        return torch.cat(levels)


class ControlledTask:
    def __init__(self, geometry=Geometry(), seed=17):
        geometry.validate()
        self.geometry = geometry
        self.seed = seed
        g = torch.Generator().manual_seed(seed)
        n = sum(geometry.vocabulary**j for j in range(geometry.depth))
        k, d = geometry.vocabulary, geometry.width
        def normal(*shape):
            return torch.randn(*shape, generator=g, dtype=torch.float64)
        def normalize_rows(x):
            return x / x.norm(dim=-1, keepdim=True) * math.sqrt(d)
        self.old = normalize_rows(normal(n, d))
        h = normal(n, d)
        perpendicular = normalize_rows(h - (h * self.old).sum(-1, keepdim=True) / d * self.old)
        rho = geometry.feature_overlap
        self.new = rho * self.old + math.sqrt(1 - rho**2) * perpendicular
        w = normal(k, d) / math.sqrt(d)
        self.w0 = geometry.concentration * (w - w.mean(0))
        other = normal(k, d) / math.sqrt(d)
        other -= other.mean(0)
        other -= (other * self.w0).sum() / self.w0.square().sum() * self.w0
        other *= self.w0.norm() / other.norm()
        rw = geometry.readout_compatibility
        self.task_readout = rw * self.w0 + math.sqrt(1 - rw**2) * other
        rankings = (self.new @ self.task_readout.T).argsort(-1, descending=True)
        prescribed = torch.full((n, k), .55 / (k - 2), dtype=torch.float64)
        prescribed.scatter_(1, rankings[:, :1], .30)
        prescribed.scatter_(1, rankings[:, 1:2], .15)
        log_target = prescribed.log()
        log_target -= log_target.mean(-1, keepdim=True)
        self.wref = torch.linalg.lstsq(self.new, log_target, driver="gelsd").solution.T
        self.target = self.policy(self.wref, self.new)
        self.old_target = self.policy(self.w0, self.old)
        p0 = self.policy(self.w0, self.new)
        q0 = (1 - geometry.context_strength) * p0 + geometry.context_strength * self.target
        log_odds = lambda p: p[:, :-1].log() - p[:, -1:].log()
        r0 = self.w0[:-1] - self.w0[-1:]
        if torch.linalg.matrix_rank(r0) != k - 1:
            raise ValueError("Context calibration lacks full row rank")
        self.displacement = (log_odds(q0) - log_odds(p0)) @ torch.linalg.pinv(r0).T
        self.teacher_features = self.new + self.displacement
        self.calibration_error = float((self.policy(self.w0, self.teacher_features) - q0).abs().max())
        if self.calibration_error > 1e-10:
            raise ValueError("Initial contextual teacher calibration failed")
        self.initial_metrics = self.metrics(self.w0, self.w0)

    @staticmethod
    def policy(w, features):
        return (features @ w.T).softmax(-1)

    @torch.no_grad()
    def metrics(self, student, teacher):
        depth = self.geometry.depth
        p = self.policy(student, self.new)
        q = self.policy(teacher, self.teacher_features)
        old = self.policy(student, self.old)
        similarity = lambda a, b: 1 - .5 * (a - b).abs().sum(-1)
        weighted = lambda weights, values: float((weights * values).sum() / depth)
        nu_p = occupancy(p, depth)
        return {
            "new_similarity": weighted(nu_p, similarity(p, self.target)),
            "target_path_fidelity": weighted(occupancy(self.target, depth), similarity(p, self.target)),
            "old_similarity": weighted(occupancy(self.old_target, depth), similarity(old, self.old_target)),
            "teacher_utility": weighted(nu_p, similarity(q, self.target)),
            "teacher_entropy_nats": weighted(nu_p, -(q * q.clamp_min(1e-300).log()).sum(-1)),
        }

    def run(self, rollout_source="student", direction="reverse", ema_rate=0., steps=200, learning_rate=.001):
        validate_rate(ema_rate)
        if rollout_source not in {"student", "teacher"} or direction not in {"forward", "reverse"}:
            raise ValueError("Invalid independent axis")
        if type(steps) is not int or not 1 <= steps <= 1000 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Invalid optimizer budget")
        w = self.w0.clone().requires_grad_()
        wt = self.w0.clone()
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        history = []
        for step in range(steps):
            p_logits = self.new @ w.T
            with torch.no_grad():
                q_logits = self.teacher_features @ wt.T
                source = p_logits.detach() if rollout_source == "student" else q_logits
                weights = occupancy(source.softmax(-1), self.geometry.depth)
            loss = (weights * token_kl(p_logits, q_logits, direction)).sum() / self.geometry.depth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([w], 10., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                if ema_rate == 1:
                    wt.copy_(w)
                elif ema_rate:
                    wt.mul_(1 - ema_rate).add_(w, alpha=ema_rate)
            if step in {0, steps - 1}:
                history.append({"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(norm)})
        final = self.metrics(w, wt)
        return {"evidence": "CONTROLLED_MODEL_ONLY", "geometry": asdict(self.geometry), "seed": self.seed,
                "rollout_source": rollout_source, "kl_direction": direction, "ema_rate": ema_rate,
                "steps": steps, "learning_rate": learning_rate, "calibration_max_error": self.calibration_error,
                "initial": self.initial_metrics, "final": final, "trace": history,
                "acquisition_pp": 100 * (final["new_similarity"] - self.initial_metrics["new_similarity"]),
                "retention_pp": 100 * (final["old_similarity"] - 1),
                "target_path_gain_pp": 100 * (final["target_path_fidelity"] - self.initial_metrics["target_path_fidelity"]),
                "native_rineya_executed": False}
