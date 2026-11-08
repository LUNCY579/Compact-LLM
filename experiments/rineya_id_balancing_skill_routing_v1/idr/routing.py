"""Top-1 whole-call routing. Gold outcomes are deliberately absent from its API."""
from dataclasses import asdict, dataclass
from typing import Callable
import numpy as np
from .control import ControlConfig, CountController, digest, vector


@dataclass(frozen=True)
class Expert:
    name: str
    operation: str
    artifact_sha256: str
    qualified: bool
    prior: float = 1.0
    strength: float = 1.0

    def __post_init__(self):
        if not self.name or not self.operation or len(self.artifact_sha256) != 64:
            raise ValueError("expert name, operation and SHA256 are required")
        if any(c not in "0123456789abcdef" for c in self.artifact_sha256):
            raise ValueError("invalid artifact SHA256")
        if type(self.qualified) is not bool:
            raise ValueError("qualification must be explicit boolean")
        if not np.isfinite(self.prior) or self.prior < 0 or not np.isfinite(self.strength):
            raise ValueError("invalid prior/strength")


@dataclass(frozen=True)
class RoutingConfig:
    quality_slack: float = 0.05
    cost_weight: float = 0.0
    target_policy: str = "prior"

    def __post_init__(self):
        if any(not np.isfinite(x) or x < 0 for x in
               (self.quality_slack, self.cost_weight)):
            raise ValueError("slack/cost_weight must be finite and nonnegative")
        if self.target_policy not in {"uniform", "prior"}:
            raise ValueError("target policy must be uniform or prior")


class PoolRouter:
    """Qualified alternatives for ONE operation; separate instance per pool.

    quality must be an externally calibrated *prediction*, never a gold outcome.
    Guard uses predicted quality; it is NOT a guarantee of actual correctness.
    Top-1 selection does not change the chosen expert's composition strength.
    """
    def __init__(self, experts, routing=RoutingConfig(), control=ControlConfig()):
        self.experts = tuple(experts)
        if not self.experts or len({e.operation for e in self.experts}) != 1:
            raise ValueError("a pool must contain exactly one operation")
        if len({e.name for e in self.experts}) != len(self.experts):
            raise ValueError("duplicate expert name")
        self.routing = routing
        self.identity = digest({"experts": [asdict(e) for e in self.experts],
                                "routing": asdict(routing)})
        self.controller = CountController([e.name for e in self.experts], control, self.identity)

    def choose(self, operation, quality, predicted_cost, applicable=None, record=False):
        if operation != self.experts[0].operation:
            raise ValueError("cross-operation routing prohibited")
        n = len(self.experts)
        q = vector(quality, n, "predicted quality")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("quality predictions must be in [0, 1]")
        c = vector(predicted_cost, n, "predicted cost", True)
        app = np.ones(n, dtype=bool) if applicable is None else np.asarray(applicable)
        if app.shape != (n,) or app.dtype != bool:
            raise ValueError("applicable must be a boolean mask")
        eligible = app & np.array([e.qualified and e.prior > 0 for e in self.experts])
        if not eligible.any():
            raise ValueError("no qualified eligible alternative: abstain, do not force routing")
        safe = eligible & (q >= q[eligible].max() - self.routing.quality_slack - 1e-12)
        utility = q - self.routing.cost_weight * c
        biased = utility + self.controller.bias
        pick = int(np.argmax(np.where(safe, biased, -np.inf)))
        baseline = int(np.argmax(np.where(safe, utility, -np.inf)))
        target = np.where(safe, [e.prior for e in self.experts], 0.0)
        if self.routing.target_policy == "uniform":
            target = safe.astype(float)
        target /= target.sum()
        if record:
            counts = np.zeros(n)
            counts[pick] = 1
            self.controller.observe(counts, target)
        return {"expert_index": pick, "expert": self.experts[pick].name,
                "strength": self.experts[pick].strength, "eligible": eligible.tolist(),
                "safe": safe.tolist(), "target": target.tolist(),
                "overrode_unbiased": pick != baseline,
                "predicted_quality_gap": float(q[eligible].max() - q[pick]),
                "control_updates": self.controller.updates}

    def state_dict(self):
        return self.controller.state_dict()


class CallExecutor:
    """Dispatch one complete call; no KV-cache reuse between adapter selections.

    Backend signature: (expert_name, strength, request_dict) -> response_dict.
    No hidden memoization: all backend calls are charged. The configuration key
    is supplied for audit and for externally scoped cache implementations.
    """
    def __init__(self, router: PoolRouter, backend: Callable):
        self.router, self.backend = router, backend

    def __call__(self, operation, quality, predicted_cost, request, applicable=None):
        choice = self.router.choose(operation, quality, predicted_cost, applicable)
        response = self.backend(choice["expert"], choice["strength"], request)
        return {"choice": choice, "response": response,
                "cache_namespace": digest({"pool": self.router.identity,
                                            "expert": choice["expert"],
                                            "strength": choice["strength"]})}


def diagnostic(outcomes, selected, eligible):
    """Analysis-only oracle headroom, never imported into choose()."""
    y = np.asarray(outcomes, dtype=float)
    mask = np.asarray(eligible)
    sel = np.asarray(selected)
    if (y.ndim != 2 or len(y) == 0 or not np.isfinite(y).all()
            or mask.shape != y.shape or mask.dtype != bool
            or sel.shape != (len(y),) or not np.issubdtype(sel.dtype, np.integer)
            or (sel < 0).any() or (sel >= y.shape[1]).any()
            or not mask.any(axis=1).all()):
        raise ValueError("invalid outcome/selection/eligibility matrix")
    if (y < 0).any() or (y > 1).any() or not mask[np.arange(len(y)), sel].all():
        raise ValueError("outcomes must be [0,1]; selected experts must be eligible")
    best = np.where(mask, y, -np.inf).max(axis=1)
    actual = y[np.arange(len(y)), sel]
    return {"examples": len(y), "observed_success": float(actual.mean()),
            "oracle_headroom_only": float(best.mean()),
            "routing_regret": float((best - actual).mean()),
            "competence_gap_fraction": float((best == 0).mean()),
            "missed_available_success_fraction": float(((best == 1) & (actual == 0)).mean())}
