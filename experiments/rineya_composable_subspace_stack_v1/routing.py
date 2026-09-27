"""Task-prototype routing baseline inspired by LORAUTER, not its reproduction.

Only prompt embeddings and CAL task labels enter fitting. Answers/correctness are
excluded from the routing interface. Feature provenance must be audited by the host.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .core import require
from .protocol import digest, hash_arg


def unit(x):
    a = np.asarray(x, dtype=np.float64)
    require(a.ndim == 1 and a.size > 0 and np.isfinite(a).all(), "finite feature vector required")
    n = float(np.linalg.norm(a))
    require(n > 1e-12, "zero feature vector")
    return a/n


@dataclass(frozen=True)
class PromptFeature:
    row_id: str
    parent_id: str
    prompt_sha256: str
    embedding: tuple[float, ...]

    def validate(self):
        require(bool(self.row_id) and bool(self.parent_id), "routing identity missing")
        hash_arg(self.prompt_sha256, "prompt digest")
        unit(self.embedding)


@dataclass
class TaskRouter:
    tasks: list[str]
    prototypes: list[list[float]]
    skill_weights: list[list[float]]
    feature_receipt: str
    cal_ids: list[str]
    cal_parents: list[str]
    cal_prompts: list[str]
    threshold: float = 0.2
    margin: float = 0.05
    top_k: int = 1
    temperature: float = 0.2

    def validate(self):
        require(self.tasks and len(set(self.tasks)) == len(self.tasks), "unique tasks required")
        require(len(self.tasks) == len(self.prototypes) == len(self.skill_weights), "router dimensions mismatch")
        p = [unit(x) for x in self.prototypes]
        require(all(x.shape == p[0].shape for x in p), "prototype width mismatch")
        w = np.asarray(self.skill_weights, dtype=float)
        require(w.ndim == 2 and w.shape[1] >= 2 and np.isfinite(w).all() and (w >= 0).all(), "invalid task-skill weights")
        require(np.allclose(w.sum(axis=1), 1, rtol=0, atol=1e-12), "task-skill weights must sum to one")
        require(type(self.top_k) is int and 1 <= self.top_k <= len(self.tasks), "invalid routing top-k")
        require(-1 <= self.threshold <= 1 and 0 <= self.margin <= 2 and math.isfinite(self.temperature) and self.temperature > 0, "invalid routing thresholds")
        hash_arg(self.feature_receipt, "feature receipt")
        return self

    def route(self, feature: PromptFeature, *, split="CHECK"):
        self.validate()
        feature.validate()
        require(split in ("CAL", "CHECK"), "unknown routing split")
        if split == "CHECK":
            require(feature.row_id not in self.cal_ids and feature.parent_id not in self.cal_parents and feature.prompt_sha256 not in self.cal_prompts, "routing CAL/CHECK leakage")
        z = unit(feature.embedding)
        p = np.array([unit(x) for x in self.prototypes])
        require(z.shape[0] == p.shape[1], "feature width mismatch")
        scores = p @ z
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.tasks[i]))
        gap = scores[order[0]]-scores[order[1]] if len(order) > 1 else 2.0
        fallback = scores[order[0]] < self.threshold or gap < self.margin
        weights = np.zeros(len(self.skill_weights[0]))
        selected = []
        if not fallback:
            indices = order[:self.top_k]
            logits = np.array([scores[i] for i in indices])/self.temperature
            probs = np.exp(logits-logits.max())
            probs /= probs.sum()
            weights = probs @ np.array([self.skill_weights[i] for i in indices])
            selected = [self.tasks[i] for i in indices]
        return {"weights": weights.tolist(), "fallback_base": bool(fallback),
                "selected_tasks": selected, "scores": scores.tolist(), "margin": float(gap),
                "router_sha256": digest(self.__dict__)}


def fit_router(features, task_labels, task_skills, *, split, feature_receipt, **kwargs):
    require(split == "CAL", "router fitting is CAL-only")
    require(len(features) == len(task_labels) and features, "routing training count mismatch")
    require(len({f.row_id for f in features}) == len(features), "duplicate calibration ids")
    require(len({f.parent_id for f in features}) == len(features), "one canonical routing view per CAL parent required")
    for f in features:
        f.validate()
    tasks = sorted(set(task_labels))
    require(set(task_skills) == set(tasks), "task mapping mismatch")
    protos = []
    for task in tasks:
        vectors = [unit(f.embedding) for f, t in zip(features, task_labels) if t == task]
        protos.append(unit(np.mean(vectors, axis=0)).tolist())
    return TaskRouter(tasks, protos, [task_skills[t] for t in tasks], feature_receipt,
                      [f.row_id for f in features], [f.parent_id for f in features],
                      [f.prompt_sha256 for f in features], **kwargs).validate()


def route_and_predict(router, features, prompts, predictor, *, split="CHECK"):
    """predictor(prompt, coefficients) can use modular or materialized skill deltas."""
    require(len(features) == len(prompts), "feature/prompt count mismatch")
    rows = []
    for f, prompt in zip(features, prompts):
        require(digest(prompt) == f.prompt_sha256, "feature not bound to supplied prompt")
        decision = router.route(f, split=split)
        rows.append({"id": f.row_id, "route": decision, "prediction": predictor(prompt, decision["weights"])})
    return rows


def routing_regret(candidate_losses, selected_losses):
    x, chosen = np.asarray(candidate_losses, float), np.asarray(selected_losses, float)
    require(x.ndim == 2 and x.shape[0] > 0 and x.shape[1] > 0 and chosen.shape == (x.shape[0],), "routing losses shape mismatch")
    require(np.isfinite(x).all() and np.isfinite(chosen).all(), "nonfinite routing loss")
    oracle = x.min(axis=1)
    best_static = float(x.mean(axis=0).min())
    return {"oracle_loss": float(oracle.mean()), "best_static_loss": best_static,
            "oracle_headroom": best_static-float(oracle.mean()),
            "routing_regret": float((chosen-oracle).mean()), "oracle_is_diagnostic_only": True}


class LinearSkillBank:
    """Additive inference hooks for already-bound torch.nn.Linear skill matrices.

    No checkpoint replacement, no base-weight mutation, one route per request.
    Host serialization/locking must ensure requests never share mutable routes.
    Each bank holds effective deltas, not separate LoRA factors.
    """
    def __init__(self, module, deltas_by_module):
        import torch
        require(not module.training, "routing bank is inference-only")
        modules = dict(module.named_modules())
        require(deltas_by_module, "empty skill bank")
        self.handles = []
        self.closed = False
        self.weights = None
        self.bank = {}
        self.width = None
        for name, deltas in deltas_by_module.items():
            require(name in modules and isinstance(modules[name], torch.nn.Linear), "bank target must be Linear")
            layer = modules[name]
            require(len(deltas) >= 2 and (self.width is None or len(deltas) == self.width), "bank donor count mismatch")
            self.width = len(deltas)
            converted = [torch.as_tensor(d, device=layer.weight.device, dtype=layer.weight.dtype).detach().clone() for d in deltas]
            require(all(d.shape == layer.weight.shape and bool(torch.isfinite(d).all()) for d in converted), "bank delta shape/value mismatch")
            self.bank[name] = converted
        try:
            for name in self.bank:
                def hook(layer, args, output, key=name):
                    require(self.weights is not None and not self.closed, "route must be selected before forward")
                    update = sum((a*d for a, d in zip(self.weights, self.bank[key])), torch.zeros_like(self.bank[key][0]))
                    return output + torch.nn.functional.linear(args[0], update)
                self.handles.append(modules[name].register_forward_hook(hook))
        except Exception:
            self.close()
            raise

    def select(self, weights):
        require(not self.closed and len(weights) == self.width, "closed/mismatched bank")
        require(all(type(a) in (float, int) and math.isfinite(a) and a >= 0 for a in weights), "invalid bank route")
        require(sum(weights) <= 1+1e-12, "route exceeds registered dose")
        self.weights = tuple(weights)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
