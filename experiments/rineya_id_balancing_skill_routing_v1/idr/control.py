"""Paper Eq. 12 plus explicitly optional Rineya safeguards.

Biases affect selection, never mixture weights. Updates use completed windows.
No model weights, gradients, rewards or confirmation labels enter this module.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def vector(value, size: int, name: str, nonnegative=False):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected {size} finite values")
    if nonnegative and (a < 0).any():
        raise ValueError(f"{name}: negative value")
    return a.copy()


@dataclass(frozen=True)
class ControlConfig:
    method: str = "id"
    ki: float = 0.01
    kd: float = 0.02
    bias_limit: float | None = 0.10
    min_expected: float = 4.0

    def __post_init__(self):
        if self.method not in {"none", "sign", "integral", "id"}:
            raise ValueError("unknown control method")
        if any(not np.isfinite(v) or v < 0 for v in
               (self.ki, self.kd, self.min_expected)):
            raise ValueError("gains and min_expected must be finite and nonnegative")
        if self.bias_limit is not None and (not np.isfinite(self.bias_limit)
                                           or self.bias_limit <= 0):
            raise ValueError("bias_limit must be positive or None")


def center_bound(value, limit):
    """Projection onto zero-sum bounded biases; the bias IS the integrator state."""
    x = value - value.mean()
    if limit is None or np.max(np.abs(x)) <= limit:
        return x
    lo, hi = float(x.min() - limit), float(x.max() + limit)
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.clip(x - mid, -limit, limit).sum() > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(x - (lo + hi) / 2, -limit, limit)


class CountController:
    """One controller per operation/pool (or per token-routed projection).

    begin/observe/commit groups microbatches into one optimizer-step window.
    abort discards an unsuccessful window. Evaluation cannot open a window.
    Rare target windows are deferred *as a whole*, preserving count mass.
    identity binds expert order, artifacts and target policy to the checkpoint.
    """
    def __init__(self, expert_ids, config=ControlConfig(), identity=""):
        self.expert_ids = tuple(expert_ids)
        if not self.expert_ids or len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("expert ids must be nonempty and unique")
        self.config, self.identity = config, str(identity)
        self.bias = np.zeros(len(self.expert_ids))
        self.previous = self.bias.copy()
        self.pending_counts = self.bias.copy()
        self.pending_targets = self.bias.copy()
        self.updates = 0
        self._window = None

    def begin(self, phase="train"):
        if phase not in {"train", "calibration"}:
            raise ValueError("evaluation/confirmation controller updates prohibited")
        if self._window is not None:
            raise RuntimeError("window already open")
        self._window = [np.zeros_like(self.bias), np.zeros_like(self.bias)]
        return self.bias.copy()

    def observe(self, counts, targets):
        if self._window is None:
            raise RuntimeError("open a window before recording completed assignments")
        n = vector(counts, len(self.bias), "counts", True)
        m = vector(targets, len(self.bias), "targets", True)
        if not np.allclose(n, np.round(n), atol=1e-9, rtol=0):
            raise ValueError("counts must be integer token/call assignments")
        if not np.isclose(n.sum(), m.sum(), atol=1e-7, rtol=1e-9):
            raise ValueError("count and target mass differ")
        if (n[m == 0] != 0).any():
            raise ValueError("assigned to a zero-target expert")
        self._window[0] += n
        self._window[1] += m

    def abort(self):
        self._window = None

    def commit(self):
        if self._window is None:
            raise RuntimeError("no completed window")
        n = self.pending_counts + self._window[0]
        m = self.pending_targets + self._window[1]
        self._window = None
        active = m > 0
        if not active.any():
            return {"updated": False, "reason": "empty", "updates": self.updates}
        if (m[active] < self.config.min_expected).any():
            self.pending_counts, self.pending_targets = n, m
            return {"updated": False, "reason": "rare_target", "updates": self.updates}
        e = np.zeros_like(m)
        e[active] = (m[active] - n[active]) / m[active]
        delta = e - self.previous
        gate = (self.previous * delta > 0) & active
        step = np.zeros_like(e)
        if self.config.method == "sign":
            step = self.config.ki * np.sign(e)
        elif self.config.method in {"integral", "id"}:
            step = self.config.ki * e
            if self.config.method == "id":
                step += self.config.kd * gate * delta
        raw = self.bias + step
        bounded = center_bound(raw, self.config.bias_limit)
        self.bias = bounded
        self.previous = e  # inactive experts get zero derivative history
        self.pending_counts.fill(0)
        self.pending_targets.fill(0)
        self.updates += 1
        return {"updated": True, "updates": self.updates,
                "error": e.tolist(), "gate": gate.tolist(),
                "counts": n.tolist(), "targets": m.tolist(),
                "bias": bounded.tolist(),
                "projection_linf": float(np.max(np.abs(bounded - (raw - raw.mean())))),
                "target_max_overload": float(max(0, -e.min())),
                "target_max_underload": float(max(0, e.max()))}

    def state_dict(self):
        if self._window is not None:
            raise RuntimeError("checkpoint only at completed optimizer-step boundaries")
        data = {"schema": 1, "expert_ids": list(self.expert_ids),
                "config": asdict(self.config), "identity": self.identity,
                "bias": self.bias.tolist(), "previous": self.previous.tolist(),
                "pending_counts": self.pending_counts.tolist(),
                "pending_targets": self.pending_targets.tolist(), "updates": self.updates}
        return {"payload": data, "sha256": digest(data)}

    def load_state_dict(self, saved):
        if self._window is not None:
            raise RuntimeError("cannot restore into an open window")
        d = saved["payload"]
        if digest(d) != saved["sha256"] or d["schema"] != 1:
            raise ValueError("controller checkpoint integrity/schema failure")
        if (tuple(d["expert_ids"]) != self.expert_ids or d["config"] != asdict(self.config)
                or d["identity"] != self.identity):
            raise ValueError("controller expert/config/artifact identity mismatch")
        vals = {k: vector(d[k], len(self.bias), k, k.startswith("pending"))
                for k in ("bias", "previous", "pending_counts", "pending_targets")}
        if abs(vals["bias"].mean()) > 1e-9:
            raise ValueError("uncentered checkpoint")
        limit = self.config.bias_limit
        if limit is not None and np.max(np.abs(vals["bias"])) > limit + 1e-9:
            raise ValueError("out-of-bound checkpoint")
        if not np.isclose(vals["pending_counts"].sum(), vals["pending_targets"].sum()):
            raise ValueError("corrupt deferred mass")
        if type(d["updates"]) is not int or d["updates"] < 0:
            raise ValueError("invalid update counter")
        for k, v in vals.items():
            setattr(self, k, v)
        self.updates = d["updates"]


def quantile_bias(scores, old_bias, k):
    """Paper Eq. 7 exact-quantile reference, uniform pool, next-window use only.

    This is not a distributed histogram implementation. Ties prevent exact loads.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or not len(s) or not np.isfinite(s).all():
        raise ValueError("nonempty finite score matrix required")
    if type(k) is not int or not 1 <= k < s.shape[1]:
        raise ValueError("quantile baseline requires 1 <= K < E")
    b = vector(old_bias, s.shape[1], "bias")
    cutoff = np.sort(s + b, axis=1)[:, -(k + 1)]
    target = -np.quantile(s - cutoff[:, None], 1 - k / s.shape[1], axis=0)
    return target - target.mean()
