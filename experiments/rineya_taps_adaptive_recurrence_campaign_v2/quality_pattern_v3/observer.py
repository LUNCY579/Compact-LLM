"""Passive last-token measurements; the inherited recurrent operator is intact."""
from __future__ import annotations

import math
import time

import torch

from ..control import BlockRecurrence

SCHEMA = "PASSIVE_INPUT_FEATURES_V1"


def _finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("passive observer nonfinite tensor")


def last_token_stats(old, new, normal, previous=None):
    """All operands are current forward states, never labels or continuations."""
    if old.shape != new.shape or normal.shape != new.shape or old.ndim != 3 or old.shape[0] != 1:
        raise ValueError("observer requires same-shape batch1 [prefix,channel] states")
    a, b, ref = (value.detach().float()[0, -1] for value in (old, new, normal))
    _finite(a, b, ref)
    delta = b - a
    _finite(delta)
    norm = lambda value: float(value.norm())
    an, bn, dn = norm(a), norm(b), norm(delta)
    dot = float((a * b).sum())
    denominator = an * bn
    result = {
        "old_l2": an, "new_l2": bn, "delta_l2": dn,
        "old_rms": float(a.square().mean().sqrt()),
        "new_rms": float(b.square().mean().sqrt()),
        "normal_rms": float(ref.square().mean().sqrt()),
        "delta_rms": float(delta.square().mean().sqrt()),
        "state_dot": dot, "state_cosine_denominator": denominator,
        "state_cosine": None if denominator == 0 else max(-1., min(1., dot / denominator)),
        "delta_over_old_l2_denominator": an,
        "delta_over_old_l2": None if an == 0 else dn / an,
        "zero_old_norm": an == 0, "zero_new_norm": bn == 0, "zero_delta_norm": dn == 0,
        "denominator_policy": "NULL_IF_ZERO_NO_EPSILON_SUBSTITUTION",
        "previous_delta_l2": None, "delta_dot_previous": None,
        "delta_cosine_denominator": None, "delta_cosine_previous": None,
        "delta_over_previous_l2": None,
    }
    if previous is not None:
        if previous.shape != delta.shape:
            raise ValueError("previous last-token delta shape differs")
        _finite(previous)
        pn, dp = norm(previous), float((delta * previous).sum())
        den = dn * pn
        result.update(previous_delta_l2=pn, delta_dot_previous=dp, delta_cosine_denominator=den,
                      delta_cosine_previous=None if den == 0 else max(-1., min(1., dp / den)),
                      delta_over_previous_l2=None if pn == 0 else dn / pn)
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("passive observer nonfinite reduction")
    result["finite"] = True
    return result, delta.clone()


def readout_stats(logits, unknown_id=51378):
    """Actual head logits; nonUNKNOWN rival set includes every other head row."""
    if logits.ndim != 1 or logits.numel() < 2 or type(unknown_id) is not int or not 0 <= unknown_id < logits.numel():
        raise ValueError("passive readout vocabulary/UNKNOWN index differs")
    z = logits.detach().float()
    _finite(z)
    logp = z.log_softmax(0)
    p = logp.exp()
    values, indices = z.topk(2)
    # Build a separate candidate vector. The actual model logits stay untouched.
    rivals = torch.cat((z[:unknown_id], z[unknown_id + 1:]))
    rival_offset = int(rivals.argmax())
    rival_id = rival_offset if rival_offset < unknown_id else rival_offset + 1
    unknown, rival = z[unknown_id], z[rival_id]
    result = {
        "schema": SCHEMA, "vocabulary_rows": z.numel(), "UNKNOWN_token_id": unknown_id,
        "UNKNOWN_logit": float(unknown), "rival_logit": float(rival), "rival_token_id": rival_id,
        "UNKNOWN_minus_rival_margin": float(unknown - rival),
        "rival_set": "ALL_HEAD_ROWS_EXCEPT_UNKNOWN_INCLUDING_EOS_AND_SPECIAL_UNFILTERED",
        "top1_token_id": int(z.argmax()), "top1_probability": float(p[z.argmax()]),
        "top2_token_ids": indices.tolist(), "top2_logits": values.tolist(),
        "top2_margin": float(values[0] - values[1]),
        "entropy_nats": float(-(p * logp).sum()), "logsumexp": float(z.logsumexp(0)),
        "logits_changed": False, "target_independent": True, "finite": True,
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("passive readout nonfinite reduction")
    return result


class PassiveBlock(BlockRecurrence):
    """Override only recording; inherited math, hooks and guards execute first."""

    def _record(self, old, new, event, normal, core_seconds, controller_seconds):
        super()._record(old, new, event, normal, core_seconds, controller_seconds)
        started = time.perf_counter()
        iteration = event["iteration"]
        try:
            if self.config.scheduler_mode != "unit" or not bool((event["alpha"] == 1).all()):
                raise RuntimeError("passive probe permits exact unit updates only")
            if iteration not in (0, 1):
                raise RuntimeError("passive probe permits ordinary plus one extra block only")
            previous = None if iteration == 0 else getattr(self, "_observer_previous", None)
            if iteration == 1 and previous is None:
                raise RuntimeError("unit proposal has no same-forward ordinary delta")
            stats, delta = last_token_stats(old, new, normal, previous)
            self._observer_previous = delta if iteration == 0 and self.extra_loops == 1 else None
            feature = {
                "schema": SCHEMA, "sample_id": self.sample_id, "forward_index": self.forward_count,
                "iteration": iteration, "prefix_tokens": old.shape[1], "last_token_index": old.shape[1] - 1,
                "stage": "ORDINARY_PRE_EXTRA" if iteration == 0 else "UNIT_PROPOSAL_POST_EXTRA",
                "available_before_extra": iteration == 0, "extra_block_already_computed": iteration == 1,
                "prefix_kind": "ORIGINAL_INPUT_ONLY", "preceding_gold_tokens": 0,
                "target_independent": True, "teacher_conditioned": False,
                "alpha": 1., "statistics": stats,
            }
            if hasattr(self, "observer_sink") and self.observer_sink is not None:
                self.observer_sink(feature)
        except BaseException:
            self._observer_previous = None
            raise
        finally:
            self.observer_seconds = getattr(self, "observer_seconds", 0.) + time.perf_counter() - started


def observed_block(model, *, extra_loops, sample_id, sink, guard=None):
    from ..control import TAPSConfig
    wrapper = PassiveBlock(model, extra_loops=extra_loops, sample_id=sample_id,
                           config=TAPSConfig(scheduler_mode="unit"), guard=guard, arm_id="PASSIVE")
    wrapper.observer_sink, wrapper.observer_seconds = sink, 0.
    class Cleanup:
        def remove(self):
            wrapper._observer_previous = None
    wrapper.handles.append(Cleanup())
    return wrapper
