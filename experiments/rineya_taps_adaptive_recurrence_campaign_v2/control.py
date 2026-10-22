"""Successor control apparatus; the frozen v1 operator and A3 stay unchanged.

Random control draws form a stream over generated prefixes within one example.
Moment history still starts again at every full-prefix model call. This fixes
the v1 native A5 reset without changing its adaptive formula or recurrent depth.
"""
from __future__ import annotations

import hashlib
import json
import random

import torch

from experiments.rineya_taps_adaptive_recurrence_v1.integration import (
    BlockRecurrence as ReferenceBlockRecurrence,
)
from experiments.rineya_taps_adaptive_recurrence_v1.scheduler import (
    TAPSScheduler as ReferenceScheduler,
    TAPSConfig,
    apply_update,
    scalar_events,
)


class TAPSScheduler(ReferenceScheduler):
    """V1 moments plus independently seeded per-sequence random streams.

    ``reset_moments`` permits a new prefix shape while preserving RNG position.
    ``reset`` begins a new example evaluation and reproduces the initial stream.
    An example identity cannot change until the latter, even between prefixes.
    No random draw is consumed by the ordinary unit warmup observation.
    """

    def __init__(self, config: TAPSConfig, *, arm_id: str = "arm"):
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("nonempty arm_id required")
        self.arm_id = arm_id
        super().__init__(config)

    def reset(self):
        super().reset()
        self._stream_ids = ()
        self._streams = []
        self.random_draws = []

    def reset_moments(self):
        # Call the inherited reset directly, without restarting our RNG stream.
        super().reset()

    @torch.inference_mode()
    def observe(self, delta, sequence_ids, *, mask=None, force_unit=False,
                alpha_override=None):
        ids = tuple(sequence_ids)
        random_mode = self.config.scheduler_mode == "random"
        if random_mode and self._stream_ids and ids != self._stream_ids:
            raise ValueError("random stream identity changed: reset required")
        draw = (random_mode and self.config.enabled and not force_unit
                and alpha_override is None)
        # Validate and update inherited moments before consuming any randomness.
        alpha, event = super().observe(
            delta, ids, mask=mask, force_unit=force_unit or draw,
            alpha_override=alpha_override,
        )
        if random_mode and not self._stream_ids:
            self._stream_ids = ids
            for name in ids:
                binding = json.dumps([self.config.seed, self.arm_id, name],
                                     ensure_ascii=False, separators=(",", ":"))
                seed = int.from_bytes(hashlib.sha256(binding.encode("utf-8")).digest(), "big")
                self._streams.append(random.Random(seed))
            self.random_draws = [0] * len(ids)
        if draw:
            alpha = torch.tensor([rng.choice(self.config.random_pool) for rng in self._streams],
                                 dtype=alpha.dtype, device=alpha.device)
            self.random_draws = [n + 1 for n in self.random_draws]
            self.state.alpha = alpha
            event["alpha"] = alpha
        if random_mode:
            event["random_draw_index"] = torch.tensor(self.random_draws, device=alpha.device)
        return alpha, event


class BlockRecurrence(ReferenceBlockRecurrence):
    """One context is one example/arm/seed generation trajectory.

    Keep cache, frozen-state, shapes, overlap and restoration guards inherited.
    Native qualification remains required; this class never loads model assets.
    A caller must bind the actual stochastic replicate through ``config.seed``.
    Re-entering a context restarts that replicate deterministically.
    """

    def __init__(self, model, layers=(12, 13), extra_loops=1, *, config=None,
                 enabled=True, sink=None, sample_id="sample", guard=None,
                 arm_id="arm"):
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("nonempty sample_id required")
        cfg = config or TAPSConfig()
        self.arm_id = arm_id
        self._controller = TAPSScheduler(cfg, arm_id=arm_id)
        super().__init__(model, layers, extra_loops, config=cfg, enabled=enabled,
                         sink=sink, sample_id=sample_id, guard=guard)

    @property
    def random_binding(self):
        return {"seed": self.config.seed, "arm_id": self.arm_id,
                "sample_id": self.sample_id, "prefix_stream": True}

    def __enter__(self):
        result = super().__enter__()
        self._controller.reset()
        return result

    def _entry(self, module, args, kwargs):
        super()._entry(module, args, kwargs)
        self._controller.reset_moments()
        self.scheduler = self._controller

    def __exit__(self, *exc):
        try:
            super().__exit__(*exc)
        finally:
            self._controller.reset_moments()
