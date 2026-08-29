"""Mandatory safety instrumentation for the sequential arms (brief section 42).

Why this module is not optional
-------------------------------
The staged recurrence writes ``alpha_s * G_s(X_{s-1})`` back into the tensor that the
next stage's ``q_proj``/``k_proj``/``v_proj`` read, with **no normalisation between
stages** (see ``wrapper.py``). Nothing in the architecture bounds ``X_s``: an arm with
``alphas = 1`` and sixteen stages can walk the post-``input_layernorm`` activation far
outside the range the weights were trained on, and the failure mode is not a crash but
a quietly degraded score that looks like a negative scientific result. So every arm run
is instrumented, and the instrumentation is reported next to the scores.

What is measured, per run
-------------------------
* per-layer activation RMS, and its **ratio** to the parallel baseline on the same input
* per-layer max ``|activation|``
* NaN / Inf counts, per layer and on the logits
* KL divergence of the next-token distribution against the baseline
* a hard abort when the RMS ratio leaves :attr:`StabilityConfig.rms_ratio_band`

A destabilised arm is a **result**, not an error to be smoothed over. Nothing here
clamps, rescales or retries; :class:`StabilityAbort` carries the full
:class:`StabilityReport` so the caller records exactly what happened and moves on.
``abort=False`` turns the hard stop into a flag when a sweep deliberately wants to
score an unstable arm anyway.

The band is two-sided, and the lower bound bites first
------------------------------------------------------
Measured on a randomly initialised Qwen3 of this shape, the ``alphas = 1/n_stages``
("compute-matched") variants leave the default band from **below** — a four-stage arm
lands near ``0.44``, a sixteen-stage arm near ``0.41`` — while the ``alphas = 1``
("naive") variants sit at ``1.00``. That is the arithmetic of ``arms.py``: the stages
partition the heads, so ``1/n`` scaling shrinks the whole attention write by ``1/n``.
On a *trained* checkpoint the residual stream dominates each layer's output and the
same arm moves the ratio far less, so the default ``(0.5, 2.0)`` is a live tripwire
rather than a formality either way. A sweep that intends to measure the compute-matched
convention should pass ``StabilityConfig(abort=False)`` and report the ratio, not widen
the band until nothing fires.

DEVIATION: brief section 42 says "activation RMS per layer". The probe reads each
decoder layer's **output** (the residual stream after that block), which is the tensor
whose growth actually compounds, rather than the attention block's internal ``X_s``.
The intermediate ``X_s`` is not observable from outside the wrapper without a second
hook mechanism, and its excursion shows up in the layer output on the same forward,
one layer later at the latest. Layer outputs are also directly comparable between the
parallel baseline and every arm, which the per-stage tensors are not (they do not
exist in the baseline).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from rineya_psa.sequential_attention.arms import SequentialArm
from rineya_psa.sequential_attention.wrapper import (
    decoder_layers,
    resolve_layers,
    sequential_session,
)

__all__ = [
    "ActivationProfile",
    "ArmRunResult",
    "DEFAULT_STABILITY_CONFIG",
    "LayerStats",
    "StabilityAbort",
    "StabilityConfig",
    "StabilityReport",
    "compare_profiles",
    "profile_arm",
    "profile_forward",
    "run_arm_with_stability",
]


class StabilityAbort(RuntimeError):
    """Raised when an arm leaves the configured safety band.

    Carries ``.report`` so the caller writes the measurement to the results table
    before re-raising or continuing; an abort with no record would turn a finding into
    a gap.
    """

    def __init__(self, report: StabilityReport, result: ArmRunResult | None = None) -> None:
        super().__init__(report.reason() or "stability violation")
        self.report = report
        self.result = result


@dataclass(frozen=True)
class StabilityConfig:
    """The safety band, and whether leaving it stops the run.

    The default band ``(0.5, 2.0)`` is a factor of two either way on per-layer RMS.
    That is wide enough that ``alphas = 1/n_stages`` shrinking the attention write does
    not trip it on its own, and narrow enough to catch a recurrence that is compounding.
    ``max_abs_limit`` is a separate, absolute tripwire for the case where RMS still
    looks sane because one position blew up and the rest did not.
    """

    rms_ratio_band: tuple[float, float] = (0.5, 2.0)
    max_abs_limit: float = 1.0e4
    kl_limit: float | None = None
    abort: bool = True

    def __post_init__(self) -> None:
        lo, hi = (float(self.rms_ratio_band[0]), float(self.rms_ratio_band[1]))
        if not 0.0 < lo < hi:
            raise ValueError(f"rms_ratio_band must satisfy 0 < lo < hi, got {(lo, hi)}")
        object.__setattr__(self, "rms_ratio_band", (lo, hi))
        if self.kl_limit is not None and float(self.kl_limit) < 0.0:
            raise ValueError("kl_limit must be non-negative")

    def to_json(self) -> dict[str, Any]:
        return {
            "rms_ratio_band": list(self.rms_ratio_band),
            "max_abs_limit": float(self.max_abs_limit),
            "kl_limit": None if self.kl_limit is None else float(self.kl_limit),
            "abort": bool(self.abort),
        }


DEFAULT_STABILITY_CONFIG = StabilityConfig()


@dataclass(frozen=True)
class LayerStats:
    """One decoder layer's output, reduced to four numbers on the device."""

    layer: int
    rms: float
    max_abs: float
    n_nan: int
    n_inf: int

    @property
    def finite(self) -> bool:
        return self.n_nan == 0 and self.n_inf == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "rms": float(self.rms),
            "max_abs": float(self.max_abs),
            "n_nan": int(self.n_nan),
            "n_inf": int(self.n_inf),
        }


@dataclass(frozen=True, eq=False)
class ActivationProfile:
    """Everything one instrumented forward produced. ``eq=False``: array payload."""

    label: str
    layer_stats: tuple[LayerStats, ...]
    next_token_logprobs: np.ndarray
    logits_max_abs: float
    n_nonfinite_logits: int
    n_tokens: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def rms(self) -> np.ndarray:
        return np.array([s.rms for s in self.layer_stats], dtype=np.float64)

    @property
    def max_abs(self) -> float:
        values = [s.max_abs for s in self.layer_stats]
        return float(max(values)) if values else float("nan")

    @property
    def n_nan(self) -> int:
        return int(sum(s.n_nan for s in self.layer_stats))

    @property
    def n_inf(self) -> int:
        return int(sum(s.n_inf for s in self.layer_stats))

    @property
    def finite(self) -> bool:
        return self.n_nan == 0 and self.n_inf == 0 and self.n_nonfinite_logits == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_tokens": int(self.n_tokens),
            "layer_stats": [s.to_json() for s in self.layer_stats],
            "logits_max_abs": float(self.logits_max_abs),
            "n_nonfinite_logits": int(self.n_nonfinite_logits),
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "meta": dict(self.meta),
        }


def _reduce_layer_output(tensor: torch.Tensor, layer: int) -> LayerStats:
    """Four scalars, computed in float32 so a bf16 sum of squares cannot overflow."""
    values = tensor.detach()
    n_nan = int(torch.isnan(values).sum().item())
    n_inf = int(torch.isinf(values).sum().item())
    finite = values.to(torch.float32)
    if n_nan or n_inf:
        finite = torch.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(torch.sqrt(finite.pow(2).mean()).item()) if finite.numel() else float("nan")
    max_abs = float(finite.abs().max().item()) if finite.numel() else float("nan")
    return LayerStats(layer=layer, rms=rms, max_abs=max_abs, n_nan=n_nan, n_inf=n_inf)


@contextlib.contextmanager
def _layer_probes(model: Any, sink: dict[int, LayerStats]) -> Iterator[None]:
    """Forward hooks on every decoder layer, removed on exit including on exception."""
    handles = []
    for index, block in enumerate(decoder_layers(model)):

        def hook(_module: Any, _inputs: Any, output: Any, index: int = index) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(tensor, torch.Tensor):
                sink[index] = _reduce_layer_output(tensor, index)

        handles.append(block.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def profile_forward(
    model: Any,
    input_ids: torch.Tensor,
    *,
    label: str,
    attention_mask: torch.Tensor | None = None,
    meta: Mapping[str, Any] | None = None,
) -> ActivationProfile:
    """One forward with per-layer probes attached, reduced to an :class:`ActivationProfile`.

    ``input_ids`` is ``[B, T]`` (batch 1 is the lab's default, SPEC 12.2). Only the
    final position's logits are kept — the KL comparison is on the next-token
    distribution, and keeping ``[B, T, V]`` is the memory wall of SPEC 11.1.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, T], got {tuple(input_ids.shape)}")
    sink: dict[int, LayerStats] = {}
    with _layer_probes(model, sink), torch.inference_mode():
        try:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        except TypeError:
            # A trust_remote_code head that predates `logits_to_keep`. Costs the full
            # [B, T, V] tensor, so it is a fallback rather than the default.
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
    logits = out.logits[:, -1, :].to(torch.float32)
    n_nonfinite = int((~torch.isfinite(logits)).sum().item())
    safe = torch.nan_to_num(logits, nan=-1.0e30, posinf=1.0e30, neginf=-1.0e30)
    logprobs = torch.log_softmax(safe, dim=-1).mean(dim=0)
    return ActivationProfile(
        label=label,
        layer_stats=tuple(sink[i] for i in sorted(sink)),
        next_token_logprobs=logprobs.detach().to("cpu", torch.float64).numpy(),
        logits_max_abs=float(safe.abs().max().item()),
        n_nonfinite_logits=n_nonfinite,
        n_tokens=int(input_ids.shape[1]),
        meta=dict(meta or {}),
    )


def _kl(baseline_logprobs: np.ndarray, arm_logprobs: np.ndarray) -> float:
    """``KL(baseline || arm)`` in nats, from two log-probability vectors.

    Baseline-first because the question is "how much of the reference model's
    behaviour did this arm lose", which is the direction that punishes an arm for
    putting no mass where the baseline was confident.
    """
    if baseline_logprobs.shape != arm_logprobs.shape:
        raise ValueError(
            f"logprob shapes differ: {baseline_logprobs.shape} vs {arm_logprobs.shape}"
        )
    p = np.exp(baseline_logprobs)
    total = float(p.sum())
    if not math.isfinite(total) or total <= 0.0:
        return float("nan")
    p = p / total
    diff = baseline_logprobs - arm_logprobs
    contributions = p * np.where(np.isfinite(diff), diff, 0.0)
    return float(np.clip(contributions.sum(), 0.0, None))


@dataclass(frozen=True, eq=False)
class StabilityReport:
    """The verdict plus every number behind it. ``stable`` is never inferred later."""

    label: str
    baseline_label: str
    rms_ratio: np.ndarray
    rms: np.ndarray
    baseline_rms: np.ndarray
    worst_layer: int
    worst_rms_ratio: float
    max_abs: float
    kl_next_token: float
    n_nan: int
    n_inf: int
    n_nonfinite_logits: int
    violations: tuple[str, ...]
    config: StabilityConfig
    arm: Mapping[str, Any] | None = None

    @property
    def stable(self) -> bool:
        return not self.violations

    def reason(self) -> str | None:
        return None if self.stable else "; ".join(self.violations)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "baseline_label": self.baseline_label,
            "rms": [float(x) for x in self.rms],
            "baseline_rms": [float(x) for x in self.baseline_rms],
            "rms_ratio": [float(x) for x in self.rms_ratio],
            "worst_layer": int(self.worst_layer),
            "worst_rms_ratio": float(self.worst_rms_ratio),
            "max_abs": float(self.max_abs),
            "kl_next_token": float(self.kl_next_token),
            "n_nan": int(self.n_nan),
            "n_inf": int(self.n_inf),
            "n_nonfinite_logits": int(self.n_nonfinite_logits),
            "violations": list(self.violations),
            "stable": self.stable,
            "config": self.config.to_json(),
            "arm": None if self.arm is None else dict(self.arm),
        }


def compare_profiles(
    profile: ActivationProfile,
    baseline: ActivationProfile,
    *,
    config: StabilityConfig = DEFAULT_STABILITY_CONFIG,
    arm: SequentialArm | None = None,
) -> StabilityReport:
    """Score one arm's profile against the parallel baseline's. Never raises on data.

    A layer whose baseline RMS is zero yields a NaN ratio rather than an infinity, and
    NaN ratios are treated as violations: an unmeasurable ratio is not a passing one.
    """
    if len(profile.layer_stats) != len(baseline.layer_stats):
        raise ValueError(
            f"profile has {len(profile.layer_stats)} layers, baseline has "
            f"{len(baseline.layer_stats)}"
        )
    rms = profile.rms
    base_rms = baseline.rms
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(base_rms > 0.0, rms / base_rms, np.nan)

    lo, hi = config.rms_ratio_band
    # "Worst" is the largest deviation from 1 in log space, so a 0.25x collapse and a
    # 4x explosion are ranked equally badly instead of the explosion always winning.
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = np.abs(np.log(np.where(ratio > 0.0, ratio, np.nan)))
    if deviation.size and np.isfinite(deviation).any():
        worst = int(np.nanargmax(np.where(np.isfinite(deviation), deviation, -np.inf)))
    else:
        worst = 0
    worst_ratio = float(ratio[worst]) if ratio.size else float("nan")

    violations: list[str] = []
    bad_band = [
        int(i)
        for i, value in enumerate(ratio)
        if not np.isfinite(value) or not (lo <= float(value) <= hi)
    ]
    if bad_band:
        violations.append(
            f"rms_ratio outside [{lo}, {hi}] at layers {bad_band[:8]}"
            + ("..." if len(bad_band) > 8 else "")
            + f" (worst {worst_ratio:.4g} at layer {worst})"
        )
    if profile.n_nan or profile.n_inf:
        violations.append(
            f"non-finite activations: {profile.n_nan} NaN, {profile.n_inf} Inf"
        )
    if profile.n_nonfinite_logits:
        violations.append(f"non-finite logits: {profile.n_nonfinite_logits}")
    if math.isfinite(profile.max_abs) and profile.max_abs > float(config.max_abs_limit):
        violations.append(
            f"max |activation| {profile.max_abs:.4g} > {config.max_abs_limit:.4g}"
        )

    kl = _kl(baseline.next_token_logprobs, profile.next_token_logprobs)
    if config.kl_limit is not None and (not math.isfinite(kl) or kl > float(config.kl_limit)):
        violations.append(f"next-token KL {kl:.4g} > {config.kl_limit:.4g}")

    return StabilityReport(
        label=profile.label,
        baseline_label=baseline.label,
        rms_ratio=ratio,
        rms=rms,
        baseline_rms=base_rms,
        worst_layer=worst,
        worst_rms_ratio=worst_ratio,
        max_abs=profile.max_abs,
        kl_next_token=kl,
        n_nan=profile.n_nan,
        n_inf=profile.n_inf,
        n_nonfinite_logits=profile.n_nonfinite_logits,
        violations=tuple(violations),
        config=config,
        arm=None if arm is None else arm.describe(),
    )


def profile_arm(
    model: Any,
    input_ids: torch.Tensor,
    layers: Iterable[int] | str | None,
    arm: SequentialArm,
    *,
    baseline: ActivationProfile | None = None,
    config: StabilityConfig = DEFAULT_STABILITY_CONFIG,
    attention_mask: torch.Tensor | None = None,
    label: str | None = None,
) -> tuple[ActivationProfile, StabilityReport]:
    """Install ``arm``, profile one forward, uninstall, and score it.

    ``baseline=None`` profiles the untouched model first, on the same input, which is
    the correct reference: the comparison must be against *this* model on *this*
    prompt, not against a stored curve from another run.
    """
    if baseline is None:
        baseline = profile_forward(
            model, input_ids, label="parallel_baseline", attention_mask=attention_mask
        )
    with sequential_session(model, layers, arm):
        profile = profile_forward(
            model,
            input_ids,
            label=label or arm.name,
            attention_mask=attention_mask,
            meta={"arm": arm.describe()},
        )
    return profile, compare_profiles(profile, baseline, config=config, arm=arm)


@dataclass(frozen=True, eq=False)
class ArmRunResult:
    """Scores and safety instrumentation, together — brief section 42's actual demand.

    ``records`` is whatever ``evaluation.runner.run_items`` returned, or ``None`` when
    the arm aborted before scoring. ``aborted`` distinguishes "scored, and it was
    unstable" from "not scored, because it was unstable".
    """

    arm: SequentialArm
    layers: tuple[int, ...]
    report: StabilityReport
    profile: ActivationProfile
    baseline: ActivationProfile
    records: Sequence[Any] | None
    aborted: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm.describe(),
            "layers": list(self.layers),
            "aborted": bool(self.aborted),
            "stable": self.report.stable,
            "n_records": None if self.records is None else len(self.records),
            "stability": self.report.to_json(),
            "profile": self.profile.to_json(),
            "baseline_profile": self.baseline.to_json(),
        }


def run_arm_with_stability(
    model: Any,
    arm: SequentialArm,
    layers: Iterable[int] | str | None,
    probe_input_ids: torch.Tensor,
    *,
    score: Any | None = None,
    baseline: ActivationProfile | None = None,
    config: StabilityConfig = DEFAULT_STABILITY_CONFIG,
    attention_mask: torch.Tensor | None = None,
) -> ArmRunResult:
    """Profile an arm, then score it — the entry point every arm sweep should use.

    ``score`` is a callable taking the installed model and returning the run's records
    (typically ``lambda m: run_items(m, tok, items, ...)``). It is invoked **inside** a
    second installation of the same arm, after the safety check, so an arm that leaves
    the band never gets to spend GPU-hours producing numbers nobody should quote.

    With ``config.abort=True`` a violation raises :class:`StabilityAbort` carrying both
    the report and a fully populated :class:`ArmRunResult` (``records=None``,
    ``aborted=True``), so the caller writes the finding to the results table instead of
    losing it. With ``abort=False`` the arm is scored anyway and ``result.report.stable``
    is ``False`` — which is the right setting for the sweep that wants to *measure* how
    badly a destabilised arm does.
    """
    resolved = resolve_layers(model, layers)
    if baseline is None:
        baseline = profile_forward(
            model, probe_input_ids, label="parallel_baseline", attention_mask=attention_mask
        )
    profile, report = profile_arm(
        model,
        probe_input_ids,
        resolved,
        arm,
        baseline=baseline,
        config=config,
        attention_mask=attention_mask,
    )

    if not report.stable and config.abort:
        raise StabilityAbort(
            report,
            ArmRunResult(
                arm=arm,
                layers=resolved,
                report=report,
                profile=profile,
                baseline=baseline,
                records=None,
                aborted=True,
            ),
        )

    records = None
    if score is not None:
        with sequential_session(model, resolved, arm):
            records = score(model)
    return ArmRunResult(
        arm=arm,
        layers=resolved,
        report=report,
        profile=profile,
        baseline=baseline,
        records=records,
        aborted=False,
    )
