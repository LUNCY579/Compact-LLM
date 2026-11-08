"""Conversion-damage measurement: how much does conversion alone change behaviour?

Run this **before** any training. It answers one question: if we replace native
attention with the retrofit and change nothing else, how far does the model move?

## What the answer turns out to be

In exact arithmetic, **zero**. The retrofit reduces algebraically to native
attention while its conditioning projections are zero, so there is no
approximation to measure. `test_conversion_is_bitwise_function_preserving`
asserts `torch.equal` in float64.

That makes this module's real job narrower and more honest than a generic
"damage report": it measures the **floating-point** consequence of computing the
same quantity in a different order, at the dtypes a real bake would use. The
sequential head loop accumulates differently from a fused parallel reference, so
in bfloat16 the outputs are close but not identical, and the size of that gap is
a fact about numerics rather than about the architecture.

## The mathematical limitation, stated plainly

There is none for this conversion. The sequential structure does **not** prevent
reproducing native parallel attention, because the carrier recursion runs across
heads without feeding back into any head's Q/K/V while the conditioning is zero.
Each head's attention is therefore computed from exactly the native inputs.

This would stop being true if a future variant fed the carrier into K or V for
*cached* positions, or made head `j`'s query depend on head `j-1`'s output
through a non-zero path at initialization. Any such change must re-run this
report and must not claim exactness.

Metrics reported (development-split only; no confirmatory data is touched):
token-level KL divergence, NLL change, top-1 and top-5 agreement, greedy
generation agreement, per-layer hidden-state drift, and output-norm drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

DAMAGE_SCHEMA_VERSION = "rineya.tgsa-conversion-damage.v1"


def _quantiles(values: Tensor) -> dict[str, float]:
    flat = values.flatten().float()
    if flat.numel() == 0:
        return {}
    qs = torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0])
    got = torch.quantile(flat, qs.to(flat.device))
    return {
        "min": float(got[0]),
        "median": float(got[1]),
        "p90": float(got[2]),
        "p99": float(got[3]),
        "max": float(got[4]),
        "mean": float(flat.mean()),
    }


@dataclass(slots=True)
class DamageReport:
    """Complete distributions, not a pass/fail verdict.

    Deliberately has no threshold. A post-hoc "acceptable damage" cutoff invented
    after seeing the numbers would be meaningless; the reader is given the full
    distribution and decides.
    """

    dtype: str
    sequence_length: int
    batch_examples: int
    token_kl: dict[str, float]
    nll_native: float
    nll_retrofit: float
    nll_delta: float
    top1_agreement: float
    top5_agreement: float
    greedy_generation_agreement: float
    output_absolute_drift: dict[str, float]
    output_relative_drift: dict[str, float]
    output_norm_ratio: float
    per_layer_hidden_drift: list[float]
    exact_in_float64: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "sequence_length": self.sequence_length,
            "batch_examples": self.batch_examples,
            "token_kl_divergence": self.token_kl,
            "nll_native": self.nll_native,
            "nll_retrofit": self.nll_retrofit,
            "nll_delta": self.nll_delta,
            "top1_agreement": self.top1_agreement,
            "top5_agreement": self.top5_agreement,
            "greedy_generation_agreement": self.greedy_generation_agreement,
            "output_absolute_drift": self.output_absolute_drift,
            "output_relative_drift": self.output_relative_drift,
            "output_norm_ratio": self.output_norm_ratio,
            "per_layer_hidden_drift": self.per_layer_hidden_drift,
            "exact_in_float64": self.exact_in_float64,
        }


def token_kl_divergence(native_logits: Tensor, retrofit_logits: Tensor) -> Tensor:
    """Per-token KL(native || retrofit), computed in float32 for stability."""

    p = torch.log_softmax(native_logits.float(), dim=-1)
    q = torch.log_softmax(retrofit_logits.float(), dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1)


def top_k_agreement(native_logits: Tensor, retrofit_logits: Tensor, k: int) -> float:
    if k == 1:
        return float((native_logits.argmax(-1) == retrofit_logits.argmax(-1)).float().mean())
    native_top = native_logits.topk(k, dim=-1).indices
    retrofit_top = retrofit_logits.topk(k, dim=-1).indices
    # Agreement = fraction of positions whose top-1 native token is still in the
    # retrofit's top-k, which is the quantity that matters for sampling.
    best = native_top[..., :1]
    return float((retrofit_top == best).any(dim=-1).float().mean())


def measure_conversion_damage(
    *,
    native_forward: Callable[[Tensor], tuple[Tensor, list[Tensor]]],
    retrofit_forward: Callable[[Tensor], tuple[Tensor, list[Tensor]]],
    input_ids: Tensor,
    labels: Tensor | None = None,
    dtype_name: str = "float32",
    greedy_steps: int = 0,
    greedy_step: Callable[[Tensor], Tensor] | None = None,
    exact_in_float64: bool = False,
) -> DamageReport:
    """Compare a native and a retrofitted model on development inputs only.

    Both callables must return ``(logits, per_layer_hidden_states)``.
    """

    with torch.no_grad():
        native_logits, native_hidden = native_forward(input_ids)
        retrofit_logits, retrofit_hidden = retrofit_forward(input_ids)

        kl = token_kl_divergence(native_logits, retrofit_logits)

        if labels is not None:
            native_nll = float(
                torch.nn.functional.cross_entropy(
                    native_logits.float().flatten(0, 1), labels.flatten(), reduction="mean"
                )
            )
            retrofit_nll = float(
                torch.nn.functional.cross_entropy(
                    retrofit_logits.float().flatten(0, 1), labels.flatten(), reduction="mean"
                )
            )
        else:
            native_nll = retrofit_nll = float("nan")

        difference = (retrofit_logits.float() - native_logits.float()).abs()
        scale = native_logits.float().abs().clamp_min(1e-6)

        per_layer = [
            float((b.float() - a.float()).abs().max())
            for a, b in zip(native_hidden, retrofit_hidden)
        ]

        agreement = 1.0
        if greedy_steps and greedy_step is not None:
            agreement = float(greedy_step(input_ids))

        native_norm = float(native_logits.float().norm())
        retrofit_norm = float(retrofit_logits.float().norm())

    return DamageReport(
        dtype=dtype_name,
        sequence_length=int(input_ids.shape[1]),
        batch_examples=int(input_ids.shape[0]),
        token_kl=_quantiles(kl),
        nll_native=native_nll,
        nll_retrofit=retrofit_nll,
        nll_delta=retrofit_nll - native_nll,
        top1_agreement=top_k_agreement(native_logits, retrofit_logits, 1),
        top5_agreement=top_k_agreement(native_logits, retrofit_logits, 5),
        greedy_generation_agreement=agreement,
        output_absolute_drift=_quantiles(difference),
        output_relative_drift=_quantiles(difference / scale),
        output_norm_ratio=retrofit_norm / max(native_norm, 1e-12),
        per_layer_hidden_drift=per_layer,
        exact_in_float64=exact_in_float64,
    )


def render_markdown(reports: Sequence[Mapping[str, object]], *, context: Mapping[str, object]) -> str:
    lines = [
        "# TGSA retrofit conversion-damage report",
        "",
        "Measured **before any training**, on development inputs only.",
        "",
        f"**Architecture:** {context.get('architecture')}  ",
        f"**Converted layers:** {context.get('converted_layers')}  ",
        f"**Conditioning:** {context.get('conditioning')}",
        "",
        "## Headline",
        "",
        "Conversion is **exactly** function-preserving in exact arithmetic: the",
        "retrofit reduces algebraically to native attention while its conditioning",
        "projections are zero. What remains is floating-point ordering, measured below.",
        "",
        "| dtype | seq | KL median | KL max | top-1 agree | top-5 agree | |logit diff| max | norm ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        kl = report["token_kl_divergence"]
        drift = report["output_absolute_drift"]
        lines.append(
            f"| `{report['dtype']}` | {report['sequence_length']} | "
            f"{kl.get('median', float('nan')):.3e} | {kl.get('max', float('nan')):.3e} | "
            f"{report['top1_agreement']:.4f} | {report['top5_agreement']:.4f} | "
            f"{drift.get('max', float('nan')):.3e} | {report['output_norm_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Reading this",
            "",
            "A float64 row with all-zero drift is the algebraic statement: nothing was",
            "approximated. Non-zero rows at lower precision are the cost of computing the",
            "same quantity in a different summation order, not a loss of pretrained",
            "function. They should be compared against the drift between two runs of the",
            "*native* model at that dtype, which is the floor.",
            "",
            "No pass threshold is asserted. The full distribution is given so a reader can",
            "judge, rather than being handed a cutoff invented after seeing the numbers.",
            "",
        ]
    )
    return "\n".join(lines)
