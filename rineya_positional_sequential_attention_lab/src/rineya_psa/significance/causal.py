"""The causal Significance Matrix ``I[l, h, b]`` (brief sections 11-12, 15).

``Delta[l,h,b] = L_normal - L_perturbed`` on the chosen objective, averaged over items,
with the **per-item** deltas retained so ``statistics/`` can bootstrap them over
``scenario_id``. Note the sign: for a loss objective (``gold_nll``) an important cell
gives a *negative* Delta, so every ranking goes through
``SignificanceMatrix.importance()``, which re-signs by objective.

Two tiers, exactly as SPEC section 11.2 requires:

* **Tier 1, attribution.** One forward + one backward per item yields every cell at
  once, because for the Mode-2 gate ``s`` on cell ``(l,h,b)``,
  ``dL/ds = sum_{j in bin b} <dL/dV[l,h,j], V[l,h,j]>``. Implemented with a forward
  hook on each attention module's ``v_proj`` — the kernel's ``value`` tensor is a
  reshape of that output, so the inner product is exact. Linear response only, and
  labelled as such everywhere it is written down.
* **Tier 2, ablation.** Real ``eps`` attenuation, one cell per forward.
  ``validate_attribution_against_ablation`` reports the Spearman agreement that decides
  whether Tier 1 may be trusted.

**Cost.** Brute-force Tier 2 is ``n_items * (1 + L*H*B)`` forwards — 458 k at the
screening scale, i.e. days. Two accelerators, both here:

* ``mode="ladder"``: the coarse-to-fine ladder of brief section 15. Representative
  layers first (2, 7, 12, 17, 22, 27 for a 28-layer model), then recursive refinement of
  the neighbourhood of the winning layers, restricted to the winning (head, bin) pairs.
  Cells the ladder never visits stay ``NaN`` in ``per_item`` and ``False`` in
  ``evaluated``; they are *not* silently reported as zero.
* ``mode="grouped"``: several cells attenuated in **one** forward.

**Honesty about the grouped mode.** Distinct cells within one layer touch disjoint
entries of that layer's ``V`` (different KV head, or different source bin, or both), so
the combined Mode-2 table is exactly the product of the individual ones — the
*perturbation* composes exactly. The *loss delta* does not: ``L`` is a non-linear
function of ``V``, so a group delta equals the sum of its members' deltas only to first
order in ``eps``. Across layers it is worse still, because layer ``l``'s perturbed output
is layer ``l+1``'s input, so the two interact at first order rather than second.
Grouping is therefore **restricted to one layer per group** and is used as a *screen*:
groups whose joint delta is below threshold have all their cells written off as zero and
flagged in ``notes["screened_out"]``; surviving groups are re-run one cell at a time.
``validate_group_additivity`` measures the error against the sequential reference and is
what ``tests/test_significance.py`` pins.

DEVIATION (perturbation="key"). Scaling the key vectors is not expressible through the
sanctioned interface: ``k_proj``'s output is followed by Qwen3's ``k_norm`` RMSNorm,
which is scale-invariant and would undo a pre-projection hook, and the post-RoPE ``key``
tensor is only visible inside ``hooks/registry.py``, which this module must not edit. The
nearest valid alternative that removes the same thing — the attention mass flowing out of
bin ``b`` — is Mode 1 with a negative bias, ``z' = z - log(1/(1-eps))`` on that bin, which
multiplies the bin's pre-normalisation weight by exactly ``1 - eps``. ``eps`` keeps its
"fraction removed" meaning; the units of the underlying knob differ, and the result
records ``perturbation="key_as_logit_penalty"``.

DEVIATION (perturbation="head_out"). Mode 3 scales a whole query head's output and has no
bin axis; a *bin-restricted* head-output attenuation is algebraically identical to Mode 2
and would duplicate ``perturbation="value"``. So ``head_out`` collapses the bin axis: the
returned matrix has one bin (``AbsoluteBinning(1)``, i.e. the whole sequence) and cell
``(l, h)`` scales every query head in ``arch.q_heads_for_kv(h)`` by ``1 - eps``.

DEVIATION (SPEC section 12.1). SPEC says ``evaluation/runner.py`` owns the
``sdpa_kernel`` backend-forcing context manager. It does not implement one, and this
module must not edit it, so ``forced_sdpa_backend`` is defined here and wraps every
forward. It is a no-op on CPU and on torch builds without the backend enum.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rineya_psa.evaluation.runner import prepare_items
from rineya_psa.significance.matrices import OBJECTIVE_SIGN, OBJECTIVES, SignificanceMatrix
from rineya_psa.statistics.bootstrap import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    PairedBootstrapResult,
    bootstrap_ci_of_statistic,
    paired_bootstrap,
)
from rineya_psa.statistics.provenance import json_default
from rineya_psa.types import BenchmarkItem, TokenizedItem

__all__ = [
    "DEFAULT_EPS",
    "GPU_HOUR_BUDGET",
    "PERTURBATIONS",
    "REPRESENTATIVE_LAYER_STRIDE",
    "LadderConfig",
    "SignificanceResult",
    "assert_train_eval_equivalent",
    "bootstrap_cell",
    "cost_estimate",
    "estimate_causal_significance",
    "estimate_causal_significance_ablation",
    "estimate_causal_significance_attribution",
    "forced_sdpa_backend",
    "representative_layers",
    "select_validation_cells",
    "validate_attribution_against_ablation",
    "validate_group_additivity",
]

#: SPEC 5.3's default attenuation. Duplicated as a plain float so this module stays
#: importable without torch; ``hooks.specs.DEFAULT_ATTENUATION_EPS`` is the same number.
DEFAULT_EPS: float = 0.25

PERTURBATIONS: tuple[str, ...] = ("value", "key", "head_out")

#: Brief section 15's coarse grid is every fifth layer starting at 2.
REPRESENTATIVE_LAYER_STRIDE: int = 5

#: Measured forward seconds at batch 1 with the backend forced (SPEC 12.1).
_FORWARD_SECONDS: Mapping[int, float] = {
    2048: 0.119,
    4096: 0.288,
    8192: 0.834,
    16384: 2.27,
    32768: 7.06,
}

#: SPEC 11.2: a backward is roughly three forwards.
_BACKWARD_MULTIPLIER: float = 3.0

#: SPEC 11.3: a script above this must print its cost estimate and demand a flag.
GPU_HOUR_BUDGET: float = 2.0


# ======================================================================================
# small helpers
# ======================================================================================


@contextlib.contextmanager
def forced_sdpa_backend(enabled: bool = True, device: Any = None):
    """SPEC 12.1's cuDNN/mem-efficient pin. Silently inert where it cannot apply.

    ``device`` matters: MATH is the only SDPA backend available on CPU, so pinning
    CUDNN/EFFICIENT on a CPU tensor does not fall back — it raises "no viable backend".
    A CUDA-capable *machine* running a CPU model is exactly the CI and smoke-test case,
    so availability of a GPU is not on its own a licence to pin.
    """
    if not enabled:
        yield
        return
    import torch

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover - very old torch
        yield
        return
    if device is not None:
        kind = getattr(device, "type", None) or str(device).split(":")[0]
        if kind != "cuda":
            yield
            return
    elif not torch.cuda.is_available():
        yield
        return
    backends = [
        b
        for b in (
            getattr(SDPBackend, "CUDNN_ATTENTION", None),
            getattr(SDPBackend, "EFFICIENT_ATTENTION", None),
        )
        if b is not None
    ]
    if not backends:  # pragma: no cover - both enums exist in every supported build
        yield
        return
    with sdpa_kernel(backends):
        yield


def _nanmean_over_items(per_item: np.ndarray) -> np.ndarray:
    """``nanmean`` over the item axis, quiet about cells the ladder never visited.

    An all-NaN cell is the *intended* representation of "not measured" here, so the
    "Mean of empty slice" warning is noise that would train a reader to ignore warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(per_item, axis=0)


def representative_layers(
    n_layers: int, *, stride: int = REPRESENTATIVE_LAYER_STRIDE, start: int = 2
) -> tuple[int, ...]:
    """Brief section 15's coarse layer grid; ``(2, 7, 12, 17, 22, 27)`` at 28 layers."""
    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    return tuple(range(min(start, n_layers - 1), n_layers, stride))


def _binning_for(n_bins: int, binning: Any | None) -> Any:
    from rineya_psa.hooks.binning import AbsoluteBinning

    if binning is not None:
        return binning
    return AbsoluteBinning(int(n_bins))


def _arch_of(model: Any) -> Any:
    from rineya_psa.loading import arch_from_config

    return arch_from_config(model.config)


# ======================================================================================
# result container
# ======================================================================================


@dataclass(frozen=True, eq=False)
class SignificanceResult:
    """A significance matrix plus the per-item evidence behind every cell."""

    matrix: SignificanceMatrix
    per_item: np.ndarray  # [n_items, L, H, B]; NaN where the cell was never evaluated
    item_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    baseline: np.ndarray  # [n_items] L_normal
    evaluated: np.ndarray  # [L, H, B] bool
    tier: str
    perturbation: str
    objective: str
    eps: float
    n_forwards: int
    wall_seconds: float
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        per_item = np.asarray(self.per_item, dtype=np.float64)
        if per_item.shape != (len(self.item_ids), *self.matrix.shape):
            raise ValueError(
                f"per_item is {per_item.shape}, expected "
                f"{(len(self.item_ids), *self.matrix.shape)}"
            )
        object.__setattr__(self, "per_item", per_item)
        object.__setattr__(self, "evaluated", np.asarray(self.evaluated, dtype=bool))
        object.__setattr__(self, "baseline", np.asarray(self.baseline, dtype=np.float64))
        object.__setattr__(self, "notes", dict(self.notes))

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def cell_index(self, cell: tuple[int, int, int]) -> tuple[int, int, int]:
        """Model-space ``(layer, kv_head, bin)`` -> matrix-space indices."""
        layer, head, b = (int(x) for x in cell)
        try:
            return (
                self.matrix.layers.index(layer),
                self.matrix.kv_heads.index(head),
                self.matrix.bins.index(b),
            )
        except ValueError:
            raise KeyError(f"cell {cell} is outside this matrix's grid") from None

    def deltas_for(self, cell: tuple[int, int, int]) -> np.ndarray:
        """Per-item Deltas for one cell, in ``item_ids`` order."""
        li, hi, bi = self.cell_index(cell)
        return self.per_item[:, li, hi, bi].copy()

    def importance(self) -> np.ndarray:
        return self.matrix.importance()

    def coverage(self) -> float:
        """Fraction of the grid the estimator actually measured."""
        return float(self.evaluated.mean()) if self.evaluated.size else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "perturbation": self.perturbation,
            "objective": self.objective,
            "eps": self.eps,
            "n_items": self.n_items,
            "n_forwards": self.n_forwards,
            "wall_seconds": self.wall_seconds,
            "coverage": self.coverage(),
            "baseline_mean": float(np.nanmean(self.baseline)) if self.baseline.size else None,
            "matrix": self.matrix.summary(),
            "top_cells": [
                {"layer": c[0], "kv_head": c[1], "bin": c[2], "importance": v}
                for c, v in self.matrix.top_cells(16)
            ],
            "notes": dict(self.notes),
        }

    def save(self, path: str | Path, *, config: Mapping[str, Any] | None = None) -> Path:
        """Write ``values`` + ``per_item`` + metadata to ``.npz`` with a provenance sidecar.

        ``per_item`` is the payload ``statistics/`` needs: the matrix alone cannot be
        bootstrapped, so a result that saved only its mean would be unfalsifiable.
        """
        from rineya_psa.significance.matrices import write_npz_with_provenance

        payload = {
            "values": self.matrix.values,
            "per_item": self.per_item,
            "baseline": self.baseline,
            "evaluated": self.evaluated,
            "meta": np.asarray(json.dumps(self._meta(), default=json_default)),
        }
        return write_npz_with_provenance(payload, path, config=config)

    def _meta(self) -> dict[str, Any]:
        return {
            "matrix": {
                "layers": list(self.matrix.layers),
                "kv_heads": list(self.matrix.kv_heads),
                "bins": list(self.matrix.bins),
                "binning_desc": dict(self.matrix.binning_desc),
                "objective": self.matrix.objective,
                "model_key": self.matrix.model_key,
                "provenance": dict(self.matrix.provenance),
            },
            "item_ids": list(self.item_ids),
            "scenario_ids": list(self.scenario_ids),
            "tier": self.tier,
            "perturbation": self.perturbation,
            "objective": self.objective,
            "eps": self.eps,
            "n_forwards": self.n_forwards,
            "wall_seconds": self.wall_seconds,
            "notes": dict(self.notes),
        }

    @classmethod
    def load(cls, path: str | Path) -> SignificanceResult:
        """Inverse of :meth:`save`. Round-trips the per-item evidence, not just the mean."""
        with np.load(Path(path), allow_pickle=False) as archive:
            values = np.asarray(archive["values"], dtype=np.float64)
            per_item = np.asarray(archive["per_item"], dtype=np.float64)
            baseline = np.asarray(archive["baseline"], dtype=np.float64)
            evaluated = np.asarray(archive["evaluated"], dtype=bool)
            meta = json.loads(str(archive["meta"]))
        m = meta["matrix"]
        matrix = SignificanceMatrix(
            values=values,
            layers=tuple(m["layers"]),
            kv_heads=tuple(m["kv_heads"]),
            bins=tuple(m["bins"]),
            binning_desc=m["binning_desc"],
            objective=m["objective"],
            model_key=m["model_key"],
            provenance=m.get("provenance", {}),
        )
        return cls(
            matrix=matrix,
            per_item=per_item,
            item_ids=tuple(meta["item_ids"]),
            scenario_ids=tuple(meta["scenario_ids"]),
            baseline=baseline,
            evaluated=evaluated,
            tier=meta["tier"],
            perturbation=meta["perturbation"],
            objective=meta["objective"],
            eps=float(meta["eps"]),
            n_forwards=int(meta["n_forwards"]),
            wall_seconds=float(meta["wall_seconds"]),
            notes=meta.get("notes", {}),
        )


def bootstrap_cell(
    result: SignificanceResult,
    cell: tuple[int, int, int],
    *,
    n: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
) -> PairedBootstrapResult:
    """Scenario-clustered BCa interval for one cell's mean Delta (SPEC section 7).

    The per-item Deltas are already differences, so the paired bootstrap runs them
    against a zero comparator; clustering on ``scenario_id`` keeps the positional
    variants of one source example from being counted as independent evidence.
    """
    deltas = result.deltas_for(cell)
    keep = np.isfinite(deltas)
    if not keep.any():
        raise ValueError(f"cell {cell} was never evaluated; nothing to bootstrap")
    scenarios = [s for s, k in zip(result.scenario_ids, keep, strict=True) if k]
    values = deltas[keep]
    return paired_bootstrap(
        np.zeros_like(values), values, scenarios, n=n, seed=seed, ci=ci
    )


# ======================================================================================
# scoring
# ======================================================================================


def _objective_value(
    model: Any,
    tok: Any,
    item: BenchmarkItem,
    tok_item: TokenizedItem,
    objective: str,
    *,
    device: Any,
    differentiable: bool = False,
) -> Any:
    """One item's objective. Returns a float, or a torch scalar when differentiable."""
    import torch

    from rineya_psa.evaluation.metrics import accuracy, answer_logit_window

    if not tok_item.answer_ids:
        raise ValueError(f"{tok_item.item_id}: no answer tokens to score")
    ids = list(tok_item.input_ids) + list(tok_item.answer_ids)
    n_answer = len(tok_item.answer_ids)
    total = len(ids)
    start, stop = answer_logit_window(total, n_answer)
    seq = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    # Batch 1, unpadded, attention_mask=None (SPEC 12.2), and only the answer window of
    # the LM head is materialised (SPEC 11.1).
    out = model(input_ids=seq, use_cache=False, logits_to_keep=n_answer + 1)
    logits = out.logits[0]
    offset = total - int(logits.shape[0])
    rows = logits[start - offset : stop - offset]
    log_probs = torch.log_softmax(rows.float(), dim=-1)

    if objective == "gold_nll":
        gold = torch.tensor(list(tok_item.answer_ids), dtype=torch.long, device=log_probs.device)
        value = -log_probs.gather(-1, gold.unsqueeze(-1)).squeeze(-1).sum()
        return value if differentiable else float(value)
    if objective == "margin":
        gold_id = int(tok_item.answer_ids[0])
        first = log_probs[0]
        gold_lp = first[gold_id]
        competitor = first.clone()
        competitor[gold_id] = -torch.inf
        value = gold_lp - competitor.max()
        return value if differentiable else float(value)
    if objective == "correct":
        if differentiable:
            raise ValueError(
                "objective='correct' is a step function and has no gradient; use "
                "tier='ablation' for it"
            )
        prediction = tok.decode(
            [int(i) for i in log_probs.argmax(dim=-1).tolist()], skip_special_tokens=True
        )
        return 1.0 if accuracy(prediction, item) else 0.0
    raise ValueError(f"unknown objective {objective!r}; known: {list(OBJECTIVES)}")


def _score_all(
    model: Any,
    tok: Any,
    pairs: Sequence[tuple[BenchmarkItem, TokenizedItem]],
    objective: str,
    *,
    device: Any,
    ctx: Any | None = None,
    force_backend: bool = True,
) -> np.ndarray:
    """Score every item under one context. One ``psa_session`` for the whole sweep."""
    import torch

    from rineya_psa.hooks.registry import psa_session

    out = np.empty(len(pairs), dtype=np.float64)
    session = psa_session(model, ctx) if ctx is not None else contextlib.nullcontext()
    with session, torch.inference_mode(), forced_sdpa_backend(force_backend, device):
        for i, (item, tok_item) in enumerate(pairs):
            out[i] = float(
                _objective_value(model, tok, item, tok_item, objective, device=device)
            )
    return out


# ======================================================================================
# cell / group construction
# ======================================================================================


def _cell_grid(
    arch: Any, layers: Sequence[int] | None, kv_heads: Sequence[int] | None, n_bins: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    ls = tuple(range(arch.num_layers)) if layers is None else tuple(int(x) for x in layers)
    hs = tuple(range(arch.num_kv_heads)) if kv_heads is None else tuple(int(x) for x in kv_heads)
    bs = tuple(range(int(n_bins)))
    if any(not 0 <= x < arch.num_layers for x in ls):
        raise IndexError(f"layers out of range for {arch.num_layers} layers: {ls}")
    if any(not 0 <= x < arch.num_kv_heads for x in hs):
        raise IndexError(f"kv_heads out of range for {arch.num_kv_heads} KV heads: {hs}")
    return ls, hs, bs


def _make_context(
    arch: Any,
    binning: Any,
    cells: Sequence[tuple[int, int, int]],
    *,
    perturbation: str,
    eps: float,
) -> Any:
    """A ``PSAContext`` attenuating every cell in ``cells`` (all in the same layer)."""
    import numpy as _np

    from rineya_psa.hooks.specs import (
        HeadScaleSpec,
        LogitBiasSpec,
        PSAContext,
        ValueScaleSpec,
    )

    if not cells:
        raise ValueError("no cells to perturb")
    layers = sorted({int(c[0]) for c in cells})
    n_bins = binning.n_bins

    if perturbation == "value":
        table = _np.ones((arch.num_layers, arch.num_kv_heads, n_bins), dtype=_np.float64)
        for layer, head, b in cells:
            table[int(layer), int(head), int(b)] = 1.0 - float(eps)
        spec = ValueScaleSpec(table=table, binning=binning, layers=tuple(layers))
        return PSAContext(enabled=True, value_scale=spec, position_bins=binning)

    if perturbation == "key":
        # -log(1 - eps) subtracted from the bin's pre-softmax logits multiplies that
        # bin's unnormalised weight by exactly (1 - eps): the same fraction of mass the
        # Mode-2 gate removes from the value side.
        penalty = -math.log(max(1.0 - float(eps), 1e-6))
        table = _np.zeros((arch.num_layers, arch.num_kv_heads, n_bins), dtype=_np.float64)
        for layer, head, b in cells:
            table[int(layer), int(head), int(b)] = -penalty
        spec = LogitBiasSpec(table=table, lam=1.0, binning=binning, layers=tuple(layers))
        return PSAContext(enabled=True, logit_bias=spec, position_bins=binning)

    if perturbation == "head_out":
        gamma = _np.ones((arch.num_layers, arch.num_q_heads), dtype=_np.float64)
        for layer, head, _b in cells:
            for q in arch.q_heads_for_kv(int(head)):
                gamma[int(layer), int(q)] = 1.0 - float(eps)
        spec = HeadScaleSpec(gamma=gamma, layers=tuple(layers))
        return PSAContext(enabled=True, head_scale=spec, position_bins=binning)

    raise ValueError(f"unknown perturbation {perturbation!r}; known: {list(PERTURBATIONS)}")


def _group_cells(
    cells: Sequence[tuple[int, int, int]], group_size: int
) -> list[list[tuple[int, int, int]]]:
    """Partition into single-layer groups of at most ``group_size`` cells.

    Single-layer because a cross-layer group interacts at *first* order (layer ``l``
    feeds layer ``l+1``), which would make the screen unsound rather than merely
    approximate. See the module docstring.
    """
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    by_layer: dict[int, list[tuple[int, int, int]]] = {}
    for cell in cells:
        by_layer.setdefault(int(cell[0]), []).append(tuple(int(x) for x in cell))
    groups: list[list[tuple[int, int, int]]] = []
    for layer in sorted(by_layer):
        members = by_layer[layer]
        for start in range(0, len(members), group_size):
            groups.append(members[start : start + group_size])
    return groups


# ======================================================================================
# Tier 2: ablation
# ======================================================================================


@dataclass(frozen=True)
class LadderConfig:
    """Brief section 15's coarse-to-fine schedule."""

    stride: int = REPRESENTATIVE_LAYER_STRIDE
    start: int = 2
    #: How many coarse layers count as "winning" and seed the refinement.
    top_layers: int = 2
    #: Layers within this distance of a winner are refined in the next round.
    radius: int = 2
    #: Only the best this-many (head, bin) pairs are carried into refined layers.
    top_pairs: int = 8
    max_rounds: int = 3

    def __post_init__(self) -> None:
        for name in ("stride", "top_layers", "top_pairs", "max_rounds"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if int(self.radius) < 0:
            raise ValueError("radius must be non-negative")

    def describe(self) -> dict[str, Any]:
        return {
            "stride": self.stride,
            "start": self.start,
            "top_layers": self.top_layers,
            "radius": self.radius,
            "top_pairs": self.top_pairs,
            "max_rounds": self.max_rounds,
        }


def _evaluate_groups(
    model: Any,
    tok: Any,
    pairs: Sequence[tuple[BenchmarkItem, TokenizedItem]],
    groups: Sequence[Sequence[tuple[int, int, int]]],
    *,
    arch: Any,
    binning: Any,
    baseline: np.ndarray,
    perturbation: str,
    eps: float,
    objective: str,
    device: Any,
    force_backend: bool,
    progress: Any | None = None,
) -> list[np.ndarray]:
    """Per-item joint Delta for each group. ``len(result) == len(groups)``."""
    out: list[np.ndarray] = []
    for group in groups:
        ctx = _make_context(arch, binning, list(group), perturbation=perturbation, eps=eps)
        perturbed = _score_all(
            model, tok, pairs, objective, device=device, ctx=ctx, force_backend=force_backend
        )
        out.append(baseline - perturbed)
        if progress is not None:
            progress(len(pairs))
    return out


def estimate_causal_significance_ablation(
    model: Any,
    tok: Any,
    items: Iterable[BenchmarkItem | tuple[BenchmarkItem, TokenizedItem]],
    layers: Sequence[int] | None = None,
    kv_heads: Sequence[int] | None = None,
    n_bins: int = 8,
    *,
    eps: float = DEFAULT_EPS,
    perturbation: str = "value",
    objective: str = "gold_nll",
    binning: Any | None = None,
    mode: str = "ladder",
    ladder: LadderConfig | None = None,
    group_size: int = 1,
    screen_quantile: float = 0.5,
    model_key: str = "unknown",
    device: Any | None = None,
    force_backend: bool = True,
    progress: Any | None = None,
) -> SignificanceResult:
    """Tier 2: real ``eps`` attenuation, one cell (or one screened group) per forward.

    ``mode`` is ``"full"`` (every cell), ``"ladder"`` (brief section 15's coarse-to-fine
    schedule, the default because ``"full"`` is unaffordable at the screening scale), or
    ``"grouped"`` (screen groups of ``group_size`` cells, then refine the survivors).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; known: {list(OBJECTIVES)}")
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {perturbation!r}; known: {list(PERTURBATIONS)}")
    if mode not in ("full", "ladder", "grouped"):
        raise ValueError(f"unknown mode {mode!r}; known: ['full', 'ladder', 'grouped']")

    arch = _arch_of(model)
    if perturbation == "head_out":
        # See the module DEVIATION note: Mode 3 has no bin axis.
        binning = _binning_for(1, None)
        n_bins = 1
    binning = _binning_for(n_bins, binning)
    n_bins = int(binning.n_bins)
    ls, hs, bs = _cell_grid(arch, layers, kv_heads, n_bins)

    pairs = prepare_items(tok, items)
    if not pairs:
        raise ValueError("no items to score")
    if device is None:
        device = next(model.parameters()).device

    li_of = {v: i for i, v in enumerate(ls)}
    hi_of = {v: i for i, v in enumerate(hs)}
    bi_of = {v: i for i, v in enumerate(bs)}

    per_item = np.full((len(pairs), len(ls), len(hs), len(bs)), np.nan, dtype=np.float64)
    evaluated = np.zeros((len(ls), len(hs), len(bs)), dtype=bool)
    sign = float(OBJECTIVE_SIGN.get(objective, 1.0))

    started = time.perf_counter()
    baseline = _score_all(
        model, tok, pairs, objective, device=device, ctx=None, force_backend=force_backend
    )
    n_forwards = len(pairs)
    notes: dict[str, Any] = {"mode": mode, "grid": {"layers": list(ls), "kv_heads": list(hs)}}

    def record(cells: Sequence[tuple[int, int, int]], deltas: np.ndarray) -> None:
        for layer, head, b in cells:
            per_item[:, li_of[layer], hi_of[head], bi_of[b]] = deltas
            evaluated[li_of[layer], hi_of[head], bi_of[b]] = True

    def run(cells: Sequence[tuple[int, int, int]], size: int) -> None:
        nonlocal n_forwards
        groups = _group_cells(cells, size)
        results = _evaluate_groups(
            model, tok, pairs, groups,
            arch=arch, binning=binning, baseline=baseline, perturbation=perturbation,
            eps=eps, objective=objective, device=device, force_backend=force_backend,
            progress=progress,
        )
        n_forwards += len(groups) * len(pairs)
        for group, deltas in zip(groups, results, strict=True):
            record(group, deltas)

    all_cells = [(lay, h, b) for lay in ls for h in hs for b in bs]

    if mode == "full":
        run(all_cells, 1)

    elif mode == "grouped":
        groups = _group_cells(all_cells, max(2, int(group_size)))
        group_deltas = _evaluate_groups(
            model, tok, pairs, groups,
            arch=arch, binning=binning, baseline=baseline, perturbation=perturbation,
            eps=eps, objective=objective, device=device, force_backend=force_backend,
            progress=progress,
        )
        n_forwards += len(groups) * len(pairs)
        scores = np.array([abs(float(np.nanmean(d))) for d in group_deltas])
        threshold = float(np.quantile(scores, float(screen_quantile))) if scores.size else 0.0
        survivors: list[tuple[int, int, int]] = []
        screened: list[list[int]] = []
        for group, group_score, deltas in zip(groups, scores, group_deltas, strict=True):
            if group_score >= threshold:
                survivors.extend(group)
            else:
                # Written off as zero, not as NaN: the screen *did* look at them.
                for layer, head, b in group:
                    per_item[:, li_of[layer], hi_of[head], bi_of[b]] = 0.0
                    evaluated[li_of[layer], hi_of[head], bi_of[b]] = True
                screened.extend([[layer, head, b] for layer, head, b in group])
                del deltas
        notes["screen"] = {
            "group_size": max(2, int(group_size)),
            "n_groups": len(groups),
            "threshold_abs_mean_delta": threshold,
            "quantile": float(screen_quantile),
            "n_screened_out": len(screened),
            "caveat": (
                "screened cells are recorded as exactly zero; the screen assumes "
                "first-order additivity of within-layer group deltas"
            ),
        }
        notes["screened_out"] = screened
        run(survivors, 1)

    else:  # "ladder"
        cfg = ladder or LadderConfig()
        coarse = [
            lay
            for lay in representative_layers(
                arch.num_layers, stride=cfg.stride, start=cfg.start
            )
            if lay in li_of
        ]
        if not coarse:
            coarse = list(ls)
        rounds: list[dict[str, Any]] = []
        visited: set[int] = set()
        frontier = coarse
        radius = cfg.radius
        # Round 0 sweeps the whole (head, bin) grid and then narrows this; binding it
        # up front means a future edit to the round-0 branch cannot leave it unbound.
        carried_pairs: list[tuple[int, int]] = [(h, b) for h in hs for b in bs]
        for round_index in range(cfg.max_rounds):
            frontier = [lay for lay in frontier if lay not in visited]
            if not frontier:
                break
            if round_index == 0:
                cells = [(lay, h, b) for lay in frontier for h in hs for b in bs]
            else:
                cells = [(lay, h, b) for lay in frontier for (h, b) in carried_pairs]
            # Size 1, deliberately. `run(cells, k>1)` records the GROUP's joint delta
            # into every member cell, which is only defensible as a screen that is then
            # refined -- that is what mode="grouped" does. The ladder has no refinement
            # pass, so grouping here would write one number into k cells, mark them all
            # evaluated, and report coverage 1.0. Observed on a tiny model: a cell whose
            # true delta was +4.4e-3 was recorded as -7.2e-3, sign and all.
            run(cells, 1)
            visited.update(frontier)
            rounds.append(
                {"round": round_index, "layers": list(frontier), "n_cells": len(cells)}
            )

            mean = sign * _nanmean_over_items(per_item)  # importance, [L, H, B]
            layer_score = np.nanmax(
                np.where(evaluated, mean, np.nan), axis=(1, 2), initial=-np.inf
            )
            ranked = [
                ls[i]
                for i in np.argsort(np.where(np.isfinite(layer_score), layer_score, -np.inf))[::-1]
                if ls[i] in visited
            ]
            winners = ranked[: cfg.top_layers]
            pair_score = np.nanmax(
                np.where(evaluated, mean, np.nan)[[li_of[w] for w in winners]],
                axis=0,
                initial=-np.inf,
            )  # [H, B]
            order = np.argsort(pair_score, axis=None)[::-1][: cfg.top_pairs]
            carried_pairs = [
                (hs[int(i // len(bs))], bs[int(i % len(bs))]) for i in order
            ]
            frontier = sorted(
                {
                    ls[i]
                    for w in winners
                    for i, layer in enumerate(ls)
                    if abs(layer - w) <= radius and layer not in visited
                }
            )
            radius = max(1, radius // 2)
            notes.setdefault("ladder_winners", []).append(
                {"round": round_index, "winner_layers": winners,
                 "carried_pairs": [list(p) for p in carried_pairs]}
            )
        notes["ladder"] = cfg.describe()
        notes["ladder_rounds"] = rounds
        notes["ladder_caveat"] = (
            "cells outside the refined region were never evaluated and are NaN, not zero"
        )

    wall = time.perf_counter() - started
    values = _nanmean_over_items(per_item)
    matrix = SignificanceMatrix(
        values=values,
        layers=ls,
        kv_heads=hs,
        bins=bs,
        binning_desc=binning.describe(),
        objective=objective,
        model_key=model_key,
        provenance={
            "tier": "ablation",
            "perturbation": (
                "key_as_logit_penalty" if perturbation == "key" else perturbation
            ),
            "eps": float(eps),
            "mode": mode,
            "n_items": len(pairs),
        },
    )
    return SignificanceResult(
        matrix=matrix,
        per_item=per_item,
        item_ids=tuple(t.item_id for _, t in pairs),
        scenario_ids=tuple(i.scenario_id for i, _ in pairs),
        baseline=baseline,
        evaluated=evaluated,
        tier="ablation",
        perturbation=("key_as_logit_penalty" if perturbation == "key" else perturbation),
        objective=objective,
        eps=float(eps),
        n_forwards=n_forwards,
        wall_seconds=wall,
        notes=notes,
    )


def validate_group_additivity(
    model: Any,
    tok: Any,
    items: Iterable[BenchmarkItem | tuple[BenchmarkItem, TokenizedItem]],
    cells: Sequence[tuple[int, int, int]],
    *,
    eps: float = DEFAULT_EPS,
    perturbation: str = "value",
    objective: str = "gold_nll",
    n_bins: int = 8,
    binning: Any | None = None,
    device: Any | None = None,
    force_backend: bool = True,
) -> dict[str, Any]:
    """Group Delta versus the sum of the one-cell-at-a-time reference.

    This is the honesty check the module docstring promises. ``cells`` must all lie in
    one layer; the returned ``relative_error`` is what a caller should look at before
    trusting ``mode="grouped"`` at a given ``eps``.
    """
    arch = _arch_of(model)
    if perturbation == "head_out":
        binning, n_bins = _binning_for(1, None), 1
    binning = _binning_for(n_bins, binning)
    pairs = prepare_items(tok, items)
    if device is None:
        device = next(model.parameters()).device
    cells = [tuple(int(x) for x in c) for c in cells]
    if len({c[0] for c in cells}) > 1:
        raise ValueError(
            "group additivity is only claimed within a layer; cross-layer groups "
            "interact at first order"
        )
    if len(set(cells)) != len(cells):
        raise ValueError("cells must be distinct; a repeated cell is not a disjoint group")

    baseline = _score_all(
        model, tok, pairs, objective, device=device, ctx=None, force_backend=force_backend
    )
    singles = _evaluate_groups(
        model, tok, pairs, [[c] for c in cells],
        arch=arch, binning=binning, baseline=baseline, perturbation=perturbation,
        eps=eps, objective=objective, device=device, force_backend=force_backend,
    )
    joint = _evaluate_groups(
        model, tok, pairs, [list(cells)],
        arch=arch, binning=binning, baseline=baseline, perturbation=perturbation,
        eps=eps, objective=objective, device=device, force_backend=force_backend,
    )[0]
    summed = np.sum(np.stack(singles, axis=0), axis=0)
    scale = float(np.max(np.abs(summed))) if summed.size else 0.0
    diff = joint - summed
    return {
        "cells": [list(c) for c in cells],
        "eps": float(eps),
        "perturbation": perturbation,
        "objective": objective,
        "group_delta_mean": float(np.mean(joint)),
        "sum_of_singles_mean": float(np.mean(summed)),
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "relative_error": float(np.max(np.abs(diff)) / scale) if scale > 0 else 0.0,
        "per_item_group": joint.tolist(),
        "per_item_sum_of_singles": summed.tolist(),
        "n_items": len(pairs),
    }


# ======================================================================================
# Tier 1: attribution
# ======================================================================================


def assert_train_eval_equivalent(
    model: Any, input_ids: Any, *, atol: float = 0.0
) -> dict[str, Any]:
    """SPEC 12.4's precondition for gradient checkpointing.

    Checkpointing needs ``model.train()``, which is only safe if training mode changes
    nothing. Qwen3-0.6B has ``attention_dropout=0.0`` and no other dropout, so it should
    be bitwise identical — but "should" is why this function exists. Returns the measured
    difference and never flips the model's mode permanently.
    """
    import torch

    was_training = bool(model.training)
    try:
        model.eval()
        with torch.inference_mode():
            eval_logits = model(input_ids=input_ids, use_cache=False).logits.float().clone()
        model.train()
        with torch.inference_mode():
            train_logits = model(input_ids=input_ids, use_cache=False).logits.float().clone()
    finally:
        model.train(was_training)
    diff = float((eval_logits - train_logits).abs().max())
    return {"max_abs_diff": diff, "bitwise_equal": diff == 0.0, "within_atol": diff <= atol}


@contextlib.contextmanager
def _value_gradient_hooks(model: Any, layers: Sequence[int]):
    """Retain each attention module's ``v_proj`` output and its gradient.

    The kernel's ``value`` tensor is a view of this output (Qwen3 applies no norm and no
    RoPE to V), so ``<dL/dV, V>`` computed here is the exact derivative of the Mode-2
    gate. With every parameter frozen, the output carries no ``grad_fn``, hence
    ``requires_grad_(True)`` makes it a leaf whose ``.grad`` autograd will populate.
    """
    from rineya_psa.hooks.registry import attention_modules

    wanted = {int(x) for x in layers}
    saved: dict[int, Any] = {}
    handles = []

    def make(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not output.requires_grad:
                output.requires_grad_(True)
            output.retain_grad()
            saved[layer] = output

        return hook

    for module in attention_modules(model):
        layer = int(getattr(module, "layer_idx", -1))
        if layer not in wanted:
            continue
        if not hasattr(module, "v_proj"):
            raise RuntimeError(
                f"layer {layer} has no v_proj; attribution needs the value projection"
            )
        handles.append(module.v_proj.register_forward_hook(make(layer)))
    if len(handles) != len(wanted):
        raise RuntimeError(
            f"hooked {len(handles)} layers but {len(wanted)} were requested; "
            "layer_idx numbering does not match the requested layers"
        )
    try:
        yield saved
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def _frozen_parameters(model: Any):
    """Freeze every parameter and restore the flags on exit (SPEC 12.4's mitigation)."""
    saved = [(p, bool(p.requires_grad)) for p in model.parameters()]
    for param, _ in saved:
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param, flag in saved:
            param.requires_grad_(flag)


def estimate_causal_significance_attribution(
    model: Any,
    tok: Any,
    items: Iterable[BenchmarkItem | tuple[BenchmarkItem, TokenizedItem]],
    layers: Sequence[int] | None = None,
    kv_heads: Sequence[int] | None = None,
    n_bins: int = 8,
    *,
    eps: float = DEFAULT_EPS,
    objective: str = "gold_nll",
    binning: Any | None = None,
    model_key: str = "unknown",
    device: Any | None = None,
    force_backend: bool = True,
    progress: Any | None = None,
) -> SignificanceResult:
    """Tier 1: every cell from one forward + one backward per item (SPEC 11.2).

    ``I_hat[l,h,b] = eps * dL/ds`` with ``dL/ds = sum_{j in bin b} <dL/dV[l,h,j], V[l,h,j]>``.
    First-order only — never report it without
    :func:`validate_attribution_against_ablation`.
    """
    import torch

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; known: {list(OBJECTIVES)}")
    if objective == "correct":
        raise ValueError(
            "objective='correct' is a step function with zero gradient almost everywhere; "
            "attribution cannot estimate it. Use estimate_causal_significance_ablation."
        )
    arch = _arch_of(model)
    binning = _binning_for(n_bins, binning)
    if binning.is_query_dependent:
        raise ValueError(
            f"{binning.name} is query-dependent, so a source token has no single bin and "
            "dL/ds is not defined per cell. Attribution requires an AbsoluteBinning."
        )
    n_bins = int(binning.n_bins)
    ls, hs, bs = _cell_grid(arch, layers, kv_heads, n_bins)

    pairs = prepare_items(tok, items)
    if not pairs:
        raise ValueError("no items to score")
    if device is None:
        device = next(model.parameters()).device

    per_item = np.zeros((len(pairs), len(ls), len(hs), len(bs)), dtype=np.float64)
    baseline = np.empty(len(pairs), dtype=np.float64)
    head_index = torch.tensor(list(hs), dtype=torch.long, device=device)

    started = time.perf_counter()
    was_training = bool(model.training)
    model.eval()
    try:
        with _frozen_parameters(model), _value_gradient_hooks(model, ls) as saved:
            for i, (item, tok_item) in enumerate(pairs):
                saved.clear()
                with torch.enable_grad(), forced_sdpa_backend(force_backend, device):
                    loss = _objective_value(
                        model, tok, item, tok_item, objective,
                        device=device, differentiable=True,
                    )
                    baseline[i] = float(loss.detach())
                    model.zero_grad(set_to_none=True)
                    loss.backward()

                    total = len(tok_item.input_ids) + len(tok_item.answer_ids)
                    bin_ids = binning.bin_ids(total, device=device)
                    for li, layer in enumerate(ls):
                        value = saved.get(layer)
                        if value is None or value.grad is None:
                            raise RuntimeError(
                                f"no gradient reached layer {layer}'s v_proj output; the "
                                "graph was cut (inference_mode, or a parameter still "
                                "requires grad)"
                            )
                        # [B, T, H_kv * D] -> per (token, kv head) inner product with V.
                        flat = (value.detach() * value.grad).view(
                            value.shape[0], value.shape[1], arch.num_kv_heads, -1
                        )
                        per_token = flat.sum(dim=-1)[0]  # [T, H_kv]
                        per_bin = torch.zeros(
                            (arch.num_kv_heads, n_bins),
                            dtype=per_token.dtype,
                            device=per_token.device,
                        )
                        per_bin.index_add_(1, bin_ids, per_token.transpose(0, 1))
                        per_item[i, li] = (
                            float(eps)
                            * per_bin.index_select(0, head_index).double().cpu().numpy()
                        )
                if progress is not None:
                    progress(1)
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)

    wall = time.perf_counter() - started
    matrix = SignificanceMatrix(
        values=per_item.mean(axis=0),
        layers=ls,
        kv_heads=hs,
        bins=bs,
        binning_desc=binning.describe(),
        objective=objective,
        model_key=model_key,
        provenance={
            "tier": "attribution",
            "approximation": "first-order linear response of the Mode-2 gate",
            "eps": float(eps),
            "n_items": len(pairs),
        },
    )
    return SignificanceResult(
        matrix=matrix,
        per_item=per_item,
        item_ids=tuple(t.item_id for _, t in pairs),
        scenario_ids=tuple(i.scenario_id for i, _ in pairs),
        baseline=baseline,
        evaluated=np.ones((len(ls), len(hs), len(bs)), dtype=bool),
        tier="attribution",
        perturbation="value",
        objective=objective,
        eps=float(eps),
        n_forwards=len(pairs),
        wall_seconds=wall,
        notes={
            "estimator": "attribution_patching",
            "caveat": (
                "linear response: I_hat = eps * dL/ds recovers L(1) - L(1-eps) only to "
                "first order"
            ),
            "n_backwards": len(pairs),
        },
    )


# ======================================================================================
# Tier 1 / Tier 2 agreement
# ======================================================================================


def select_validation_cells(
    attribution: SignificanceResult,
    *,
    k: int = 64,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[tuple[int, int, int]]]:
    """SPEC 11.2's validation subset: the top-K cells plus K random controls."""
    imp = np.abs(attribution.matrix.values)
    flat = np.where(np.isfinite(imp), imp, -np.inf).reshape(-1)
    order = np.argsort(flat)[::-1]
    k = min(int(k), int(np.isfinite(imp).sum()))
    top_flat = order[:k]
    rest = np.array([i for i in order[k:] if np.isfinite(flat[i])], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    control_flat = rng.choice(rest, size=min(k, rest.size), replace=False) if rest.size else []

    def to_cells(indices: Any) -> list[tuple[int, int, int]]:
        cells = []
        for idx in np.atleast_1d(np.asarray(indices, dtype=np.int64)):
            li, hi, bi = np.unravel_index(int(idx), imp.shape)
            cells.append(
                (
                    int(attribution.matrix.layers[li]),
                    int(attribution.matrix.kv_heads[hi]),
                    int(attribution.matrix.bins[bi]),
                )
            )
        return cells

    return {"top": to_cells(top_flat), "control": to_cells(control_flat)}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation. scipy when present, otherwise Pearson on average ranks."""
    try:
        from scipy import stats

        return float(stats.spearmanr(x, y).statistic)
    except Exception:  # noqa: BLE001 - scipy is optional at runtime by SPEC's package list
        return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse = np.unique(values, return_inverse=True)
    if unique.size < values.size:
        sums = np.zeros(unique.size)
        counts = np.zeros(unique.size)
        np.add.at(sums, inverse, ranks)
        np.add.at(counts, inverse, 1.0)
        ranks = (sums / counts)[inverse]
    return ranks


def validate_attribution_against_ablation(
    attribution: SignificanceResult,
    ablation: SignificanceResult,
    *,
    n_resamples: int = 2000,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
    accept_rho: float = 0.5,
) -> dict[str, Any]:
    """SPEC 11.2's gate on Tier 1.

    Returns the Spearman rank correlation over the cells both estimators measured, a
    bootstrap CI for it, and the Bland-Altman agreement summary. ``verdict`` is
    ``"attribution_usable"`` only when the CI's lower bound clears ``accept_rho``; the
    caller is expected to fall back to coarse-to-fine ablation and say so otherwise.

    The CI resample is **clustered on layer**, not on individual cells. Cells are not
    exchangeable: every attribution value in a layer is a contraction of that layer's
    ``v_proj`` gradients, so residuals share a layer-level component by construction and
    a cell-level resample produces an anticonservative interval — which then makes the
    ``accept_rho`` gate on ``ci_low`` looser than this docstring claims. Clustering on
    layer is the same discipline SPEC section 7 applies to ``scenario_id``.

    ``eps`` is checked across the two estimators, because the Bland-Altman block reports
    their difference "in objective units" and that comparison is meaningless if the two
    were measured at different perturbation strengths.
    """
    if attribution.objective != ablation.objective:
        raise ValueError(
            f"objectives differ: {attribution.objective!r} vs {ablation.objective!r}"
        )
    a_eps, b_eps = float(attribution.eps), float(ablation.eps)
    if not math.isclose(a_eps, b_eps, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"eps differs between the two estimators ({a_eps} vs {b_eps}); the "
            "Bland-Altman difference would not be in comparable units. Re-measure at a "
            "single eps, or compare ranks only."
        )
    shared_bins = set(attribution.matrix.bins) & set(ablation.matrix.bins)
    xs: list[float] = []
    ys: list[float] = []
    used: list[tuple[int, int, int]] = []
    for li, layer in enumerate(ablation.matrix.layers):
        for hi, head in enumerate(ablation.matrix.kv_heads):
            for bi, b in enumerate(ablation.matrix.bins):
                if b not in shared_bins or not ablation.evaluated[li, hi, bi]:
                    continue
                try:
                    ai, aj, ak = attribution.cell_index((layer, head, b))
                except KeyError:
                    continue
                if not attribution.evaluated[ai, aj, ak]:
                    continue
                x = float(attribution.matrix.values[ai, aj, ak])
                y = float(ablation.matrix.values[li, hi, bi])
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                xs.append(x)
                ys.append(y)
                used.append((int(layer), int(head), int(b)))

    n = len(xs)
    if n < 4:
        raise ValueError(
            f"only {n} cells were measured by both estimators; the agreement statistic "
            "needs at least 4"
        )
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    rho = _spearman(x_arr, y_arr)
    pearson = float(np.corrcoef(x_arr, y_arr)[0, 1]) if n > 1 else float("nan")

    pairs = list(zip(xs, ys, strict=True))

    def rho_of(rows: list[Any]) -> float:
        a = np.asarray([r[0] for r in rows], dtype=np.float64)
        b = np.asarray([r[1] for r in rows], dtype=np.float64)
        if np.unique(a).size < 2 or np.unique(b).size < 2:
            raise ValueError("degenerate resample")
        return _spearman(a, b)

    layer_of = [c[0] for c in used]
    interval = bootstrap_ci_of_statistic(
        rho_of, pairs, cluster_ids=layer_of, n=int(n_resamples), seed=int(seed), ci=ci
    )

    diff = y_arr - x_arr
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    verdict = (
        "attribution_usable"
        if interval.ci_low >= float(accept_rho)
        else "fallback_to_coarse_to_fine_ablation"
    )
    return {
        "n_cells": n,
        "cells": [list(c) for c in used],
        "spearman": rho,
        "spearman_ci": interval.to_json(),
        "ci_clustered_on": "layer",
        "n_layers": len({c[0] for c in used}),
        "pearson": pearson,
        "bland_altman": {
            "mean_difference": mean_diff,
            "sd_difference": sd_diff,
            "limits_of_agreement": [mean_diff - 1.96 * sd_diff, mean_diff + 1.96 * sd_diff],
            "note": "difference is ablation minus attribution, in objective units",
        },
        "accept_rho": float(accept_rho),
        "verdict": verdict,
        "attribution": {"tier": attribution.tier, "eps": attribution.eps},
        "ablation": {
            "tier": ablation.tier,
            "eps": ablation.eps,
            "mode": ablation.notes.get("mode"),
        },
    }


# ======================================================================================
# the dispatcher and the cost gate
# ======================================================================================


def estimate_causal_significance(
    model: Any,
    tok: Any,
    items: Iterable[BenchmarkItem | tuple[BenchmarkItem, TokenizedItem]],
    layers: Sequence[int] | None = None,
    kv_heads: Sequence[int] | None = None,
    n_bins: int = 8,
    eps: float = DEFAULT_EPS,
    *,
    perturbation: str = "value",
    objective: str = "gold_nll",
    binning: Any | None = None,
    tier: str = "ablation",
    **kwargs: Any,
) -> SignificanceResult:
    """The Significance Matrix of brief sections 11-12.

    ``tier="ablation"`` (default) runs the real attenuation through the coarse-to-fine
    ladder; ``tier="attribution"`` runs the one-backward linear-response estimator. Both
    return the same :class:`SignificanceResult`, with ``tier`` recorded on it and in the
    matrix provenance so a downstream table can never mistake one for the other.
    """
    if tier == "attribution":
        if perturbation != "value":
            raise ValueError(
                f"attribution is derived for the Mode-2 value gate; perturbation="
                f"{perturbation!r} has no closed-form linear response here"
            )
        return estimate_causal_significance_attribution(
            model, tok, items, layers, kv_heads, n_bins,
            eps=eps, objective=objective, binning=binning, **kwargs,
        )
    if tier == "ablation":
        return estimate_causal_significance_ablation(
            model, tok, items, layers, kv_heads, n_bins,
            eps=eps, perturbation=perturbation, objective=objective, binning=binning,
            **kwargs,
        )
    raise ValueError(f"unknown tier {tier!r}; known: ['ablation', 'attribution']")


def _forward_seconds(seq_len: int) -> float:
    """Interpolate SPEC 12.1's measured table; extrapolate linearly in ``T`` outside it."""
    keys = sorted(_FORWARD_SECONDS)
    t = float(max(1, int(seq_len)))
    if t <= keys[0]:
        return _FORWARD_SECONDS[keys[0]] * t / keys[0]
    if t >= keys[-1]:
        return _FORWARD_SECONDS[keys[-1]] * t / keys[-1]
    lo = max(k for k in keys if k <= t)
    hi = min(k for k in keys if k >= t)
    if lo == hi:
        return _FORWARD_SECONDS[lo]
    w = (t - lo) / (hi - lo)
    return (1 - w) * _FORWARD_SECONDS[lo] + w * _FORWARD_SECONDS[hi]


def cost_estimate(
    *,
    n_items: int,
    n_layers: int,
    n_kv_heads: int,
    n_bins: int,
    seq_len: int = 4096,
    tier: str = "ablation",
    mode: str = "ladder",
    group_size: int = 1,
    ladder: LadderConfig | None = None,
    perturbation: str = "value",
) -> dict[str, Any]:
    """What this run will cost, before it is launched (SPEC 11.3's gate).

    Every caller that could exceed ``GPU_HOUR_BUDGET`` is expected to print this dict and
    require an explicit ``--yes-i-costed-this`` flag, per SPEC 11.3.
    """
    if perturbation == "head_out":
        n_bins = 1
    cells = int(n_layers) * int(n_kv_heads) * int(n_bins)
    per_forward = _forward_seconds(seq_len)

    if tier == "attribution":
        n_passes = int(n_items)
        seconds = n_passes * per_forward * (1.0 + _BACKWARD_MULTIPLIER)
        plan = {"n_backwards": n_passes, "cells_per_pass": cells}
    else:
        cfg = ladder or LadderConfig()
        if mode == "full":
            n_cells_visited = cells
            n_perturbed_passes = math.ceil(cells / max(1, group_size))
        elif mode == "grouped":
            size = max(2, int(group_size))
            screen = math.ceil(cells / size)
            # Half the groups survive a median screen by construction.
            n_perturbed_passes = screen + math.ceil(cells / 2)
            n_cells_visited = cells
        else:  # ladder
            if int(group_size) > 1:
                raise ValueError(
                    "mode='ladder' does not support group_size > 1: the ladder has no "
                    "refinement pass, so a group's joint delta would be recorded as every "
                    "member cell's individual delta. Use mode='grouped', which screens "
                    "groups and then re-runs the survivors one cell at a time."
                )
            coarse_layers = len(representative_layers(
                int(n_layers), stride=cfg.stride, start=cfg.start
            ))
            coarse_cells = coarse_layers * int(n_kv_heads) * int(n_bins)
            refined_layers = min(
                int(n_layers) - coarse_layers, cfg.top_layers * (2 * cfg.radius + 1)
            )
            refined_cells = max(0, refined_layers) * cfg.top_pairs
            n_cells_visited = coarse_cells + refined_cells
            # NOT divided by group_size: the ladder runs one cell per forward (see the
            # note at the run(cells, 1) call). Dividing here is what made a 9.5 GPU-hour
            # plan advertise itself as 1.2 hours and point the SPEC 11.3 budget gate
            # straight at a path that records group deltas as if they were cell deltas.
            n_perturbed_passes = n_cells_visited
        n_passes = int(n_items) * (1 + n_perturbed_passes)
        seconds = n_passes * per_forward
        plan = {
            "n_perturbed_configurations": n_perturbed_passes,
            "n_cells_visited": n_cells_visited,
            "coverage": n_cells_visited / cells if cells else 0.0,
        }

    brute_passes = int(n_items) * (1 + cells)
    gpu_hours = seconds / 3600.0
    return {
        "tier": tier,
        "mode": mode if tier == "ablation" else "all_cells_one_backward",
        "n_items": int(n_items),
        "n_cells": cells,
        "seq_len": int(seq_len),
        "seconds_per_forward": per_forward,
        # For tier="attribution" each pass is a forward AND a backward; report both so
        # "n_forwards: 256" cannot be read as the whole cost of a 256-forward-plus-
        # 256-backward plan.
        "n_forwards": n_passes,
        "n_backwards": n_passes if tier == "attribution" else 0,
        "estimated_seconds": seconds,
        "estimated_gpu_hours": gpu_hours,
        "exceeds_budget": gpu_hours > GPU_HOUR_BUDGET,
        "gpu_hour_budget": GPU_HOUR_BUDGET,
        "brute_force_forwards": brute_passes,
        "brute_force_gpu_hours": brute_passes * per_forward / 3600.0,
        # Ratio of SECONDS, not of passes. For tier="attribution" a pass is a forward
        # plus a backward, so a pass-ratio overstated the speedup by exactly the backward
        # multiplier (1793x reported against an actual 448x).
        "speedup_vs_brute_force": (
            (brute_passes * per_forward) / seconds if seconds else float("inf")
        ),
        "plan": plan,
        "warning": (
            "over the 2 GPU-hour budget of SPEC 11.3; pass --yes-i-costed-this"
            if gpu_hours > GPU_HOUR_BUDGET
            else None
        ),
    }
