"""Phase 4b — content-independent positional bias ``B`` and distance significance ``D``.

Brief sections 13 and 14, for the whole panel. Two matrices that separate *this head
matters here* (the causal significance matrix ``I`` of Phase 4a) from *this head always
looks here* (what this script measures).

1. **``B[l, h, b]`` — positional bias** (``rineya_psa.significance.bias``). Attention
   statistics on contexts that carry no retrievable content, so whatever structure
   survives belongs to the position machinery rather than to the text. Two control
   families, measured and reported **separately** so the agreement between them is a
   number rather than an assumption: ``random_ids`` (i.i.d. draws from the *real*
   unigram distribution, taken from the benchmark's own prompts by default) and
   ``shuffled_filler`` (the benchmark's inert filler prose, tokenized then
   token-shuffled). Three statistics through the sanctioned hooks capture path:
   attention mass by absolute source bin, mean pre-softmax qk logit by source bin, and
   attention entropy. ``B`` is reported at ``[l, q_head, b]`` **and** rolled up to
   ``[l, kv_head, b]``; the rollup is an approximation and every artifact says so (see
   "The KV rollup" below).

2. **``D[l, h, r]`` — distance significance** (``rineya_psa.significance.distance``),
   plus the additive decomposition ``S = A[abs] + D[dist]`` under the sum-to-zero
   identifiability constraint. Variance explained by absolute-only, distance-only and
   additive is reported per model, which is the direct numeric answer to brief
   question 7.

Both are **inference only** — no gradients — so SPEC 12.4's 2K backward ceiling does
not apply and the default context length is 8192.

The KV rollup
-------------
Attention probabilities exist per *query* head. SPEC 5.3's Mode 1 and Mode 2
interventions act at *KV*-head granularity. The bridge is the mean over the
``kv_group_size`` query heads that read each KV head, which is a fiction whenever two
query heads in one group have opposite positional profiles. Every artifact this script
writes carries ``kv_rollup_is_an_approximation: true``, the prose note, and the
within-group dispersion, so the rollup can never be read as a measurement.

SPEC 12.3 does NOT restrict this script
---------------------------------------
SPEC 12.3 restricts *distance-binned interventions* to ``T <= 2048``, because a
query-dependent Mode-1 bias needs a full ``[1, H, T, T]`` mask and a query-dependent
Mode-2 value scale is not expressible at all. **Measuring** ``D`` by capture has no such
limit: the capture path already chunks over query rows and never materialises a
``[T, T]`` tensor. Do not read the 2K number in SPEC 12.3 as a bound on this script.

Cost
----
Capture runs the eager chunked kernel, not SDPA, so SPEC 12.1's pinned-forward numbers
are a floor rather than an estimate. See :func:`capture_seconds` for the model and its
measured basis. At the defaults (four models, 8 contexts, 8192 tokens) the estimate is
about 11 GPU-minutes. ``--dry-run`` prints it and exits; anything over two GPU-hours
refuses to launch without ``--yes-i-costed-this``.

    PYTHONPATH=src python scripts/phase4b_bias_and_distance.py --dry-run

DEVIATION: brief section 46 names ``kv_head_bias_matrix.npz`` and
``distance_significance.npz`` without a model axis, but the panel has four checkpoints.
Those two files are therefore written with a leading model axis and a ``meta`` record
naming the models in order; the per-model, per-kind intermediates go to
``results/phase4b/`` so a single checkpoint can be re-read without unstacking.

DEVIATION: ``measure_positional_bias`` returns the across-context standard deviation but
not the per-context arrays, so a pooled-over-kinds :class:`PositionalBias` cannot carry a
correct within-set ``attn_by_bin_sd``. The pooled object carries ``None`` and each
kind's own sd survives in its own npz and in the summary, rather than a plausible
number that is actually a between-kind spread.

DEVIATION: Supa-A1.1 runs as one arm, not the gated/ungated pair Phase 3 used. The
abstention offset is a constant added to one row of the *output* logits inside
``forward``; it cannot reach the attention probabilities or the qk logits, so ``B`` and
``D`` are bit-identical in the two modes and a second arm would buy nothing but cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rineya_psa import paths  # noqa: E402
from rineya_psa.backend import efficient_attention, probe_backends  # noqa: E402
from rineya_psa.loading import (  # noqa: E402
    dtype_is_approximation,
    free_model,
    load_model,
    load_tokenizer,
)
from rineya_psa.significance.bias import (  # noqa: E402
    CONTEXT_KINDS,
    DEFAULT_BIAS_BINS,
    DEFAULT_QUERY_TAIL_FRACTION,
    BiasContext,
    PositionalBias,
    edge_bias_score,
    measure_positional_bias,
    middle_mass,
    random_token_contexts,
    shuffled_filler_contexts,
    unigram_distribution,
)
from rineya_psa.significance.distance import (  # noqa: E402
    DEFAULT_ABSOLUTE_BINS,
    DEFAULT_DISTANCE_BINS,
    DistanceProfile,
    measure_distance_profile,
    variance_explained_summary,
)
from rineya_psa.significance.matrices import write_npz_with_provenance  # noqa: E402
from rineya_psa.statistics.provenance import json_default, write_result  # noqa: E402
from rineya_psa.types import BenchmarkItem  # noqa: E402

#: One arm per checkpoint. See the gated/ungated DEVIATION in the module docstring.
PANEL: tuple[str, ...] = paths.PANEL

#: Both transforms of the additive fit are reported. ``log`` is primary: attention
#: weights come out of a softmax, so an additive effect on the pre-softmax logit — which
#: is what Mode 1 applies and what ``S`` parameterises — is multiplicative on the weight.
#: Fitting probabilities directly would reject additivity for a model that is exactly
#: additive in the natural parameter, so quoting only ``identity`` understates the fit
#: and quoting only ``log`` hides the choice.
TRANSFORMS: tuple[str, ...] = ("log", "identity")

KV_ROLLUP_NOTE = (
    "attention probabilities are per QUERY head; SPEC 5.3 Mode 1 and Mode 2 act at "
    "KV-head granularity. The [l, kv_head, b] arrays are the mean over the query heads "
    "of each GQA group -- an APPROXIMATION, not a measurement. kv_head_dispersion_* "
    "reports the within-group spread the mean threw away; where it is comparable to the "
    "group mean the rollup summarises nothing and any Mode 1/2 table built from it is "
    "unsafe."
)

SPEC_12_3_NOTE = (
    "SPEC 12.3's T<=2048 restriction is on distance-binned INTERVENTIONS, which need a "
    "[1, H, T, T] mask. Distance-binned CAPTURE chunks over query rows and has no such "
    "limit; these arrays were measured at the context length recorded in "
    "context_tokens."
)


# ======================================================================================
# cost model
# ======================================================================================

# SPEC 12.1's pinned SDPA forwards (0.288 s at T=4096, 0.834 s at T=8192) fitted with
# t = a*T + b*T^2; the quadratic term is attention, the linear one everything else. The
# fit reproduces the measured 2048 point at 0.112 s against 0.119 s measured, so it is
# honest to a few percent across the range this script runs in.
_PINNED_LINEAR = 3.881e-5
_PINNED_QUADRATIC = 7.69e-9

# Capture cannot use SDPA: it needs the probabilities, so ``hooks/registry.py`` runs an
# eager chunked kernel. SPEC 12.1's MATH column is the closest measured proxy for a full
# eager pass and is 8.0x the pinned forward at T=4096 (2.30 s vs 0.288 s).
_EAGER_OVER_PINNED = 8.0

# The per-chunk reductions on top of the kernel: the bin scatter, the fp32 softmax the
# entropy term needs, and the float64 numpy accumulate. Estimated, not measured, and
# deliberately generous -- an under-estimate here is what makes a sweep overrun.
_CAPTURE_REDUCTION_OVERHEAD = 1.5

# Load, .to(cuda) and free, per checkpoint. Measured in Phase 3's arm loop.
_MODEL_LOAD_SECONDS = 30.0


def pinned_forward_seconds(n_tokens: int) -> float:
    """SPEC 12.1's forward cost at an arbitrary length, from the two-point fit above."""
    t = float(n_tokens)
    return _PINNED_LINEAR * t + _PINNED_QUADRATIC * t * t


def capture_seconds(n_tokens: int) -> float:
    """One capture pass over one context: eager chunked attention plus the reductions."""
    return pinned_forward_seconds(n_tokens) * _EAGER_OVER_PINNED * _CAPTURE_REDUCTION_OVERHEAD


def estimate_seconds(
    n_models: int, n_contexts: int, n_tokens: int, *, per_context: float | None = None
) -> float:
    """Two passes per context — one for ``B``, one for the joint ``(abs, dist)`` table.

    Both passes visit every query-row chunk of every captured layer, so they cost the
    same even though the bias pass reduces only the query tail: ``valid_mask`` zeroes
    rows inside the reduction, it does not skip the kernel.
    """
    per = capture_seconds(n_tokens) if per_context is None else float(per_context)
    return n_models * (_MODEL_LOAD_SECONDS + 2.0 * n_contexts * per)


# ======================================================================================
# contexts
# ======================================================================================


def load_split_items(split: str, path: Path | None = None) -> list[BenchmarkItem]:
    """Benchmark items for the unigram distribution. Never touches the sealed split."""
    if split == "sealed":  # defence in depth; argparse already refuses it  # sealed-ok: refusal
        raise SystemExit("the sealed split is opened only by scripts/phase14_*.py")  # sealed-ok:
    target = path if path is not None else paths.DATA / split / "position_examples.jsonl"
    if not target.exists():
        return []
    return [BenchmarkItem.from_json(json.loads(line)) for line in target.open(encoding="utf-8")]


def benchmark_unigram(
    tok: object, split: str, n_items: int, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, object]]:
    """Empirical unigram over real benchmark prompts, with a filler fallback.

    "Random token ids drawn from the real token distribution" is only true if the
    distribution comes from the material the study actually scores on. Uniform draws
    over the vocabulary would concentrate on barely-trained embedding rows, and the
    filler-only default inside ``bias.py`` measures the filler's vocabulary rather than
    the benchmark's. Both fallbacks are recorded rather than silently substituted.
    """
    items = load_split_items(split)
    if not items:
        print(f"  no benchmark at data/{split}/; unigram falls back to the inert filler")
        return unigram_distribution(tok, seed=seed), {
            "unigram_source": "filler", "unigram_n_texts": 0, "unigram_split": None,
        }
    rng = np.random.default_rng(int(seed))
    take = min(int(n_items), len(items))
    idx = rng.choice(len(items), size=take, replace=False)
    texts = [f"{items[int(i)].system}\n{items[int(i)].user}" for i in idx]
    return unigram_distribution(tok, texts, seed=seed), {
        "unigram_source": "benchmark",
        "unigram_n_texts": take,
        "unigram_split": split,
        "unigram_item_ids": sorted(items[int(i)].item_id for i in idx),
    }


def build_contexts(
    tok: object,
    kind: str,
    n_contexts: int,
    n_tokens: int,
    *,
    seed: int,
    distribution: tuple[np.ndarray, np.ndarray] | None,
) -> list[BiasContext]:
    if kind == "random_ids":
        return random_token_contexts(
            tok, n_contexts, n_tokens, seed=seed, distribution=distribution
        )
    if kind == "shuffled_filler":
        return shuffled_filler_contexts(tok, n_contexts, n_tokens, seed=seed)
    raise ValueError(f"unknown context kind {kind!r}; known: {list(CONTEXT_KINDS)}")


# ======================================================================================
# pooling and summaries
# ======================================================================================


def pool_bias(parts: Sequence[PositionalBias], extra: Mapping[str, object]) -> PositionalBias:
    """Context-count-weighted mean of per-kind measurements.

    Weighting by ``n_contexts`` rather than averaging the two kind means keeps the pooled
    figure equal to what a single run over the union would have produced when the kinds
    are unbalanced (``--limit`` can unbalance them).
    """
    ref = parts[0]
    for other in parts[1:]:
        if other.shape != ref.shape or other.layers != ref.layers:
            raise ValueError("cannot pool bias measurements taken on different grids")
        if dict(other.binning_desc) != dict(ref.binning_desc):
            raise ValueError("cannot pool bias measurements taken on different binnings")
    counts = np.asarray([p.n_contexts for p in parts], dtype=np.float64)
    w = counts / counts.sum()
    return PositionalBias(
        attn_by_bin=sum(wi * p.attn_by_bin for wi, p in zip(w, parts, strict=True)),
        qk_logit_mean=sum(wi * p.qk_logit_mean for wi, p in zip(w, parts, strict=True)),
        entropy=sum(wi * p.entropy for wi, p in zip(w, parts, strict=True)),
        layers=ref.layers,
        q_heads=ref.q_heads,
        bins=ref.bins,
        binning_desc=ref.binning_desc,
        model_key=ref.model_key,
        n_kv_heads=ref.n_kv_heads,
        context_kinds=tuple(sorted({k for p in parts for k in p.context_kinds})),
        n_contexts=int(counts.sum()),
        provenance={
            **dict(ref.provenance),
            "pooled_over_kinds": {p.context_kinds[0]: int(p.n_contexts) for p in parts},
            "pooling_weights": [float(x) for x in w],
            "sd_note": (
                "attn_by_bin_sd is None on a pooled object: the per-kind runs return a "
                "mean and an sd but not the per-context arrays, so a correct within-set "
                "spread cannot be reconstructed here. Each kind's own sd is in its own "
                "npz and in the summary."
            ),
            "kv_rollup_is_an_approximation": True,
            "kv_rollup_note": KV_ROLLUP_NOTE,
            **dict(extra),
        },
        attn_by_bin_sd=None,
    )


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho; scipy when present, average-rank Pearson when it is not."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    try:
        from scipy import stats

        return float(stats.spearmanr(a[ok], b[ok]).statistic)
    except Exception:  # noqa: BLE001 - SPEC lists scipy, but it is optional at runtime
        def ranks(v: np.ndarray) -> np.ndarray:
            order = np.argsort(v, kind="stable")
            out = np.empty(v.size, dtype=np.float64)
            out[order] = np.arange(v.size, dtype=np.float64)
            return out

        return float(np.corrcoef(ranks(a[ok]), ranks(b[ok]))[0, 1])


def kind_agreement(by_kind: Mapping[str, PositionalBias]) -> dict[str, object]:
    """Does ``B`` depend on which flavour of nonsense produced it?

    If it does, the "content-independent" label is wrong and every downstream use of
    ``B`` as a content-free reference has to be qualified. Reported as a number so that
    judgement is available rather than assumed.
    """
    kinds = sorted(by_kind)
    if len(kinds) < 2:
        return {"note": f"only {kinds} measured; no cross-kind check possible"}
    out: dict[str, object] = {"kinds": kinds}
    for i, a in enumerate(kinds):
        for b in kinds[i + 1 :]:
            pa, pb = by_kind[a].attn_by_bin, by_kind[b].attn_by_bin
            qa, qb = by_kind[a].qk_logit_mean, by_kind[b].qk_logit_mean
            out[f"{a}_vs_{b}"] = {
                "attn_pearson": float(np.corrcoef(pa.reshape(-1), pb.reshape(-1))[0, 1]),
                "attn_spearman": rank_correlation(pa, pb),
                "attn_max_abs_diff": float(np.abs(pa - pb).max()),
                "attn_mean_abs_diff": float(np.abs(pa - pb).mean()),
                "qk_spearman": rank_correlation(qa, qb),
                "entropy_mean_abs_diff": float(
                    np.abs(by_kind[a].entropy - by_kind[b].entropy).mean()
                ),
            }
    return out


def bias_summary(bias: PositionalBias) -> dict[str, object]:
    """``PositionalBias.summary`` plus the KV-rollup honesty block."""
    attn_disp = bias.kv_head_dispersion("attn_by_bin")
    kv_attn = bias.to_kv_heads("attn_by_bin")
    # Dispersion is only interpretable next to the thing it disperses around: 0.01 is
    # tiny beside a group mean of 0.5 and fatal beside one of 0.02.
    scale = np.abs(kv_attn).mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(scale > 0, attn_disp / np.where(scale > 0, scale, 1.0), np.nan)
    out = dict(bias.summary())
    out["kv_rollup"] = {
        "is_an_approximation": True,
        "note": KV_ROLLUP_NOTE,
        "attn_dispersion_mean": float(np.nanmean(attn_disp)),
        "attn_dispersion_max": float(np.nanmax(attn_disp)),
        "attn_dispersion_over_group_mean_median": float(np.nanmedian(ratio)),
        "attn_dispersion_over_group_mean_max": float(np.nanmax(ratio)),
        "qk_dispersion_mean": float(np.nanmean(bias.kv_head_dispersion("qk_logit_mean"))),
    }
    edge = edge_bias_score(bias)
    mid = middle_mass(bias, width=0.5, reduce="sum")
    worst = np.unravel_index(int(np.nanargmax(edge)), edge.shape)
    out["edge_bias"] = {
        "mean": float(np.nanmean(edge)),
        "max": float(np.nanmax(edge)),
        "max_at": {"layer": int(bias.layers[int(worst[0])]),
                   "q_head": int(bias.q_heads[int(worst[1])])},
        "middle_mass_mean": float(np.nanmean(mid)),
        "middle_mass_min": float(np.nanmin(mid)),
        "n_heads_edge_biased": int(np.count_nonzero(edge > 0)),
        "n_heads": int(edge.size),
    }
    return out


def distance_summary(profile: DistanceProfile) -> dict[str, object]:
    """Question 7's number, under both transforms, plus the marginal profiles."""
    out = dict(profile.summary(transform="log"))
    variance: dict[str, object] = {}
    for transform in TRANSFORMS:
        fit = profile.fit_additive(transform=transform)
        variance[transform] = {
            "identifiability": fit.identifiability,
            "weighting": fit.weighting,
            "converged": fit.converged,
            **variance_explained_summary(fit),
        }
    out["variance_explained"] = variance
    out["spec_12_3_note"] = SPEC_12_3_NOTE
    out["kv_rollup"] = {"is_an_approximation": True, "note": KV_ROLLUP_NOTE}
    return out


# ======================================================================================
# per-model measurement
# ======================================================================================


def run_model(
    key: str,
    args: argparse.Namespace,
    contexts_by_kind: Mapping[str, list[BiasContext]],
    layers: tuple[int, ...] | None,
) -> tuple[PositionalBias, dict[str, PositionalBias], DistanceProfile, dict[str, object]]:
    """Load one checkpoint, run the two capture passes, free it again."""
    model, arch = load_model(key, device=args.device)
    on_cuda = str(next(model.parameters()).device).startswith("cuda")
    per_kind: dict[str, PositionalBias] = {}
    timings: dict[str, float] = {}
    try:
        # SPEC 12.1: the pin is process-global and CUDA-only, hence ``active``. Capture
        # itself is eager, but ``psa_session`` leaves un-captured layers on SDPA and the
        # dispatcher would pick MATH for those.
        with efficient_attention(active=on_cuda) as pinned:
            for kind in sorted(contexts_by_kind):
                ctxs = contexts_by_kind[kind]
                seen = [0]
                n_ctx = len(ctxs)

                # Defaults bind the loop variables: a closure over ``kind`` would report
                # every arm under the last kind's name.
                def tick(n: int, _seen=seen, _k=kind, _tot=n_ctx, _key=key) -> None:
                    _seen[0] += n
                    print(f"  {_key}/bias/{_k} {_seen[0]}/{_tot}", flush=True)

                t0 = time.perf_counter()
                per_kind[kind] = measure_positional_bias(
                    model,
                    contexts=ctxs,
                    layers=layers,
                    n_bins=args.bias_bins,
                    chunk=args.chunk,
                    query_tail_fraction=args.query_tail_fraction,
                    model_key=key,
                    progress=tick,
                )
                timings[f"bias_{kind}_seconds"] = time.perf_counter() - t0
                print(
                    f"  {key}/bias/{kind}: {len(ctxs)} contexts in "
                    f"{timings[f'bias_{kind}_seconds']:.1f} s "
                    f"({timings[f'bias_{kind}_seconds'] / max(len(ctxs), 1):.2f} s/context)",
                    flush=True,
                )

            pooled_contexts = [c for kind in sorted(contexts_by_kind)
                               for c in contexts_by_kind[kind]]
            seen_d = [0]
            n_pooled = len(pooled_contexts)

            def tick_d(n: int, _seen=seen_d, _tot=n_pooled, _key=key) -> None:
                _seen[0] += n
                print(f"  {_key}/distance {_seen[0]}/{_tot}", flush=True)

            t0 = time.perf_counter()
            profile = measure_distance_profile(
                model,
                contexts=pooled_contexts,
                layers=layers,
                n_absolute_bins=args.abs_bins,
                n_distance_bins=args.dist_bins,
                chunk=args.chunk,
                # 1.0 on purpose: the distance axis is DEFINED relative to the query row,
                # so early rows are the short-look-back end of the distribution, not a
                # causal-masking artefact to be excluded. cell_counts carries their reach.
                query_tail_fraction=1.0,
                model_key=key,
                progress=tick_d,
            )
            timings["distance_seconds"] = time.perf_counter() - t0
            print(
                f"  {key}/distance: {len(pooled_contexts)} contexts in "
                f"{timings['distance_seconds']:.1f} s",
                flush=True,
            )
    finally:
        free_model(model)

    meta = {
        "model_key": key,
        "backend_pinned": bool(pinned),
        "dtype_approximated": dtype_is_approximation(key),
        "arch": {
            "num_layers": arch.num_layers,
            "num_q_heads": arch.num_q_heads,
            "num_kv_heads": arch.num_kv_heads,
            "kv_group_size": arch.kv_group_size,
        },
        **timings,
    }
    pooled = pool_bias(
        [per_kind[k] for k in sorted(per_kind)],
        extra={"spec_12_3_note": SPEC_12_3_NOTE, **{k: v for k, v in meta.items()
                                                    if k != "model_key"}},
    )
    return pooled, per_kind, profile, meta


# ======================================================================================
# artifacts
# ======================================================================================


def write_panel_bias(
    path: Path, biases: Mapping[str, PositionalBias], config: Mapping[str, object]
) -> Path:
    """Brief section 46's ``kv_head_bias_matrix.npz``, with a leading model axis."""
    keys = list(biases)
    ref = biases[keys[0]]
    stack = {name: np.stack([getattr(biases[k], name) for k in keys]) for name in
             ("attn_by_bin", "qk_logit_mean", "entropy")}
    arrays: dict[str, object] = {
        "attn_by_bin_q": stack["attn_by_bin"],
        "qk_logit_mean_q": stack["qk_logit_mean"],
        "entropy_q": stack["entropy"],
        "attn_by_bin_kv": np.stack([biases[k].to_kv_heads("attn_by_bin") for k in keys]),
        "qk_logit_mean_kv": np.stack([biases[k].to_kv_heads("qk_logit_mean") for k in keys]),
        "kv_head_dispersion_attn": np.stack(
            [biases[k].kv_head_dispersion("attn_by_bin") for k in keys]
        ),
        "kv_head_dispersion_qk": np.stack(
            [biases[k].kv_head_dispersion("qk_logit_mean") for k in keys]
        ),
        "edge_bias_score_q": np.stack([edge_bias_score(biases[k]) for k in keys]),
        "middle_mass_q": np.stack(
            [middle_mass(biases[k], width=0.5, reduce="sum") for k in keys]
        ),
        "bin_centers": ref.bin_centers,
        "meta": np.asarray(json.dumps({
            "models": keys,
            "axes": {
                "attn_by_bin_q": ["model", "layer", "q_head", "bin"],
                "attn_by_bin_kv": ["model", "layer", "kv_head", "bin"],
                "kv_head_dispersion_attn": ["model", "layer", "kv_head"],
                "entropy_q": ["model", "layer", "q_head"],
            },
            "layers": list(ref.layers),
            "q_heads": list(ref.q_heads),
            "n_kv_heads": int(ref.n_kv_heads),
            "bins": list(ref.bins),
            "binning_desc": dict(ref.binning_desc),
            "context_kinds": list(ref.context_kinds),
            "n_contexts": int(ref.n_contexts),
            "kv_rollup_is_an_approximation": True,
            "kv_rollup_note": KV_ROLLUP_NOTE,
            "per_model_provenance": {k: dict(biases[k].provenance) for k in keys},
        }, default=json_default)),
    }
    return write_npz_with_provenance(arrays, path, config=config)


def write_panel_distance(
    path: Path, profiles: Mapping[str, DistanceProfile], config: Mapping[str, object]
) -> Path:
    """Brief section 46's ``distance_significance.npz``: ``D``, ``A`` and the fit."""
    keys = list(profiles)
    ref = profiles[keys[0]]
    fits = {k: profiles[k].fit_additive(transform="log") for k in keys}
    arrays: dict[str, object] = {
        "distance_marginal_q": np.stack([profiles[k].distance_marginal() for k in keys]),
        "distance_marginal_kv": np.stack(
            [profiles[k].to_kv_heads(profiles[k].distance_marginal()) for k in keys]
        ),
        "absolute_marginal_q": np.stack([profiles[k].absolute_marginal() for k in keys]),
        "absolute_marginal_kv": np.stack(
            [profiles[k].to_kv_heads(profiles[k].absolute_marginal()) for k in keys]
        ),
        "joint_mass_q": np.stack([profiles[k].joint_mass for k in keys]),
        "cell_counts": np.stack([profiles[k].cell_counts for k in keys]),
        "additive_mu": np.stack([fits[k].mu for k in keys]),
        "additive_absolute": np.stack([fits[k].absolute for k in keys]),
        "additive_distance": np.stack([fits[k].distance for k in keys]),
        "r2_absolute_only": np.stack([fits[k].r2_absolute_only for k in keys]),
        "r2_distance_only": np.stack([fits[k].r2_distance_only for k in keys]),
        "r2_additive": np.stack([fits[k].r2_additive for k in keys]),
        "meta": np.asarray(json.dumps({
            "models": keys,
            "axes": {
                "distance_marginal_q": ["model", "layer", "q_head", "distance_bin"],
                "distance_marginal_kv": ["model", "layer", "kv_head", "distance_bin"],
                "joint_mass_q": ["model", "layer", "q_head", "abs_bin", "distance_bin"],
                "additive_absolute": ["model", "layer", "q_head", "abs_bin"],
                "additive_distance": ["model", "layer", "q_head", "distance_bin"],
            },
            "layers": list(ref.layers),
            "q_heads": list(ref.q_heads),
            "n_kv_heads": int(ref.n_kv_heads),
            "absolute_desc": dict(ref.absolute_desc),
            "distance_desc": dict(ref.distance_desc),
            "additive_transform": "log",
            "identifiability": fits[keys[0]].identifiability,
            "weighting": fits[keys[0]].weighting,
            "spec_12_3_note": SPEC_12_3_NOTE,
            "kv_rollup_is_an_approximation": True,
            "kv_rollup_note": KV_ROLLUP_NOTE,
            "per_model_provenance": {k: dict(profiles[k].provenance) for k in keys},
        }, default=json_default)),
    }
    return write_npz_with_provenance(arrays, path, config=config)


# ======================================================================================
# driver
# ======================================================================================


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="calibration", choices=("calibration", "validation"),
                    help="unigram source split; the sealed split is refused")  # sealed-ok:
    ap.add_argument("--models", default=None, help="comma-separated subset of the panel")
    ap.add_argument("--kinds", default=",".join(CONTEXT_KINDS),
                    help=f"comma-separated control families; known: {list(CONTEXT_KINDS)}")
    ap.add_argument("--context-tokens", type=int, default=8192,
                    help="capture is inference-only, so 8K is affordable (SPEC 12.3 note)")
    ap.add_argument("--n-per-kind", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap contexts per kind, for smoke runs; marks the run partial")
    ap.add_argument("--bias-bins", type=int, default=DEFAULT_BIAS_BINS)
    ap.add_argument("--abs-bins", type=int, default=DEFAULT_ABSOLUTE_BINS)
    ap.add_argument("--dist-bins", type=int, default=DEFAULT_DISTANCE_BINS)
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer subset; default every layer")
    ap.add_argument("--chunk", type=int, default=256,
                    help="query rows per capture chunk; 256 costs 25 MB of bin ids at 8K")
    ap.add_argument("--query-tail-fraction", type=float, default=DEFAULT_QUERY_TAIL_FRACTION,
                    help="share of query rows the BIAS pass averages; the distance pass "
                         "always uses all rows")
    ap.add_argument("--unigram-items", type=int, default=32)
    ap.add_argument("--unigram-source", default="benchmark", choices=("benchmark", "filler"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out-dir", default=None, help="default results/")
    ap.add_argument("--seconds-per-context", type=float, default=None,
                    help="override the cost model with a measured s/context/pass")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate and exit")
    ap.add_argument("--yes-i-costed-this", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.split == "sealed":  # defence in depth; argparse already refuses it  # sealed-ok:
        raise SystemExit("the sealed split is opened only by scripts/phase14_*.py")  # sealed-ok:

    models = tuple(m.strip() for m in args.models.split(",")) if args.models else PANEL
    unknown = [m for m in models if m not in paths.MODELS]
    if unknown:
        raise SystemExit(f"unknown model keys {unknown}; known: {sorted(paths.MODELS)}")
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    bad = [k for k in kinds if k not in CONTEXT_KINDS]
    if bad:
        raise SystemExit(f"unknown context kinds {bad}; known: {list(CONTEXT_KINDS)}")
    layers = (tuple(int(x) for x in args.layers.split(",")) if args.layers else None)

    n_per_kind = min(args.n_per_kind, args.limit) if args.limit else args.n_per_kind
    if n_per_kind < 1:
        raise SystemExit("need at least one context per kind")
    n_contexts = n_per_kind * len(kinds)

    est = estimate_seconds(
        len(models), n_contexts, args.context_tokens, per_context=args.seconds_per_context
    )
    print(
        f"{len(models)} model(s) x {n_contexts} context(s) x 2 capture passes at "
        f"T={args.context_tokens}; estimated {est / 60:.1f} min "
        f"({capture_seconds(args.context_tokens):.2f} s/context/pass modelled)"
    )
    if args.dry_run:
        return 0
    if est > 2 * 3600 and not args.yes_i_costed_this:
        raise SystemExit(
            f"estimate is {est / 3600:.1f} GPU-hours, over the 2-hour guard; "
            "re-run with --yes-i-costed-this once you have looked at that number"
        )

    is_partial = bool(
        args.limit
        or len(models) != len(PANEL)
        or len(kinds) != len(CONTEXT_KINDS)
        or layers is not None
    )

    # Probed before the first checkpoint is resident: the MATH entry of the probe
    # allocates 2.7 GB, which is affordable only while nothing else is on the card, and
    # a machine that offers neither fast backend should fail now rather than after an
    # hour of eight-times-slower capture. Skipped entirely off CUDA so a CPU smoke run
    # cannot disturb a GPU job that is already running.
    on_cuda_run = str(args.device).startswith("cuda")
    backends = (
        probe_backends() if on_cuda_run
        else {"cuda": False, "note": f"device={args.device!r}; backend probe skipped"}
    )
    if on_cuda_run and not backends.get("pinned"):
        raise SystemExit(
            "neither CUDNN_ATTENTION nor EFFICIENT_ATTENTION is available; capture would "
            "fall through to MATH. See rineya_psa.backend."
        )

    paths.ensure_dirs()
    out_dir = Path(args.out_dir) if args.out_dir else paths.RESULTS
    per_model_dir = out_dir / "phase4b"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    # A partial run must not be able to overwrite the section-46 deliverables. Phase 3
    # learned this the expensive way: a --limit smoke run silently replaced a real
    # summary with a small one that looked entirely plausible.
    suffix = "_partial" if is_partial else ""

    tok = load_tokenizer("qwen3")  # one vocabulary serves the panel; SPEC 11.4
    if args.unigram_source == "benchmark":
        distribution, unigram_meta = benchmark_unigram(
            tok, args.split, args.unigram_items, args.seed
        )
    else:
        distribution = unigram_distribution(tok, seed=args.seed)
        unigram_meta = {"unigram_source": "filler", "unigram_n_texts": 0,
                        "unigram_split": None}
    print(f"unigram: {unigram_meta['unigram_source']} "
          f"({unigram_meta['unigram_n_texts']} texts, "
          f"{len(distribution[0])} distinct ids)")

    contexts_by_kind = {
        kind: build_contexts(
            tok, kind, n_per_kind, args.context_tokens, seed=args.seed,
            distribution=distribution,
        )
        for kind in kinds
    }
    for kind, ctxs in sorted(contexts_by_kind.items()):
        print(f"  {kind}: {len(ctxs)} contexts of {ctxs[0].n_tokens} tokens")

    config = {
        "script": Path(__file__).name,
        "models": list(models),
        "kinds": list(kinds),
        "context_tokens": int(args.context_tokens),
        "n_per_kind": int(n_per_kind),
        "bias_bins": int(args.bias_bins),
        "abs_bins": int(args.abs_bins),
        "dist_bins": int(args.dist_bins),
        "layers": list(layers) if layers else None,
        "chunk": int(args.chunk),
        "query_tail_fraction": float(args.query_tail_fraction),
        "seed": int(args.seed),
        "split": args.split,
        **unigram_meta,
    }

    biases: dict[str, PositionalBias] = {}
    profiles: dict[str, DistanceProfile] = {}
    per_model: dict[str, object] = {}
    t_start = time.perf_counter()

    for n_done, key in enumerate(models):
        print(f"[{n_done + 1}/{len(models)}] {key}", flush=True)
        pooled, per_kind, profile, meta = run_model(key, args, contexts_by_kind, layers)
        biases[key], profiles[key] = pooled, profile

        for kind, part in sorted(per_kind.items()):
            part.save_npz(per_model_dir / f"bias_{key}_{kind}{suffix}.npz", config=config)
        pooled.save_npz(per_model_dir / f"bias_{key}{suffix}.npz", config=config)
        profile.save_npz(per_model_dir / f"distance_{key}{suffix}.npz", config=config)

        per_model[key] = {
            **meta,
            "bias_pooled": bias_summary(pooled),
            "bias_by_kind": {k: bias_summary(v) for k, v in sorted(per_kind.items())},
            "kind_agreement": kind_agreement(per_kind),
            "distance": distance_summary(profile),
        }
        q7 = per_model[key]["distance"]["variance_explained"]["log"]
        print(
            f"  {key} question 7 (log transform): absolute-only R2 "
            f"{q7['absolute_only_pooled_r2']:.4f}, distance-only "
            f"{q7['distance_only_pooled_r2']:.4f}, additive "
            f"{q7['additive_pooled_r2']:.4f}, residual "
            f"{q7['residual_after_additive']:.4f}",
            flush=True,
        )
        elapsed = time.perf_counter() - t_start
        remaining = len(models) - n_done - 1
        if remaining:
            print(f"  elapsed {elapsed / 60:.1f} min; "
                  f"{remaining} model(s) left, ~{elapsed / (n_done + 1) * remaining / 60:.1f} min",
                  flush=True)

    # Cross-model shape agreement is a precondition for stacking, and the panel is
    # architecturally identical by SPEC section 1, so a mismatch is a loader bug.
    shapes = {k: v.shape for k, v in biases.items()}
    if len({tuple(s) for s in shapes.values()}) != 1:
        raise SystemExit(f"bias shapes differ across the panel: {shapes}")

    bias_path = write_panel_bias(
        out_dir / f"kv_head_bias_matrix{suffix}.npz", biases, config
    )
    dist_path = write_panel_distance(
        out_dir / f"distance_significance{suffix}.npz", profiles, config
    )

    summary = {
        "config": config,
        "is_partial_run": is_partial,
        "item_limit": args.limit,
        "models": list(models),
        "n_contexts": n_contexts,
        "wall_seconds": time.perf_counter() - t_start,
        "estimated_seconds": est,
        "backends": backends,
        "artifacts": {
            "kv_head_bias_matrix": str(bias_path),
            "distance_significance": str(dist_path),
            "per_model_dir": str(per_model_dir),
        },
        "kv_rollup_is_an_approximation": True,
        "kv_rollup_note": KV_ROLLUP_NOTE,
        "spec_12_3_note": SPEC_12_3_NOTE,
        "per_model": per_model,
    }
    summary_path = out_dir / f"phase4b_bias_distance_summary{suffix}.json"
    write_result(summary, summary_path, config=config)

    if is_partial:
        print(
            f"NOTE: partial run (limit={args.limit}, models={len(models)}/{len(PANEL)}, "
            f"kinds={list(kinds)}, layers={'all' if layers is None else list(layers)}); "
            f"summary marked is_partial_run=true and artifacts carry the '_partial' "
            f"suffix so the section-46 deliverables are not overwritten"
        )
    print(f"wrote {bias_path}")
    print(f"wrote {dist_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
