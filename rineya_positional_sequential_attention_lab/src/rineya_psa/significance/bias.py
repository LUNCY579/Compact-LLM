"""Content-independent positional bias ``B[l, h, b]`` (brief section 13).

The significance matrix of :mod:`rineya_psa.significance.causal` measures what a head
does *on a task*. This module measures what a head does when there is **no task**: the
same attention statistics on contexts that carry no retrievable content, so whatever
structure survives is a property of the position machinery (RoPE, the causal mask, the
learned query/key geometry) rather than of the text.

Two context families, both generated here and both reported:

``random_ids``
    Token ids sampled i.i.d. from an empirical unigram distribution over the *real*
    tokenizer. Uniform sampling over the vocabulary would put most of its mass on rare
    tokens whose embeddings are barely trained, which is a different confound, not a
    smaller one.
``shuffled_filler``
    The lab's own inert filler prose, tokenized and then token-shuffled. It keeps the
    real unigram *and* the real token-length statistics and destroys every syntactic
    and semantic relation. Comparing the two families is the check that the measured
    bias does not depend on which flavour of nonsense was used.

Three statistics, all through the sanctioned :mod:`rineya_psa.hooks` capture path:
expected attention probability by source bin, expected pre-softmax qk logit by source
bin, and attention entropy.

**Query rows.** A query at position ``i`` can only attend to ``j <= i``, so averaging
over *all* rows manufactures an enormous apparent early-bin bias that is pure causal
masking. By default only the last ``query_tail_fraction`` of rows contribute
(``0.25``), which is the regime the benchmark actually scores in — the answer is
generated at the end of the prompt. Pass ``query_tail_fraction=1.0`` for the
unrestricted average; the value used is recorded in ``provenance``.

**Head granularity, stated explicitly.** Attention probabilities exist per *query*
head, so the primary array is ``[layer, q_head, bin]``. Modes 1 and 2 of SPEC 5.3 act
at *KV*-head granularity. :meth:`PositionalBias.to_kv_heads` bridges the two by
averaging the ``arch.kv_group_size`` query heads that read each KV head. That is an
**approximation**: two query heads in one group can have opposite positional profiles
and their mean is then a fiction. ``kv_head_dispersion`` reports the within-group
spread so a reader can see when the approximation is unsafe.

Torch and transformers are imported inside functions only, so this module stays
importable on a torch-free interpreter (SPEC section 2).

DEVIATION: SPEC 5.4 defines ``attn_by_bin`` and ``qk_logit_mean`` per query head and
offers no KV-head reduction. The KV-head aggregate the brief asks for is therefore
computed here, in analysis space, rather than in ``hooks/capture.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rineya_psa.significance.matrices import SignificanceMatrix, write_npz_with_provenance

__all__ = [
    "BIAS_FIELDS",
    "CONTEXT_KINDS",
    "DEFAULT_BIAS_BINS",
    "DEFAULT_QUERY_TAIL_FRACTION",
    "BiasContext",
    "PositionalBias",
    "default_bias_contexts",
    "edge_bias_score",
    "middle_mass",
    "measure_positional_bias",
    "random_token_contexts",
    "shuffled_filler_contexts",
    "tail_query_mask",
    "unigram_distribution",
]

#: Bin grid for the bias sweep. Finer than the significance grid because capture is
#: cheap: one forward per context buys every layer and every bin at once.
DEFAULT_BIAS_BINS: int = 16

#: Only the last quarter of query rows contribute by default. See the module docstring.
DEFAULT_QUERY_TAIL_FRACTION: float = 0.25

CONTEXT_KINDS: tuple[str, ...] = ("random_ids", "shuffled_filler")

#: Per-``(layer, head, bin)`` fields a :class:`PositionalBias` carries.
BIAS_FIELDS: tuple[str, ...] = ("attn_by_bin", "qk_logit_mean")

_CAPTURE_WHAT: tuple[str, ...] = ("attn_by_bin", "qk_logit_mean", "attn_entropy")


# ======================================================================================
# contexts
# ======================================================================================


@dataclass(frozen=True)
class BiasContext:
    """One semantically irrelevant prompt, as token ids."""

    context_id: str
    kind: str
    input_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in CONTEXT_KINDS:
            raise ValueError(f"unknown context kind {self.kind!r}; known: {list(CONTEXT_KINDS)}")
        if len(self.input_ids) < 2:
            raise ValueError(
                f"{self.context_id}: need at least 2 tokens, got {len(self.input_ids)}"
            )

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)


def _addressable_ids(tok: Any) -> np.ndarray:
    """Token ids the tokenizer will actually decode, as a sorted int64 array.

    ``vocab_size`` on a Qwen3 config (151936) overshoots the tokenizer's addressable
    range (151669); sampling the gap produces ids whose embedding rows were never
    trained, which is exactly the artefact this module must not measure.
    """
    vocab = tok.get_vocab()
    ids = np.fromiter((int(v) for v in vocab.values()), dtype=np.int64, count=len(vocab))
    return np.unique(ids)


def unigram_distribution(
    tok: Any, texts: Sequence[str] | None = None, *, seed: int = 0, n_chunks: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical ``(ids, probabilities)`` over the tokenizer's own output.

    ``texts=None`` falls back to the lab's inert filler, which is the only prose this
    repository owns and is guaranteed digit-free and semantically empty. If the filler
    generator is unavailable the distribution degrades to uniform over the addressable
    vocabulary and says so through a probability vector that is exactly uniform.
    """
    if texts is None:
        texts = _filler_texts(seed=seed, n_chunks=n_chunks)
    counts: dict[int, int] = {}
    for text in texts:
        for token_id in tok.encode(str(text), add_special_tokens=False):
            counts[int(token_id)] = counts.get(int(token_id), 0) + 1
    if not counts:
        ids = _addressable_ids(tok)
        return ids, np.full(ids.size, 1.0 / ids.size, dtype=np.float64)
    ids = np.asarray(sorted(counts), dtype=np.int64)
    weights = np.asarray([counts[int(i)] for i in ids], dtype=np.float64)
    return ids, weights / weights.sum()


def _filler_texts(*, seed: int, n_chunks: int) -> list[str]:
    from rineya_psa.benchmark.assemble import render_chunk
    from rineya_psa.benchmark.filler import generate_chunks

    chunks = generate_chunks(int(n_chunks), seed=int(seed))
    return [render_chunk(chunk, "natural") for chunk in chunks]


def random_token_contexts(
    tok: Any,
    n_contexts: int = 8,
    n_tokens: int = 1024,
    *,
    seed: int = 20260828,
    distribution: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[BiasContext]:
    """Contexts of i.i.d. token ids drawn from the empirical unigram distribution."""
    ids, probs = distribution if distribution is not None else unigram_distribution(tok, seed=seed)
    rng = np.random.default_rng(int(seed))
    out: list[BiasContext] = []
    for k in range(int(n_contexts)):
        drawn = rng.choice(ids, size=int(n_tokens), replace=True, p=probs)
        out.append(
            BiasContext(
                context_id=f"random_ids-{seed}-{k}",
                kind="random_ids",
                input_ids=tuple(int(x) for x in drawn),
            )
        )
    return out


def shuffled_filler_contexts(
    tok: Any,
    n_contexts: int = 8,
    n_tokens: int = 1024,
    *,
    seed: int = 20260828,
) -> list[BiasContext]:
    """Inert filler prose, tokenized and then token-shuffled.

    Shuffling *after* tokenization rather than shuffling words and re-encoding keeps the
    token multiset exactly, so the only thing that differs from real prose is order.
    """
    rng = np.random.default_rng(int(seed) + 7919)
    out: list[BiasContext] = []
    for k in range(int(n_contexts)):
        pool: list[int] = []
        chunk_seed = int(seed) + 1000 * k
        # Grow the pool until it covers the requested length; 24 chunks is ~1.5 k tokens.
        for round_index in range(64):
            texts = _filler_texts(seed=chunk_seed + round_index, n_chunks=24)
            for text in texts:
                pool.extend(int(t) for t in tok.encode(text, add_special_tokens=False))
            if len(pool) >= int(n_tokens):
                break
        if len(pool) < int(n_tokens):
            raise RuntimeError(
                f"filler produced only {len(pool)} tokens; cannot build a "
                f"{n_tokens}-token shuffled context"
            )
        arr = np.asarray(pool[: max(int(n_tokens), len(pool))], dtype=np.int64)
        rng.shuffle(arr)
        out.append(
            BiasContext(
                context_id=f"shuffled_filler-{seed}-{k}",
                kind="shuffled_filler",
                input_ids=tuple(int(x) for x in arr[: int(n_tokens)]),
            )
        )
    return out


def default_bias_contexts(
    tok: Any,
    n_per_kind: int = 4,
    n_tokens: int = 1024,
    *,
    seed: int = 20260828,
) -> list[BiasContext]:
    """Both families, balanced, so the kind-effect can be checked rather than assumed."""
    return [
        *random_token_contexts(tok, n_per_kind, n_tokens, seed=seed),
        *shuffled_filler_contexts(tok, n_per_kind, n_tokens, seed=seed),
    ]


# ======================================================================================
# result container
# ======================================================================================


@dataclass(frozen=True, eq=False)
class PositionalBias:
    """Capture statistics on content-free contexts, ``[layer, q_head, bin]``.

    ``eq=False`` for the same reason as every other array-carrying container in the lab:
    a generated ``__eq__`` would return an array and poison ``in``.
    """

    attn_by_bin: np.ndarray  # [L, H_q, B] mean attention mass on each absolute source bin
    qk_logit_mean: np.ndarray  # [L, H_q, B] mean pre-softmax qk logit by source bin
    entropy: np.ndarray  # [L, H_q] mean attention entropy, nats
    layers: tuple[int, ...]
    q_heads: tuple[int, ...]
    bins: tuple[int, ...]
    binning_desc: Mapping[str, Any]
    model_key: str
    n_kv_heads: int
    context_kinds: tuple[str, ...]
    n_contexts: int
    provenance: Mapping[str, Any] = field(default_factory=dict)
    #: Per-context spread of ``attn_by_bin``, same shape. Zero when a single context ran.
    attn_by_bin_sd: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = (len(self.layers), len(self.q_heads), len(self.bins))
        for name in BIAS_FIELDS:
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"{name} is {array.shape}, expected {shape}")
            object.__setattr__(self, name, np.ascontiguousarray(array))
        entropy = np.asarray(self.entropy, dtype=np.float64)
        if entropy.shape != shape[:2]:
            raise ValueError(f"entropy is {entropy.shape}, expected {shape[:2]}")
        object.__setattr__(self, "entropy", np.ascontiguousarray(entropy))
        object.__setattr__(self, "binning_desc", dict(self.binning_desc))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if self.attn_by_bin_sd is not None:
            object.__setattr__(
                self, "attn_by_bin_sd", np.asarray(self.attn_by_bin_sd, dtype=np.float64)
            )
        if len(self.q_heads) % int(self.n_kv_heads):
            raise ValueError(
                f"{len(self.q_heads)} query heads do not group into {self.n_kv_heads} KV heads"
            )

    @property
    def kv_group_size(self) -> int:
        return len(self.q_heads) // int(self.n_kv_heads)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.layers), len(self.q_heads), len(self.bins))

    @property
    def bin_centers(self) -> np.ndarray:
        from rineya_psa.significance.matrices import bin_centers

        return bin_centers(self.binning_desc)

    def field(self, name: str) -> np.ndarray:
        if name not in BIAS_FIELDS:
            raise ValueError(f"unknown bias field {name!r}; known: {list(BIAS_FIELDS)}")
        return getattr(self, name)

    # -- query head -> KV head ------------------------------------------------

    def to_kv_heads(self, name: str = "attn_by_bin") -> np.ndarray:
        """``[L, H_kv, B]``, the mean over the query heads reading each KV head.

        An approximation, deliberately not hidden: see the module docstring and
        :meth:`kv_head_dispersion`.
        """
        values = self.field(name)
        g = self.kv_group_size
        grouped = values.reshape(values.shape[0], int(self.n_kv_heads), g, values.shape[2])
        return grouped.mean(axis=2)

    def kv_head_dispersion(self, name: str = "attn_by_bin") -> np.ndarray:
        """``[L, H_kv]`` within-group spread: how much :meth:`to_kv_heads` threw away.

        The per-bin standard deviation across the query heads of one group, averaged over
        bins. Large next to the group mean means the KV-head aggregate is not a summary
        of anything and Mode 1/2 tables built from it should be treated with suspicion.
        """
        values = self.field(name)
        g = self.kv_group_size
        grouped = values.reshape(values.shape[0], int(self.n_kv_heads), g, values.shape[2])
        if g < 2:
            return np.zeros(grouped.shape[:2], dtype=np.float64)
        return grouped.std(axis=2, ddof=1).mean(axis=2)

    def to_significance_matrix(
        self, name: str = "attn_by_bin", *, kv_heads: bool = True
    ) -> SignificanceMatrix:
        """Wrap one field as a :class:`SignificanceMatrix` so the SVD and control
        machinery of brief sections 21 and 37 applies to ``B`` as well as to ``I``."""
        if kv_heads:
            values = self.to_kv_heads(name)
            heads = tuple(range(int(self.n_kv_heads)))
            granularity = "kv_head_mean_over_query_heads"
        else:
            values = self.field(name)
            heads = self.q_heads
            granularity = "q_head"
        return SignificanceMatrix(
            values=values,
            layers=self.layers,
            kv_heads=heads,
            bins=self.bins,
            binning_desc=self.binning_desc,
            objective=f"bias_{name}",
            model_key=self.model_key,
            provenance={
                "kind": "positional_bias",
                "field": name,
                "head_granularity": granularity,
                "context_kinds": list(self.context_kinds),
                "n_contexts": int(self.n_contexts),
                **dict(self.provenance),
            },
        )

    # -- summaries ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        centers = self.bin_centers
        attn = self.attn_by_bin
        return {
            "model_key": self.model_key,
            "shape": list(self.shape),
            "binning": dict(self.binning_desc),
            "context_kinds": list(self.context_kinds),
            "n_contexts": int(self.n_contexts),
            "bin_centers": centers.tolist(),
            "attn_by_bin_pooled": attn.mean(axis=(0, 1)).tolist(),
            "qk_logit_mean_pooled": self.qk_logit_mean.mean(axis=(0, 1)).tolist(),
            "entropy_by_layer": self.entropy.mean(axis=1).tolist(),
            "edge_bias_score_mean": float(np.nanmean(edge_bias_score(self))),
            "middle_mass_mean": float(np.nanmean(middle_mass(self))),
            "kv_head_dispersion_mean": float(np.nanmean(self.kv_head_dispersion())),
            "provenance": dict(self.provenance),
        }

    def save_npz(self, path: str | Path, *, config: Mapping[str, Any] | None = None) -> Path:
        import json

        from rineya_psa.statistics.provenance import json_default

        payload = {
            "attn_by_bin": self.attn_by_bin,
            "qk_logit_mean": self.qk_logit_mean,
            "entropy": self.entropy,
            "meta": np.asarray(
                json.dumps(
                    {
                        "layers": list(self.layers),
                        "q_heads": list(self.q_heads),
                        "bins": list(self.bins),
                        "binning_desc": dict(self.binning_desc),
                        "model_key": self.model_key,
                        "n_kv_heads": int(self.n_kv_heads),
                        "context_kinds": list(self.context_kinds),
                        "n_contexts": int(self.n_contexts),
                        "provenance": dict(self.provenance),
                    },
                    default=json_default,
                )
            ),
        }
        if self.attn_by_bin_sd is not None:
            payload["attn_by_bin_sd"] = self.attn_by_bin_sd
        return write_npz_with_provenance(payload, path, config=config)

    @classmethod
    def load(cls, path: str | Path) -> PositionalBias:
        import json

        with np.load(Path(path), allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            sd = np.asarray(archive["attn_by_bin_sd"]) if "attn_by_bin_sd" in archive else None
            return cls(
                attn_by_bin=np.asarray(archive["attn_by_bin"], dtype=np.float64),
                qk_logit_mean=np.asarray(archive["qk_logit_mean"], dtype=np.float64),
                entropy=np.asarray(archive["entropy"], dtype=np.float64),
                layers=tuple(meta["layers"]),
                q_heads=tuple(meta["q_heads"]),
                bins=tuple(meta["bins"]),
                binning_desc=meta["binning_desc"],
                model_key=meta["model_key"],
                n_kv_heads=int(meta["n_kv_heads"]),
                context_kinds=tuple(meta["context_kinds"]),
                n_contexts=int(meta["n_contexts"]),
                provenance=meta.get("provenance", {}),
                attn_by_bin_sd=sd,
            )


# ======================================================================================
# summaries for later correlations (brief section 13's deliverable)
# ======================================================================================


def _values_and_centers(
    values: Any, binning_desc: Mapping[str, Any] | None
) -> tuple[np.ndarray, np.ndarray]:
    from rineya_psa.significance.matrices import bin_centers

    if isinstance(values, PositionalBias):
        return np.asarray(values.attn_by_bin, dtype=np.float64), values.bin_centers
    if isinstance(values, SignificanceMatrix):
        return np.asarray(values.values, dtype=np.float64), values.bin_centers
    array = np.asarray(values, dtype=np.float64)
    if binning_desc is not None:
        centers = bin_centers(binning_desc)
        if centers.size != array.shape[-1]:
            raise ValueError(
                f"binning describes {centers.size} bins but the array has {array.shape[-1]}"
            )
        return array, centers
    n = array.shape[-1]
    return array, (np.arange(n, dtype=np.float64) + 0.5) / float(n)


def edge_bias_score(
    values: Any,
    *,
    binning_desc: Mapping[str, Any] | None = None,
    edge_fraction: float = 0.25,
) -> np.ndarray:
    """``mean(edge bins) - mean(interior bins)``, reduced over the last axis.

    Positive means the head prefers the two ends of the context over its middle — the
    lost-in-the-middle signature, expressed as a single number per ``(layer, head)`` so
    it can be correlated against the causal significance matrix later.

    ``edge_fraction`` is the share of the normalised position axis at *each* end that
    counts as an edge, so the default 0.25 splits the context into 25/50/25.
    """
    array, centers = _values_and_centers(values, binning_desc)
    if not 0.0 < edge_fraction < 0.5:
        raise ValueError(f"edge_fraction must lie in (0, 0.5), got {edge_fraction}")
    lo, hi = float(centers.min()), float(centers.max())
    span = hi - lo
    if span <= 0.0:
        return np.zeros(array.shape[:-1], dtype=np.float64)
    edge = (centers <= lo + edge_fraction * span) | (centers >= hi - edge_fraction * span)
    interior = ~edge
    if not edge.any() or not interior.any():
        raise ValueError(
            f"edge_fraction={edge_fraction} leaves one side empty on a {centers.size}-bin grid"
        )
    with np.errstate(invalid="ignore"):
        return np.nanmean(array[..., edge], axis=-1) - np.nanmean(array[..., interior], axis=-1)


def middle_mass(
    values: Any,
    *,
    binning_desc: Mapping[str, Any] | None = None,
    center: float = 0.5,
    width: float = 0.5,
    reduce: str = "sum",
) -> np.ndarray:
    """Total (``reduce="sum"``) or average (``"mean"``) weight on the middle window.

    ``sum`` is the right reduction for ``attn_by_bin``, where the bins partition a unit
    of probability mass; ``mean`` is the right one for ``qk_logit_mean``, where they do
    not. Getting this backwards produces a number that scales with the bin count, which
    is why the choice is an explicit argument and not a default buried in a helper.
    """
    if reduce not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")
    array, centers = _values_and_centers(values, binning_desc)
    if not 0.0 < width <= 1.0:
        raise ValueError(f"width must lie in (0, 1], got {width}")
    span = float(centers.max() - centers.min()) or 1.0
    half = 0.5 * width * span
    window = (centers >= center - half) & (centers <= center + half)
    if not window.any():  # a coarse grid can straddle the window entirely
        window = np.zeros(centers.size, dtype=bool)
        window[int(np.argmin(np.abs(centers - center)))] = True
    with np.errstate(invalid="ignore"):
        block = array[..., window]
        return np.nansum(block, axis=-1) if reduce == "sum" else np.nanmean(block, axis=-1)


# ======================================================================================
# measurement
# ======================================================================================


def tail_query_mask(n_tokens: int, fraction: float) -> Any:
    """``[T]`` bool selecting the last ``fraction`` of query rows (at least one)."""
    import torch

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"query_tail_fraction must lie in (0, 1], got {fraction}")
    mask = torch.zeros(int(n_tokens), dtype=torch.bool)
    start = int(n_tokens) - max(1, int(math.ceil(fraction * n_tokens)))
    mask[max(0, start) :] = True
    return mask


def measure_positional_bias(
    model: Any,
    tok: Any = None,
    contexts: Iterable[BiasContext] | None = None,
    *,
    layers: Sequence[int] | None = None,
    n_bins: int = DEFAULT_BIAS_BINS,
    binning: Any | None = None,
    chunk: int = 256,
    query_tail_fraction: float = DEFAULT_QUERY_TAIL_FRACTION,
    model_key: str = "unknown",
    device: Any | None = None,
    force_backend: bool = True,
    progress: Any | None = None,
) -> PositionalBias:
    """Run the capture sweep of brief section 13 and average over contexts.

    One ``psa_session`` per context, so every context carries equal weight regardless of
    its length; pooling inside a single session would weight by query rows instead and
    silently let one long context dominate.

    ``contexts=None`` builds :func:`default_bias_contexts` from ``tok``, which then must
    not be ``None``.
    """
    import torch

    from rineya_psa.hooks.binning import AbsoluteBinning
    from rineya_psa.hooks.registry import psa_session
    from rineya_psa.hooks.specs import CaptureSpec, PSAContext
    from rineya_psa.loading import arch_from_config
    from rineya_psa.significance.causal import forced_sdpa_backend

    arch = arch_from_config(model.config)
    binning = binning if binning is not None else AbsoluteBinning(int(n_bins))
    if binning.is_query_dependent:
        raise ValueError(
            f"{binning.name} is query-dependent; B[l,h,b] is defined over absolute source "
            "bins. Use significance.distance for the relative-distance profile."
        )
    n_bins = int(binning.n_bins)

    if contexts is None:
        if tok is None:
            raise ValueError("measure_positional_bias needs either `contexts` or a tokenizer")
        contexts = default_bias_contexts(tok)
    context_list = list(contexts)
    if not context_list:
        raise ValueError("no contexts to measure")

    layer_ids = tuple(range(arch.num_layers)) if layers is None else tuple(int(x) for x in layers)
    if any(not 0 <= lay < arch.num_layers for lay in layer_ids):
        raise IndexError(f"layers out of range for {arch.num_layers} layers: {layer_ids}")
    if device is None:
        device = next(model.parameters()).device

    shape = (len(layer_ids), arch.num_q_heads, n_bins)
    attn = np.zeros((len(context_list), *shape), dtype=np.float64)
    qk = np.zeros((len(context_list), *shape), dtype=np.float64)
    entropy = np.zeros((len(context_list), len(layer_ids), arch.num_q_heads), dtype=np.float64)
    n_rows_total = 0

    spec = CaptureSpec(
        layers=layer_ids, what=_CAPTURE_WHAT, chunk=int(chunk), binning=binning
    )
    for ci, context in enumerate(context_list):
        ids = torch.tensor(list(context.input_ids), dtype=torch.long, device=device).unsqueeze(0)
        valid = tail_query_mask(context.n_tokens, query_tail_fraction).to(device)
        n_rows_total += int(valid.sum())
        ctx = PSAContext(
            enabled=True, capture=spec, position_bins=binning, valid_mask=valid
        )
        with psa_session(model, ctx), torch.inference_mode(), forced_sdpa_backend(
            force_backend, device
        ):
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        captured = ctx.captured
        for li, layer in enumerate(layer_ids):
            attn[ci, li] = _require(captured, layer, "attn_by_bin", shape[1:])
            qk[ci, li] = _require(captured, layer, "qk_logit_mean", shape[1:])
            entropy[ci, li] = _require(captured, layer, "attn_entropy", (arch.num_q_heads,))
        if progress is not None:
            progress(1)

    kinds = tuple(sorted({c.kind for c in context_list}))
    return PositionalBias(
        attn_by_bin=attn.mean(axis=0),
        qk_logit_mean=qk.mean(axis=0),
        entropy=entropy.mean(axis=0),
        layers=layer_ids,
        q_heads=tuple(range(arch.num_q_heads)),
        bins=tuple(range(n_bins)),
        binning_desc=binning.describe(),
        model_key=model_key,
        n_kv_heads=int(arch.num_kv_heads),
        context_kinds=kinds,
        n_contexts=len(context_list),
        provenance={
            "query_tail_fraction": float(query_tail_fraction),
            "chunk": int(chunk),
            "context_ids": [c.context_id for c in context_list],
            "context_lengths": [c.n_tokens for c in context_list],
            "n_query_rows": int(n_rows_total),
            "capture_what": list(_CAPTURE_WHAT),
            "caveat": (
                "measured on content-free contexts; attn_by_bin averages only the last "
                f"{query_tail_fraction:.2f} of query rows so causal masking does not "
                "masquerade as positional bias"
            ),
        },
        attn_by_bin_sd=(
            attn.std(axis=0, ddof=1) if len(context_list) > 1 else np.zeros(shape)
        ),
    )


def _require(
    captured: Mapping[tuple[int, str], np.ndarray], layer: int, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = captured.get((int(layer), name))
    if array is None:
        raise RuntimeError(
            f"capture produced no {name!r} for layer {layer}; the psa kernel did not run "
            "(is the context enabled, and does CaptureSpec cover this layer?)"
        )
    array = np.asarray(array, dtype=np.float64)
    if array.shape != shape:
        raise RuntimeError(f"{name} for layer {layer} is {array.shape}, expected {shape}")
    return array
