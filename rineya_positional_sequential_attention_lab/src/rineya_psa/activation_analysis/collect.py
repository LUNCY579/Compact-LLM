"""Per-layer activation capture at chosen token positions (brief section 16).

This module deliberately uses **torch ``nn.Module`` forward hooks**, not the ``"psa"``
attention interface of SPEC section 5. The two answer different questions: the psa
kernel sees ``query``/``key``/``value`` *inside* one attention call and is the right
tool for attention statistics, while brief section 16 asks for the vectors flowing
*between* blocks -- the residual stream, the attention block's output after
``o_proj``, and the MLP output. Those are module outputs, so a module hook is the
honest instrument. ``hooks/`` is untouched.

Sites, and exactly which tensor each one is
-------------------------------------------
``resid``
    Output of ``Qwen3DecoderLayer.forward``, i.e. the residual stream *after* layer
    ``l`` has added both its attention and its MLP contribution.
``attn_out``
    First element of ``Qwen3Attention.forward``'s return: the ``o_proj`` output,
    before the residual add.
``mlp_out``
    ``Qwen3MLP.forward``'s return, before the residual add.

So ``resid[l] = resid[l-1] + attn_out[l] + mlp_out[l]`` up to the layernorms that sit
on the *inputs* of the two sublayers. Capturing all three lets a reader attribute a
residual-stream change to a sublayer without a second forward.

Position sampling policy, and why subsampling is not optional
-------------------------------------------------------------
Storing every position is not affordable. One 4K item at 28 layers, 3 sites and
hidden size 1024 in float16 costs

    28 * 3 * 4096 * 1024 * 2 B = 704 MB

so a few hundred items would be hundreds of gigabytes. :class:`PositionPolicy`
therefore keeps a small, *named* set of positions per item -- BOS, an evenly spaced
subsample inside each evidence span, the last few prompt tokens (the final query
representation), and a seeded random control set drawn from outside the evidence.
With the defaults that is at most ``1 + 8 + 2 + 8 = 19`` positions, i.e.

    28 * 3 * 19 * 1024 * 2 B = 3.3 MB per item, ~1.0 GB for 300 items,

which is the budget brief section 16 asks for. :func:`estimate_bytes` states the cost
before a run starts and :meth:`ActivationDataset.nbytes` states it afterwards. The
control set exists so a probe has a within-item negative: if normalized position is
decodable only at hand-picked landmarks, that is a different claim from decodable
everywhere.

Storage is float16 numpy on CPU. The models run in bfloat16 or float16 anyway, so
float16 storage loses nothing that was in the activation; bfloat16 would lose
mantissa bits that float16 keeps, and numpy has no native bfloat16.

DEVIATION: SPEC section 12.1 says ``evaluation/runner.py`` owns the ``sdpa_kernel``
backend forcing and "every other module goes through the runner". ``runner.run_items``
has no hook-capable entry point -- it owns batching, scoring and record construction,
and cannot expose the raw single-item forward a module hook needs. This module
therefore re-enters the same context manager itself via :func:`forced_sdpa_backends`,
with the same backend list, rather than editing ``runner.py``. On CPU it is a no-op.

DEVIATION: batch size is fixed at 1 and ``attention_mask`` is not passed, per SPEC
section 12.2. Left padding would shift RoPE offsets, which is a positional confound in
a study about position, and this module's whole output is indexed by token position.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rineya_psa.paths import assert_readonly_target
from rineya_psa.types import BenchmarkItem, TokenizedItem, stable_id

__all__ = [
    "DEFAULT_POLICY",
    "ROLES",
    "SITES",
    "ActivationCollector",
    "ActivationDataset",
    "ActivationRow",
    "PositionPolicy",
    "PositionSelection",
    "collect_activations",
    "collect_item",
    "dataset_from_arrays",
    "decoder_layers",
    "estimate_bytes",
    "forced_sdpa_backends",
    "load_dataset_npz",
    "select_positions",
    "site_hooks",
]

#: Module outputs this file can capture or overwrite. See the module docstring.
SITES: tuple[str, ...] = ("resid", "attn_out", "mlp_out")

#: Named position roles. ``final`` is the query-side representation brief section 17
#: patches; ``control`` is the seeded random negative.
ROLES: tuple[str, ...] = ("bos", "evidence", "final", "control")

_STORE_DTYPE = np.float16


# --------------------------------------------------------------------------------------
# Model plumbing
# --------------------------------------------------------------------------------------


def decoder_layers(model: Any) -> list[nn.Module]:
    """Decoder blocks in layer order, duck-typed rather than ``isinstance``-checked.

    Supa-A1.1 arrives through ``trust_remote_code`` and defines its own class objects,
    so an ``isinstance(m, Qwen3DecoderLayer)`` test would silently find nothing.
    A block is anything owning both a ``self_attn`` with a ``layer_idx`` and an ``mlp``.
    """
    found: dict[int, nn.Module] = {}
    for module in model.modules():
        attn = getattr(module, "self_attn", None)
        mlp = getattr(module, "mlp", None)
        if attn is None or mlp is None or not hasattr(attn, "layer_idx"):
            continue
        idx = int(attn.layer_idx)
        found.setdefault(idx, module)
    if not found:
        raise RuntimeError(
            "no decoder blocks found; expected modules carrying .self_attn and .mlp "
            "(is this a Qwen3-family model?)"
        )
    return [found[i] for i in sorted(found)]


@contextlib.contextmanager
def forced_sdpa_backends(device: Any) -> Iterator[None]:
    """SPEC 12.1's backend pin, re-entered locally. No-op off CUDA."""
    kind = getattr(device, "type", str(device))
    if kind != "cuda":
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover - torch < 2.3 has no selector
        yield
        return
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        yield


#: ``fn(layer, site, tensor) -> replacement | None``. Returning ``None`` leaves the
#: forward untouched, which is what a pure reader does.
SiteFn = Callable[[int, str, torch.Tensor], "torch.Tensor | None"]


def _site_module(block: nn.Module, site: str) -> nn.Module:
    if site == "resid":
        return block
    if site == "attn_out":
        return block.self_attn
    if site == "mlp_out":
        return block.mlp
    raise ValueError(f"unknown site {site!r}; known: {list(SITES)}")


def _make_hook(fn: SiteFn, layer: int, site: str):
    def hook(module: nn.Module, args: Any, output: Any) -> Any:  # noqa: ARG001
        # Qwen3Attention returns (attn_output, attn_weights); the block and the MLP
        # return a bare tensor. Preserve whichever shape came in.
        if isinstance(output, tuple):
            new = fn(layer, site, output[0])
            return None if new is None else (new, *output[1:])
        new = fn(layer, site, output)
        return None if new is None else new

    return hook


@contextlib.contextmanager
def site_hooks(
    model: Any,
    fn: SiteFn,
    *,
    layers: Sequence[int] | None = None,
    sites: Sequence[str] = SITES,
) -> Iterator[list[int]]:
    """Register ``fn`` on every requested ``(layer, site)`` and remove it on exit.

    Yields the layer indices that were hooked. ``fn`` may return a replacement tensor,
    which is how ``patching.py`` reuses this instead of writing a second hook stack;
    a reader returns ``None``.
    """
    blocks = decoder_layers(model)
    wanted = list(range(len(blocks))) if layers is None else sorted({int(i) for i in layers})
    bad = [i for i in wanted if not 0 <= i < len(blocks)]
    if bad:
        raise IndexError(f"layers {bad} out of range for a {len(blocks)}-layer model")
    site_list = list(dict.fromkeys(str(s) for s in sites))
    unknown = [s for s in site_list if s not in SITES]
    if unknown:
        raise ValueError(f"unknown sites {unknown}; known: {list(SITES)}")

    handles = []
    try:
        for layer in wanted:
            for site in site_list:
                target = _site_module(blocks[layer], site)
                handles.append(target.register_forward_hook(_make_hook(fn, layer, site)))
        yield wanted
    finally:
        for handle in handles:
            handle.remove()


# --------------------------------------------------------------------------------------
# Position selection
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionPolicy:
    """Which token positions survive into storage. See the module docstring's arithmetic."""

    max_evidence_tokens: int = 8
    n_final_tokens: int = 2
    n_control: int = 8
    include_bos: bool = True
    seed: int = 20260828

    def __post_init__(self) -> None:
        for name in ("max_evidence_tokens", "n_final_tokens", "n_control"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.n_final_tokens < 1 and self.n_control < 1 and not self.include_bos:
            raise ValueError("policy selects no positions at all")

    @property
    def max_positions(self) -> int:
        return (
            int(self.include_bos)
            + int(self.max_evidence_tokens)
            + int(self.n_final_tokens)
            + int(self.n_control)
        )

    def describe(self) -> dict[str, Any]:
        return {**asdict(self), "max_positions": self.max_positions}


DEFAULT_POLICY = PositionPolicy()


@dataclass(frozen=True)
class PositionSelection:
    """Token indices kept for one item, each tagged with the role that put it there."""

    indices: tuple[int, ...]
    roles: tuple[str, ...]
    span_index: tuple[int, ...]  # evidence span ordinal, -1 for every other role
    n_tokens: int

    def __post_init__(self) -> None:
        if not (len(self.indices) == len(self.roles) == len(self.span_index)):
            raise ValueError("indices, roles and span_index must be the same length")

    @property
    def normalized(self) -> tuple[float, ...]:
        n = max(int(self.n_tokens), 1)
        return tuple(i / n for i in self.indices)

    @property
    def offsets_from_end(self) -> tuple[int, ...]:
        return tuple(self.n_tokens - 1 - i for i in self.indices)

    def of_role(self, role: str) -> tuple[int, ...]:
        return tuple(i for i, r in zip(self.indices, self.roles, strict=True) if r == role)


def _span_subsample(start: int, end: int, budget: int) -> list[int]:
    """Up to ``budget`` evenly spaced indices in ``[start, end)``, endpoints included."""
    width = max(0, int(end) - int(start))
    if width == 0 or budget <= 0:
        return []
    k = min(int(budget), width)
    if k == 1:
        return [int(start)]
    picks = np.linspace(int(start), int(end) - 1, k)
    return sorted({int(round(p)) for p in picks})


def select_positions(
    n_tokens: int,
    evidence_spans: Sequence[tuple[int, int]],
    policy: PositionPolicy = DEFAULT_POLICY,
    *,
    item_key: str = "",
) -> PositionSelection:
    """Apply ``policy`` to one item. Deterministic given ``(policy.seed, item_key)``.

    Role collisions resolve by priority ``bos > evidence > final > control``: a control
    draw can never shadow an evidence token, and the last prompt token keeps the
    ``final`` label even if it happens to sit inside an evidence span.
    """
    n = int(n_tokens)
    if n < 1:
        raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")

    chosen: dict[int, tuple[str, int]] = {}

    def take(index: int, role: str, span: int = -1) -> None:
        idx = int(index)
        if 0 <= idx < n:
            chosen.setdefault(idx, (role, span))

    if policy.include_bos:
        take(0, "bos")

    spans = [(int(a), int(b)) for a, b in evidence_spans if int(b) > int(a)]
    if spans and policy.max_evidence_tokens:
        per_span = max(1, policy.max_evidence_tokens // len(spans))
        for ordinal, (a, b) in enumerate(spans):
            for idx in _span_subsample(a, min(b, n), per_span):
                take(idx, "evidence", ordinal)

    for k in range(int(policy.n_final_tokens)):
        take(n - 1 - k, "final")

    if policy.n_control:
        blocked = set(chosen)
        for a, b in spans:
            blocked.update(range(max(0, a), min(n, b)))
        candidates = np.array(sorted(set(range(n)) - blocked), dtype=np.int64)
        if candidates.size:
            seed = int(stable_id(policy.seed, item_key, n), 16) % (2**63)
            rng = np.random.default_rng(seed)
            k = int(min(policy.n_control, candidates.size))
            for idx in np.sort(rng.choice(candidates, size=k, replace=False)):
                take(int(idx), "control")

    order = sorted(chosen)
    return PositionSelection(
        indices=tuple(order),
        roles=tuple(chosen[i][0] for i in order),
        span_index=tuple(chosen[i][1] for i in order),
        n_tokens=n,
    )


def estimate_bytes(
    n_items: int, n_layers: int, n_sites: int, n_positions: int, hidden_size: int
) -> int:
    """Storage a collection run will occupy, in bytes. Print this before you start."""
    return int(n_items) * int(n_layers) * int(n_sites) * int(n_positions) * int(hidden_size) * 2


# --------------------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------------------


class ActivationCollector:
    """Forward-hook reader over ``(layer, site)`` pairs, one item at a time.

    Used as a context manager so the hooks come off even if a forward raises::

        with ActivationCollector(model, layers=[0, 13, 27]) as collector:
            vectors = collector.run(input_ids, selection.indices)
    """

    def __init__(
        self,
        model: Any,
        *,
        layers: Sequence[int] | None = None,
        sites: Sequence[str] = SITES,
        store_dtype: Any = _STORE_DTYPE,
    ) -> None:
        self.model = model
        self.sites = tuple(dict.fromkeys(str(s) for s in sites))
        # float16 is the *storage* default (see the module docstring's budget). Patching
        # asks for float32 instead: a transplanted vector is compared against itself, so
        # a rounding step would land directly in the measured effect.
        self.store_dtype = np.dtype(store_dtype)
        n_blocks = len(decoder_layers(model))
        self.layers = tuple(
            range(n_blocks) if layers is None else sorted({int(i) for i in layers})
        )
        self._indices: torch.Tensor | None = None
        self._buffer: dict[tuple[int, str], np.ndarray] = {}
        self._stack: contextlib.ExitStack | None = None
        #: Logits of the most recent :meth:`run`, kept so a caller can score without a
        #: second forward.
        self.last_logits: torch.Tensor | None = None

    # -- hook body -----------------------------------------------------------------

    def _read(self, layer: int, site: str, tensor: torch.Tensor) -> None:
        if self._indices is None:
            return
        if tensor.ndim != 3:
            raise RuntimeError(
                f"layer {layer} site {site!r} produced a {tensor.ndim}D tensor; "
                "expected [batch, tokens, hidden]"
            )
        if tensor.shape[0] != 1:
            raise RuntimeError(
                f"activation collection runs at batch size 1 (SPEC 12.2); got "
                f"batch {tensor.shape[0]}"
            )
        picked = tensor[0].index_select(0, self._indices.to(tensor.device))
        self._buffer[(int(layer), site)] = (
            picked.detach().to(torch.float32).cpu().numpy().astype(self.store_dtype)
        )
        return None

    def __enter__(self) -> ActivationCollector:
        stack = contextlib.ExitStack()
        stack.enter_context(
            site_hooks(
                self.model,
                lambda layer, site, tensor: self._read(layer, site, tensor),
                layers=self.layers,
                sites=self.sites,
            )
        )
        self._stack = stack
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._indices = None

    # -- driver --------------------------------------------------------------------

    def run(
        self,
        input_ids: Sequence[int] | torch.Tensor,
        indices: Sequence[int],
        *,
        logits_to_keep: int = 1,
    ) -> dict[tuple[int, str], np.ndarray]:
        """One unpadded forward; returns ``{(layer, site): [n_positions, hidden] f16}``.

        ``logits_to_keep`` defaults to 1 because a capture run does not need the LM
        head: at T = 4096 the full ``[1, T, 151936]`` logit tensor is 1.2 GB in bf16
        and is the memory wall SPEC section 11.1 describes.
        """
        if self._stack is None:
            raise RuntimeError("use ActivationCollector as a context manager before run()")
        device = next(self.model.parameters()).device
        ids = (
            input_ids
            if isinstance(input_ids, torch.Tensor)
            else torch.tensor(list(input_ids), dtype=torch.long)
        )
        ids = ids.reshape(1, -1).to(device)
        idx = torch.tensor([int(i) for i in indices], dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError("no positions requested")
        if int(idx.max()) >= int(ids.shape[1]):
            raise IndexError(f"position {int(idx.max())} outside a {int(ids.shape[1])}-token item")

        self._buffer = {}
        self._indices = idx
        try:
            with torch.inference_mode(), forced_sdpa_backends(device):
                out = self.model(
                    input_ids=ids, use_cache=False, logits_to_keep=int(logits_to_keep)
                )
            self.last_logits = out.logits.detach()
        finally:
            self._indices = None

        missing = [
            (layer, site)
            for layer in self.layers
            for site in self.sites
            if (layer, site) not in self._buffer
        ]
        if missing:  # pragma: no cover - a hook that never fired means a wrong module
            raise RuntimeError(f"no activation captured for {missing[:4]}")
        return dict(self._buffer)


# --------------------------------------------------------------------------------------
# Tidy dataset
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationRow:
    """One (item, token position). The row unit of every probe design matrix."""

    item_id: str
    scenario_id: str
    family: str
    fmt: str
    length_bucket: str
    split: str
    token_index: int
    n_tokens: int
    normalized_position: float
    offset_from_end: int
    role: str
    span_index: int
    evidence_position: float
    target_position: float
    answer: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _evidence_position(item: BenchmarkItem, tok_item: TokenizedItem) -> float:
    """Normalized midpoint of the first evidence span, measured if possible."""
    if tok_item.evidence_spans and tok_item.n_tokens:
        a, b = tok_item.evidence_spans[0]
        return float((a + b) / 2.0 / tok_item.n_tokens)
    if tok_item.actual_positions:
        return float(tok_item.actual_positions[0])
    if item.target_positions:
        return float(item.target_positions[0])
    return float("nan")


@dataclass
class ActivationDataset:
    """Row metadata plus one ``[n_rows, hidden]`` float16 matrix per ``(layer, site)``."""

    rows: tuple[ActivationRow, ...]
    matrices: dict[tuple[int, str], np.ndarray]
    layers: tuple[int, ...]
    sites: tuple[str, ...]
    hidden_size: int
    policy: PositionPolicy = DEFAULT_POLICY
    model_key: str | None = None

    def __post_init__(self) -> None:
        n = len(self.rows)
        for key, matrix in self.matrices.items():
            if matrix.shape != (n, self.hidden_size):
                raise ValueError(
                    f"matrix {key} is {matrix.shape}, expected ({n}, {self.hidden_size})"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def matrix(self, layer: int, site: str) -> np.ndarray:
        try:
            return self.matrices[(int(layer), str(site))]
        except KeyError:
            raise KeyError(
                f"no activations for layer {layer} site {site!r}; have "
                f"layers {self.layers} sites {self.sites}"
            ) from None

    @property
    def nbytes(self) -> int:
        return int(sum(m.nbytes for m in self.matrices.values()))

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(r.scenario_id for r in self.rows)

    def column(self, name: str) -> np.ndarray:
        """One metadata field as an array, dtype chosen by the field's Python type."""
        values = [getattr(r, name) for r in self.rows]
        if values and isinstance(values[0], str):
            return np.asarray(values, dtype=object)
        return np.asarray(values)

    def role_mask(self, roles: Sequence[str] | None) -> np.ndarray:
        if roles is None:
            return np.ones(len(self.rows), dtype=bool)
        wanted = {str(r) for r in roles}
        return np.asarray([r.role in wanted for r in self.rows], dtype=bool)

    def subset(self, mask: Sequence[bool] | np.ndarray) -> ActivationDataset:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != (len(self.rows),):
            raise ValueError(f"mask is {keep.shape}, expected ({len(self.rows)},)")
        return replace(
            self,
            rows=tuple(r for r, k in zip(self.rows, keep, strict=True) if k),
            matrices={k: m[keep] for k, m in self.matrices.items()},
        )

    def metadata_frame(self) -> Any:
        """Row metadata as a pandas DataFrame (lazy import; pandas is optional)."""
        import pandas as pd

        return pd.DataFrame([r.to_json() for r in self.rows])

    def describe(self) -> dict[str, Any]:
        return {
            "n_rows": len(self.rows),
            "n_items": len({r.item_id for r in self.rows}),
            "n_scenarios": len({r.scenario_id for r in self.rows}),
            "layers": list(self.layers),
            "sites": list(self.sites),
            "hidden_size": self.hidden_size,
            "bytes": self.nbytes,
            "policy": self.policy.describe(),
            "model_key": self.model_key,
            "role_counts": {
                role: int(sum(1 for r in self.rows if r.role == role)) for role in ROLES
            },
        }

    # -- serialisation ---------------------------------------------------------------

    def save_npz(self, path: str | Path) -> Path:
        """Compressed npz: one array per ``(layer, site)`` plus a JSON metadata blob."""
        target = Path(path)
        assert_readonly_target(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"m::{layer}::{site}": m for (layer, site), m in self.matrices.items()}
        meta = {
            "rows": [r.to_json() for r in self.rows],
            "layers": list(self.layers),
            "sites": list(self.sites),
            "hidden_size": self.hidden_size,
            "policy": asdict(self.policy),
            "model_key": self.model_key,
        }
        np.savez_compressed(target, __meta__=np.asarray(json.dumps(meta)), **arrays)
        return target


def load_dataset_npz(path: str | Path) -> ActivationDataset:
    """Inverse of :meth:`ActivationDataset.save_npz`."""
    with np.load(Path(path), allow_pickle=False) as handle:
        meta = json.loads(str(handle["__meta__"]))
        matrices = {}
        for key in handle.files:
            if not key.startswith("m::"):
                continue
            _, layer, site = key.split("::", 2)
            matrices[(int(layer), site)] = np.asarray(handle[key], dtype=_STORE_DTYPE)
    return ActivationDataset(
        rows=tuple(ActivationRow(**r) for r in meta["rows"]),
        matrices=matrices,
        layers=tuple(int(i) for i in meta["layers"]),
        sites=tuple(meta["sites"]),
        hidden_size=int(meta["hidden_size"]),
        policy=PositionPolicy(**meta["policy"]),
        model_key=meta.get("model_key"),
    )


def collect_item(
    collector: ActivationCollector,
    item: BenchmarkItem,
    tok_item: TokenizedItem,
    policy: PositionPolicy = DEFAULT_POLICY,
) -> tuple[list[ActivationRow], dict[tuple[int, str], np.ndarray]]:
    """Capture one item. Returns its rows and ``{(layer, site): [n_pos, hidden]}``."""
    selection = select_positions(
        tok_item.n_tokens, tok_item.evidence_spans, policy, item_key=tok_item.item_id
    )
    vectors = collector.run(tok_item.input_ids, selection.indices)
    evidence = _evidence_position(item, tok_item)
    target = float(item.target_positions[0]) if item.target_positions else float("nan")
    rows = [
        ActivationRow(
            item_id=item.item_id,
            scenario_id=item.scenario_id,
            family=item.family,
            fmt=item.fmt,
            length_bucket=item.length_bucket,
            split=item.split,
            token_index=int(index),
            n_tokens=int(selection.n_tokens),
            normalized_position=float(index) / max(int(selection.n_tokens), 1),
            offset_from_end=int(selection.n_tokens - 1 - index),
            role=role,
            span_index=int(span),
            evidence_position=evidence,
            target_position=target,
            answer=item.answer,
        )
        for index, role, span in zip(
            selection.indices, selection.roles, selection.span_index, strict=True
        )
    ]
    return rows, vectors


def collect_activations(
    model: Any,
    pairs: Sequence[tuple[BenchmarkItem, TokenizedItem]],
    *,
    layers: Sequence[int] | None = None,
    sites: Sequence[str] = SITES,
    policy: PositionPolicy = DEFAULT_POLICY,
    model_key: str | None = None,
    progress: Callable[[int], None] | None = None,
) -> ActivationDataset:
    """Run every item once and stack the results into an :class:`ActivationDataset`.

    Items are processed one at a time (SPEC 12.2) and each item's vectors are moved to
    CPU float16 inside the hook, so peak device memory is one item's activations, not
    the whole set.
    """
    if not pairs:
        raise ValueError("no items to collect")
    hidden = int(model.config.hidden_size)

    all_rows: list[ActivationRow] = []
    chunks: dict[tuple[int, str], list[np.ndarray]] = {}
    with ActivationCollector(model, layers=layers, sites=sites) as collector:
        for item, tok_item in pairs:
            rows, vectors = collect_item(collector, item, tok_item, policy)
            all_rows.extend(rows)
            for key, block in vectors.items():
                chunks.setdefault(key, []).append(block)
            if progress is not None:
                progress(len(rows))
        used_layers, used_sites = collector.layers, collector.sites

    return ActivationDataset(
        rows=tuple(all_rows),
        matrices={key: np.concatenate(blocks, axis=0) for key, blocks in chunks.items()},
        layers=used_layers,
        sites=used_sites,
        hidden_size=hidden,
        policy=policy,
        model_key=model_key,
    )


def dataset_from_arrays(
    rows: Sequence[ActivationRow],
    matrices: Mapping[tuple[int, str], np.ndarray],
    *,
    policy: PositionPolicy = DEFAULT_POLICY,
    model_key: str | None = None,
) -> ActivationDataset:
    """Build a dataset from pre-computed arrays. Used by tests and by offline re-analysis."""
    if not rows:
        raise ValueError("rows must be non-empty")
    keys = sorted(matrices)
    if not keys:
        raise ValueError("matrices must be non-empty")
    hidden = int(next(iter(matrices.values())).shape[1])
    return ActivationDataset(
        rows=tuple(rows),
        matrices={
            (int(layer), str(site)): np.asarray(m, dtype=_STORE_DTYPE)
            for (layer, site), m in matrices.items()
        },
        layers=tuple(sorted({int(layer) for layer, _ in keys})),
        sites=tuple(dict.fromkeys(str(site) for _, site in keys)),
        hidden_size=hidden,
        policy=policy,
        model_key=model_key,
    )
