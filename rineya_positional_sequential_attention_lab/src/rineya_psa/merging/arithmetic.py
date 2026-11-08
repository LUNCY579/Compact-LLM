"""Brief section 34 -- blockwise checkpoint arithmetic ``theta' = theta_base + alpha_l * Delta_l``.

``Delta_l`` is ``theta_other - theta_base`` restricted to a *broad layer block* (early /
middle / late decoder layers, plus the stack-level tensors), and ``alpha_l`` is that
block's interpolation coefficient drawn from :data:`ALPHA_GRID` = ``(0, 0.5, 1.0, 1.2)``.
``alpha = 0`` recovers the base, ``1.0`` recovers the other checkpoint on that block, and
``1.2`` extrapolates past it -- the point of the grid is to test whether the published
core's positional behaviour is a *direction* in weight space or only an endpoint.

**Prefer the in-memory path.** :func:`apply_blockwise_delta` rewrites a live model's
parameters in place from a base snapshot and a delta dict, and :meth:`AppliedMerge.restore`
puts the base back exactly. A 0.6B variant is ~1.2 GB on disk; a 4-block x 4-alpha screen
is 256 combinations, i.e. 300 GB if every one is materialised. Screening therefore never
touches the filesystem.

On-disk materialisation (:func:`materialise_merge`) is genuinely needed in exactly four
situations, and is worth its 1.2 GB only then:

1. handing a variant to a **separate process** -- a `from_pretrained` in an eval harness,
   a different CUDA context, or another machine;
2. running the Phase 1 audit (``model_audit``) against it, which reads safetensors bytes
   rather than a live module;
3. publishing or archiving a variant that a result depends on, so the recipe's SHA-256s
   can be re-verified later;
4. reproducing a run after the base and other checkpoints have moved on.

Everything is written under ``experiments/merges/<merge_id>/``. Every write is preceded by
``paths.assert_readonly_target``, so a merge can never land next to a source checkpoint.

Arithmetic is float32 and streamed tensor-by-tensor (and chunk-by-chunk inside a tensor),
so peak RSS is tens of megabytes regardless of checkpoint size; the result is stored back
in the **base tensor's own dtype**, because a variant that silently upcast bf16 to fp32
would be four times the size and would not be comparable to its parent.

DEVIATION: the safetensors *writer* here is hand-rolled (an 8-byte length, a JSON header,
then contiguous payload) rather than ``safetensors.torch.save_file``. ``save_file`` takes a
fully materialised ``dict[str, Tensor]``, which is the 1.2 GB peak this module exists to
avoid; writing the container directly keeps the peak at one chunk. The reader side reuses
``model_audit.hashing``, and ``tests/test_merging.py`` round-trips the output through the
real ``safetensors`` package so the format claim is checked rather than asserted.

DEVIATION: torch is imported lazily inside :func:`apply_blockwise_delta` and friends.
SPEC section 2 does not whitelist ``merging/`` for a module-scope torch import, and the
on-disk path genuinely does not need it -- bfloat16 encoding is done in numpy
(``round-to-nearest-even`` on the top 16 bits of the float32 word), which the tests pin
bit-for-bit against ``torch.Tensor.to(torch.bfloat16)``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from rineya_psa.model_audit.distances import NO_LAYER, parse_tensor_name
from rineya_psa.model_audit.hashing import (
    FLOAT_DTYPES,
    TensorLocation,
    collect_tensor_locations,
    discover_model_files,
    open_memmap,
    sha256_file,
)
from rineya_psa.paths import EXPERIMENTS, assert_readonly_target
from rineya_psa.statistics.provenance import write_result
from rineya_psa.types import stable_id

__all__ = [
    "ALPHA_GRID",
    "DEFAULT_CHUNK_BYTES",
    "GLOBAL_BLOCK",
    "AppliedMerge",
    "BlockAlphas",
    "LayerBlock",
    "MergeArtifact",
    "apply_blockwise_delta",
    "block_alphas_grid",
    "default_blocks",
    "deltas_from_states",
    "encode_from_float",
    "full_block_grid",
    "materialise_merge",
    "merge_workspace",
    "single_block_sweep",
    "snapshot_state",
    "uniform_alpha_sweep",
]

#: Brief section 34's coefficient grid.
ALPHA_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 1.2)

#: Block that owns every tensor outside the decoder stack (embeddings, final norm,
#: a materialised tied ``lm_head``). It is kept separate from the layer blocks because
#: moving the embedding matrix changes *every* position at once, which is precisely the
#: confound a positional study must be able to switch off.
GLOBAL_BLOCK: str = "global"

#: ~64 MiB of float32 per streamed chunk, matching ``model_audit.distances``.
DEFAULT_CHUNK_BYTES: int = 64 << 20

#: Files copied beside the weights so a materialised merge is loadable in place.
_AUX_GLOBS: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "*.py",
)

# safetensors dtype tag -> the numpy dtype its payload is written as. BF16 has no numpy
# dtype and is handled by ``_encode_bf16``.
_ENCODE_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
}


# --------------------------------------------------------------------------------------
# Layer blocks and their coefficients
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerBlock:
    """A contiguous span of decoder layers that shares one interpolation coefficient."""

    name: str
    layers: tuple[int, ...] = ()
    is_global: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a layer block needs a name")
        if self.is_global and self.layers:
            raise ValueError(f"{self.name}: the global block owns no decoder layers")
        if not self.is_global and not self.layers:
            raise ValueError(f"{self.name}: a layer block must own at least one layer")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layers": list(self.layers),
            "is_global": self.is_global,
            "span": (
                None if self.is_global else [int(min(self.layers)), int(max(self.layers))]
            ),
        }


def default_blocks(num_layers: int, *, n_layer_blocks: int = 3) -> tuple[LayerBlock, ...]:
    """Broad early/middle/late blocks over ``num_layers``, plus the global block.

    Three blocks, not twenty-eight: the brief asks for *broad* blocks because a per-layer
    coefficient has 4^28 settings and no statistical power behind any of them. For
    Qwen3-0.6B this gives layers 0-8 / 9-18 / 19-27.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if not 1 <= n_layer_blocks <= num_layers:
        raise ValueError(
            f"n_layer_blocks must lie in [1, {num_layers}], got {n_layer_blocks}"
        )
    names = (
        ("early", "middle", "late")
        if n_layer_blocks == 3
        else tuple(f"block{i}" for i in range(n_layer_blocks))
    )
    bounds = np.linspace(0, num_layers, n_layer_blocks + 1).round().astype(int)
    blocks = [
        LayerBlock(name=names[i], layers=tuple(range(int(bounds[i]), int(bounds[i + 1]))))
        for i in range(n_layer_blocks)
    ]
    blocks.append(LayerBlock(name=GLOBAL_BLOCK, is_global=True))
    return tuple(blocks)


@dataclass(frozen=True)
class BlockAlphas:
    """One coefficient per block, with the tensor -> block resolution baked in.

    ``for_tensor`` is the only sanctioned way to ask "what alpha does this state-dict key
    get"; it routes through ``model_audit.distances.parse_tensor_name`` so the layer index
    is parsed in exactly one place in the lab.
    """

    blocks: tuple[LayerBlock, ...]
    alphas: Mapping[str, float]

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if not blocks:
            raise ValueError("BlockAlphas needs at least one block")
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate block names: {names}")
        if sum(1 for b in blocks if b.is_global) > 1:
            raise ValueError("at most one global block is allowed")

        seen: dict[int, str] = {}
        for block in blocks:
            for layer in block.layers:
                if layer in seen:
                    raise ValueError(
                        f"layer {layer} is claimed by both {seen[layer]!r} and {block.name!r}"
                    )
                seen[layer] = block.name

        supplied = dict(self.alphas)
        unknown = sorted(set(supplied) - set(names))
        if unknown:
            raise ValueError(f"alphas for unknown blocks {unknown}; known: {names}")
        missing = sorted(set(names) - set(supplied))
        if missing:
            raise ValueError(f"no alpha given for blocks {missing}")
        resolved = {}
        for name, value in supplied.items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"alpha for {name!r} is not finite: {value}")
            resolved[name] = value

        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "alphas", MappingProxyType(resolved))
        object.__setattr__(self, "_layer_to_block", MappingProxyType(seen))
        object.__setattr__(
            self,
            "_global_name",
            next((b.name for b in blocks if b.is_global), None),
        )

    # -- resolution ---------------------------------------------------------------

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.blocks)

    @property
    def global_block(self) -> str | None:
        return self._global_name  # type: ignore[attr-defined]  # set in __post_init__

    def block_for_layer(self, layer: int) -> str:
        """Block owning decoder ``layer``; ``NO_LAYER`` (-1) resolves to the global block."""
        mapping: Mapping[int, str] = self._layer_to_block  # type: ignore[attr-defined]
        if int(layer) == NO_LAYER:
            name = self.global_block
            if name is None:
                raise KeyError("no global block is configured for stack-level tensors")
            return name
        try:
            return mapping[int(layer)]
        except KeyError:
            raise KeyError(
                f"layer {layer} belongs to no block; blocks cover {sorted(mapping)}"
            ) from None

    def block_for_tensor(self, name: str) -> str:
        return self.block_for_layer(parse_tensor_name(str(name)).layer)

    def for_layer(self, layer: int) -> float:
        return self.alphas[self.block_for_layer(layer)]

    def for_tensor(self, name: str) -> float:
        return self.alphas[self.block_for_tensor(name)]

    # -- derivation ---------------------------------------------------------------

    def with_alpha(self, block: str, value: float) -> BlockAlphas:
        if block not in self.alphas:
            raise KeyError(f"unknown block {block!r}; known: {self.block_names}")
        updated = dict(self.alphas)
        updated[block] = float(value)
        return BlockAlphas(blocks=self.blocks, alphas=updated)

    @classmethod
    def uniform(cls, blocks: Sequence[LayerBlock], value: float) -> BlockAlphas:
        return cls(blocks=tuple(blocks), alphas={b.name: float(value) for b in blocks})

    # -- identity -----------------------------------------------------------------

    def is_identity(self) -> bool:
        """True when the merge provably reproduces the base checkpoint."""
        return all(a == 0.0 for a in self.alphas.values())

    def slug(self) -> str:
        """Filesystem-safe tag, e.g. ``early0p5-middle1p0-late1p2-global0p0``."""
        parts = [
            f"{name}{self.alphas[name]:.2f}".replace(".", "p") for name in self.block_names
        ]
        return "-".join(parts)

    def describe(self) -> dict[str, Any]:
        return {
            "blocks": [b.describe() for b in self.blocks],
            "alphas": dict(self.alphas),
            "slug": self.slug(),
            "is_identity": self.is_identity(),
        }


def uniform_alpha_sweep(
    blocks: Sequence[LayerBlock], grid: Sequence[float] = ALPHA_GRID
) -> list[BlockAlphas]:
    """One coefficient applied to every block: the plain task-arithmetic interpolation."""
    return [BlockAlphas.uniform(blocks, a) for a in grid]


def single_block_sweep(
    blocks: Sequence[LayerBlock],
    grid: Sequence[float] = ALPHA_GRID,
    *,
    background: float = 0.0,
    include_background: bool = True,
) -> list[BlockAlphas]:
    """Move one block at a time off ``background``: the block-attribution screen.

    This is the cheap ``|blocks| * (|grid| - 1) + 1`` design, and it is the one that
    answers "which part of the stack carries the positional change".
    """
    blocks = tuple(blocks)
    out: list[BlockAlphas] = []
    if include_background:
        out.append(BlockAlphas.uniform(blocks, background))
    for block in blocks:
        for alpha in grid:
            if alpha == background:
                continue
            out.append(BlockAlphas.uniform(blocks, background).with_alpha(block.name, alpha))
    return out


def full_block_grid(
    blocks: Sequence[LayerBlock],
    grid: Sequence[float] = ALPHA_GRID,
    *,
    max_combinations: int = 512,
) -> list[BlockAlphas]:
    """Cartesian product over blocks. Refuses to silently build a huge sweep."""
    blocks = tuple(blocks)
    total = len(grid) ** len(blocks)
    if total > max_combinations:
        raise ValueError(
            f"{len(grid)}^{len(blocks)} = {total} combinations exceeds max_combinations="
            f"{max_combinations}; use single_block_sweep or raise the ceiling deliberately"
        )
    return [
        BlockAlphas(blocks=blocks, alphas=dict(zip((b.name for b in blocks), combo, strict=True)))
        for combo in product(grid, repeat=len(blocks))
    ]


def block_alphas_grid(
    num_layers: int,
    *,
    grid: Sequence[float] = ALPHA_GRID,
    n_layer_blocks: int = 3,
    design: str = "single_block",
) -> list[BlockAlphas]:
    """Convenience: blocks plus a sweep, in one call. ``design`` selects the sweep shape."""
    blocks = default_blocks(num_layers, n_layer_blocks=n_layer_blocks)
    if design == "uniform":
        return uniform_alpha_sweep(blocks, grid)
    if design == "single_block":
        return single_block_sweep(blocks, grid)
    if design == "full":
        return full_block_grid(blocks, grid)
    raise ValueError(f"unknown design {design!r}; expected uniform|single_block|full")


# --------------------------------------------------------------------------------------
# In-memory application (the preferred path)
# --------------------------------------------------------------------------------------


@dataclass
class AppliedMerge:
    """Handle for an in-place blockwise merge, and the means to undo it exactly.

    ``restore()`` writes ``base_state`` back rather than replaying a saved copy, so it
    costs no extra memory and is exact by construction. Usable as a context manager.
    """

    model: Any
    base_state: Mapping[str, Any]
    alphas: BlockAlphas
    touched: tuple[str, ...]
    skipped: tuple[str, ...]
    tensor_alpha: Mapping[str, float]
    chunk_bytes: int = DEFAULT_CHUNK_BYTES
    restored: bool = False

    def restore(self) -> None:
        """Put the base weights back. Idempotent."""
        if self.restored:
            return
        _write_blend(
            self.model,
            self.base_state,
            deltas={},
            tensor_alpha=dict.fromkeys(self.touched, 0.0),
            names=self.touched,
            chunk_bytes=self.chunk_bytes,
        )
        self.restored = True

    def __enter__(self) -> AppliedMerge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.restore()

    def describe(self) -> dict[str, Any]:
        return {
            "alphas": self.alphas.describe(),
            "n_touched": len(self.touched),
            "n_skipped": len(self.skipped),
            "skipped": list(self.skipped[:20]),
            "restored": self.restored,
            "alpha_by_block_tensor_count": _count_by_alpha(self.tensor_alpha),
        }


def _count_by_alpha(tensor_alpha: Mapping[str, float]) -> dict[str, int]:
    out: dict[str, int] = {}
    for alpha in tensor_alpha.values():
        out[f"{alpha:g}"] = out.get(f"{alpha:g}", 0) + 1
    return out


def _model_tensors(model: Any) -> dict[str, Any]:
    """Live parameter/buffer objects by state-dict name (``keep_vars`` so writes stick)."""
    return dict(model.state_dict(keep_vars=True))


def snapshot_state(
    model: Any,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    include: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Detached CPU float32 copy of a model's state dict -- the ``theta_base`` snapshot.

    float32 and off the accelerator on purpose: the snapshot outlives the merge sweep and
    must not compete with activations for VRAM, and a bf16 snapshot would re-round the base
    every time it is written back, so ``restore()`` would drift.
    """
    import torch

    torch_dtype = getattr(torch, dtype)
    out: dict[str, Any] = {}
    with torch.no_grad():
        for name, tensor in _model_tensors(model).items():
            if include is not None and not include(name):
                continue
            out[name] = tensor.detach().to(device=device, dtype=torch_dtype).clone()
    return out


def deltas_from_states(
    base_state: Mapping[str, Any],
    other_state: Mapping[str, Any],
    *,
    include: Callable[[str], bool] | None = None,
    drop_zero: bool = True,
) -> dict[str, Any]:
    """``Delta = other - base`` over shared keys, in float32 on the base's device.

    Keys present on one side only are dropped: the panel's tied ``lm_head.weight`` is
    materialised by two of the four cores and not the other two, and pretending the
    missing side is zero would inject a spurious delta the size of the embedding matrix.
    ``drop_zero`` discards exactly-equal tensors so the delta dict costs nothing for the
    parts of the stack a fine-tune never touched.
    """
    import torch

    deltas: dict[str, Any] = {}
    with torch.no_grad():
        for name, base in base_state.items():
            if include is not None and not include(name):
                continue
            other = other_state.get(name)
            if other is None:
                continue
            if tuple(base.shape) != tuple(other.shape):
                raise ValueError(
                    f"{name}: shape {tuple(base.shape)} vs {tuple(other.shape)}; the two "
                    "checkpoints are not arithmetic-compatible"
                )
            delta = other.to(device=base.device, dtype=torch.float32) - base.to(torch.float32)
            if drop_zero and not bool(torch.any(delta != 0)):
                continue
            deltas[name] = delta
    return deltas


def _write_blend(
    model: Any,
    base_state: Mapping[str, Any],
    deltas: Mapping[str, Any],
    tensor_alpha: Mapping[str, float],
    names: Sequence[str],
    *,
    chunk_bytes: int,
) -> None:
    """``param <- base + alpha * delta``, chunked, accumulating in float32.

    Chunking matters: ``embed_tokens`` is 151936 x 1024, which is 622 MB in float32. A
    whole-tensor blend would need 1.2 GB of scratch on an 8.59 GB card that is also holding
    the model.
    """
    import torch

    tensors = _model_tensors(model)
    elements = max(1, chunk_bytes // 4)
    with torch.no_grad():
        for name in names:
            param = tensors[name]
            base = base_state[name]
            delta = deltas.get(name)
            alpha = float(tensor_alpha.get(name, 0.0))
            flat_p = param.reshape(-1)
            flat_b = base.reshape(-1)
            flat_d = None if (delta is None or alpha == 0.0) else delta.reshape(-1)
            n = flat_p.numel()
            if flat_b.numel() != n:
                raise ValueError(
                    f"{name}: base snapshot has {flat_b.numel()} elements, model has {n}"
                )
            for start in range(0, n, elements):
                stop = min(start + elements, n)
                # ``Tensor.to`` returns *self* when the device and dtype already match, and
                # the snapshot is float32 on CPU by construction -- so without copy=True the
                # add_ below would write into base_state and restore() would stop restoring.
                buf = flat_b[start:stop].to(
                    device=param.device, dtype=torch.float32, copy=flat_d is not None
                )
                if flat_d is not None:
                    buf.add_(
                        flat_d[start:stop].to(device=param.device, dtype=torch.float32),
                        alpha=alpha,
                    )
                flat_p[start:stop].copy_(buf)


def apply_blockwise_delta(
    model: Any,
    base_state: Mapping[str, Any],
    deltas: Mapping[str, Any],
    alphas: BlockAlphas,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> AppliedMerge:
    """Set ``model``'s weights to ``theta_base + alpha_block * Delta`` in place.

    This is the screening path: no bytes hit the disk, the blend runs in float32 chunk by
    chunk, and :meth:`AppliedMerge.restore` (or a ``with`` block) returns the model to
    ``base_state`` exactly. Every name in ``base_state`` is rewritten, including the ones
    with no delta and the ones whose alpha is 0, so the model's state after the call
    depends only on ``(base_state, deltas, alphas)`` and not on what it held before.

    Names in ``deltas`` that the model does not expose are reported in
    :attr:`AppliedMerge.skipped` rather than raising: a tied ``lm_head.weight`` legitimately
    disappears from ``state_dict`` on the cores that do not materialise it.
    """
    tensors = _model_tensors(model)
    names = [n for n in base_state if n in tensors]
    missing_base = sorted(set(base_state) - set(tensors))
    skipped = sorted(set(deltas) - set(tensors)) + missing_base
    if not names:
        raise ValueError(
            "base_state shares no tensor names with the model; snapshot_state(model) is "
            "the intended source of base_state"
        )

    tensor_alpha = {name: alphas.for_tensor(name) for name in names}
    _write_blend(
        model, base_state, deltas, tensor_alpha, names, chunk_bytes=chunk_bytes
    )
    return AppliedMerge(
        model=model,
        base_state=base_state,
        alphas=alphas,
        touched=tuple(names),
        skipped=tuple(sorted(set(skipped))),
        tensor_alpha=MappingProxyType(tensor_alpha),
        chunk_bytes=chunk_bytes,
    )


# --------------------------------------------------------------------------------------
# float encoding (numpy-only, so the on-disk path needs no torch)
# --------------------------------------------------------------------------------------


def _encode_bf16(values: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit pattern with round-to-nearest-even, as ``<u2``.

    bfloat16 *is* the top 16 bits of a float32, so rounding is "add half an ulp, plus one
    more when the retained bit is odd, then truncate". NaN is re-emitted as a quiet NaN
    because the naive shift can turn a signalling NaN into an infinity.
    """
    f32 = np.ascontiguousarray(values, dtype="<f4")
    u = f32.view("<u4")
    with np.errstate(over="ignore"):
        biased = u + 0x7FFF + ((u >> 16) & 1)
    out = (biased >> 16).astype("<u2")
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    if nan.any():
        out[nan] = 0x7FC0
    return out


def encode_from_float(values: np.ndarray, st_dtype: str) -> bytes:
    """Serialise a float array as raw little-endian bytes in safetensors ``st_dtype``."""
    if st_dtype == "BF16":
        return _encode_bf16(values).tobytes()
    try:
        nd = _ENCODE_DTYPES[st_dtype]
    except KeyError:
        raise ValueError(f"cannot encode floats as safetensors dtype {st_dtype!r}") from None
    return np.ascontiguousarray(values, dtype=nd).tobytes()


def _decode_to_float32(raw: np.ndarray, st_dtype: str) -> np.ndarray:
    """uint8 view -> float32. bf16 widens exactly; every other float dtype casts."""
    if st_dtype == "BF16":
        wide = raw.view("<u2").astype("<u4")
        np.left_shift(wide, 16, out=wide)
        return wide.view("<f4")
    try:
        nd = _ENCODE_DTYPES[st_dtype]
    except KeyError:
        raise ValueError(f"cannot decode safetensors dtype {st_dtype!r} to float32") from None
    return raw.view(nd).astype("<f4")


# --------------------------------------------------------------------------------------
# On-disk materialisation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeArtifact:
    """Where a materialised variant landed and what it hashes to."""

    merge_id: str
    directory: Path
    weights_file: Path
    recipe_file: Path
    n_tensors: int
    n_params: int
    bytes_written: int
    weights_sha256: str
    tensor_sha256: Mapping[str, str]
    aux_files: tuple[str, ...]
    base_only: tuple[str, ...]
    other_only: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "merge_id": self.merge_id,
            "directory": str(self.directory),
            "weights_file": str(self.weights_file),
            "recipe_file": str(self.recipe_file),
            "n_tensors": self.n_tensors,
            "n_params": self.n_params,
            "bytes_written": self.bytes_written,
            "weights_sha256": self.weights_sha256,
            "tensor_sha256": dict(self.tensor_sha256),
            "aux_files": list(self.aux_files),
            "base_only": list(self.base_only),
            "other_only": list(self.other_only),
        }


def merge_workspace(merge_id: str) -> Path:
    """``experiments/merges/<merge_id>/``, guaranteed not to be inside a checkpoint."""
    if not merge_id or any(sep in merge_id for sep in ("/", "\\", "..")):
        raise ValueError(f"merge_id must be a single safe path component, got {merge_id!r}")
    target = EXPERIMENTS / "merges" / merge_id
    assert_readonly_target(target)
    return target


def default_merge_id(base_key: str, other_key: str, alphas: BlockAlphas) -> str:
    """Deterministic, readable, and short enough for Windows' path limit."""
    slug = alphas.slug()
    stem = f"{base_key}+{other_key}-{slug}".replace(" ", "")
    if len(stem) <= 96:
        return stem
    return f"{base_key}+{other_key}-{stable_id(slug)}"


def _safetensors_shards(model_dir: Path) -> list[Path]:
    files = discover_model_files(model_dir)
    if not files:
        raise FileNotFoundError(f"no weight files under {model_dir}")
    bad = [f.name for f in files if f.suffix != ".safetensors"]
    if bad:
        raise ValueError(
            f"{model_dir}: blockwise arithmetic reads safetensors only, found {bad[:3]}"
        )
    return files


def _build_header(
    order: Sequence[str], locs: Mapping[str, TensorLocation], metadata: Mapping[str, str]
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    """safetensors container header: 8-byte length, JSON, padded to an 8-byte boundary."""
    entries: dict[str, Any] = {}
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name in order:
        loc = locs[name]
        begin, end = cursor, cursor + loc.nbytes
        entries[name] = {
            "dtype": loc.st_dtype,
            "shape": list(loc.shape),
            "data_offsets": [begin, end],
        }
        spans[name] = (begin, end)
        cursor = end
    entries["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    body = json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    pad = (-len(body)) % 8
    body = body + b" " * pad
    return struct.pack("<Q", len(body)) + body, spans


def _iter_element_chunks(
    n_elements: int, itemsize: int, chunk_bytes: int
) -> Iterator[tuple[int, int]]:
    # base + delta + result, all float32, plus the raw source views.
    per_element = max(1, 4 * 3 + 2 * itemsize)
    step = max(1, chunk_bytes // per_element)
    for start in range(0, n_elements, step):
        yield start, min(start + step, n_elements)


def _copy_aux_files(base_dir: Path, out_dir: Path) -> tuple[str, ...]:
    copied: list[str] = []
    for pattern in _AUX_GLOBS:
        for src in sorted(base_dir.glob(pattern)):
            if not src.is_file() or src.suffix == ".safetensors":
                continue
            dst = out_dir / src.name
            assert_readonly_target(dst)
            shutil.copy2(src, dst)
            copied.append(src.name)
    return tuple(sorted(set(copied)))


def materialise_merge(
    base_dir: str | Path,
    other_dir: str | Path,
    alphas: BlockAlphas,
    *,
    base_key: str = "base",
    other_key: str = "other",
    merge_id: str | None = None,
    out_dir: str | Path | None = None,
    copy_aux: bool = True,
    overwrite: bool = False,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    include: Callable[[str], bool] | None = None,
) -> MergeArtifact:
    """Write ``theta_base + alpha_block * (theta_other - theta_base)`` to an experiment dir.

    Reads both checkpoints through memory-mapped safetensors, blends one chunk at a time in
    float32, and stores each tensor back in **the base tensor's own dtype**. Output goes to
    ``experiments/merges/<merge_id>/`` unless ``out_dir`` overrides it; either way
    ``paths.assert_readonly_target`` runs on the directory and on every file before it is
    opened, so a source checkpoint can never be the destination.

    Tensors present only in the base are copied through unchanged (and listed as
    ``base_only``); tensors present only in ``other`` are not invented (listed as
    ``other_only``). Non-float tensors are copied byte-for-byte -- an integer buffer has no
    meaningful interpolation.
    """
    base_dir = Path(base_dir).resolve()
    other_dir = Path(other_dir).resolve()
    merge_id = merge_id or default_merge_id(base_key, other_key, alphas)
    directory = Path(out_dir).resolve() if out_dir is not None else merge_workspace(merge_id)
    assert_readonly_target(directory)

    weights_file = directory / "model.safetensors"
    if weights_file.exists() and not overwrite:
        raise FileExistsError(f"{weights_file} exists; pass overwrite=True to replace it")
    assert_readonly_target(weights_file)

    base_locs = collect_tensor_locations(_safetensors_shards(base_dir))
    other_locs = collect_tensor_locations(_safetensors_shards(other_dir))
    if include is not None:
        base_locs = {n: loc for n, loc in base_locs.items() if include(n)}
        other_locs = {n: loc for n, loc in other_locs.items() if include(n)}

    order = sorted(base_locs)
    base_only = tuple(n for n in order if n not in other_locs)
    other_only = tuple(sorted(set(other_locs) - set(base_locs)))
    mismatched = [
        n for n in order if n in other_locs and base_locs[n].shape != other_locs[n].shape
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} tensors have different shapes in the two checkpoints "
            f"(e.g. {mismatched[:3]}); they are not arithmetic-compatible"
        )

    directory.mkdir(parents=True, exist_ok=True)
    header, spans = _build_header(
        order,
        base_locs,
        {
            "format": "pt",
            "merge_id": merge_id,
            "produced_by": "rineya_psa.merging.arithmetic.materialise_merge",
            "base": base_key,
            "other": other_key,
            "alphas": json.dumps(dict(alphas.alphas), sort_keys=True),
        },
    )

    tensor_alpha = {n: alphas.for_tensor(n) for n in order}
    tensor_sha: dict[str, str] = {}
    n_params = 0
    with open(weights_file, "wb") as out, open_memmap_pool() as pool:
        out.write(header)
        for name in order:
            loc = base_locs[name]
            other = other_locs.get(name)
            alpha = tensor_alpha[name]
            digest = hashlib.sha256()
            written = _stream_tensor(out, digest, pool, loc, other, alpha, chunk_bytes)
            if written != loc.nbytes:
                raise RuntimeError(
                    f"{name}: wrote {written} bytes, header reserved {loc.nbytes}"
                )
            tensor_sha[name] = digest.hexdigest()
            n_params += loc.n_params

    total = weights_file.stat().st_size
    expected = len(header) + (spans[order[-1]][1] if order else 0)
    if total != expected:
        raise RuntimeError(f"{weights_file}: wrote {total} bytes, header implies {expected}")

    aux = _copy_aux_files(base_dir, directory) if copy_aux else ()

    recipe = {
        "merge_id": merge_id,
        "formula": "theta' = theta_base + alpha_block * (theta_other - theta_base)",
        "base": {"key": base_key, "path": str(base_dir), "n_tensors": len(base_locs)},
        "other": {"key": other_key, "path": str(other_dir), "n_tensors": len(other_locs)},
        "alphas": alphas.describe(),
        "tensor_alpha": tensor_alpha,
        "store_dtype": {n: base_locs[n].st_dtype for n in order},
        "base_only_tensors": list(base_only),
        "other_only_tensors": list(other_only),
        "output": {
            "directory": str(directory),
            "weights_file": weights_file.name,
            "bytes": total,
            "n_tensors": len(order),
            "n_params": n_params,
            "sha256": sha256_file(weights_file),
            "tensor_sha256": tensor_sha,
        },
        "aux_files": list(aux),
    }
    recipe_file = write_result(
        recipe,
        directory / "merge_recipe.json",
        config={"merge_id": merge_id, "alphas": dict(alphas.alphas)},
        extra={"base_dir": str(base_dir), "other_dir": str(other_dir)},
    )

    return MergeArtifact(
        merge_id=merge_id,
        directory=directory,
        weights_file=weights_file,
        recipe_file=recipe_file,
        n_tensors=len(order),
        n_params=n_params,
        bytes_written=total,
        weights_sha256=recipe["output"]["sha256"],
        tensor_sha256=MappingProxyType(tensor_sha),
        aux_files=aux,
        base_only=base_only,
        other_only=other_only,
    )


class open_memmap_pool:  # noqa: N801 - used as a context manager, not a type
    """One read-only memmap per source file, closed together."""

    def __init__(self) -> None:
        self._maps: dict[Path, np.memmap] = {}

    def get(self, path: Path) -> np.memmap:
        path = Path(path)
        mm = self._maps.get(path)
        if mm is None:
            mm = open_memmap(path)
            self._maps[path] = mm
        return mm

    def __enter__(self) -> open_memmap_pool:
        return self

    def __exit__(self, *exc: object) -> None:
        self._maps.clear()


def _stream_tensor(
    out: Any,
    digest: Any,
    pool: open_memmap_pool,
    base: TensorLocation,
    other: TensorLocation | None,
    alpha: float,
    chunk_bytes: int,
) -> int:
    """Blend and write one tensor; returns the byte count actually written."""
    base_mm = pool.get(base.file)
    passthrough = other is None or alpha == 0.0 or not base.is_float
    if passthrough and other is not None and alpha != 0.0 and not base.is_float:
        # An integer buffer has no interpolation; copying the base is the honest choice
        # and is recorded in the recipe by ``store_dtype`` staying integral.
        pass

    written = 0
    itemsize = base.itemsize
    for start, stop in _iter_element_chunks(base.n_params, itemsize, chunk_bytes):
        raw = np.asarray(base_mm[base.begin + start * itemsize : base.begin + stop * itemsize])
        if passthrough:
            payload = raw.tobytes()
        else:
            other_mm = pool.get(other.file)
            other_raw = np.asarray(
                other_mm[other.begin + start * other.itemsize : other.begin + stop * other.itemsize]
            )
            b = _decode_to_float32(raw, base.st_dtype)
            o = _decode_to_float32(other_raw, other.st_dtype)
            payload = encode_from_float(b + np.float32(alpha) * (o - b), base.st_dtype)
        out.write(payload)
        digest.update(payload)
        written += len(payload)
    return written


def iter_float_families(names: Iterable[str], families: Sequence[str]) -> list[str]:
    """Names whose parsed family is in ``families`` -- an ``include`` predicate helper."""
    wanted = set(families)
    return [n for n in names if parse_tensor_name(n).family in wanted]


def attention_only(name: str) -> bool:
    """``include`` predicate restricting a merge to the attention projections."""
    return parse_tensor_name(name).family in {"q_proj", "k_proj", "v_proj", "o_proj"}


def float_dtype_tags() -> frozenset[str]:
    """safetensors dtype tags this module will interpolate."""
    return FLOAT_DTYPES
