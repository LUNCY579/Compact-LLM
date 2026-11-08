"""Compute accounting — brief section 36.

An arm that helps because it is three times as expensive has not shown that
sequentiality matters. This module produces the number that settles that, next to the
wall-clock and memory figures that show whether the model of cost is honest.

The FLOPs model
---------------
Multiply-accumulate counted as 2 FLOPs. Batch 1, prompt length ``T``, hidden ``H``,
intermediate ``I``, query heads ``Hq``, KV heads ``Hkv``, head dim ``D``, vocabulary
``V``, ``Dq = Hq*D``, ``Dkv = Hkv*D``, ``c`` the causal factor and ``k`` the number of
positions kept by the LM head. **Per decoder layer**::

    q_proj    = 2 * T * H * Dq          * n_q_passes
    k_proj    = 2 * T * H * Dkv         * n_kv_passes
    v_proj    = 2 * T * H * Dkv         * n_kv_passes
    scores    = 2 * c * Hq * T^2 * D
    context   = 2 * c * Hq * T^2 * D
    o_proj    = 2 * T * Dq * H
    mlp       = 6 * T * H * I                     (gate, up, down; SwiGLU)

and once for the whole model::

    lm_head   = 2 * k * H * V

Elementwise work — RMSNorm, ``q_norm``/``k_norm``, RoPE, softmax, SiLU, the residual
adds, and every intervention's gather — is ``O(T*H)`` or ``O(Hq*T^2)`` with a small
constant and is *excluded* from the layer terms; :func:`intervention_flops` accounts
for the intervention side separately so it can be reported rather than buried.

**The fact that makes this worth computing.** A staged arm partitions the query heads,
so summed over stages its attention matmuls and its ``o_proj`` are *exactly* the
parallel cost — the entire overhead of sequential attention on this architecture lives
in the repeated Q/K/V projections. That is why ``n_q_passes`` and ``n_kv_passes`` are the
only arm-dependent quantities above, and why a 4-stage arm costs far less than 4x.

**Two projection counts, both real.** Algorithmically a staged arm needs one query
projection in total (each head is projected once, from its own stage's hidden state) and
``kv_projection_cost()`` key/value projections. The wrapper as written calls
``_project_qkv`` on the full hidden state at every stage and discards what it does not
use, so the *executed* cost is ``n_stages`` of each. ``as_implemented=True`` (the
default) reports the executed cost, because it is the number that has to agree with the
wall-clock and tokens/sec in the same row; ``as_implemented=False`` reports the
algorithmic floor. Both are one keyword apart and the row records which was used.

``causal_factor`` defaults to 1.0 — the dense ``T x T`` count. A causal kernel that skips
the upper triangle approaches 0.5; the efficient SDPA backends this lab forces (SPEC
12.1) do skip it, so pass ``causal_factor=0.5`` when comparing against a kernel-level
measurement and say which was used. The default is dense because it is the figure that
does not depend on which backend the dispatcher picked.

DEVIATION: SPEC section 2 permits an import-time ``torch`` only under ``hooks/``,
``loading.py``, ``sequential_attention/``, ``activation_analysis/`` and
``interventions/runtime*.py``, so ``torch`` and ``psutil`` are imported inside functions.
The FLOPs model is pure arithmetic and needs neither, which means cost planning runs on
a torch-free interpreter.

DEVIATION: ``psutil`` is optional. When it is absent the host-RAM columns are ``None``
and ``psutil_available`` is ``False`` in every row, rather than the row being dropped or
the number being faked.

DEVIATION: ``SequentialArm`` is referenced only in annotations, so it is imported under
``TYPE_CHECKING``. A module-scope import would execute
``sequential_attention/__init__.py``, which imports torch — permitted there by SPEC
section 2, but not here.
"""

from __future__ import annotations

import contextlib
import importlib.util
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rineya_psa.paths import RESULTS
from rineya_psa.types import Arch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rineya_psa.sequential_attention.arms import SequentialArm

__all__ = [
    "COMPUTE_METRICS_COLUMNS",
    "COMPUTE_METRICS_PATH",
    "ComputeConfig",
    "ComputeMeasurement",
    "ComputeProbe",
    "FlopsBreakdown",
    "arm_projection_counts",
    "compute_metrics_row",
    "estimate_flops",
    "intervention_flops",
    "measure_compute",
    "psutil_available",
    "write_compute_metrics",
]

#: Where section 36's table lands.
COMPUTE_METRICS_PATH: Path = RESULTS / "compute_metrics.csv"


def psutil_available() -> bool:
    """True when host-RAM columns can be filled. Checked, never assumed."""
    return importlib.util.find_spec("psutil") is not None


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeConfig:
    """Everything the FLOPs model needs. ``Arch`` plus the one field it lacks."""

    arch: Arch
    intermediate_size: int

    def __post_init__(self) -> None:
        if int(self.intermediate_size) < 1:
            raise ValueError(f"intermediate_size must be >= 1, got {self.intermediate_size}")
        object.__setattr__(self, "intermediate_size", int(self.intermediate_size))

    @classmethod
    def from_hf_config(cls, config: Any) -> ComputeConfig:
        from rineya_psa.loading import arch_from_config

        return cls(
            arch=arch_from_config(config),
            intermediate_size=int(config.intermediate_size),
        )

    @classmethod
    def from_model(cls, model: Any) -> ComputeConfig:
        return cls.from_hf_config(model.config)

    def describe(self) -> dict[str, Any]:
        a = self.arch
        return {
            "num_layers": a.num_layers,
            "hidden_size": a.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_q_heads": a.num_q_heads,
            "num_kv_heads": a.num_kv_heads,
            "head_dim": a.head_dim,
            "vocab_size": a.vocab_size,
        }


@dataclass(frozen=True)
class FlopsBreakdown:
    """Per-forward FLOPs, split so a reader can see where an arm's overhead went."""

    q_proj: int
    kv_proj: int
    attention: int
    o_proj: int
    mlp: int
    lm_head: int
    intervention: int
    n_q_passes: int
    n_kv_passes: int
    arm_layers: int
    seq_len: int

    @property
    def total(self) -> int:
        return int(
            self.q_proj
            + self.kv_proj
            + self.attention
            + self.o_proj
            + self.mlp
            + self.lm_head
            + self.intervention
        )

    @property
    def per_token(self) -> float:
        return self.total / max(int(self.seq_len), 1)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total"] = self.total
        d["per_token"] = self.per_token
        return d


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


def arm_projection_counts(
    arm: SequentialArm | None, *, as_implemented: bool = True
) -> tuple[int, int]:
    """``(query projection passes, key/value projection passes)`` per staged layer.

    See the module docstring for why there are two answers and which one is the default.
    """
    if arm is None or arm.n_stages == 1:
        return 1, 1
    if arm.query_refinement:
        # One Q/K/V projection, then one q_proj per refinement step (all but the last
        # stage). The hidden state never moves, so K/V are computed exactly once.
        return arm.n_stages, 1
    if as_implemented:
        return arm.n_stages, arm.n_stages
    return 1, arm.kv_projection_cost()


def estimate_flops(
    config: ComputeConfig,
    seq_len: int,
    *,
    arm: SequentialArm | None = None,
    arm_layers: Sequence[int] | int | None = None,
    logits_to_keep: int = 1,
    causal_factor: float = 1.0,
    as_implemented: bool = True,
    intervention: int = 0,
) -> FlopsBreakdown:
    """Forward-pass FLOPs for one prompt of ``seq_len`` tokens, batch 1.

    ``arm_layers`` is the layer set the arm is installed on (a sequence of indices, a
    count, or ``None`` for every layer). The layers without the arm are billed at the
    parallel rate, which is what makes a six-layer sequential install comparable with a
    full-stack one.
    """
    T = int(seq_len)
    if T < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if not 0.0 < float(causal_factor) <= 1.0:
        raise ValueError(f"causal_factor must lie in (0, 1], got {causal_factor}")
    k = int(logits_to_keep)
    if k < 0:
        raise ValueError(f"logits_to_keep must be >= 0, got {logits_to_keep}")

    a = config.arch
    H = a.hidden_size
    D = a.head_dim
    Dq = a.num_q_heads * D
    Dkv = a.num_kv_heads * D
    L = a.num_layers

    if arm is None:
        n_staged = 0
    elif arm_layers is None:
        n_staged = L
    elif isinstance(arm_layers, int):
        n_staged = int(arm_layers)
    else:
        n_staged = len({int(i) for i in arm_layers})
    if not 0 <= n_staged <= L:
        raise ValueError(f"{n_staged} staged layers but the model has {L}")
    n_plain = L - n_staged

    n_q, n_kv = arm_projection_counts(arm, as_implemented=as_implemented)

    q_unit = 2 * T * H * Dq
    kv_unit = 2 * (2 * T * H * Dkv)  # k_proj and v_proj together
    q_proj = q_unit * (n_plain + n_staged * n_q)
    kv_proj = kv_unit * (n_plain + n_staged * n_kv)

    # Stage-invariant: the stages partition the query heads, so these sum back to the
    # parallel cost no matter how the partition is drawn.
    attention = int(2 * (2 * float(causal_factor) * a.num_q_heads * T * T * D) * L)
    o_proj = 2 * T * Dq * H * L
    mlp = 6 * T * H * config.intermediate_size * L
    lm_head = 2 * k * H * a.vocab_size

    return FlopsBreakdown(
        q_proj=int(q_proj),
        kv_proj=int(kv_proj),
        attention=int(attention),
        o_proj=int(o_proj),
        mlp=int(mlp),
        lm_head=int(lm_head),
        intervention=int(intervention),
        n_q_passes=int(n_q),
        n_kv_passes=int(n_kv),
        arm_layers=int(n_staged),
        seq_len=T,
    )


def intervention_flops(
    config: ComputeConfig,
    seq_len: int,
    *,
    modes: Iterable[str] = (),
    n_layers: int | None = None,
    low_rank: int = 0,
    mean_centering: bool = False,
    qk_targets: int = 2,
    causal_factor: float = 1.0,
) -> int:
    """Arithmetic added by the interventions, so their cost is reported not assumed.

    ``modes`` are SPEC 5.3's kernel modes by name (``logit_bias``, ``value_scale``,
    ``head_scale``); ``low_rank`` is the rank of a ``cheap.low_rank_positional_projection``
    (0 = absent) and ``mean_centering`` flags ``cheap.position_mean_centering``. Every
    term here is linear in ``T`` except the Mode-1 bias, which is the one intervention
    whose cost really is ``O(Hq * T^2)``.
    """
    T = int(seq_len)
    a = config.arch
    layers = a.num_layers if n_layers is None else int(n_layers)
    if not 0 <= layers <= a.num_layers:
        raise ValueError(f"{layers} intervened layers but the model has {a.num_layers}")

    total = 0
    names = {str(m) for m in modes}
    unknown = names - {"logit_bias", "value_scale", "head_scale"}
    if unknown:
        raise ValueError(f"unknown intervention modes {sorted(unknown)}")
    if "logit_bias" in names:
        # One add per live attention cell.
        total += int(float(causal_factor) * a.num_q_heads * T * T) * layers
    if "value_scale" in names:
        total += a.num_kv_heads * T * a.head_dim * layers
    if "head_scale" in names:
        total += a.num_q_heads * T * a.head_dim * layers
    if int(low_rank) > 0:
        # x @ U then (xU) @ U^T, per hooked projection input.
        total += 2 * (2 * T * a.hidden_size * int(low_rank)) * layers * int(qk_targets)
    if mean_centering:
        total += T * a.hidden_size * layers * int(qk_targets)
    return int(total)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeMeasurement:
    """What actually happened, as opposed to what the model predicted."""

    wall_clock_s: float
    peak_vram_bytes: int | None
    host_rss_bytes: int | None
    host_rss_delta_bytes: int | None
    device: str
    psutil_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ComputeProbe:
    """Filled in by :func:`measure_compute` on block exit."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._result: ComputeMeasurement | None = None

    @property
    def result(self) -> ComputeMeasurement:
        if self._result is None:
            raise RuntimeError("the measure_compute block has not finished yet")
        return self._result


def _rss_bytes() -> int | None:
    if not psutil_available():
        return None
    import psutil

    return int(psutil.Process().memory_info().rss)


@contextlib.contextmanager
def measure_compute(device: Any = None) -> Iterator[ComputeProbe]:
    """Time a block, and record peak VRAM and host RSS around it.

    ``torch.cuda.synchronize`` is called before the clock stops — without it the timer
    measures how long it took to *queue* the work, which on this card is roughly an
    order of magnitude less than the work itself. Peak VRAM is
    ``torch.cuda.max_memory_allocated`` after a ``reset_peak_memory_stats``, so it is the
    peak of *this* block and not of the process.
    """
    import torch

    dev = torch.device(device) if device is not None else None
    on_cuda = torch.cuda.is_available() and (dev is None or dev.type == "cuda")
    probe = ComputeProbe(str(dev) if dev is not None else ("cuda" if on_cuda else "cpu"))

    rss_before = _rss_bytes()
    if on_cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    try:
        yield probe
    finally:
        if on_cuda:
            torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - started
        rss_after = _rss_bytes()
        probe._result = ComputeMeasurement(
            wall_clock_s=float(elapsed),
            peak_vram_bytes=int(torch.cuda.max_memory_allocated(dev)) if on_cuda else None,
            host_rss_bytes=rss_after,
            host_rss_delta_bytes=(
                None if rss_after is None or rss_before is None else rss_after - rss_before
            ),
            device=probe.device,
            psutil_available=psutil_available(),
        )


# ---------------------------------------------------------------------------
# the row
# ---------------------------------------------------------------------------

#: Column order of ``results/compute_metrics.csv``. Fixed so appended runs line up.
COMPUTE_METRICS_COLUMNS: tuple[str, ...] = (
    "label",
    "model_key",
    "dtype",
    "device",
    "arm",
    "n_stages",
    "alphas",
    "recompute_kv",
    "query_refinement",
    "condition",
    "lam",
    "arm_layers",
    "seq_len",
    "n_items",
    "n_tokens",
    "wall_clock_s",
    "tokens_per_s",
    "items_per_s",
    "flops_total",
    "flops_per_token",
    "flops_ratio_vs_parallel",
    "flops_q_proj",
    "flops_kv_proj",
    "flops_attention",
    "flops_o_proj",
    "flops_mlp",
    "flops_lm_head",
    "flops_intervention",
    "n_q_passes",
    "n_kv_passes",
    "peak_vram_bytes",
    "peak_vram_gib",
    "host_rss_bytes",
    "host_rss_delta_bytes",
    "psutil_available",
    "causal_factor",
    "logits_to_keep",
    "as_implemented",
    "notes",
)


def compute_metrics_row(
    label: str,
    config: ComputeConfig,
    *,
    seq_len: int,
    n_items: int = 1,
    n_tokens: int | None = None,
    measurement: ComputeMeasurement | ComputeProbe | None = None,
    arm: SequentialArm | None = None,
    arm_layers: Sequence[int] | int | None = None,
    logits_to_keep: int = 1,
    causal_factor: float = 1.0,
    as_implemented: bool = True,
    intervention: int = 0,
    model_key: str | None = None,
    dtype: str | None = None,
    condition: str | None = None,
    lam: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """One row of ``results/compute_metrics.csv``.

    ``n_tokens`` defaults to ``n_items * seq_len`` — the prompt tokens actually pushed
    through the model — so ``tokens_per_s`` is prefill throughput, which is the quantity
    that is comparable across arms. It is *not* generated tokens per second; a study that
    mixes the two produces a table nobody can read.
    """
    if isinstance(measurement, ComputeProbe):
        measurement = measurement.result

    flops = estimate_flops(
        config,
        seq_len,
        arm=arm,
        arm_layers=arm_layers,
        logits_to_keep=logits_to_keep,
        causal_factor=causal_factor,
        as_implemented=as_implemented,
        intervention=intervention,
    )
    parallel = estimate_flops(
        config,
        seq_len,
        arm=None,
        logits_to_keep=logits_to_keep,
        causal_factor=causal_factor,
    )
    items = int(n_items)
    tokens = int(n_tokens) if n_tokens is not None else items * int(seq_len)
    total = flops.total * items

    wall = None if measurement is None else float(measurement.wall_clock_s)
    peak_vram = None if measurement is None else measurement.peak_vram_bytes

    return {
        "label": str(label),
        "model_key": model_key,
        "dtype": dtype,
        "device": None if measurement is None else measurement.device,
        "arm": None if arm is None else arm.name,
        "n_stages": 1 if arm is None else arm.n_stages,
        "alphas": None if arm is None else ",".join(f"{a:g}" for a in arm.alphas),
        "recompute_kv": None if arm is None else bool(arm.recompute_kv),
        "query_refinement": None if arm is None else bool(arm.query_refinement),
        "condition": condition,
        "lam": None if lam is None else float(lam),
        "arm_layers": flops.arm_layers,
        "seq_len": int(seq_len),
        "n_items": items,
        "n_tokens": tokens,
        "wall_clock_s": wall,
        "tokens_per_s": (None if not wall else tokens / wall),
        "items_per_s": (None if not wall else items / wall),
        "flops_total": int(total),
        "flops_per_token": (total / tokens) if tokens else None,
        "flops_ratio_vs_parallel": flops.total / parallel.total,
        "flops_q_proj": flops.q_proj,
        "flops_kv_proj": flops.kv_proj,
        "flops_attention": flops.attention,
        "flops_o_proj": flops.o_proj,
        "flops_mlp": flops.mlp,
        "flops_lm_head": flops.lm_head,
        "flops_intervention": flops.intervention,
        "n_q_passes": flops.n_q_passes,
        "n_kv_passes": flops.n_kv_passes,
        "peak_vram_bytes": peak_vram,
        "peak_vram_gib": (None if peak_vram is None else peak_vram / 2**30),
        "host_rss_bytes": None if measurement is None else measurement.host_rss_bytes,
        "host_rss_delta_bytes": (
            None if measurement is None else measurement.host_rss_delta_bytes
        ),
        "psutil_available": psutil_available(),
        "causal_factor": float(causal_factor),
        "logits_to_keep": int(logits_to_keep),
        "as_implemented": bool(as_implemented),
        "notes": str(notes),
    }


def write_compute_metrics(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Write the section-36 table plus its provenance sidecar.

    Column order is pinned to :data:`COMPUTE_METRICS_COLUMNS` and unknown keys raise
    rather than being dropped, so a typo in a caller's row shows up here instead of as a
    silently missing column three phases later. Falls back to the stdlib ``csv`` writer
    when pandas is absent.
    """
    target = Path(path) if path is not None else COMPUTE_METRICS_PATH
    ordered: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        unknown = set(row) - set(COMPUTE_METRICS_COLUMNS)
        if unknown:
            raise ValueError(f"row {i} has unknown columns {sorted(unknown)}")
        ordered.append({col: row.get(col) for col in COMPUTE_METRICS_COLUMNS})

    from rineya_psa.statistics.provenance import write_result

    payload: Any = ordered
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        payload = pd.DataFrame(ordered, columns=list(COMPUTE_METRICS_COLUMNS))
    else:  # pragma: no cover - pandas is present in the lab environment
        payload = _CsvRows(ordered)
    return write_result(payload, target, config=config)


class _CsvRows(list):
    """Minimal stand-in with the ``to_csv`` surface ``write_result`` looks for.

    Only reached when pandas is missing; it keeps the artifact writable rather than
    turning an optional dependency into a hard one.
    """

    def to_parquet(self, target: Any, index: bool = False) -> None:  # noqa: D102
        raise RuntimeError("parquet output needs pandas + pyarrow; write .csv instead")

    def to_csv(self, target: Any, index: bool = False) -> None:  # noqa: D102
        import csv

        with Path(target).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(COMPUTE_METRICS_COLUMNS))
            writer.writeheader()
            writer.writerows(self)
