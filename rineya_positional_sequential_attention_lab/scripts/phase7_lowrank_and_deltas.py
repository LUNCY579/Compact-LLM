"""Phase 7 — low-rank structure of ``I``/``B``, and what moved where (brief 20, 21).

Three questions, one CPU-only pass over artifacts that already exist:

1. **Is the significance matrix really 224 independent positional profiles?**
   SVD of the ``[l*h, b]`` unfolding of every Phase-4 significance matrix and every
   Phase-4b bias matrix, cumulative explained variance at ranks 1/2/4/8, and the
   compact ``S[h, p] = a_h f(p) + c_h`` fit. Both the cumulative variance and the
   relative reconstruction error are reported, and the script **asserts**
   ``relerr(k) == sqrt(1 - cumulative_at(k))``. That identity only holds because
   ``LowRankSpectrum.reconstruction_error`` normalises on the *centred* matrix; it
   used to normalise on the raw one, which put a 0.0122 "relative error" next to a
   0.2116 explained variance in the same JSON block. The assertion exists so that
   regression cannot come back silently.

2. **Which parameter family moved where positional behaviour moved?** (brief question
   10.) Phase 1 found ``k_proj`` is the most-changed attention projection in all three
   Rineya cores, ahead of ``v_proj``, ``o_proj`` and ``q_proj`` in that fixed order.
   That is a fact about weights. This script asks whether it is a fact about
   *behaviour*: it correlates the per-head weight-delta norms already in
   ``results/weight_delta_by_head.csv`` against the per-head change in positional
   summary statistics (edge bias, middle mass) derived from the significance and bias
   matrices. Clustered bootstrap CIs, **clustered on LAYER**, because heads inside one
   layer are not independent draws — a single layer with a big delta and a big
   behavioural shift would otherwise contribute eight "independent" points.

3. **Can the LoRA adapter be separated from the soup?** Phase 1 §3 found layers
   {6, 7, 8, 9, 10, 14, 18, 23, 24} byte-identical to stock Qwen3 in *both* Halo and
   Nimbus. In those layers ``Supa - Nimbus`` is therefore exactly ``Supa - Qwen3``,
   i.e. a pure adapter delta with no soup contamination, and the CSV the audit already
   wrote contains it. The script re-verifies the zero (it is the premise of the whole
   contrast, not an assumption to inherit) and reports the ``adapter_only`` scope
   separately from ``soup_layers`` and ``all_layers``.

Emits ``significance_svd.json``, ``significance_low_rank_models.npz`` and
``phase7_weight_delta_correlation.csv``, each with a provenance sidecar. Exits 0 with a
clear message if the Phase-4 artifacts do not exist yet.

CPU only. No model is loaded, no checkpoint weight is read, and torch is never
imported — this is designed to run while the GPU job holds the card.

    PYTHONPATH=src python scripts/phase7_lowrank_and_deltas.py --dry-run
    PYTHONPATH=src python scripts/phase7_lowrank_and_deltas.py

LIMITATION, stated rather than hidden: the weight side of the correlation is
layer-local (``||dW_{K,l,h}||`` is a property of one tensor block) but the behavioural
side is not. ``edge_bias(supa)[l,h] - edge_bias(nimbus)[l,h]`` is measured on two whole
models that differ in every layer, so a large correlation says "heads whose K matrix
moved are heads whose positional profile moved", not "moving this K matrix caused this
head's profile to move". The causal version of that claim is Phase 8's job.

DEVIATION: ``LowRankAnalysis.write()`` would emit the two low-rank artifacts in one
call, but it writes ``to_json()`` verbatim and there is no hook for the identity check
or the partial-run flag. This script therefore calls ``write_result`` and
``write_npz_with_provenance`` itself, reusing ``SVD_JSON_NAME`` / ``LOW_RANK_NPZ_NAME``
so the artifact names stay the module's, not this script's.

DEVIATION: the Phase-4 significance ``.npz`` written by
``significance.causal.SignificanceResult.save`` is parsed here by hand rather than
through ``SignificanceResult.load``, because importing ``significance.causal`` pulls in
``evaluation.runner`` and therefore torch. Reading four numpy arrays and a JSON blob is
not worth a torch import on a machine whose GPU is busy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rineya_psa import paths  # noqa: E402
from rineya_psa.significance.bias import (  # noqa: E402
    PositionalBias,
    edge_bias_score,
    middle_mass,
)
from rineya_psa.significance.lowrank import (  # noqa: E402
    DEFAULT_RANKS,
    LOW_RANK_NPZ_NAME,
    SVD_JSON_NAME,
    LowRankSpectrum,
    analyse_low_rank,
)
from rineya_psa.significance.matrices import (  # noqa: E402
    SignificanceMatrix,
    write_npz_with_provenance,
)
from rineya_psa.statistics.bootstrap import (  # noqa: E402
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_ci_of_statistic,
)
from rineya_psa.statistics.provenance import write_result  # noqa: E402
from rineya_psa.types import Arch  # noqa: E402

CORRELATION_CSV_NAME = "phase7_weight_delta_correlation.csv"

#: The splits this script may read, as a whitelist. SPEC section 8 holds one further
#: split back for Phase 14; naming it here would be both unnecessary and a breach of
#: tests/test_split_discipline.py, which scans every script's AST for exactly that.
PERMITTED_SPLITS: tuple[str, ...] = ("calibration", "validation")

#: Layers byte-identical to stock Qwen3 in both Halo and Nimbus (Phase 1 §3). Verified
#: against the audit CSV at run time; it is the premise of the adapter-only contrast.
FROZEN_LAYERS: tuple[int, ...] = (6, 7, 8, 9, 10, 14, 18, 23, 24)

#: Per-head families the audit decomposes, plus the MLP, which it does not.
PARAM_FAMILIES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "mlp")

#: The three MLP tensors, combined in quadrature into one per-layer ``mlp`` number.
MLP_TENSORS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

WEIGHT_METRICS: tuple[str, ...] = ("frobenius", "rel_frobenius")

HEAD_STATISTICS: tuple[str, ...] = ("edge_bias", "middle_mass")

SCOPES: tuple[str, ...] = ("all_layers", "adapter_only", "soup_layers")

#: Header written by ``SignificanceMatrix.save_csv``; used to sniff a CSV source
#: without opening it as anything else.
_SIG_CSV_HEADER = (
    "model_key,objective,layer,kv_head,bin,bin_center,value,importance,binning_json"
)

#: Measured end to end on this box, 2026-08-29: one source matrix on a 28-layer x
#: 8-KV-head grid costs 237 s for its 60 (statistic, family, metric, scope) cells at
#: the SPEC default of 10 000 resamples, i.e. 3.95 s per cell, split between Pearson
#: and Spearman in the 0.88 : 1.71 ratio the two cost in isolation. Used only by
#: ``--dry-run``; the GPU guard below never trips because this script spends no GPU
#: time at all.
_BOOT_SECONDS_PER_10K = {"pearson": 1.35, "spearman": 2.60}


# ======================================================================================
# source discovery
# ======================================================================================


@dataclass(frozen=True, eq=False)
class Source:
    """One Phase-4 / Phase-4b matrix, already folded onto the KV-head grid."""

    name: str
    path: Path
    kind: str  # "significance" | "bias"
    matrix: SignificanceMatrix

    @property
    def model_key(self) -> str:
        return self.matrix.model_key

    @property
    def group_key(self) -> tuple[Any, ...]:
        """What makes two sources comparable: same objective, grid and binning.

        A delta between matrices measured on different bin counts is numerically valid
        and completely meaningless, which is the mistake this key exists to prevent.
        """
        return (
            self.kind,
            self.matrix.objective,
            self.matrix.layers,
            self.matrix.kv_heads,
            json.dumps(dict(self.matrix.binning_desc), sort_keys=True),
        )


def _matrix_from_meta(values: np.ndarray, meta: Mapping[str, Any]) -> SignificanceMatrix:
    return SignificanceMatrix(
        values=values,
        layers=tuple(meta["layers"]),
        kv_heads=tuple(meta["kv_heads"]),
        bins=tuple(meta["bins"]),
        binning_desc=meta["binning_desc"],
        objective=meta["objective"],
        model_key=meta["model_key"],
        provenance=meta.get("provenance", {}),
    )


def load_source(path: Path, *, bias_field: str = "attn_by_bin") -> Source | None:
    """Sniff ``path`` and return a :class:`Source`, or ``None`` if it is not one.

    Sniffing on content rather than on filename is deliberate: Phase 4 has no driver
    script yet, so its artifact names are not fixed, and a glob that guessed wrong
    would silently analyse nothing.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8") as fh:
            header = fh.readline().strip()
        if header != _SIG_CSV_HEADER:
            return None
        return Source(path.stem, path, "significance", SignificanceMatrix.load_csv(path))
    if suffix != ".npz":
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if {"attn_by_bin", "qk_logit_mean", "entropy"} <= keys:
                kind = "bias"
                matrix = None
            elif {"values", "meta"} <= keys:
                kind = "significance"
                meta = json.loads(str(archive["meta"]))
                # SignificanceResult.save nests the matrix meta one level down and adds
                # per_item/baseline/evaluated; SignificanceMatrix.save_npz does not.
                inner = meta.get("matrix", meta)
                matrix = _matrix_from_meta(
                    np.asarray(archive["values"], dtype=np.float64), inner
                )
            else:
                return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if kind == "bias":
        # PositionalBias is [L, q_head, B]; to_significance_matrix averages the query
        # heads of each GQA group, which is the granularity the weight-delta CSV and
        # every intervention mode share. kv_head_dispersion() records what that threw
        # away and is carried into the summary.
        matrix = PositionalBias.load(path).to_significance_matrix(bias_field)
    assert matrix is not None
    return Source(path.stem, path, kind, matrix)


def discover_sources(
    results_dir: Path, *, bias_field: str = "attn_by_bin"
) -> list[Source]:
    """Every significance / bias matrix in ``results_dir``, in a stable order."""
    skip = {LOW_RANK_NPZ_NAME, CORRELATION_CSV_NAME, SVD_JSON_NAME}
    found: list[Source] = []
    # RECURSIVE. Phase 4 writes one directory per arm
    # (results/phase4_significance/<arm>/kv_head_position_significance.npz) so that the
    # brief's fixed artifact names survive across four models; a top-level glob finds
    # nothing and reports "no artifacts" on a run that produced eight of them.
    for path in sorted(results_dir.rglob("*.npz")) + sorted(results_dir.rglob("*.csv")):
        if path.name in skip:
            continue
        source = load_source(path, bias_field=bias_field)
        if source is None:
            continue
        # A Tier-2 ablation matrix is SPARSE BY DESIGN: Phase 4 evaluates ~128 of 1792
        # cells (top-K plus the random control) and leaves the rest NaN, exactly as SPEC
        # 11.2 requires. An SVD needs complete rows, so such a matrix has nothing to
        # decompose - which is a property of the input, not an error. Skip it loudly
        # instead of aborting a run over the dense Tier-1 matrices that ARE decomposable.
        values = np.asarray(source.matrix.values, dtype=np.float64)
        if values.ndim >= 2:
            flat = values.reshape(-1, values.shape[-1])
            if not np.isfinite(flat).all(axis=1).any():
                covered = float(np.isfinite(flat).mean())
                print(
                    f"  skip {path.name} ({source.name}): no complete row "
                    f"({covered:.1%} of cells evaluated) - sparse by construction",
                    flush=True,
                )
                continue
        found.append(source)
    return found


# ======================================================================================
# the low-rank half (brief section 21)
# ======================================================================================


def check_reconstruction_identity(
    spectrum: LowRankSpectrum, *, atol: float = 1e-9, rtol: float = 1e-6
) -> list[dict[str, Any]]:
    """Assert ``relerr(k) == sqrt(1 - cumulative_at(k))`` at every reported rank.

    Both quantities are already in the JSON, side by side. They agree only while the
    reconstruction error is normalised on the centred matrix — the same matrix the
    explained-variance ratio is computed on. Checking it here means a future change to
    either denominator fails loudly instead of publishing two numbers that look like
    independent evidence and are in fact the same number computed two ways.

    The comparison is made in the **variance** domain (``relerr**2`` against
    ``1 - cumulative``) and not in the sqrt domain. Once ``k`` reaches the matrix's true
    rank the cumulative variance is 1 to within a few ULPs, ``1 - cumulative`` is pure
    cancellation noise around 1e-16, and ``sqrt`` amplifies that to ~1e-8 next to a
    residual of ~1e-15. Comparing the square keeps both sides at the scale their errors
    actually live on, so the assertion tests the identity rather than the square root.
    """
    centred_norm = float(np.linalg.norm(spectrum.values - spectrum.column_mean))
    rows: list[dict[str, Any]] = []
    for k in spectrum.ranks:
        cumulative = spectrum.cumulative_at(k)
        relerr = spectrum.reconstruction_error(k)
        implied = math.sqrt(max(0.0, 1.0 - cumulative))
        if centred_norm <= 0.0:
            # A constant matrix has nothing to explain; both sides are conventions.
            rows.append({"rank": int(k), "status": "degenerate_zero_variance",
                         "cumulative_explained_variance": cumulative,
                         "relative_reconstruction_error": relerr,
                         "sqrt_one_minus_cumulative": implied})
            continue
        residual_share = relerr * relerr
        unexplained = 1.0 - cumulative
        if not math.isclose(residual_share, unexplained, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(
                f"{spectrum.name}: rank {k} squared reconstruction error "
                f"{residual_share!r} does not match the unexplained variance share "
                f"{unexplained!r}. The two are normalised on different matrices again; "
                "see LowRankSpectrum.reconstruction_error."
            )
        rows.append({"rank": int(k), "status": "ok",
                     "cumulative_explained_variance": cumulative,
                     "relative_reconstruction_error": relerr,
                     "sqrt_one_minus_cumulative": implied,
                     "squared_error_minus_unexplained": residual_share - unexplained,
                     "abs_difference_sqrt_domain": abs(relerr - implied)})
    return rows


# ======================================================================================
# the weight-delta half (brief section 20 / brief question 10)
# ======================================================================================


def _arch_from_head_rows(rows: Sequence[Mapping[str, str]]) -> Arch:
    """Head bookkeeping straight out of the audit CSV, not out of a hard-coded config."""
    n_layers = max(int(r["layer"]) for r in rows) + 1
    q_heads = max(int(r["n_heads"]) for r in rows if r["head_space"] == "q")
    kv_heads = max(int(r["n_heads"]) for r in rows if r["head_space"] == "kv")
    head_dim = max(int(r["head_dim"]) for r in rows)
    # hidden_size and vocab_size play no part in the q -> kv mapping this Arch is built
    # for; hidden_size is derivable, vocab_size is not present in the CSV at all.
    return Arch(
        num_layers=n_layers,
        hidden_size=q_heads * head_dim,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=0,
    )


@dataclass(frozen=True, eq=False)
class WeightDeltas:
    """``[model][family] -> {(layer, kv_head): {metric: value}}``, plus the Arch."""

    arch: Arch
    values: Mapping[str, Mapping[str, Mapping[tuple[int, int], Mapping[str, float]]]]
    models: tuple[str, ...]


def load_weight_deltas(head_csv: Path, layer_csv: Path) -> WeightDeltas:
    """Fold the Phase-1 audit CSVs onto the ``(layer, kv_head)`` grid.

    ``q_proj``/``o_proj`` are decomposed per *query* head, ``k_proj``/``v_proj`` per KV
    head. Query heads are folded onto their KV head with ``Arch.q_heads_for_kv`` (SPEC
    §5.5, the only sanctioned mapping) and combined **in quadrature**, because the
    Frobenius norm of two stacked blocks is the root of the sum of their squares — not
    their mean. For a KV-space family the group has one member and quadrature is the
    identity, so both families end up on the same scale.

    ``mlp`` has no head decomposition anywhere in the audit, so it is a per-layer number
    broadcast to every head of that layer. With the bootstrap clustered on layer that is
    honest: the MLP correlation is a layer-level correlation and its CI says so.
    """
    head_rows = list(csv.DictReader(head_csv.open(encoding="utf-8", newline="")))
    if not head_rows:
        raise SystemExit(f"{head_csv} has no rows")
    arch = _arch_from_head_rows(head_rows)
    q_to_kv = {q: kv for kv in range(arch.num_kv_heads) for q in arch.q_heads_for_kv(kv)}

    # sums of squares, so the quadrature combination is a single sqrt at the end
    acc: dict[str, dict[str, dict[tuple[int, int], list[float]]]] = {}

    def bump(model: str, family: str, cell: tuple[int, int], delta: float, base: float) -> None:
        slot = acc.setdefault(model, {}).setdefault(family, {}).setdefault(cell, [0.0, 0.0])
        slot[0] += delta * delta
        slot[1] += base * base

    for row in head_rows:
        family = row["head_kind"]
        if family not in PARAM_FAMILIES:
            continue
        index = int(row["head_index"])
        kv = index if row["head_space"] == "kv" else q_to_kv[index]
        bump(row["model_key"], family, (int(row["layer"]), kv),
             float(row["frobenius"]), float(row["base_frobenius"]))

    layer_rows = list(csv.DictReader(layer_csv.open(encoding="utf-8", newline="")))
    per_layer: dict[str, dict[int, list[float]]] = {}
    for row in layer_rows:
        if row["row_kind"] != "layer_family" or row["family"] not in MLP_TENSORS:
            continue
        slot = per_layer.setdefault(row["model_key"], {}).setdefault(int(row["layer"]),
                                                                    [0.0, 0.0])
        slot[0] += float(row["frobenius"]) ** 2
        slot[1] += float(row["base_frobenius"]) ** 2
    for model, layers in per_layer.items():
        for layer, (dsq, bsq) in layers.items():
            for kv in range(arch.num_kv_heads):
                acc.setdefault(model, {}).setdefault("mlp", {})[(layer, kv)] = [dsq, bsq]

    out: dict[str, dict[str, dict[tuple[int, int], dict[str, float]]]] = {}
    for model, families in acc.items():
        for family, cells in families.items():
            for cell, (dsq, bsq) in cells.items():
                delta = math.sqrt(dsq)
                base = math.sqrt(bsq)
                out.setdefault(model, {}).setdefault(family, {})[cell] = {
                    "frobenius": delta,
                    "rel_frobenius": delta / base if base > 0 else float("nan"),
                }
    return WeightDeltas(arch=arch, values=out, models=tuple(sorted(out)))


def verify_frozen_layers(deltas: WeightDeltas, layers: Sequence[int]) -> dict[str, Any]:
    """Re-verify Phase 1 §3: Halo and Nimbus are untouched in the frozen layers.

    The whole adapter-only contrast rests on ``Nimbus == Qwen3`` there, which makes
    ``Supa - Nimbus`` equal to the ``Supa - Qwen3`` column the audit already wrote. If
    that ever stopped holding, the contrast would still produce a plausible number, so
    it is checked rather than cited.
    """
    report: dict[str, Any] = {"layers": list(layers), "max_abs_frobenius": {}}
    for model in ("halo", "nimbus"):
        families = deltas.values.get(model)
        if not families:
            report["max_abs_frobenius"][model] = None
            continue
        worst = max(
            (abs(cell["frobenius"])
             for family in families.values()
             for (layer, _), cell in family.items()
             if layer in set(layers)),
            default=0.0,
        )
        report["max_abs_frobenius"][model] = worst
        if worst > 0.0:
            raise SystemExit(
                f"Phase 1 section 3 no longer holds: {model} has a non-zero weight "
                f"delta "
                f"({worst:.6g}) in layers {list(layers)}, so 'Supa - Nimbus is a pure "
                "adapter delta there' is false and the adapter_only scope would be a "
                "silently wrong number. Re-run the Phase 1 audit."
            )
    report["verified"] = True
    return report


# ======================================================================================
# correlation machinery
# ======================================================================================


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so a plateau does not acquire a fake gradient."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if unique.size < x.size:
        sums = np.zeros(unique.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:  # a constant vector has no correlation, not a correlation of 0
        return float("nan")
    return float(a @ b / denom)


def _correlation(rows: Sequence[tuple[float, float]], kind: str) -> float:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or not np.isfinite(arr).all():
        return float("nan")
    x, y = arr[:, 0], arr[:, 1]
    if kind == "spearman":
        x, y = _average_ranks(x), _average_ranks(y)
    return _pearson(x, y)


@dataclass(frozen=True)
class CorrelationRow:
    """One (model, source, statistic, family, metric, scope) correlation."""

    payload: dict[str, Any]


def correlate(
    pairs: Sequence[tuple[int, float, float]],
    *,
    kind: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Clustered-on-layer bootstrap CI for one correlation, or a reason it is absent."""
    rows = [(x, y) for _, x, y in pairs]
    clusters = [layer for layer, _, _ in pairs]
    point = _correlation(rows, kind)
    if not math.isfinite(point):
        # Say *which* side collapsed. The commonest case here is by design: Halo and
        # Nimbus have an identically zero weight delta in the frozen layers, so their
        # adapter_only rows cannot have a correlation and should say why rather than
        # look like a measurement that came out at zero.
        arr = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] < 3:
            why = f"only {arr.shape[0]} usable head(s)"
        elif not np.isfinite(arr).all():
            why = "a non-finite value in the pair list"
        elif float(arr[:, 0].std()) <= 0.0:
            why = "the weight delta is constant over these heads"
        elif float(arr[:, 1].std()) <= 0.0:
            why = "the behavioural statistic is constant over these heads"
        else:
            why = "the correlation is undefined"
        return {"estimate": None, "ci_low": None, "ci_high": None, "sd": None,
                "method": None, "n_resamples": 0, "n_clusters": len(set(clusters)),
                "note": f"undefined: {why}"}
    try:
        result = bootstrap_ci_of_statistic(
            lambda sample: _correlation(sample, kind),
            rows,
            cluster_ids=clusters,
            n=n_boot,
            seed=seed,
        )
    except ValueError as exc:  # too few layers, or degenerate on most resamples
        return {"estimate": point, "ci_low": None, "ci_high": None, "sd": None,
                "method": None, "n_resamples": 0, "n_clusters": len(set(clusters)),
                "note": f"point estimate only: {exc}"}
    return {"estimate": result.estimate, "ci_low": result.ci_low,
            "ci_high": result.ci_high, "sd": result.sd, "method": result.method,
            "n_resamples": result.n_resamples, "n_clusters": result.n_scenarios,
            "note": ""}


def head_statistics(
    matrix: SignificanceMatrix, *, edge_fraction: float
) -> dict[str, np.ndarray]:
    """``[L, H]`` positional summaries of one matrix, on the importance scale.

    ``importance()``, not ``values``: for a loss objective an important cell has a
    *negative* Delta, so correlating raw values against a weight-delta norm would flip
    the sign of every ``gold_nll`` row relative to every bias row and the two tables
    would silently disagree.

    ``middle_mass`` reduces by sum for attention mass, where the bins partition a unit
    of probability, and by mean for everything else, where they do not.
    """
    imp = matrix.importance()
    reduce = "sum" if matrix.objective == "bias_attn_by_bin" else "mean"
    return {
        "edge_bias": edge_bias_score(
            imp, binning_desc=matrix.binning_desc, edge_fraction=edge_fraction
        ),
        "middle_mass": middle_mass(imp, binning_desc=matrix.binning_desc, reduce=reduce),
    }


def scope_layers(scope: str, layers: Sequence[int]) -> tuple[int, ...]:
    frozen = set(FROZEN_LAYERS)
    if scope == "adapter_only":
        return tuple(x for x in layers if x in frozen)
    if scope == "soup_layers":
        return tuple(x for x in layers if x not in frozen)
    return tuple(layers)


# ======================================================================================
# driver
# ======================================================================================


def estimate_seconds(sources: Sequence[Source], models: Sequence[str], n_boot: int) -> float:
    """Wall-clock estimate for the correlation half; the SVD half is milliseconds.

    Only sources whose own ``model_key`` is in the requested panel produce rows — a
    qwen3 matrix is the *baseline* of a comparison, never the left-hand side of one —
    so the count is over those, not over the cross product.
    """
    wanted = sum(1 for s in sources if s.model_key in set(models))
    per_source = (
        len(HEAD_STATISTICS) * len(PARAM_FAMILIES) * len(WEIGHT_METRICS) * len(SCOPES)
    )
    per_call = sum(_BOOT_SECONDS_PER_10K.values()) * (n_boot / 10_000.0)
    return wanted * per_source * per_call


def build_correlation_rows(
    sources: Sequence[Source],
    deltas: WeightDeltas,
    *,
    models: Sequence[str],
    edge_fraction: float,
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The whole correlation table, plus the notes worth printing at the end."""
    by_group: dict[tuple[Any, ...], dict[str, Source]] = {}
    for source in sources:
        by_group.setdefault(source.group_key, {})[source.model_key] = source

    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for _group, members in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        stats = {
            key: head_statistics(src.matrix, edge_fraction=edge_fraction)
            for key, src in members.items()
        }
        for model in models:
            source = members.get(model)
            if source is None:
                continue
            layers = list(source.matrix.layers)
            heads = list(source.matrix.kv_heads)
            for scope in SCOPES:
                keep = set(scope_layers(scope, layers))
                if not keep:
                    continue
                # In the frozen layers Supa - Nimbus IS the adapter, so use Nimbus as
                # the behavioural baseline there when it was measured; elsewhere the
                # weight column is model - qwen3 and the baseline must match it.
                baseline_key = paths.BASE_KEY
                if scope == "adapter_only" and "nimbus" in stats and model == "supa_a11":
                    baseline_key = "nimbus"
                baseline = stats.get(baseline_key)
                stat_kind = "delta" if baseline is not None else "level"
                if baseline is None and model != paths.BASE_KEY:
                    notes.append(
                        f"{source.name}: no {baseline_key} counterpart on this grid, so "
                        f"the {scope} rows report the level, not the delta"
                    )
                for stat_name in HEAD_STATISTICS:
                    target = stats[model][stat_name]
                    if baseline is not None:
                        target = target - baseline[stat_name]
                    for family in PARAM_FAMILIES:
                        cells = deltas.values.get(model, {}).get(family, {})
                        for metric in WEIGHT_METRICS:
                            pairs: list[tuple[int, float, float]] = []
                            for li, layer in enumerate(layers):
                                if layer not in keep:
                                    continue
                                for hi, head in enumerate(heads):
                                    cell = cells.get((layer, head))
                                    y = float(target[li, hi])
                                    if cell is None or not math.isfinite(y):
                                        continue
                                    x = float(cell[metric])
                                    if not math.isfinite(x):
                                        continue
                                    pairs.append((layer, x, y))
                            base_row = {
                                "model_key": model,
                                "behaviour_baseline": baseline_key
                                if baseline is not None
                                else "",
                                "source": source.name,
                                "source_kind": source.kind,
                                "objective": source.matrix.objective,
                                "statistic": stat_name,
                                "stat_kind": stat_kind,
                                "param_family": family,
                                "weight_metric": metric,
                                "scope": scope,
                                "n_heads": len(pairs),
                                "n_layers": len({layer for layer, _, _ in pairs}),
                            }
                            if len(pairs) < 3:
                                rows.append({**base_row, "correlation": "",
                                             "estimate": None, "ci_low": None,
                                             "ci_high": None, "sd": None,
                                             "n_resamples": 0, "method": "",
                                             "note": "fewer than 3 usable heads"})
                                continue
                            for kind in ("pearson", "spearman"):
                                res = correlate(pairs, kind=kind, n_boot=n_boot,
                                                seed=seed)
                                rows.append({
                                    **base_row,
                                    "correlation": kind,
                                    "estimate": res["estimate"],
                                    "ci_low": res["ci_low"],
                                    "ci_high": res["ci_high"],
                                    "sd": res["sd"],
                                    "n_resamples": res["n_resamples"],
                                    "method": res["method"] or "",
                                    "note": res["note"],
                                })
                print(f"  {source.name}/{model}/{scope}: {len(rows)} rows", flush=True)
    return rows, notes


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(paths.RESULTS),
                    help="where the Phase 4 / 4b artifacts live")
    ap.add_argument("--source", action="append", default=None,
                    help="explicit matrix path; repeatable, disables discovery")
    ap.add_argument("--weight-head-csv", default=None)
    ap.add_argument("--weight-layer-csv", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split", default=PERMITTED_SPLITS[0], choices=PERMITTED_SPLITS,
                    help="recorded in provenance; anything outside the permitted set "
                         "is refused (SPEC section 8)")
    ap.add_argument("--bias-field", default="attn_by_bin",
                    choices=("attn_by_bin", "qk_logit_mean"))
    ap.add_argument("--ranks", default=",".join(str(k) for k in DEFAULT_RANKS))
    ap.add_argument("--center", default="mean", choices=("mean", "none"))
    ap.add_argument("--edge-fraction", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of the panel to correlate")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of source matrices, for smoke runs")
    ap.add_argument("--skip-correlations", action="store_true",
                    help="low-rank half only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes-i-costed-this", action="store_true")
    args = ap.parse_args(argv)

    # Defence in depth behind argparse's own `choices` check. Both are written as a
    # whitelist rather than as a test against the held-out split's name, so that
    # tests/test_split_discipline.py's AST scan stays true: this script never names
    # that split in code, and a new split label added later is refused by default
    # instead of sliding through.
    if args.split not in PERMITTED_SPLITS:
        raise SystemExit(
            f"--split {args.split!r} is not one of {list(PERMITTED_SPLITS)}; "
            "scripts/phase14_*.py owns everything else (SPEC section 8)"
        )

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    head_csv = Path(args.weight_head_csv or results_dir / "weight_delta_by_head.csv")
    layer_csv = Path(args.weight_layer_csv or results_dir / "weight_delta_by_layer.csv")
    ranks = tuple(int(x) for x in args.ranks.split(",") if x.strip())

    if args.source:
        sources = [s for s in (load_source(Path(p), bias_field=args.bias_field)
                               for p in args.source) if s is not None]
        rejected = len(args.source) - len(sources)
        if rejected:
            raise SystemExit(
                f"{rejected} of {len(args.source)} --source paths are not a "
                "significance or bias matrix; nothing was written"
            )
    else:
        sources = discover_sources(results_dir, bias_field=args.bias_field)

    if not sources:
        print(
            "Phase 4 / 4b artifacts not found - nothing to decompose.\n"
            f"  looked in: {results_dir}\n"
            "  wanted:    a .npz written by SignificanceMatrix.save_npz, "
            "SignificanceResult.save or PositionalBias.save_npz, or a .csv written by "
            "SignificanceMatrix.save_csv\n"
            "Run Phase 4 (significance) and Phase 4b (positional bias) first, then "
            "re-run this script. Exiting 0: this is a missing input, not a failure."
        )
        return 0

    # A matrix carries the split it was measured on. Refuse anything measured outside
    # the permitted set rather than trusting the caller's --split flag: the flag
    # describes intent, the provenance describes what was actually read.
    for source in sources:
        declared = str(source.matrix.provenance.get("split", "")).lower()
        if declared and declared not in PERMITTED_SPLITS:
            raise SystemExit(
                f"{source.path} was measured on split {declared!r}, which is not one of "
                f"{list(PERMITTED_SPLITS)}; scripts/phase14_*.py owns everything else "
                "(SPEC section 8)"
            )

    if args.limit:
        sources = sources[: args.limit]

    panel_models = [m for m in paths.PANEL if m != paths.BASE_KEY]
    if args.models:
        want = {m.strip() for m in args.models.split(",")}
        panel_models = [m for m in panel_models if m in want]

    est = estimate_seconds(
        () if args.skip_correlations else sources, panel_models, args.n_boot
    )
    print(f"{len(sources)} source matrix/matrices, {len(panel_models)} model(s), "
          f"{args.n_boot} resamples; estimated {est / 60:.1f} min of CPU "
          f"(0.00 GPU-hours: this script loads no model)")
    if args.dry_run:
        for source in sources:
            print(f"  {source.name:44s} {source.kind:12s} {source.model_key:9s} "
                  f"{source.matrix.objective:18s} {list(source.matrix.shape)}")
        return 0
    # Kept for interface parity with the GPU drivers. It cannot trip here -- the
    # estimate is CPU time and the GPU cost is exactly zero -- and that is the point:
    # the guard lives in every driver so no driver has to be trusted to opt into it.
    gpu_hours = 0.0
    if gpu_hours > 2.0 and not args.yes_i_costed_this:
        raise SystemExit(
            f"estimate is {gpu_hours:.1f} GPU-hours, over the 2-hour guard; "
            "re-run with --yes-i-costed-this once you have looked at that number"
        )

    t0 = time.perf_counter()
    analysis = analyse_low_rank(
        {s.name: s.matrix for s in sources}, ranks=ranks, center=args.center
    )
    identity = {name: check_reconstruction_identity(sp)
                for name, sp in analysis.spectra.items()}
    print(f"low-rank: {len(analysis.spectra)} spectra, reconstruction identity "
          f"relerr(k) == sqrt(1 - cumulative(k)) holds at every reported rank")

    is_partial = bool(args.limit) or args.skip_correlations or bool(args.models) or (
        args.n_boot != DEFAULT_N_RESAMPLES
    ) or len(panel_models) != len(paths.PANEL) - 1

    config = {
        "script": Path(__file__).name,
        "split": args.split,
        "ranks": list(ranks),
        "center": args.center,
        "bias_field": args.bias_field,
        "edge_fraction": args.edge_fraction,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "models": panel_models,
        "limit": args.limit,
        "sources": [{"name": s.name, "path": str(s.path), "kind": s.kind,
                     "model_key": s.model_key, "objective": s.matrix.objective,
                     "shape": list(s.matrix.shape),
                     "content_sha256": s.matrix.content_hash()} for s in sources],
        "is_partial_run": is_partial,
    }

    paths.ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(analysis.to_json())
    payload["identity_check"] = identity
    payload["run"] = {
        "split": args.split,
        "is_partial_run": is_partial,
        "item_limit": args.limit,
        "models": panel_models,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "skip_correlations": args.skip_correlations,
        "sources": config["sources"],
    }
    json_path = write_result(payload, out_dir / SVD_JSON_NAME, config=config)
    npz_path = write_npz_with_provenance(
        analysis.arrays(), out_dir / LOW_RANK_NPZ_NAME, config=config
    )
    print(f"wrote {json_path.name} and {npz_path.name}")

    csv_path: Path | None = None
    if not args.skip_correlations:
        if not head_csv.exists() or not layer_csv.exists():
            print(
                f"weight-delta CSVs not found ({head_csv}, {layer_csv}); the low-rank "
                "half is written, the correlation half is skipped. Run the Phase 1 "
                "audit to produce them."
            )
        else:
            deltas = load_weight_deltas(head_csv, layer_csv)
            frozen = verify_frozen_layers(deltas, FROZEN_LAYERS)
            print(f"frozen-layer check: {frozen['max_abs_frobenius']} (expect all 0.0)")
            rows, notes = build_correlation_rows(
                sources, deltas, models=panel_models,
                edge_fraction=args.edge_fraction, n_boot=args.n_boot, seed=args.seed,
            )
            csv_path = write_result(
                rows, out_dir / CORRELATION_CSV_NAME,
                config={**config, "frozen_layer_check": frozen},
            )
            print(f"wrote {csv_path.name}: {len(rows)} correlations")
            for note in dict.fromkeys(notes):
                print(f"  NOTE {note}")
            summarise_correlations(rows)

    if is_partial:
        print(f"NOTE: partial run (limit={args.limit}, models={panel_models}, "
              f"n_boot={args.n_boot}, skip_correlations={args.skip_correlations}); "
              "artifacts marked is_partial_run=true")
    print(f"done in {time.perf_counter() - t0:.1f} s")
    return 0


def summarise_correlations(rows: Sequence[Mapping[str, Any]]) -> None:
    """Print the brief-question-10 table: which family moved where behaviour moved."""
    wanted = [r for r in rows
              if r.get("correlation") == "spearman"
              and r.get("statistic") == "edge_bias"
              and r.get("weight_metric") == "rel_frobenius"
              and r.get("estimate") is not None]
    print("\nspearman(||dW||_rel, delta edge_bias), clustered on layer:")
    if not wanted:
        print("  (no row of that shape produced a correlation)")
    print(f"{'model':10s} {'scope':16s} {'source':28s} "
          + " ".join(f"{f:>16s}" for f in PARAM_FAMILIES))
    seen: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in wanted:
        key = (str(row["model_key"]), str(row["scope"]), str(row["source"]))
        seen.setdefault(key, {})[str(row["param_family"])] = row
    for (model, scope, source), families in sorted(seen.items()):
        cells = []
        for family in PARAM_FAMILIES:
            row = families.get(family)
            if row is None or row.get("estimate") is None:
                cells.append(f"{'-':>16s}")
                continue
            lo, hi = row.get("ci_low"), row.get("ci_high")
            star = "*" if lo is not None and hi is not None and (lo > 0 or hi < 0) else " "
            cells.append(f"{float(row['estimate']):+15.3f}{star}")
        print(f"{model:10s} {scope:16s} {source[:28]:28s} " + " ".join(cells))
    # Undefined rows are results too -- an all-zero weight delta in the frozen layers is
    # the Phase 1 §3 geometry showing up, not a gap in the table.
    undefined: dict[str, int] = {}
    for row in rows:
        note = str(row.get("note") or "")
        if row.get("estimate") is None and note:
            undefined[note] = undefined.get(note, 0) + 1
    for note, count in sorted(undefined.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d} rows without a correlation: {note}")


if __name__ == "__main__":
    raise SystemExit(main())
