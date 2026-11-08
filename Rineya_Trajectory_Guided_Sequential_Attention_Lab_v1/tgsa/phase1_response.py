"""Secondary difficulty-response analysis. Never gates the primary endpoint.

The mechanistic question behind Phase 1 is not only *whether* trajectory
guidance adds capability but *where*. If the mechanism is doing what its name
claims, the paired advantage should grow with iterative depth and stay near zero
on the one-hop guardrail. That signature is informative, so it is preregistered
rather than discovered.

Two contamination risks are structurally excluded:

* **Bins are never chosen after looking at outcomes.** Every bin is a value the
  registered generator already emits into example metadata — an effective hop
  count, an effective step count, a context bucket. There is no binning
  parameter to tune, and :func:`registered_bins` derives the bin edges from the
  study configuration alone.
* **Nothing here can rescue a failed primary endpoint.** The analyzer records
  these results under a key marked non-gating, and no advancement gate reads
  them. A monotone difficulty response with a null primary contrast is a
  hypothesis for the next study, not evidence for this one.

The trend statistic is an orthogonal polynomial contrast across ordered bins,
which is a fixed-weight linear functional of exactly the same paired panel the
primary estimand uses, so it is estimated with the same registered interval
estimator and inherits its calibration.
"""

from __future__ import annotations

from typing import Mapping, Sequence, cast

from .phase1_inference import IntervalResult, estimate_interval

RESPONSE_SCHEMA_VERSION = "tgsa.phase1-difficulty-response.v1"

# Registered difficulty variables, in the order they are reported. Each entry is
# (metadata key, families it applies to, human-readable description).
REGISTERED_DIFFICULTY_VARIABLES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("effective_hops", ("pointer_chase",), "pointer-chase effective hop count"),
    ("effective_steps", ("state_update",), "state-update effective step count"),
    ("context_bucket", ("pointer_chase", "state_update"), "registered context-length bucket"),
    ("evidence_position", ("pointer_chase",), "position of the last required pair"),
    ("evidence_span", ("pointer_chase",), "span between first and last required pair"),
    ("distractor_count", ("pointer_chase",), "non-chain pairs in the prompt"),
    ("node_count", ("pointer_chase",), "pairs presented in the prompt"),
)

# Continuous variables are bucketed into a small fixed number of quantile bins
# computed from the MATERIALIZED DATASET ONLY, before any model is trained.
CONTINUOUS_VARIABLES = frozenset({"evidence_position", "evidence_span", "distractor_count", "node_count"})
CONTINUOUS_BIN_COUNT = 4


def registered_bins(
    examples_metadata: Mapping[str, Mapping[str, object]],
    *,
    variable: str,
) -> dict[str, str]:
    """Map example ID to a bin label, using dataset structure only.

    Discrete registered variables use their own value as the bin. Continuous
    variables are cut at quantiles of the *materialized dataset*, which is fixed
    before training begins and carries no outcome information.
    """

    values = {
        example_id: metadata[variable]
        for example_id, metadata in examples_metadata.items()
        if variable in metadata
    }
    if not values:
        return {}
    if variable not in CONTINUOUS_VARIABLES:
        return {example_id: f"{variable}={value}" for example_id, value in values.items()}

    ordered = sorted(float(value) for value in values.values())  # type: ignore[arg-type]
    edges: list[float] = []
    for index in range(1, CONTINUOUS_BIN_COUNT):
        position = int(round(index * (len(ordered) - 1) / CONTINUOUS_BIN_COUNT))
        edges.append(ordered[position])
    edges = sorted(set(edges))

    def label(value: float) -> str:
        for index, edge in enumerate(edges):
            if value <= edge:
                return f"{variable}=q{index}"
        return f"{variable}=q{len(edges)}"

    return {example_id: label(float(cast(float, value))) for example_id, value in values.items()}


def _orthogonal_linear_contrast(bin_count: int) -> list[float]:
    """Centred, unit-norm linear contrast weights across ordered bins."""

    if bin_count < 2:
        return [0.0] * bin_count
    centred = [index - (bin_count - 1) / 2.0 for index in range(bin_count)]
    norm = sum(value * value for value in centred) ** 0.5
    return [value / norm for value in centred]


def _bin_panel(
    candidate_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    baseline_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    bin_of: Mapping[str, str],
    families: frozenset[str],
) -> dict[int, dict[str, dict[str, float]]]:
    """Paired differences keyed by seed then BIN then example.

    Reusing the seed/stratum/example panel shape lets the registered interval
    estimator treat difficulty bins exactly as it treats registered strata.
    """

    panel: dict[int, dict[str, dict[str, float]]] = {}
    for seed in sorted(candidate_by_seed):
        candidate = {str(row["example_id"]): row for row in candidate_by_seed[seed]}
        baseline = {str(row["example_id"]): row for row in baseline_by_seed[seed]}
        cells: dict[str, dict[str, float]] = {}
        for example_id, row in candidate.items():
            if str(row["family"]) not in families:
                continue
            label = bin_of.get(example_id)
            if label is None or example_id not in baseline:
                continue
            cells.setdefault(label, {})[example_id] = float(
                bool(row["correct"])
            ) - float(bool(baseline[example_id]["correct"]))
        if cells:
            panel[seed] = dict(sorted(cells.items()))
    if not panel:
        return {}
    reference = set(next(iter(panel.values())))
    for cells in panel.values():
        if set(cells) != reference:
            return {}
    return panel


def difficulty_response(
    candidate_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    baseline_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    examples_metadata: Mapping[str, Mapping[str, object]],
    example_families: Mapping[str, str],
    estimator: str,
    confidence: float,
    bootstrap_samples: int,
    rng_seed: int,
) -> dict[str, object]:
    """Effect size within each preregistered difficulty bin, plus a trend test."""

    results: dict[str, object] = {}
    for variable, families, description in REGISTERED_DIFFICULTY_VARIABLES:
        applicable = frozenset(families)
        scoped_metadata = {
            example_id: metadata
            for example_id, metadata in examples_metadata.items()
            if example_families.get(example_id) in applicable
        }
        bin_of = registered_bins(scoped_metadata, variable=variable)
        if not bin_of:
            continue
        panel = _bin_panel(
            candidate_by_seed,
            baseline_by_seed,
            bin_of=bin_of,
            families=applicable,
        )
        if not panel:
            results[variable] = {
                "description": description,
                "families": sorted(applicable),
                "status": "unavailable",
                "reason": "bins are not shared across every seed",
            }
            continue

        labels = sorted(next(iter(panel.values())))
        per_bin: dict[str, object] = {}
        for label in labels:
            single = {seed: {label: cells[label]} for seed, cells in panel.items()}
            estimate = estimate_interval(
                single,
                estimator=estimator,
                confidence=confidence,
                samples=bootstrap_samples,
                rng_seed=rng_seed,
            )
            per_bin[label] = {
                **estimate.to_dict(),
                "examples": len(next(iter(single.values()))[label]),
            }

        trend: dict[str, object] | None = None
        if len(labels) >= 2:
            weights = _orthogonal_linear_contrast(len(labels))
            trend_estimate = _contrast_estimate(
                panel,
                labels=labels,
                weights=weights,
                estimator=estimator,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                rng_seed=rng_seed + 13,
            )
            trend = {
                **trend_estimate.to_dict(),
                "contrast_weights": dict(zip(labels, weights)),
                "interpretation": (
                    "positive means the paired advantage grows with the registered "
                    "difficulty ordering"
                ),
            }
        results[variable] = {
            "description": description,
            "families": sorted(applicable),
            "status": "estimated",
            "bins": per_bin,
            "linear_trend": trend,
        }

    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "gating": False,
        "claim_scope": (
            "Secondary and mechanistic. Bins are registered generator values, fixed "
            "before training. These results cannot establish the primary endpoint and "
            "cannot rescue a failed primary contrast."
        ),
        "estimator": estimator,
        "variables": results,
    }


def _contrast_estimate(
    panel: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    labels: Sequence[str],
    weights: Sequence[float],
    estimator: str,
    confidence: float,
    bootstrap_samples: int,
    rng_seed: int,
) -> IntervalResult:
    """Estimate a fixed linear contrast across ordered bins.

    The contrast is folded into the per-example values by scaling each bin by
    ``len(labels) * weight``, which turns the registered macro-average over bins
    into exactly ``sum_j weight_j * theta_j``. The estimator therefore sees a
    standard panel and its calibration carries over unchanged.
    """

    scale = {label: len(labels) * weight for label, weight in zip(labels, weights)}
    scaled = {
        seed: {
            label: {
                example_id: value * scale[label] for example_id, value in cells[label].items()
            }
            for label in labels
        }
        for seed, cells in panel.items()
    }
    return estimate_interval(
        scaled,
        estimator=estimator,
        confidence=confidence,
        samples=bootstrap_samples,
        rng_seed=rng_seed,
    )
