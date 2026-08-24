"""Cluster-aware method-by-compression-round interaction analysis."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .clustered import cluster_bootstrap_interval, exact_or_sampled_sign_flip

METHODS = ("B2", "B3")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_matched_schedule_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove the intended pairing and held-constant contract from output rows."""

    if not rows:
        raise ValueError("schedule-control rows must be non-empty")
    pairs: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    cluster_by_case: dict[str, str] = {}
    cases: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id", ""))
        cluster_id = str(row.get("cluster_id", ""))
        schedule_id = str(row.get("schedule_id", ""))
        method = str(row.get("method", ""))
        if not case_id or not cluster_id or not schedule_id or method not in METHODS:
            raise ValueError("each row requires case_id, cluster_id, schedule_id, and B2/B3 method")
        previous_cluster = cluster_by_case.setdefault(case_id, cluster_id)
        if previous_cluster != cluster_id:
            raise ValueError(f"case {case_id!r} maps to multiple immutable clusters")
        pair = pairs[(case_id, schedule_id)]
        if method in pair:
            raise ValueError(f"duplicate cell for {(case_id, schedule_id, method)}")
        pair[method] = row
        cases[case_id].append(row)

    for key, pair in pairs.items():
        if tuple(sorted(pair)) != METHODS:
            raise ValueError(f"matched cell {key} must contain exactly B2 and B3")
        left, right = pair["B2"], pair["B3"]
        for field in ("schedule", "held_constant", "query_access"):
            if _mapping(left.get(field), name=field) != _mapping(right.get(field), name=field):
                raise ValueError(f"{field} differs inside matched B2/B3 pair {key}")
        if float(left.get("count_exponent", math.nan)) != 0.0:
            raise ValueError(f"B2 must be raw accumulated significance at {key}")
        if float(right.get("count_exponent", math.nan)) != 1.0:
            raise ValueError(f"B3 must use realized-count normalization at {key}")
        schedule = _mapping(left["schedule"], name="schedule")
        access = _mapping(left["query_access"], name="query_access")
        rounds = int(schedule.get("compression_rounds", -1))
        probes = int(access.get("query_probes", -1))
        query_tokens = int(access.get("query_tokens_per_probe", -1))
        evaluations = int(access.get("query_probe_token_evaluations", -1))
        if rounds <= 0 or probes != rounds or query_tokens <= 0:
            raise ValueError(f"invalid schedule/query access accounting at {key}")
        if evaluations != rounds * query_tokens:
            raise ValueError(f"query token-evaluation accounting mismatch at {key}")
        scoring_events = access.get("scoring_events_per_layer")
        if scoring_events is not None and (
            not isinstance(scoring_events, Sequence)
            or isinstance(scoring_events, (str, bytes))
            or not scoring_events
            or any(int(value) != rounds for value in scoring_events)
        ):
            raise ValueError(f"per-layer scoring events do not realize the schedule at {key}")
        planned_positions = [int(value) for value in schedule.get("boundary_positions", [])]
        for method, method_row in pair.items():
            accounting = _mapping(method_row.get("accounting"), name="accounting")
            if int(accounting.get("resident_entries_after_prefill", -1)) != int(
                schedule.get("final_resident_entries", -2)
            ):
                raise ValueError(
                    f"runtime did not realize the fixed final cache size at {key}/{method}"
                )
            boundaries = accounting.get("boundary_accounting")
            reports = accounting.get("compression_reports")
            if not isinstance(boundaries, Sequence) or isinstance(boundaries, (str, bytes)):
                raise ValueError(f"boundary accounting is invalid at {key}/{method}")
            if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
                raise ValueError(f"compression reports are invalid at {key}/{method}")
            if len(boundaries) != rounds or len(reports) != rounds:
                raise ValueError(
                    f"runtime compression count differs from the schedule at {key}/{method}"
                )
            realized_positions = [
                int(_mapping(value, name="boundary row")["absolute_token_position"])
                for value in boundaries
            ]
            if realized_positions != planned_positions:
                raise ValueError(
                    f"runtime boundary positions differ from the schedule at {key}/{method}"
                )

    schedules_per_case: dict[str, int] = {}
    round_counts_per_case: dict[str, list[int]] = {}
    for case_id, case_rows in cases.items():
        b2_rows = [row for row in case_rows if row["method"] == "B2"]
        reference = _mapping(b2_rows[0]["held_constant"], name="held_constant")
        final_resident: set[int] = set()
        round_counts: set[int] = set()
        schedule_ids: set[str] = set()
        for row in b2_rows:
            if _mapping(row["held_constant"], name="held_constant") != reference:
                raise ValueError(f"held-constant fields drift across schedules for {case_id}")
            schedule = _mapping(row["schedule"], name="schedule")
            schedule_ids.add(str(row["schedule_id"]))
            final_resident.add(int(schedule.get("final_resident_entries", -1)))
            round_counts.add(int(schedule.get("compression_rounds", -1)))
        if len(final_resident) != 1 or next(iter(final_resident)) <= 0:
            raise ValueError(f"final resident cache size is not held constant for {case_id}")
        if len(round_counts) < 2:
            raise ValueError(f"case {case_id} needs at least two distinct compression-round counts")
        schedules_per_case[case_id] = len(schedule_ids)
        round_counts_per_case[case_id] = sorted(round_counts)

    return {
        "status": "PASS",
        "rows": len(rows),
        "matched_pairs": len(pairs),
        "cases": len(cases),
        "independent_clusters": len(set(cluster_by_case.values())),
        "methods": list(METHODS),
        "schedules_per_case": schedules_per_case,
        "round_counts_per_case": round_counts_per_case,
        "final_cache_size_held_constant": True,
        "query_access_equal_within_each_method_pair": True,
    }


def _nested_float(row: Mapping[str, Any], path: str) -> float:
    value: Any = row
    for component in path.split("."):
        value = _mapping(value, name=f"outcome path {path!r}").get(component)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"outcome {path!r} must be finite")
    return result


def _cluster_inference(
    values_by_cluster: Mapping[str, float],
    *,
    minimum_clusters: int,
    confidence: float,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict[str, Any]:
    ordered = np.asarray([values_by_cluster[key] for key in sorted(values_by_cluster)], dtype=np.float64)
    if ordered.size == 0:
        raise ValueError("cluster-level interaction values must be non-empty")
    interval = cluster_bootstrap_interval(
        ordered.tolist(),
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    p_value, method, samples = exact_or_sampled_sign_flip(
        ordered.tolist(),
        monte_carlo_samples=randomization_samples,
        seed=seed,
    )
    eligible = len(ordered) >= minimum_clusters
    return {
        "cluster_count": len(ordered),
        "cluster_values": dict(sorted(values_by_cluster.items())),
        "mean": float(ordered.mean()),
        "median": float(np.median(ordered)),
        "confidence_interval": list(interval),
        "p_value_two_sided": p_value,
        "test_method": method,
        "randomization_samples": samples,
        "eligible": eligible,
        "eligibility_reason": (
            "PASS" if eligible else f"requires at least {minimum_clusters} independent clusters"
        ),
    }


def clustered_method_round_interaction(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome_path: str,
    minimum_clusters: int,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    randomization_samples: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate B3-minus-B2 change as compression rounds increase.

    The row-level method contrasts are first collapsed by immutable cluster and
    round count.  A within-cluster slope and max-minus-min endpoint contrast are
    then computed.  Bootstrap resampling and sign flipping operate only on
    these independent cluster summaries, never on repeated schedule rows.
    """

    matching = validate_matched_schedule_rows(rows)
    paired: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(str(row["case_id"]), str(row["schedule_id"]))][str(row["method"])] = row

    by_cluster_round: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pair in paired.values():
        baseline, candidate = pair["B2"], pair["B3"]
        cluster_id = str(baseline["cluster_id"])
        schedule = _mapping(baseline["schedule"], name="schedule")
        rounds = int(schedule["compression_rounds"])
        difference = _nested_float(candidate, outcome_path) - _nested_float(baseline, outcome_path)
        by_cluster_round[cluster_id][rounds].append(difference)

    common_rounds = sorted(set.intersection(*(set(values) for values in by_cluster_round.values())))
    if len(common_rounds) < 2:
        raise ValueError("interaction endpoint needs at least two round counts shared by every cluster")
    lowest_common_round, highest_common_round = common_rounds[0], common_rounds[-1]
    slopes: dict[str, float] = {}
    endpoint_contrasts: dict[str, float] = {}
    collapsed: dict[str, dict[str, float]] = {}
    cluster_round_spans: dict[str, list[int]] = {}
    for cluster_id in sorted(by_cluster_round):
        per_round = {
            rounds: float(np.mean(values))
            for rounds, values in sorted(by_cluster_round[cluster_id].items())
        }
        if len(per_round) < 2:
            raise ValueError(f"cluster {cluster_id!r} needs at least two round counts")
        x = np.asarray(list(per_round), dtype=np.float64)
        y = np.asarray(list(per_round.values()), dtype=np.float64)
        centered = x - x.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= 0:
            raise ValueError(f"cluster {cluster_id!r} has no round-count variation")
        slopes[cluster_id] = float(np.dot(centered, y - y.mean()) / denominator)
        endpoint_contrasts[cluster_id] = float(
            per_round[highest_common_round] - per_round[lowest_common_round]
        )
        collapsed[cluster_id] = {str(rounds): value for rounds, value in per_round.items()}
        cluster_round_spans[cluster_id] = [int(x.min()), int(x.max())]

    by_round: dict[str, Any] = {}
    for offset, rounds in enumerate(common_rounds):
        values = {
            cluster_id: float(np.mean(by_cluster_round[cluster_id][rounds]))
            for cluster_id in sorted(by_cluster_round)
        }
        by_round[str(rounds)] = _cluster_inference(
            values,
            minimum_clusters=minimum_clusters,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples,
            seed=seed + 100 + offset,
        )

    return {
        "status": "PASS",
        "outcome_path": outcome_path,
        "contrast_direction": "B3_minus_B2",
        "matching_audit": matching,
        "cluster_round_method_contrasts": collapsed,
        "linear_interaction_per_additional_compression_round": _cluster_inference(
            slopes,
            minimum_clusters=minimum_clusters,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples,
            seed=seed,
        ),
        "highest_common_round_minus_lowest_common_round_method_contrast": {
            "lowest_common_round": lowest_common_round,
            "highest_common_round": highest_common_round,
            **_cluster_inference(
                endpoint_contrasts,
                minimum_clusters=minimum_clusters,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                randomization_samples=randomization_samples,
                seed=seed + 1,
            ),
        },
        "cluster_round_spans_used_by_linear_interaction": cluster_round_spans,
        "method_contrast_by_common_round_count": by_round,
        "independence_unit": "immutable cluster",
        "row_level_schedule_repeats_treated_as_independent": False,
    }
