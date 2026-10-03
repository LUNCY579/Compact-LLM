"""Descriptive fidelity/behavior join over already verified Track B evidence.

This module reads no files, imports no model code and never selects a candidate.
The caller must verify provenance and the complete raw-evidence screen audit.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import groupby
import math

GATES = (0.03, 0.10, 0.20, 0.40)
OBJECTIVES = {"XG": "B3", "XS": "BS", "XR": "BR"}
STRATA = ("all_tokens", "ordinary_tokens", "special_tokens", "final_prompt_token")
PAIR = {"supa_layer": 19, "donor_layer": 24}
CELLS = {"X0": {"arm": "X0", "gate": 0.0}, **{
    f"{arm}_g{gate:.2f}": {"arm": arm, "gate": gate}
    for gate in GATES for arm in (*OBJECTIVES, "XM")}}


def _number(value, name, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (minimum is not None and value < minimum
            or maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _count(value, name, maximum=128):
    if type(value) is not int or value < 0 or maximum is not None and value > maximum:
        raise ValueError(f"{name} must be a nonnegative integer within coverage")
    return value


def _same(left, right, name):
    _number(left, name)
    _number(right, name)
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"inconsistent {name}")


def _alignment(matrix):
    pairs = matrix["pairs"]
    expected = {(s, d) for s in (8, 12, 16, 19) for d in (12, 16, 20, 24)}
    if (not isinstance(pairs, list) or len(pairs) != 16
            or {(p["supa_layer"], p["donor_layer"]) for p in pairs} != expected
            or any(set(p["objectives"]) != {"B0", "B1", "B2", "B3", "BS", "BR"}
                   for p in pairs)):
        raise ValueError("complete 16-pair, six-objective matrix required")
    selected = next(p for p in pairs if p["supa_layer"] == 19 and p["donor_layer"] == 24)
    result = {}
    for arm, objective in OBJECTIVES.items():
        delta = selected["objectives"][objective]["heldout"]["delta"]
        if set(delta) != set(STRATA):
            raise ValueError("all four held-out delta strata required")
        result[arm] = {}
        for stratum in STRATA:
            source = delta[stratum]
            tokens = _count(source["tokens"], "alignment tokens", maximum=None)
            if tokens == 0:
                raise ValueError("selected alignment strata must have observations")
            result[arm][stratum] = {
                "tokens": tokens,
                "mean_token_cosine": _number(source["mean_token_cosine"], "delta cosine", -1.000001, 1.000001),
                "mean_token_relative_mse": _number(source["mean_token_relative_mse"], "token-relative MSE", 0),
                "global_relative_mse": _number(source["global_relative_mse"], "global-relative MSE", 0),
            }
        counts = result[arm]
        if counts["all_tokens"]["tokens"] != counts["ordinary_tokens"]["tokens"] + counts["special_tokens"]["tokens"]:
            raise ValueError("alignment token strata do not partition all tokens")
        if counts["final_prompt_token"]["tokens"] != 32:
            raise ValueError("32 held-out final prompt tokens required")
    return result


def _complete(summary, baseline):
    paired, diagnostic = summary["paired_statistics"], summary["diagnostics"]
    overall = paired["overall"]
    if (paired["analysis_role"] != "candidate_selection"
            or any(type(value) is not int or value != 128 for value in
                   (overall["n"], overall["independent_scenarios"], diagnostic["examples"]))):
        raise ValueError("complete 128-case paired scenario coverage required")
    base_accuracy = _number(baseline["accuracy"], "baseline accuracy", 0, 1)
    accuracy = _number(summary["accuracy"], "accuracy", 0, 1)
    wins, losses, both, neither = [_count(overall[key], key) for key in
                                  ("paired_wins", "paired_losses", "both_correct", "both_wrong")]
    if wins + losses + both + neither != 128:
        raise ValueError("paired outcome counts do not cover 128 cases")
    for left, right, label in (
        (overall["candidate_accuracy"], accuracy, "candidate accuracy"),
        (overall["baseline_accuracy"], base_accuracy, "baseline accuracy"),
        ((wins + both) / 128, accuracy, "candidate counts"),
        ((losses + both) / 128, base_accuracy, "baseline counts"),
        (overall["accuracy_difference"], accuracy - base_accuracy, "accuracy gain"),
    ):
        _same(left, right, label)
    nll, identity, validity = overall["nll"], overall["output_identity"], overall["validity"]
    if (_count(nll["pairs"], "NLL pairs") != 128
            or _count(identity["pairs"], "output identity pairs") != 128
            or nll["available_case_analysis"] is not False
            or nll["direction"] != "negative favors candidate"):
        raise ValueError("complete paired NLL/output coverage and recorded NLL direction required")
    base_diag = baseline["diagnostics"]
    base_nll = _number(base_diag["mean_gold_nll"], "baseline NLL", 0)
    candidate_nll = _number(diagnostic["mean_gold_nll"], "candidate NLL", 0)
    _same(nll["candidate_minus_baseline"], candidate_nll - base_nll, "paired NLL sign and value")
    identical = _count(identity["identical"], "identical outputs")
    _same(identity["rate"], identical / 128, "output identity rate")
    invalid = _count(diagnostic["invalid_outputs"], "invalid outputs")
    base_invalid = _count(base_diag["invalid_outputs"], "baseline invalid outputs")
    if (_count(validity["candidate_invalid"], "paired invalid outputs") != invalid
            or _count(validity["baseline_invalid"], "paired baseline invalid outputs") != base_invalid
            or _count(diagnostic["safety_stops"], "safety stops") != 0):
        raise ValueError("inconsistent validity or safety-stopped complete cell")
    cost = {key: _number(diagnostic[key], key, 0) for key in
            ("mean_wall_seconds", "mean_generation_dense_matmul_flops_estimate",
             "peak_rss_bytes", "peak_cuda_allocated_bytes")}
    cost["operator_inventory"] = {key: _count(diagnostic["operator_inventory"][key], key, maximum=None)
                                  for key in ("added_parameter_count", "added_active_parameters",
                                              "extra_matmul_flops_per_token")}
    for paired_key, diagnostic_key in (("wall_seconds", "mean_wall_seconds"),
                                      ("generation_dense_matmul_flops_estimate", "mean_generation_dense_matmul_flops_estimate")):
        item = overall[paired_key]
        _same(item["candidate_mean"], diagnostic[diagnostic_key], paired_key)
        _same(item["baseline_mean"], base_diag[diagnostic_key], f"baseline {paired_key}")
        ratio = diagnostic[diagnostic_key] / base_diag[diagnostic_key] if base_diag[diagnostic_key] > 0 else None
        if ratio is None:
            if item["ratio"] is not None:
                raise ValueError("zero baseline cost has undefined ratio")
        else:
            _same(item["ratio"], ratio, f"{paired_key} ratio")
        cost[f"{paired_key}_ratio_to_X0"] = ratio
    behavior = {
        "examples": 128, "accuracy": accuracy, "accuracy_gain_vs_X0": accuracy - base_accuracy,
        "paired_wins": wins, "paired_losses": losses,
        "mean_gold_nll": candidate_nll, "nll_gain_X0_minus_cell": base_nll - candidate_nll,
        "output_token_identity": {"identical": identical, "pairs": 128, "rate": identical / 128},
        "mean_kl_X0_to_cell": _number(diagnostic["mean_kl_X0_to_cell"], "KL", 0),
        "mean_first_token_cosine": _number(diagnostic["mean_first_token_cosine"], "first-token cosine", -1.000001, 1.000001),
        "invalid_outputs": invalid, "additional_invalid_vs_X0": invalid - base_invalid,
        "safety_stops": 0, "token_cap_terminations": _count(diagnostic["token_cap_terminations"], "token-cap terminations"),
    }
    return behavior, cost


def _ranking(rows, metric):
    available = [(row["cell_id"], metric(row)) for row in rows if metric(row) is not None]
    available.sort(key=lambda item: (-item[1], item[0]))
    groups = []
    rank = 1
    for value, items in groupby(available, key=lambda item: item[1]):
        names = [name for name, _ in items]
        groups.append({"rank": rank, "cell_ids": names, "value": value, "tied": len(names) > 1})
        rank += len(names)
    return {"direction": "descending", "tie_rule": "exact equality of reported values",
            "groups": groups, "unavailable_cells": [row["cell_id"] for row in rows if metric(row) is None]}


def analyze(matrix, screen_report, screen_config):
    """Join fixed S19/D24 bridge fidelity with all 17 terminal screen cells.

    Failed-cell partial outputs never contribute a behavioral estimate or rank.
    This is descriptive ordering of three formulations, not an independent-dose
    correlation analysis. The existing screen decision is copied unchanged.
    """
    try:
        if (screen_config["schema_version"] != "cross_scale_generation_screen_v1"
                or screen_config["selected_pair"] != PAIR or screen_config["selected_objective"] != "B3"
                or screen_config["cells"] != CELLS or screen_config["gate_grid"] != list(GATES)
                or screen_config["upstream_export"]["source_objectives"] != OBJECTIVES):
            raise ValueError("fixed S19/D24 B3/BS/BR mapping and 17-cell dose grid required")
        if (screen_report["schema_version"] != "cross_scale_generation_screen_analysis_v1"
                or screen_report["analysis_role"] != "candidate_selection"
                or screen_report["status"] not in ("SCREEN_GO", "SCREEN_NO_GO")
                or any(type(screen_report[key]) is not int or screen_report[key] != 128
                       for key in ("examples", "independent_scenarios"))
                or set(screen_report["cells"]) != set(CELLS)
                or screen_report["cells"]["X0"]["status"] != "COMPLETE"):
            raise ValueError("complete terminal 17-cell screen with intact 128-case X0 required")
        if (screen_report["status"] == "SCREEN_GO") != (screen_report["selected"] is not None):
            raise ValueError("inconsistent existing screen status and selection")
        alignment = _alignment(matrix)
        rows = []
        for name, cell in CELLS.items():
            summary = screen_report["cells"][name]
            row = {"cell_id": name, **cell, "objective": OBJECTIVES.get(cell["arm"]),
                   "alignment": deepcopy(alignment.get(cell["arm"])), "status": summary["status"],
                   "failure": None, "retained_partial_examples": 0, "behavior": None, "cost": None}
            if summary["status"] == "COMPLETE":
                row["behavior"], row["cost"] = _complete(summary, screen_report["cells"]["X0"])
            elif summary["status"] == "FAILED":
                failure = summary["failure"]
                if failure["status"] != "FAILED" or failure["phase"] not in ("qualification", "generation"):
                    raise ValueError("terminal qualification/generation failure receipt required")
                row["failure"] = deepcopy(failure)
                row["retained_partial_examples"] = _count(summary["retained_partial_examples"], "retained partial examples")
            else:
                raise ValueError("unfinished cells cannot enter a fidelity report")
            rows.append(row)
        baseline = rows[0]["behavior"]
        if baseline["paired_wins"] or baseline["paired_losses"] or baseline["output_token_identity"]["identical"] != 128:
            raise ValueError("X0 self-comparison must have zero discordance and exact token identity")
        doses = []
        for gate in GATES:
            bridges = [row for row in rows if row["gate"] == gate and row["arm"] in OBJECTIVES]
            doses.append({"gate": gate, "bridge_cells": [row["cell_id"] for row in bridges],
                "ordinary_delta_cosine_rank": _ranking(bridges, lambda row: row["alignment"]["ordinary_tokens"]["mean_token_cosine"]),
                "accuracy_gain_rank": _ranking(bridges, lambda row: None if row["behavior"] is None else row["behavior"]["accuracy_gain_vs_X0"]),
                "nll_gain_rank": _ranking(bridges, lambda row: None if row["behavior"] is None else row["behavior"]["nll_gain_X0_minus_cell"])})
        return {"schema_version": "cross_scale_fidelity_relation_v1", "analysis_role": "descriptive",
            "selected_pair": deepcopy(PAIR), "cells": rows, "doses": doses,
            "screen_status": screen_report["status"], "screen_selection": deepcopy(screen_report["selected"]),
            "screen_ranked_eligible_gates": deepcopy(screen_report["ranked_eligible_gates"]),
            "fidelity_formulations": 3, "repeated_doses_are_independent_fidelity_observations": False,
            "efficacy_claim_licensed": False,
            "limitations": ["Three bridge formulations at one selected depth pair; doses repeat the same fidelity observations.",
                "Architecture-screen outcomes are descriptive and do not establish confirmatory efficacy.",
                "Failed cells preserve failure evidence and have no behavioral estimates.",
                "Fidelity and outcome orderings do not establish a causal relation; no correlation test or fitted slope is computed."]}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed or incomplete fidelity-report input: {exc}") from exc
