"""PARENTQ decision rule (PREREG_PARENTQ.md section 6). numpy plus the frozen GEO benchmark.

Q1 is the posterior predictive probability that the frozen GEO screen's parent gate
(`benchmark.parent_gate`, unmodified) passes on a screen drawn from the same constructors
with the screen's composition: one T row per subtype, and H rows answerable, answerable,
distractor, cue. Each subtype's paired (BASE credited, parent credited) cell
probabilities get an independent Jeffreys Dirichlet(1/2, 1/2, 1/2, 1/2) posterior from
the qualification rows. The gate's truth tables come from calling the frozen
parent_gate on the frozen screen rows with synthetic credited values, so this rule
cannot drift from the screen's own gate.

Q2 is the H contract diagnostic: does H abstain on missing-value rows under the screen's
contract, and if not, under its own abstention training prompts (descriptive, coded labels).
This module is self-contained (no parentq_rows import) so the worker can load it from
pinned bytes; a CPU test checks its constants against parentq_rows.
"""
from __future__ import annotations

from collections import Counter
import itertools
import re
from typing import Any, Mapping, Sequence

import numpy as np

T_SUBTYPES = ("ledger_replay:transfer", "ledger_replay:scale", "register_machine:add",
              "register_machine:conditional")
H_KINDS = ("answerable", "distractor", "cue")
CELLS = ((False, False), (False, True), (True, False), (True, True))   # (BASE, parent) credited
PRIOR = 0.5                    # Jeffreys
QUALIFIED_AT = 0.70            # posterior predictive P(screen parent gate passes)
NOT_QUALIFIED_BELOW = 0.40
CANARY_MIN_MATCHES = 7         # of 8 stored CONFIRM-002 texts, per arm
MC_DRAWS, MC_SEED = 20000, 20261004
ALPHA = 0.05
ARMS = ("BASE", "H", "T")      # canary and schedule order
_INTEGERS = re.compile(r"-?\d+")


def subtype(row) -> str:
    if row.task_kind == "transition_parent":
        return f"{row.metadata['transition_family']}:{row.metadata['event_kind']}"
    return row.metadata["h_row_kind"]


def cell_index(base: bool, parent: bool) -> int:
    return CELLS.index((bool(base), bool(parent)))


def gate_tables(benchmark) -> dict[str, dict[str, Any]]:
    """Which 4-row cell combinations pass the frozen gate, for T and for H separately."""
    rows = benchmark.build_screen()
    groups = benchmark.row_groups(rows)
    t_rows, h_rows = groups["T"], groups["H"]

    def run(t_cells, h_cells):
        scores = {"BASE": {}, "T": {}, "H": {}}
        for group_rows, cells, arm, other in ((t_rows, t_cells, "T", "H"), (h_rows, h_cells, "H", "T")):
            for row, (base, parent) in zip(group_rows, cells, strict=True):
                scores["BASE"][row.row_id] = {"credited": base}
                scores[arm][row.row_id] = {"credited": parent}
                scores[other][row.row_id] = {"credited": False}
        return benchmark.parent_gate(rows, scores)

    combos = tuple(itertools.product(range(len(CELLS)), repeat=4))
    full = [CELLS[3]] * 4
    return {
        "T": {"subtypes": tuple(subtype(row) for row in t_rows), "combos": combos,
              "passed": tuple(run([CELLS[c] for c in combo], full)["T"]["passed"] for combo in combos)},
        "H": {"subtypes": tuple(subtype(row) for row in h_rows), "combos": combos,
              "passed": tuple(run(full, [CELLS[c] for c in combo])["H"]["passed"] for combo in combos)},
    }


def _rising(x: float, k: int) -> float:
    value = 1.0
    for j in range(k):
        value *= x + j
    return value


def predictive(table: Mapping[str, Any], alphas: Mapping[str, Sequence[float]]) -> float:
    """Exact posterior predictive pass probability (Dirichlet moments; rows of one subtype
    share that subtype's cell probabilities, rows of different subtypes are independent)."""
    total = 0.0
    for combo, passed in zip(table["combos"], table["passed"]):
        if not passed:
            continue
        probability = 1.0
        by_subtype: dict[str, list[int]] = {}
        for name, cell in zip(table["subtypes"], combo):
            by_subtype.setdefault(name, []).append(cell)
        for name, cells in by_subtype.items():
            alpha = alphas[name]
            numerator = 1.0
            for cell, k in Counter(cells).items():
                numerator *= _rising(alpha[cell], k)
            probability *= numerator / _rising(sum(alpha), len(cells))
        total += probability
    return total


def plug_in(table: Mapping[str, Any], frequencies: Mapping[str, Sequence[float]]) -> float:
    return float(sum(np.prod([frequencies[name][cell] for name, cell in zip(table["subtypes"], combo)])
                     for combo, passed in zip(table["combos"], table["passed"]) if passed))


def _draw_pass(table, draws: Mapping[str, np.ndarray]) -> np.ndarray:
    """P(gate passes | cell probabilities) for each posterior draw."""
    result = np.zeros(next(iter(draws.values())).shape[0])
    for combo, passed in zip(table["combos"], table["passed"]):
        if passed:
            term = np.ones_like(result)
            for name, cell in zip(table["subtypes"], combo):
                term = term * draws[name][:, cell]
            result += term
    return result


def cell_counts(rows, scores: Mapping[str, Mapping[str, Mapping[str, Any]]], arm: str) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {}
    for row in rows:
        counts.setdefault(subtype(row), [0, 0, 0, 0])[
            cell_index(scores["BASE"][row.row_id]["credited"], scores[arm][row.row_id]["credited"])] += 1
    return counts


def gate_probability(tables, t_counts, h_counts, *, draws=MC_DRAWS, seed=MC_SEED) -> dict[str, Any]:
    """Posterior predictive (decision quantity), plug-in, and a posterior band for P(pi)."""
    alphas = {name: [count + PRIOR for count in counts] for name, counts in (*t_counts.items(), *h_counts.items())}
    p_t, p_h = predictive(tables["T"], alphas), predictive(tables["H"], alphas)
    freqs = {name: [count / sum(counts) for count in counts] for name, counts in (*t_counts.items(), *h_counts.items())}
    rng = np.random.default_rng(seed)
    sampled = {name: rng.dirichlet(alphas[name], size=draws) for name in sorted(alphas)}
    band_t, band_h = _draw_pass(tables["T"], sampled), _draw_pass(tables["H"], sampled)
    both = band_t * band_h
    quantiles = lambda values: [float(np.quantile(values, q)) for q in (0.1, 0.5, 0.9)]
    return {"predictive": p_t * p_h, "predictive_T": p_t, "predictive_H": p_h,
            "plug_in": plug_in(tables["T"], freqs) * plug_in(tables["H"], freqs),
            "plug_in_T": plug_in(tables["T"], freqs), "plug_in_H": plug_in(tables["H"], freqs),
            "posterior_band_q10_q50_q90": quantiles(both), "posterior_band_T": quantiles(band_t),
            "posterior_band_H": quantiles(band_h), "draws": draws, "seed": seed, "prior": PRIOR,
            "cell_order": ["neither", "parent_only", "base_only", "both"]}


def min_detectable_one_sided_gains(benchmark, alpha: float = ALPHA) -> int:
    """Smallest all-one-direction discordant count with exact two-sided McNemar p <= alpha."""
    return next(g for g in range(1, 64) if benchmark.exact_paired_mcnemar(g, 0) <= alpha)


def _paired(benchmark, pairs) -> dict[str, Any]:
    """pairs: (a_credited, b_credited). Gains are a-not-b."""
    gains = sum(1 for a, b in pairs if a and not b)
    losses = sum(1 for a, b in pairs if b and not a)
    p = benchmark.exact_paired_mcnemar(gains, losses)
    return {"n": len(pairs), "gains": gains, "losses": losses, "exact_mcnemar_p_two_sided": p,
            "significant_gain": gains > losses and p <= ALPHA}


def lenient_answer(benchmark, row, text: str) -> bool:
    """Descriptive only: a non-abstaining first line whose only integer is the gold."""
    line = benchmark.first_line(text)
    return (row.expected != benchmark.UNKNOWN and not benchmark.credited_abstention(line)
            and _INTEGERS.findall(line) == [row.expected])


def h_contract_diagnostic(benchmark, h_on, h_off, scores) -> dict[str, Any]:
    off_by_on = {row.metadata["on_row_id"]: row for row in h_off}
    missing = [row for row in h_on if row.expected == benchmark.UNKNOWN]
    answerable = [row for row in h_on if row.expected != benchmark.UNKNOWN]
    credited = lambda arm, row: bool(scores[arm][row.row_id]["credited"])
    off = lambda row: off_by_on[row.row_id]
    comparisons = {
        "missing_on_H_vs_BASE": _paired(benchmark, [(credited("H", r), credited("BASE", r)) for r in missing]),
        "missing_off_H_vs_BASE": _paired(benchmark, [(credited("H", off(r)), credited("BASE", off(r))) for r in missing]),
        "missing_H_off_vs_on": _paired(benchmark, [(credited("H", off(r)), credited("H", r)) for r in missing]),
        "missing_BASE_off_vs_on": _paired(benchmark, [(credited("BASE", off(r)), credited("BASE", r)) for r in missing]),
        "answerable_H_on_vs_off": _paired(benchmark, [(credited("H", r), credited("H", off(r))) for r in answerable]),
    }
    rates = {}
    for arm in ("BASE", "H"):
        for contract, pick in (("on", lambda row: row), ("off", off)):
            rates[f"{arm}_{contract}"] = {
                "missing_abstention_credited": sum(credited(arm, pick(r)) for r in missing),
                "answerable_credited": sum(credited(arm, pick(r)) for r in answerable),
                "answerable_abstained": sum(bool(scores[arm][pick(r).row_id]["abstained"]) for r in answerable),
                "strict_correct": sum(bool(scores[arm][pick(r).row_id]["strict_correct"]) for r in (*missing, *answerable)),
                "n_missing": len(missing), "n_answerable": len(answerable)}
    if comparisons["missing_on_H_vs_BASE"]["significant_gain"]:
        label = "H_ABSTAINS_UNDER_CONTRACT"
    elif comparisons["missing_H_off_vs_on"]["significant_gain"] and comparisons["missing_off_H_vs_BASE"]["significant_gain"]:
        label = "NATIVE_PROMPT_ELICITS_H_ABSTENTION"
    elif comparisons["missing_H_off_vs_on"]["significant_gain"]:
        label = "NATIVE_PROMPT_ELICITS_ABSTENTION_NOT_SPECIFIC_TO_H"
    else:
        label = "H_ABSTENTION_NOT_SHOWN"
    return {"label": label, "comparisons": comparisons, "rates": rates,
            "min_detectable_one_direction_discordant": min_detectable_one_sided_gains(benchmark),
            "note": "descriptive; labels use exact two-sided McNemar p <= 0.05 with gains > losses"}


def canary_check(canary: Sequence[Mapping[str, Any]], outputs: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Each arm against every stored CONFIRM-002 source (stripped exact text)."""
    sources = {"BASE": "BASE", "T": "SPARSE", "H": "H25"}
    result = {}
    for arm in ARMS:
        texts = outputs.get(arm, {})
        ids = [row["row_id"] for row in canary]
        complete = all(row_id in texts for row_id in ids)
        matrix = {source: sum(texts.get(row["row_id"], "\0").strip() == row["stored"][source].strip() for row in canary)
                  for source in sources.values()}
        result[arm] = {"complete": complete, "n": len(ids), "matches_own_source": matrix[sources[arm]],
                       "matches_by_source": matrix, "passed": complete and matrix[sources[arm]] >= CANARY_MIN_MATCHES}
    return result


def decide(benchmark, bank: Mapping[str, Sequence[Any]], canary: Sequence[Mapping[str, Any]],
           outputs: Mapping[str, Mapping[str, str]], mechanics: Mapping[str, Any],
           stopped_by: str | None = None, tables=None) -> dict[str, Any]:
    """The single registered PARENTQ verdict. First match wins:
      1. padding parity: False -> STOP_NATIVE_MECHANICS_PADDING; None -> STOPPED_BEFORE:padding
      2. per arm in schedule order (BASE, H, T): canary incomplete -> STOPPED_BEFORE:canary:<arm>;
         failed -> STOP_ARM_CANARY:<arm>
      3. missing Q1 coverage -> INCOMPLETE_COVERAGE
      4. predictive P >= 0.70 -> PARENTS_QUALIFIED; >= 0.40 -> PARENTS_MARGINAL;
         else PARENTS_NOT_QUALIFIED:<the parent with the lower component probability>
    Q2 (h_contract) is reported whenever BASE and H passed their canaries and covered both
    H blocks, whatever the status. Nothing here is confirmation of any capability.
    """
    rows = {row.row_id: row for block in ("T", "H_ON", "H_OFF") for row in bank[block]}
    scores = {arm: {row_id: benchmark.score_response(rows[row_id], text)
                    for row_id, text in texts.items() if row_id in rows}
              for arm, texts in outputs.items()}
    verdict: dict[str, Any] = {"confirmation": False, "stopped_by": stopped_by, "thresholds": {
        "qualified_at": QUALIFIED_AT, "not_qualified_below": NOT_QUALIFIED_BELOW,
        "canary_min_matches": CANARY_MIN_MATCHES}}
    verdict["counts"] = {arm: {block: sum(bool(scores[arm][row.row_id]["credited"]) for row in bank[block]
                                          if row.row_id in scores.get(arm, {}))
                               for block in ("T", "H_ON", "H_OFF")} for arm in scores}
    covered = lambda arm, block: all(row.row_id in scores.get(arm, {}) for row in bank[block])

    def finish(status):
        verdict["status"] = status
        canaries = verdict.get("canary", {})
        if (all(canaries.get(arm, {}).get("passed") for arm in ("BASE", "H"))
                and all(covered(arm, block) for arm in ("BASE", "H") for block in ("H_ON", "H_OFF"))):
            verdict["h_contract"] = h_contract_diagnostic(benchmark, bank["H_ON"], bank["H_OFF"], scores)
        else:
            verdict["h_contract"] = {"label": "Q2_INCOMPLETE"}
        return verdict

    if mechanics.get("padding_parity_ok") is False:
        return finish("STOP_NATIVE_MECHANICS_PADDING")
    if mechanics.get("padding_parity_ok") is not True:
        return finish("STOPPED_BEFORE:padding")
    verdict["canary"] = canary_check(canary, outputs)
    for arm in ARMS:
        if not verdict["canary"][arm]["complete"]:
            return finish(f"STOPPED_BEFORE:canary:{arm}")
        if not verdict["canary"][arm]["passed"]:
            return finish(f"STOP_ARM_CANARY:{arm}")
    needed = {("BASE", "T"), ("T", "T"), ("BASE", "H_ON"), ("H", "H_ON")}
    missing = sorted(f"{arm}:{block}" for arm, block in needed if not covered(arm, block))
    if missing:
        verdict["missing_coverage"] = missing
        return finish("INCOMPLETE_COVERAGE")
    tables = tables or gate_tables(benchmark)
    t_counts, h_counts = cell_counts(bank["T"], scores, "T"), cell_counts(bank["H_ON"], scores, "H")
    if set(t_counts) != set(T_SUBTYPES) or set(h_counts) != set(H_KINDS):
        raise ValueError("qualification bank does not cover every screen subtype")
    gate = gate_probability(tables, t_counts, h_counts)
    verdict.update({"cells_T": t_counts, "cells_H_on": h_counts, "screen_parent_gate": gate,
                    "bank_level": {
                        "T_vs_BASE": _paired(benchmark, [(scores["T"][r.row_id]["credited"], scores["BASE"][r.row_id]["credited"]) for r in bank["T"]]),
                        "H_vs_BASE_on": _paired(benchmark, [(scores["H"][r.row_id]["credited"], scores["BASE"][r.row_id]["credited"]) for r in bank["H_ON"]])}})
    if all(covered(arm, "H_OFF") for arm in ("BASE", "H")):
        off_counts = cell_counts(bank["H_OFF"], scores, "H")
        alphas = {name: [count + PRIOR for count in counts] for name, counts in off_counts.items()}
        verdict["h_gate_predictive_under_native_prompts"] = {
            "predictive_H": predictive(tables["H"], alphas), "cells": off_counts,
            "note": "descriptive counterfactual: the screen does not use these prompts"}
        verdict["off_lenient_answers"] = {arm: sum(lenient_answer(benchmark, row, outputs[arm][row.row_id])
                                                   for row in bank["H_OFF"]) for arm in ("BASE", "H")}
    p = gate["predictive"]
    if p >= QUALIFIED_AT:
        return finish("PARENTS_QUALIFIED")
    if p >= NOT_QUALIFIED_BELOW:
        return finish("PARENTS_MARGINAL")
    weaker = "T" if gate["predictive_T"] <= gate["predictive_H"] else "H"
    return finish(f"PARENTS_NOT_QUALIFIED:{weaker}")
