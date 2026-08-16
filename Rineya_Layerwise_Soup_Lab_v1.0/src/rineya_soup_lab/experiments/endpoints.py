from __future__ import annotations

import statistics
from collections import defaultdict

from ..candidates import CandidateManager, CandidateResult
from ..evaluation.objective import align_rows, case_utility
from ..model.layer_map import MergeRecipe
from ..run_context import RunContext
from ..utils import write_csv, write_json
from .common import compare_candidates


def run_endpoints(context: RunContext, manager: CandidateManager) -> dict[str, CandidateResult]:
    anchor = manager.evaluate(
        MergeRecipe.endpoint_recipe("anchor"), alias="anchor", retain=True
    )
    greedy = manager.evaluate(
        MergeRecipe.endpoint_recipe("greedy"), alias="greedy", retain=True
    )
    comparison = compare_candidates(
        greedy,
        anchor,
        objective_config=context.config.get("objective"),
        statistics_config=context.config.get("statistics"),
    )
    anchor_rows, greedy_rows = align_rows(anchor.evidence.rows, greedy.evidence.rows)
    case_rows = []
    category_deltas: dict[str, list[float]] = defaultdict(list)
    anchor_wins = greedy_wins = ties = 0
    oracle_utilities: list[float] = []
    anchor_utilities: list[float] = []
    greedy_utilities: list[float] = []
    for a_row, g_row in zip(anchor_rows, greedy_rows):
        a_utility = case_utility(a_row, context.config.get("objective"))
        g_utility = case_utility(g_row, context.config.get("objective"))
        delta = g_utility - a_utility
        if delta > 1e-12:
            greedy_wins += 1
            winner = "greedy"
        elif delta < -1e-12:
            anchor_wins += 1
            winner = "anchor"
        else:
            ties += 1
            winner = "tie"
        category = str(a_row.get("category", "uncategorized"))
        category_deltas[category].append(delta)
        anchor_utilities.append(a_utility)
        greedy_utilities.append(g_utility)
        oracle_utilities.append(max(a_utility, g_utility))
        case_rows.append(
            {
                "id": a_row["id"],
                "category": category,
                "anchor_utility": a_utility,
                "greedy_utility": g_utility,
                "greedy_minus_anchor": delta,
                "winner": winner,
                "anchor_quality": a_row.get("quality_score"),
                "greedy_quality": g_row.get("quality_score"),
                "anchor_reply": a_row.get("reply"),
                "greedy_reply": g_row.get("reply"),
                "reply_disagreement": str(a_row.get("reply", "")).strip()
                != str(g_row.get("reply", "")).strip(),
            }
        )
    category_rows = [
        {
            "category": category,
            "case_count": len(values),
            "mean_greedy_minus_anchor": statistics.mean(values),
            "greedy_win_rate": sum(value > 1e-12 for value in values) / len(values),
            "anchor_win_rate": sum(value < -1e-12 for value in values) / len(values),
        }
        for category, values in sorted(category_deltas.items())
    ]
    best_endpoint_mean = max(statistics.mean(anchor_utilities), statistics.mean(greedy_utilities))
    complementarity = {
        "case_count": len(case_rows),
        "anchor_wins": anchor_wins,
        "greedy_wins": greedy_wins,
        "ties": ties,
        "anchor_win_rate": anchor_wins / len(case_rows),
        "greedy_win_rate": greedy_wins / len(case_rows),
        "tie_rate": ties / len(case_rows),
        "mean_anchor_case_utility": statistics.mean(anchor_utilities),
        "mean_greedy_case_utility": statistics.mean(greedy_utilities),
        "per_case_oracle_mean_utility": statistics.mean(oracle_utilities),
        "per_case_oracle_margin_over_best_endpoint": statistics.mean(oracle_utilities)
        - best_endpoint_mean,
        "interpretation": (
            "The per-case oracle is a descriptive complementarity upper bound, not a deployable "
            "weight-soup score and not used to select coefficients."
        ),
    }
    write_csv(
        context.path("analysis", "global", "endpoint_case_complementarity.csv"), case_rows
    )
    write_csv(
        context.path("analysis", "global", "endpoint_category_complementarity.csv"),
        category_rows,
    )
    write_json(
        context.path("analysis", "global", "endpoint_comparison.json"),
        {
            "anchor": anchor.to_record(),
            "greedy": greedy.to_record(),
            "greedy_minus_anchor": comparison.to_dict(),
            "complementarity": complementarity,
        },
    )
    return {"anchor": anchor, "greedy": greedy}
