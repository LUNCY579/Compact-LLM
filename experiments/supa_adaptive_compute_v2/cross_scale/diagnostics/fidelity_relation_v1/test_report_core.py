"""Small synthetic schema fixtures; no models, Torch or real outcome data."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

_SPEC = importlib.util.spec_from_file_location("fidelity_report_core", Path(__file__).with_name("report_core.py"))
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)


def complete(wins=0, losses=0, nll_gain=0.0, *, baseline=False):
    accuracy = (64 + wins - losses) / 128
    cost, flops, identical = (1.0, 100, 128) if baseline else (2.0, 200, 100)
    overall = {
        "n": 128, "independent_scenarios": 128, "baseline_accuracy": 0.5,
        "candidate_accuracy": accuracy, "accuracy_difference": accuracy - 0.5,
        "paired_wins": wins, "paired_losses": losses, "both_correct": 64 - losses,
        "both_wrong": 64 - wins,
        "nll": {"pairs": 128, "candidate_minus_baseline": -nll_gain,
                "direction": "negative favors candidate", "available_case_analysis": False},
        "output_identity": {"identical": identical, "pairs": 128, "rate": identical / 128},
        "validity": {"baseline_invalid": 0, "candidate_invalid": 0},
        "wall_seconds": {"baseline_mean": 1.0, "candidate_mean": cost, "ratio": cost},
        "generation_dense_matmul_flops_estimate": {"baseline_mean": 100, "candidate_mean": flops, "ratio": flops / 100},
    }
    diagnostics = {
        "examples": 128, "mean_gold_nll": 1.0 - nll_gain,
        "invalid_outputs": 0, "safety_stops": 0, "mean_wall_seconds": cost,
        "mean_generation_dense_matmul_flops_estimate": flops,
        "peak_rss_bytes": 1024, "peak_cuda_allocated_bytes": 2048,
        "operator_inventory": {"added_parameter_count": 0 if baseline else 30,
            "added_active_parameters": 0 if baseline else 20, "extra_matmul_flops_per_token": 0 if baseline else 40},
        "mean_kl_X0_to_cell": 0.0 if baseline else 0.01,
        "mean_first_token_cosine": 1.0 if baseline else 0.99, "token_cap_terminations": 0,
    }
    return {"status": "COMPLETE", "accuracy": accuracy, "diagnostics": diagnostics,
            "paired_statistics": {"analysis_role": "candidate_selection", "overall": overall}}


def fixtures():
    objectives = {}
    for objective, cosine in (("B0", .6), ("B1", .7), ("B2", .8), ("B3", .9), ("BS", .3), ("BR", .1)):
        delta = {name: {"tokens": tokens, "mean_token_cosine": cosine - index * .01,
                       "mean_token_relative_mse": index + .2, "global_relative_mse": index + .1}
                 for index, (name, tokens) in enumerate(zip(core.STRATA, (320, 288, 32, 32)))}
        objectives[objective] = {"heldout": {"delta": delta}}
    matrix = {"pairs": [{"supa_layer": s, "donor_layer": d, "objectives": deepcopy(objectives)}
                        for s in (8, 12, 16, 19) for d in (12, 16, 20, 24)]}
    config = {"schema_version": "cross_scale_generation_screen_v1", "selected_pair": deepcopy(core.PAIR),
        "selected_objective": "B3", "cells": deepcopy(core.CELLS), "gate_grid": list(core.GATES),
        "upstream_export": {"source_objectives": deepcopy(core.OBJECTIVES)}}
    cells = {"X0": complete(baseline=True)}
    for name, cell in core.CELLS.items():
        if name != "X0":
            wins, losses, gain = {"XG": (3, 1, .1), "XS": (1, 1, .2), "XR": (0, 1, -.1), "XM": (0, 0, 0)}[cell["arm"]]
            cells[name] = complete(wins, losses, gain)
    report = {"schema_version": "cross_scale_generation_screen_analysis_v1", "analysis_role": "candidate_selection",
        "status": "SCREEN_GO", "examples": 128, "independent_scenarios": 128, "cells": cells,
        "selected": {"arm": "XG", "gate": .1, "cell_id": "XG_g0.10", "objective": "B3", "selected_pair": deepcopy(core.PAIR)},
        "ranked_eligible_gates": [.1, .03, .2, .4]}
    return matrix, report, config


class FidelityRelationTests(unittest.TestCase):
    def test_known_signs_all_strata_and_cost(self):
        matrix, report, config = fixtures()
        result = core.analyze(matrix, report, config)
        rows = {row["cell_id"]: row for row in result["cells"]}
        self.assertEqual(len(rows), 17)
        graft = rows["XG_g0.03"]
        self.assertEqual(graft["behavior"]["accuracy_gain_vs_X0"], 2 / 128)
        self.assertAlmostEqual(graft["behavior"]["nll_gain_X0_minus_cell"], .1)
        self.assertEqual(graft["behavior"]["paired_wins"], 3)
        self.assertEqual(graft["cost"]["wall_seconds_ratio_to_X0"], 2)
        for index, stratum in enumerate(core.STRATA):
            self.assertEqual(graft["alignment"][stratum]["mean_token_relative_mse"], index + .2)
            self.assertEqual(graft["alignment"][stratum]["global_relative_mse"], index + .1)
        self.assertIsNone(rows["X0"]["alignment"])
        self.assertIsNone(rows["XM_g0.03"]["alignment"])
        dose = result["doses"][0]
        self.assertEqual(dose["ordinary_delta_cosine_rank"]["groups"][0]["cell_ids"], ["XG_g0.03"])
        self.assertEqual(dose["accuracy_gain_rank"]["groups"][0]["cell_ids"], ["XG_g0.03"])
        self.assertEqual(dose["nll_gain_rank"]["groups"][0]["cell_ids"], ["XS_g0.03"])

    def test_failure_preserves_receipt_and_nulls_all_behavior(self):
        matrix, report, config = fixtures()
        failure = {"status": "FAILED", "phase": "generation", "reason": "norm guard", "path": "attempt_000"}
        report["cells"]["XG_g0.40"].update(status="FAILED", failure=failure, retained_partial_examples=91,
                                           accuracy=.999, diagnostics={"mean_gold_nll": -500})
        result = core.analyze(matrix, report, config)
        row = next(row for row in result["cells"] if row["cell_id"] == "XG_g0.40")
        self.assertEqual(row["failure"], failure)
        self.assertEqual(row["retained_partial_examples"], 91)
        self.assertIsNone(row["behavior"])
        self.assertIsNone(row["cost"])
        self.assertIsNotNone(row["alignment"])
        self.assertEqual(result["doses"][3]["accuracy_gain_rank"]["unavailable_cells"], ["XG_g0.40"])

    def test_exact_ties_and_selection_unchanged(self):
        matrix, report, config = fixtures()
        report["cells"]["XR_g0.03"] = deepcopy(report["cells"]["XS_g0.03"])
        before = deepcopy((matrix, report, config))
        result = core.analyze(matrix, report, config)
        tied = result["doses"][0]["accuracy_gain_rank"]["groups"][1]
        self.assertEqual(tied["cell_ids"], ["XR_g0.03", "XS_g0.03"])
        self.assertTrue(tied["tied"])
        self.assertEqual(result["screen_selection"], report["selected"])
        self.assertEqual(result["screen_ranked_eligible_gates"], report["ranked_eligible_gates"])
        self.assertEqual((matrix, report, config), before)
        result["screen_selection"]["gate"] = 100
        self.assertEqual(report["selected"]["gate"], .1)
        self.assertFalse(result["repeated_doses_are_independent_fidelity_observations"])

    def test_missing_cells_failed_baseline_and_partial_coverage_rejected(self):
        for key in ("missing", "baseline", "examples", "n", "nll", "identity"):
            with self.subTest(key=key):
                matrix, report, config = fixtures()
                summary = report["cells"]["XG_g0.03"]
                if key == "missing":
                    del report["cells"]["XR_g0.40"]
                elif key == "baseline":
                    report["cells"]["X0"]["status"] = "FAILED"
                elif key == "examples":
                    summary["diagnostics"]["examples"] = 127
                elif key == "n":
                    summary["paired_statistics"]["overall"]["n"] = 127
                else:
                    summary["paired_statistics"]["overall"]["nll" if key == "nll" else "output_identity"]["pairs"] = 127
                with self.assertRaises(ValueError):
                    core.analyze(matrix, report, config)

    def test_malformed_or_inconsistent_metrics_rejected(self):
        mutations = [
            lambda s: s.update(accuracy=float("nan")),
            lambda s: s.update(accuracy=True),
            lambda s: s["diagnostics"].update(mean_kl_X0_to_cell=-1),
            lambda s: s["diagnostics"].update(mean_gold_nll=float("inf")),
            lambda s: s["diagnostics"].update(invalid_outputs=1),
            lambda s: s["diagnostics"].update(mean_wall_seconds=3),
            lambda s: s["paired_statistics"]["overall"].update(paired_wins=4),
            lambda s: s["paired_statistics"]["overall"]["nll"].update(candidate_minus_baseline=.1),
            lambda s: s["paired_statistics"]["overall"]["output_identity"].update(rate=.9),
            lambda s: s["diagnostics"]["operator_inventory"].update(added_parameter_count=None),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                matrix, report, config = fixtures()
                mutate(report["cells"]["XG_g0.03"])
                with self.assertRaises(ValueError):
                    core.analyze(matrix, report, config)

    def test_missing_matrix_stratum_and_changed_mapping_rejected(self):
        for target in ("stratum", "mapping", "pair", "incomplete_matrix"):
            with self.subTest(target=target):
                matrix, report, config = fixtures()
                if target == "stratum":
                    del matrix["pairs"][-1]["objectives"]["B3"]["heldout"]["delta"]["special_tokens"]
                elif target == "mapping":
                    config["upstream_export"]["source_objectives"]["XG"] = "BR"
                elif target == "pair":
                    config["selected_pair"]["supa_layer"] = 12
                else:
                    matrix["pairs"].pop()
                with self.assertRaises(ValueError):
                    core.analyze(matrix, report, config)


if __name__ == "__main__":
    unittest.main()
