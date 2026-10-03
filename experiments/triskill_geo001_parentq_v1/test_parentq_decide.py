"""CPU tests of the PARENTQ decision rule: every rung can pass and fail on achievable data."""
import json
import random
import unittest
from pathlib import Path

import numpy as np

import geo_frozen
import parentq_decide as d
import parentq_rows
import parentq_worker

HERE = Path(__file__).resolve().parent
benchmark = geo_frozen.load("benchmark")
BANK = parentq_rows.build_bank(benchmark)
CANARY = json.loads((HERE / parentq_worker.CANARY_JSON).read_text(encoding="utf-8"))["rows"]
TABLES = d.gate_tables(benchmark)
SOURCE = {"BASE": "BASE", "T": "SPARSE", "H": "H25"}


def t_reply(row, right):
    return row.expected if right else "{}"


def h_reply(row, mode):
    """mode: 'right' (gold answer or the abstention phrase), 'answer' (a number), 'abstain'."""
    if mode == "abstain":
        return benchmark.ABSTAIN_PHRASE
    if mode == "right":
        return row.expected if row.expected != benchmark.UNKNOWN else benchmark.ABSTAIN_PHRASE
    return row.expected if row.expected != benchmark.UNKNOWN else "41"


def outputs(*, t_right=lambda row, i: True, base_t_right=lambda row, i: False,
            h_on=lambda row, i: "right", h_off=lambda row, i: "right",
            base_on=lambda row, i: "answer", base_off=lambda row, i: "answer", canary=None):
    out = {arm: {row["row_id"]: row["stored"][(canary or SOURCE)[arm]] for row in CANARY} for arm in ("BASE", "T", "H")}
    for i, row in enumerate(BANK["T"]):
        out["T"][row.row_id] = t_reply(row, t_right(row, i))
        out["BASE"][row.row_id] = t_reply(row, base_t_right(row, i))
    for i, row in enumerate(BANK["H_ON"]):
        out["H"][row.row_id] = h_reply(row, h_on(row, i))
        out["BASE"][row.row_id] = h_reply(row, base_on(row, i))
    for i, row in enumerate(BANK["H_OFF"]):
        out["H"][row.row_id] = h_reply(row, h_off(row, i))
        out["BASE"][row.row_id] = h_reply(row, base_off(row, i))
    return out


def decide(out, mechanics=None):
    return d.decide(benchmark, BANK, CANARY, out, {"padding_parity_ok": True} if mechanics is None else mechanics,
                    tables=TABLES)


class GateTableTests(unittest.TestCase):
    def test_tables_match_direct_simulation_of_the_frozen_gate(self):
        """Simulate whole screens from fixed cell probabilities and score them with the
        frozen parent_gate; the table-based plug-in must agree within Monte Carlo error."""
        rows = benchmark.build_screen()
        groups = benchmark.row_groups(rows)
        rng = random.Random(7)
        pis = {name: np.array(rng.choices(range(1, 6), k=4), dtype=float) for name in (*d.T_SUBTYPES, *d.H_KINDS)}
        pis = {name: values / values.sum() for name, values in pis.items()}
        n, passes = 4000, {"T": 0, "H": 0}
        for _ in range(n):
            scores = {"BASE": {}, "T": {}, "H": {}}
            for group, arm, other in (("T", "T", "H"), ("H", "H", "T")):
                for row in groups[group]:
                    base, parent = d.CELLS[rng.choices(range(4), weights=pis[d.subtype(row)])[0]]
                    scores["BASE"][row.row_id], scores[arm][row.row_id] = {"credited": base}, {"credited": parent}
                    scores[other][row.row_id] = {"credited": False}
            gate = benchmark.parent_gate(rows, scores)
            passes["T"] += gate["T"]["passed"]
            passes["H"] += gate["H"]["passed"]
        for group in ("T", "H"):
            expected = d.plug_in(TABLES[group], pis)
            self.assertLess(abs(passes[group] / n - expected), 3 * (expected * (1 - expected) / n) ** 0.5 + 1e-3, group)

    def test_t_table_ignores_h_cells_and_pins_counts(self):
        self.assertEqual((sum(TABLES["T"]["passed"]), sum(TABLES["H"]["passed"])), (89, 86))
        self.assertEqual(TABLES["T"]["subtypes"], d.T_SUBTYPES)

    def test_predictive_is_the_exact_posterior_mean(self):
        counts = {"ledger_replay:transfer": [2, 3, 0, 1], "ledger_replay:scale": [1, 4, 0, 1],
                  "register_machine:add": [0, 4, 0, 2], "register_machine:conditional": [3, 2, 1, 0],
                  "answerable": [2, 2, 2, 6], "distractor": [3, 5, 0, 2], "cue": [2, 6, 0, 2]}
        alphas = {name: [c + d.PRIOR for c in values] for name, values in counts.items()}
        rng = np.random.default_rng(3)
        draws = {name: rng.dirichlet(alpha, size=300000) for name, alpha in alphas.items()}
        for group in ("T", "H"):
            self.assertAlmostEqual(d.predictive(TABLES[group], alphas), d._draw_pass(TABLES[group], draws).mean(), places=2)

    def test_min_detectable_discordance(self):
        self.assertEqual(d.min_detectable_one_sided_gains(benchmark), 6)


class DecisionLadderTests(unittest.TestCase):
    def test_strong_parents_qualify(self):
        verdict = decide(outputs())
        self.assertEqual(verdict["status"], "PARENTS_QUALIFIED", verdict.get("screen_parent_gate"))
        self.assertGreaterEqual(verdict["screen_parent_gate"]["predictive"], d.QUALIFIED_AT)
        self.assertFalse(verdict["confirmation"])

    def test_h_that_never_abstains_is_not_qualified(self):
        verdict = decide(outputs(h_on=lambda row, i: "answer"))
        self.assertEqual(verdict["status"], "PARENTS_NOT_QUALIFIED:H")

    def test_h_that_always_abstains_is_not_qualified(self):
        verdict = decide(outputs(h_on=lambda row, i: "abstain", base_on=lambda row, i: "right"))
        self.assertEqual(verdict["status"], "PARENTS_NOT_QUALIFIED:H")

    def test_t_no_better_than_base_is_not_qualified(self):
        verdict = decide(outputs(t_right=lambda row, i: False))
        self.assertEqual(verdict["status"], "PARENTS_NOT_QUALIFIED:T")

    def test_middling_parents_are_marginal(self):
        # T right on 2/3 of its rows, H abstaining on about 2/3 of the missing rows: 0.668.
        verdict = decide(outputs(t_right=lambda row, i: i % 3 != 0, h_on=lambda row, i: "right" if i % 3 else "answer"))
        p = verdict["screen_parent_gate"]["predictive"]
        self.assertEqual(verdict["status"], "PARENTS_MARGINAL", p)
        self.assertTrue(d.NOT_QUALIFIED_BELOW <= p < d.QUALIFIED_AT)

    def test_padding_and_canary_rungs(self):
        self.assertEqual(decide(outputs(), {})["status"], "STOPPED_BEFORE:padding")
        self.assertEqual(decide(outputs(), {"padding_parity_ok": False})["status"], "STOP_NATIVE_MECHANICS_PADDING")
        out = outputs()
        del out["H"][CANARY[0]["row_id"]]
        self.assertEqual(decide(out)["status"], "STOPPED_BEFORE:canary:H")
        wrong_t = outputs(canary={"BASE": "BASE", "T": "BASE", "H": "H25"})   # T arm behaving as BASE
        verdict = decide(wrong_t)
        self.assertEqual(verdict["status"], "STOP_ARM_CANARY:T")
        self.assertEqual(verdict["canary"]["T"]["matches_by_source"]["BASE"], 8)
        self.assertNotEqual(verdict["h_contract"]["label"], "Q2_INCOMPLETE")   # H data still usable
        swapped = outputs(canary={"BASE": "BASE", "T": "SPARSE", "H": "BASE"})
        self.assertEqual(decide(swapped)["status"], "STOP_ARM_CANARY:H")

    def test_one_canary_flip_is_tolerated_two_are_not(self):
        out = outputs()
        out["T"][CANARY[0]["row_id"]] = "something else"
        self.assertEqual(decide(out)["status"], "PARENTS_QUALIFIED")
        out["T"][CANARY[1]["row_id"]] = "something else"
        self.assertEqual(decide(out)["status"], "STOP_ARM_CANARY:T")

    def test_missing_q1_unit_is_incomplete(self):
        out = outputs()
        del out["T"][BANK["T"][-1].row_id]
        verdict = decide(out)
        self.assertEqual(verdict["status"], "INCOMPLETE_COVERAGE")
        self.assertEqual(verdict["missing_coverage"], ["T:T"])


class ContractDiagnosticTests(unittest.TestCase):
    def label(self, **policies):
        return decide(outputs(**policies))["h_contract"]["label"]

    def test_each_label_is_reachable(self):
        self.assertEqual(self.label(), "H_ABSTAINS_UNDER_CONTRACT")
        self.assertEqual(self.label(h_on=lambda row, i: "answer"), "NATIVE_PROMPT_ELICITS_H_ABSTENTION")
        self.assertEqual(self.label(h_on=lambda row, i: "answer", base_off=lambda row, i: "right"),
                         "NATIVE_PROMPT_ELICITS_ABSTENTION_NOT_SPECIFIC_TO_H")
        self.assertEqual(self.label(h_on=lambda row, i: "answer", h_off=lambda row, i: "answer"), "H_ABSTENTION_NOT_SHOWN")

    def test_over_abstention_under_native_prompts_is_visible(self):
        verdict = decide(outputs(h_on=lambda row, i: "answer", h_off=lambda row, i: "abstain"))
        rates = verdict["h_contract"]["rates"]["H_off"]
        self.assertEqual(rates["answerable_abstained"], rates["n_answerable"])
        self.assertEqual(rates["answerable_credited"], 0)
        self.assertLess(verdict["h_gate_predictive_under_native_prompts"]["predictive_H"], 0.5)


if __name__ == "__main__":
    unittest.main()
