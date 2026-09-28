"""Fixture gates for the DEV bank, strata, positive control and shortcut baselines.

Hermetic: the pure scoring functions need nothing, and the bank/solver paths run against
a stub solver. No release tree, no model, no CUDA.
"""
import json
from pathlib import Path

import pytest

from experiments.reasoning_posttraining_v1 import instrument as ins

PROMPT = (
    "RULES\nThis is a fictional register.\n\n"
    "RECORDS\n"
    "[R1] Verified register: item ItemAAA; field keeper; value FinnLake.\n"
    "[R2] Verified register: item ItemBBB; field keeper; value VeraHill.\n"
    "Opening: account StockZZZ held 100 units.\n"
    "Entry: receipt ReceiptQ; account StockZZZ; delta +5; state POSTED.\n"
    "Entry: receipt ReceiptR; account StockZZZ; delta -30; state POSTED.\n\n"
    "END RECORDS\n\n"
    "QUESTION\nCompute the balance of account StockZZZ. Return one integer or UNKNOWN."
)


# ---------------------------------------------------------------- strata

def test_the_four_sentinels_are_distinct_and_documented():
    assert set(ins.SENTINELS) == {"UNKNOWN", "IMPOSSIBLE", "UNREACHABLE", "INCONSISTENT"}
    assert all(isinstance(v, str) and v for v in ins.SENTINELS.values())
    for name in ins.SENTINELS:
        assert ins.stratum(name) == name
        assert ins.stratum("  " + name + " ") == name


def test_verdicts_are_answers_not_abstentions():
    for verdict in ("BOTH", "SUPPORTED", "REFUTED"):
        assert ins.stratum(verdict) == "ANSWER", verdict
    assert ins.stratum("145") == "ANSWER"
    assert ins.stratum("RoomDMKTM") == "ANSWER"
    # A sentinel embedded in a longer answer is not an abstention either.
    assert ins.stratum("UNKNOWN region") == "ANSWER"


# ---------------------------------------------------------------- scoring

def test_strict_and_tolerant_are_separately_observable():
    got = ins.score("145", "145")
    assert got["strict"] and got["tolerant"] and not got["presentation_only"]
    got = ins.score("145.", "145")
    assert not got["strict"] and got["tolerant"] and got["presentation_only"]
    got = ins.score("146", "145")
    assert not got["strict"] and not got["tolerant"]


def test_tolerant_preserves_case_so_it_cannot_manufacture_credit():
    # Identifiers are case-bearing; lowercasing would invent correctness.
    assert not ins.score("roomdmktm", "RoomDMKTM")["tolerant"]
    assert ins.score(' "RoomDMKTM" ', "RoomDMKTM")["tolerant"]


def test_whitespace_normalization_only():
    assert ins.canonical("  a   b \n c ") == "a b c"
    assert ins.tolerant("a b.") == "a b"
    assert ins.tolerant("a b?!") == "a b"


def test_score_reports_both_sentinel_sides():
    got = ins.score("UNKNOWN", "145")
    assert got["predicted_sentinel"] and not got["reference_sentinel"]
    got = ins.score("145", "UNKNOWN")
    assert not got["predicted_sentinel"] and got["reference_sentinel"]


# ---------------------------------------------------------------- summary

def bank_fixture():
    return [
        {"id": "a", "family": "signed_ledger", "reference": "145", "stratum": "ANSWER"},
        {"id": "b", "family": "signed_ledger", "reference": "211", "stratum": "ANSWER"},
        {"id": "c", "family": "signed_ledger", "reference": "UNKNOWN", "stratum": "UNKNOWN"},
        {"id": "d", "family": "deduction", "reference": "BOTH", "stratum": "ANSWER"},
    ]


def test_summary_denominators_are_explicit_and_strata_are_not_pooled():
    s = ins.summarize(bank_fixture(), ["145", "999", "UNKNOWN", "BOTH"])
    assert s["rows"] == 4 and s["strict_correct"] == 3
    assert s["per_stratum"]["ANSWER"] == {"n": 3, "strict": 2, "tolerant": 2}
    assert s["per_stratum"]["UNKNOWN"] == {"n": 1, "strict": 1, "tolerant": 1}
    assert s["per_family"]["deduction"] == {"n": 1, "strict": 1}
    assert s["quality_claim"] is False


def test_summary_separates_the_two_abstention_error_directions():
    # answers UNKNOWN on an answerable world, and answers on an unanswerable one
    s = ins.summarize(bank_fixture(), ["UNKNOWN", "UNKNOWN", "7", "UNKNOWN"])
    assert s["false_abstention_on_answerable"] == 3
    assert s["unsupported_answer_on_unanswerable"] == 1
    assert s["strict_correct"] == 0


def test_summary_refuses_mismatched_prediction_count():
    with pytest.raises(ValueError):
        ins.summarize(bank_fixture(), ["145"])


# ---------------------------------------------------------------- B-1 collision

def test_sentinel_collisions_are_counted_apart_from_specific_ones():
    bank = bank_fixture()
    # every error is the sentinel, which is also a sibling gold in signed_ledger
    got = ins.sibling_collision(bank, ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"])
    assert got["errors"] == 3
    assert got["specific_sibling_collisions"] == 0
    assert got["collision_rate"] == 0.0
    assert got["sentinel_collisions_excluded"] == 2       # the two ANSWER signed_ledger rows
    # predicting a sibling's SPECIFIC gold is the real signal
    got = ins.sibling_collision(bank, ["211", "145", "UNKNOWN", "BOTH"])
    assert got["errors"] == 2 and got["specific_sibling_collisions"] == 2
    assert got["collision_rate"] == 1.0


def test_collision_never_publishes_an_analytic_chance_level():
    got = ins.sibling_collision(bank_fixture(), ["211", "145", "UNKNOWN", "BOTH"])
    assert "chance_level_estimate" not in got
    assert got["no_analytic_chance_level"]
    assert got["baseline_required"]


def test_cross_family_match_is_not_a_collision():
    bank = bank_fixture()
    # "BOTH" is a deduction gold, predicted on a signed_ledger world
    got = ins.sibling_collision(bank, ["BOTH", "211", "UNKNOWN", "BOTH"])
    assert got["specific_sibling_collisions"] == 0


# ---------------------------------------------------------------- shortcuts

class StubSolver:
    SOLVERS = {"signed_ledger": None}

    @staticmethod
    def parts(prompt):
        body, question = prompt.split("QUESTION", 1)
        return body, question

    @staticmethod
    def solve_prompt(family, prompt):
        return "75"          # 100 + 5 - 30, the genuinely correct answer

    @staticmethod
    def validate_row(row, meta):
        return None


def shortcut_bank():
    return [{"id": "a", "family": "signed_ledger", "reference": "75",
             "stratum": "ANSWER", "prompt": [{"role": "system", "content": "s"},
                                             {"role": "user", "content": PROMPT}]}]


def test_shortcuts_never_read_a_reference():
    bank = shortcut_bank()
    preds = ins.shortcut_predictions(bank, StubSolver, train_majority="UNKNOWN")
    # remove the reference entirely; predictions must be unchanged
    stripped = [{k: v for k, v in bank[0].items() if k != "reference"}]
    again = ins.shortcut_predictions(stripped, StubSolver, train_majority="UNKNOWN")
    assert preds == again


def test_cue_copy_excludes_the_sentinel_the_question_leaks():
    bank = shortcut_bank()
    preds = ins.shortcut_predictions(bank, StubSolver, train_majority="UNKNOWN")
    # the question ends "...or UNKNOWN.", so the raw last identifier would be UNKNOWN
    assert preds["cue_copy"][0] != "UNKNOWN"
    assert preds["copy_question_sentinel"][0] == "UNKNOWN"


def test_last_delta_only_reproduces_the_rrp001_b1_arithmetic():
    preds = ins.shortcut_predictions(shortcut_bank(), StubSolver, train_majority="UNKNOWN")
    assert preds["last_delta_only"][0] == "70"           # 100 + (-30), not the correct 75
    # last_mentioned must be some identifier from the RECORDS body, never the gold
    last = preds["last_mentioned"][0]
    assert last and last != "75" and last in PROMPT.split("QUESTION")[0]


def test_constant_baselines_are_constant():
    bank = shortcut_bank() * 3
    preds = ins.shortcut_predictions(bank, StubSolver, train_majority="ZZZ")
    assert preds["constant_unknown"] == ["UNKNOWN"] * 3
    assert preds["constant_train_majority"] == ["ZZZ"] * 3


# ---------------------------------------------------------------- positive control

def test_positive_control_passes_only_when_the_solver_reproduces_every_gold():
    good = ins.positive_control(shortcut_bank(), StubSolver)
    assert good["status"] == "PASS" and good["solved"] == 1 and good["failed"] == 0
    assert good["model_calls"] == 0

    broken = [dict(shortcut_bank()[0], reference="999")]
    bad = ins.positive_control(broken, StubSolver)
    assert bad["status"] == "FAIL" and bad["failed"] == 1
    assert bad["failures"][0]["solver"] == "75"


def test_positive_control_message_names_the_broken_instrument_rule():
    got = ins.positive_control(shortcut_bank(), StubSolver)
    assert "broken instrument is not a model failure" in got["meaning"]


# ---------------------------------------------------------------- bank assembly

class BankSolver(StubSolver):
    """Serves a tiny two-family release out of memory."""

    rows = {f"w{i}": {"id": f"w{i}", "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": "75"}]} for i in range(6)}
    metas = {f"w{i}": {"id": f"w{i}", "family": "signed_ledger" if i < 3 else "deduction",
                       "answer_category": "exact", "reference_answer": "75",
                       "prompt_sha256": "p" * 64, "case_sha256": f"c{i}",
                       "actual_user_words": 10, "core_record_count": 2,
                       "group_id": f"w{i}"} for i in range(6)}

    @classmethod
    def iter_jsonl(cls, path):
        source = cls.metas if "metadata" in str(path) else cls.rows
        return iter(list(source.values()))


def test_bank_selection_is_reproducible_and_respects_exclusions():
    a = ins.build_bank(Path("."), "validation", families=("signed_ledger",), per_family=2,
                       solver=BankSolver)
    b = ins.build_bank(Path("."), "validation", families=("signed_ledger",), per_family=2,
                       solver=BankSolver)
    assert [r["id"] for r in a] == [r["id"] for r in b]     # digest-stable, order-independent
    assert len(a) == 2 and all(r["family"] == "signed_ledger" for r in a)

    excluded = frozenset([a[0]["id"]])
    c = ins.build_bank(Path("."), "validation", families=("signed_ledger",), per_family=2,
                       solver=BankSolver, exclude_ids=excluded)
    assert excluded.isdisjoint({r["id"] for r in c})


def test_bank_refuses_an_insufficient_pool():
    with pytest.raises(ValueError, match="insufficient"):
        ins.build_bank(Path("."), "validation", families=("signed_ledger",), per_family=99,
                       solver=BankSolver)


def test_bank_rows_carry_no_gold_inside_the_model_visible_prompt():
    bank = ins.build_bank(Path("."), "validation", families=("signed_ledger",), per_family=2,
                          solver=BankSolver)
    for row in bank:
        assert [m["role"] for m in row["prompt"]] == ["system", "user"]
        blob = json.dumps(row["prompt"])
        assert row["reference"] not in blob or row["reference"] == "75"  # 75 is in-prompt arithmetic only
        assert "reference_answer" not in blob
        assert row["stratum"] in ("ANSWER", *ins.SENTINELS)


# ------------------------------------------------- evaluation-window qualification

def test_window_qualification_rejects_windows_outside_its_remit():
    """Entry validation, and it also proves every lazily-imported name resolves.

    A NameError inside this function cost a real supervised launch once, because
    instrument.py must stay torch-free at module level and parameter_digest lives in a
    torch module. These calls reach past the imports before raising.
    """
    for bad in (2048, 1024, 16385, 0, -1, 4096.0, "4096", None):
        with pytest.raises(ValueError, match="evaluation-window qualification covers"):
            ins.qualify_evaluation_window(None, None, bad, lambda c: None, lambda: None)


def test_window_qualification_resolves_its_torch_dependencies():
    """Reach the tokenizer check, i.e. past `import torch` and past parameter_digest."""
    import torch

    tiny = torch.nn.Parameter(torch.zeros(2))

    class NoParams:
        @staticmethod
        def named_parameters():
            return iter([("w", tiny)])

        @staticmethod
        def parameters():
            return iter([tiny])

    class BadTokenizer:
        eos_token_id = 151645

        @staticmethod
        def encode(text, add_special_tokens=False):
            return []          # empty fixture tokens -> the documented refusal

    with pytest.raises(ValueError, match="shape fixture token identity invalid"):
        ins.qualify_evaluation_window(NoParams, BadTokenizer, 4096,
                                      lambda c: None, lambda: None)


def test_instrument_module_does_not_import_torch_at_module_level():
    """The foreground controller imports this module; torch there costs the child RAM."""
    import subprocess
    import sys

    code = ("import experiments.reasoning_posttraining_v1.instrument, sys; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(__import__("pathlib").Path(ins.__file__).parents[2]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout
