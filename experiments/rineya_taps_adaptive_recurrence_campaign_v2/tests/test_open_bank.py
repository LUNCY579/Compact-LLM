"""Independent apparatus checks; no tokenizer/model or historical task imports."""
from collections import Counter
from copy import deepcopy
import re

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import open_bank as bank


@pytest.fixture
def rows():
    return bank.generate("CPU_TEST_FIXTURE_ONLY", 17)


def test_all_gold_answers_follow_prompt_parser_and_strict_contract(rows):
    assert len(rows) == 24
    for row in rows:
        prediction = bank.solve_prompt(row["prompt"])
        assert prediction == row["gold"]
        assert bank.score_response(prediction, row) == {"correct": True, "format_valid": True}


def test_partition_balances_and_nonce_parent_provenance(rows):
    assert Counter(x["split"] for x in rows) == {"DEVELOPMENT": 16, "OPEN_VALIDATION": 8}
    assert Counter(x["category"] for x in rows) == {
        "signed_arithmetic": 6, "deduction_composition": 6,
        "program_state": 6, "document_reconciliation": 6,
    }
    assert Counter(x["gold"] for x in rows if x["category"] == "deduction_composition") == {"YES": 3, "NO": 3}
    documents = [x for x in rows if x["category"] == "document_reconciliation"]
    assert sum(x["gold"] == "UNKNOWN" for x in documents) == 3
    assert len({x["parent_id"] for x in rows}) == 24
    assert all(x["provenance"]["historical_parents"] == [] for x in rows)
    dev = {x["generator_family"] for x in rows if x["split"] == "DEVELOPMENT"}
    validation = {x["generator_family"] for x in rows if x["split"] == "OPEN_VALIDATION"}
    assert dev.isdisjoint(validation)


@pytest.mark.parametrize("text", [
    "The answer is 12", "not 12", "12 13", "YES NO", "NO YES", "UNKNOWN.",
    "12.", "+12", "012", "-0", "", "NaN", "true", '{"value":12}', "12,000",
])
def test_no_forgiving_number_choice_keyword_or_json_extraction(text):
    assert bank.normalize_output(text) is None


def test_normalization_is_gold_independent_and_unknown_is_not_blanket_credit(rows):
    assert bank.normalize_output("  yes\n") == "YES"
    assert bank.normalize_output(" -36 ") == "-36"
    assert bank.normalize_output("unknown") == "UNKNOWN"
    answerable = next(x for x in rows if re.fullmatch(r"-?\d+", x["gold"]))
    assert bank.score_response("UNKNOWN", answerable)["correct"] is False
    assert bank.score_response("UNKNOWN", answerable)["format_valid"] is False
    with pytest.raises(TypeError):
        bank.solve_prompt({"prompt": "ignored", "gold": "SECRET"})


def test_program_parser_reorders_steps_and_reads_current_state():
    prefix = "Registers: AAA=2,BBB=3,CCC=4. Execute numbered steps in increasing order; each reads current values. Steps: "
    prompt = prefix + "4 AAA=CCC-2;2 BBB=AAA-4;1 AAA=BBB+30;3 CCC=BBB+7.\nQuestion: AAA? Reply one integer."
    assert bank.solve_prompt(prompt) == "34"
    assert bank.solve_prompt(prompt.replace("AAA? Reply", "BBB? Reply")) == "29"


def test_latest_document_solver_ignores_old_and_draft_values_and_detects_conflict():
    prefix = "Use only the latest dated approved values per entity; agree=>value, absent or conflict=>UNKNOWN. Records: "
    suffix = ".\nQuestion: AAA value? Reply one integer or UNKNOWN."
    assert bank.solve_prompt(prefix + "AAA approved d1 9;AAA approved d2 17;AAA draft d3 99" + suffix) == "17"
    assert bank.solve_prompt(prefix + "AAA approved d2 17;AAA approved d2 18" + suffix) == "UNKNOWN"
    assert bank.solve_prompt(prefix + "AAA draft d2 17;BBB approved d2 18" + suffix) == "UNKNOWN"


def test_conjunctive_rule_requires_all_inputs():
    prefix = "Rules derive only their right side when all left symbols are known. Rules: AAA&BBB>CCC. Facts: "
    suffix = ".\nQuestion: derive CCC? Reply YES or NO."
    assert bank.solve_prompt(prefix + "AAA" + suffix) == "NO"
    assert bank.solve_prompt(prefix + "AAA,BBB" + suffix) == "YES"


def test_floors_recomputed_from_prompt_only_and_query_blind_independence(rows):
    report, _ = bank.audit(rows)
    assert report["positive_control_correct"] == 24 and report["blind_global_gate"]
    for attacker, count in report["blind_correct_counts"].items():
        independently_counted = sum(
            bank.score_response(bank.blind_predictions(row["prompt"])[attacker], row)["correct"]
            for row in rows
        )
        assert independently_counted == count
    for row in rows:
        context, question = row["prompt"].split("\nQuestion: ")
        entities = [x for x in re.findall(r"\b[A-Z]{3}\b", context) if x != "YES"]
        alternate = re.sub(r"[A-Z]{3}", entities[-1], question, count=1)
        blind_first = bank.blind_predictions(row["prompt"])["query_blind_first_entity_solver"]
        blind_alternate = bank.blind_predictions(context + "\nQuestion: " + alternate)["query_blind_first_entity_solver"]
        assert blind_first == blind_alternate


def test_no_target_marker_and_shallow_surface_shape_is_unqualified(rows):
    assert all("target" not in x["prompt"].lower() for x in rows)
    assert max(len(x["prompt"].encode("ascii")) for x in rows) < 400
    assert all(x["shape"]["target_max_input_tokens"] == 350 for x in rows)
    assert all(x["shape"]["native_tokenizer_binding"].startswith("NOT_RUN") for x in rows)
    assert all(x["shape"]["max_output_tokens"] == 8 for x in rows)


def test_duplicate_rows_fail_closed_and_generation_is_reproducible(rows):
    assert rows == bank.generate("CPU_TEST_FIXTURE_ONLY", 17)
    duplicate = deepcopy(rows)
    duplicate[-1] = deepcopy(duplicate[0])
    with pytest.raises(AssertionError, match="duplicate or split collision"):
        bank.audit(duplicate)
