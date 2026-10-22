import copy
import pytest
import torch

from experiments.rineya_length_generalization_regularization_v1.fixtures import ByteTokenizer, TinyLM
from experiments.rineya_length_generalization_regularization_v1.data import (
    make_bank, audit_disjoint, validate_cases, read_bank, write_bank)
from experiments.rineya_length_generalization_regularization_v1.evaluation import collect, compare


def bank(split="dev", seed=101):
    return make_bank(ByteTokenizer(), split=split, seed=seed, parents=2,
                     lengths=(256, 320), depths=(0.1, 0.9), fixture=True)


def test_exact_lengths_paired_parents_disjoint():
    rows = bank()
    assert len(rows) == 32
    assert all(len(r["prompt_ids"]) == r["length"] for r in rows)
    for parent in {r["parent_id"] for r in rows}:
        same = [r for r in rows if r["parent_id"] == parent]
        assert len(same) == 4
        assert len({r["answer"] for r in same}) == 1
    assert rows == bank()
    audit_disjoint(rows, bank("train"), bank("confirmation"))
    copy_row = copy.deepcopy(rows[0]); copy_row["split"] = "train"
    with pytest.raises(ValueError):
        audit_disjoint(rows, [copy_row])
    copy_row["parent_id"] = "unrelated"
    with pytest.raises(ValueError):
        audit_disjoint(rows, [copy_row])


@pytest.mark.parametrize("field,value", [
    ("length", 1), ("prompt_ids", []), ("prompt_ids", [-1]), ("target_ids", [True]),
    ("split", "test"), ("role", "magic"), ("answerable", 1), ("fixture", "false"),
    ("depth", float("nan")), ("answer", ""), ("id", 1)])
def test_invalid_case_schema(field, value):
    row = bank()[0]
    row[field] = value
    with pytest.raises(ValueError):
        validate_cases([row])


def test_no_truncation_or_overwrite(tmp_path):
    with pytest.raises(ValueError):
        make_bank(ByteTokenizer(), split="dev", seed=1, lengths=(10,))
    path = tmp_path / "bank.jsonl"
    rows = bank()
    write_bank(path, rows)
    assert read_bank(path) == rows
    with pytest.raises(FileExistsError):
        write_bank(path, rows)
    with pytest.raises(ValueError):
        read_bank(path, split="train")
    with pytest.raises(ValueError):
        validate_cases(rows, vocabulary=5)


def test_collection_limits_and_mode_restoration():
    model, tok, cases = TinyLM(), ByteTokenizer(), bank()[:2]
    model.train()
    model.model.layers[0].eval()
    initial = [(m, m.training) for m in model.modules()]
    records = collect(model, tok, cases, seed=1, context_limit=512, max_new_tokens=3)
    assert len(records) == 2 and all(not r["truncated"] for r in records)
    assert all(r["generated_tokens"] <= 3 for r in records)
    assert all(m.training == mode for m, mode in initial)
    with pytest.raises(ValueError):
        collect(model, tok, cases, seed=1, context_limit=cases[0]["length"], max_new_tokens=3)


def predictions(seeds=(1, 2, 3), correct=False):
    result = []
    for seed in seeds:
        for row in bank():
            from experiments.rineya_length_generalization_regularization_v1.data import digest
            result.append(dict(row, seed=seed, correct=correct, prediction="UNKNOWN", abstained=True,
                               truncated=False, prompt_sha256=digest(row["prompt_ids"])))
    return result


def test_comparison_preserves_clusters_and_rejects_promotion():
    before, after = predictions(), predictions(correct=True)
    result = compare(before, after, resamples=100)
    assert result["paired_rows"] == 96 and result["unique_parents"] == 8
    assert result["endpoints"]["long_context"]["delta"] == 1
    assert result["endpoints"]["long_context"]["ci95"] == [1, 1]
    assert result["status"] == "NOT_PROMOTABLE"
    assert "fixture_evidence" in result["reasons"] and "missing_acquisition" in result["reasons"]
    assert result == compare(before, after, resamples=100)


def test_comparison_strict_pairs_and_complete_seed_grid():
    before, after = predictions(), predictions(correct=True)
    with pytest.raises(ValueError):
        compare(before, after[:-1], resamples=100)
    with pytest.raises(ValueError):
        compare(before + before[:1], after, resamples=100)
    changed = copy.deepcopy(after); changed[0]["prompt_sha256"] = "changed"
    with pytest.raises(ValueError):
        compare(before, changed, resamples=100)
    with pytest.raises(ValueError):
        compare(before[:-1], after[:-1], resamples=100)
    changed = copy.deepcopy(after); changed[0]["truncated"] = True
    with pytest.raises(ValueError):
        compare(before, changed, resamples=100)
    result = compare(predictions((1,)), predictions((1,), True), resamples=100)
    assert "fewer_than_three_seeds" in result["reasons"]
