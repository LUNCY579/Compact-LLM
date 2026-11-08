from copy import deepcopy

import pytest

from experiments.reasoning_posttraining_v1 import evaluation_stream as stream


class Tokenizer:
    def decode(self, ids, **kw):
        return " ".join(str(x) for x in ids)


def rows():
    return [{"id": f"r{i}", "parent_id": f"p{i}", "reference": "7", "family": "fixture",
             "ids": [3, 7, 2], "prompt_ids": [3], "labels": [-100, 7, 2]} for i in range(4)]


def context(data):
    return stream.context_for({"source": "fixture", "checkout": "fixed"}, "bank", data,
        max_sequence=2048, max_new=8, serialization="single_turn", cache_policy="release_unused_between_cases")


def record(row):
    return {"id": row["id"], "parent_id": row["parent_id"], "family": row["family"],
            "reference": row["reference"], "prediction": "7", "generated_ids_including_terminal": [7, 2],
            "generated_tokens_including_terminal": 2, "terminated": True, "truncated": False,
            "canonical_training_target_match": True, "strict_target_text_match": True,
            "response_ce_sum": 1., "response_ce_targets": 2}


def evaluator(abort_after=None, seen=None):
    def evaluate(_model, data, _tok, _eos, _label, reserve, guard, *, max_new, on_record, **kw):
        for i, row in enumerate(data):
            reserve({"evaluation_and_qualification_model_calls": 1, "generated_tokens_all_candidates": max_new})
            if seen is not None:
                seen.append(row["id"])
            on_record(record(row))
            if i + 1 == abort_after:
                raise RuntimeError("fixture resource refusal")
    return evaluate


def run(directory, *, execute=None, resume=None, charges=None, ctx=None):
    data = rows()
    return stream.evaluate_streamed(None, data, Tokenizer(), [2], "fixture",
        charges.append if charges is not None else lambda _: None, lambda: {},
        directory=directory, context=ctx or context(data), evaluator=execute or evaluator(), resume=resume)


def test_refusal_preserves_completed_cases_and_resume_only_evaluates_the_suffix(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    with pytest.raises(RuntimeError, match="resource refusal"):
        run(first, execute=evaluator(abort_after=2))
    assert not (first / "COMPLETE.json").exists()
    assert len(stream.read_prefix(first, context(rows()), rows(), Tokenizer(), [2])) == 2
    before = {p.name: p.read_bytes() for p in (first / "records").iterdir()}
    seen, charges = [], []
    resumed = run(second, resume=first, execute=evaluator(seen=seen), charges=charges)
    full = run(tmp_path / "full")
    assert seen == ["r2", "r3"]
    assert resumed["records"] == full["records"]
    assert resumed["response_ce"] == full["response_ce"]
    assert resumed["reused_predictions"] == resumed["fresh_predictions"] == 2
    assert sum(c.get("evaluation_and_qualification_model_calls", 0) for c in charges) == 2
    assert resumed["generation_allowance_reserved"] == 16
    assert resumed["generation_allowance_inherited"] == 16
    assert {p.name: p.read_bytes() for p in (first / "records").iterdir()} == before


def test_a_complete_prefix_needs_no_generation(tmp_path):
    run(tmp_path / "complete")
    def forbidden(*a, **k):
        raise AssertionError("must not rerun finished rows")
    got = run(tmp_path / "reuse", resume=tmp_path / "complete", execute=forbidden)
    assert got["fresh_predictions"] == 0 and got["reused_predictions"] == 4


def test_resume_refuses_changed_binding_before_creating_output(tmp_path):
    run(tmp_path / "first")
    changed = deepcopy(context(rows()))
    changed["binding"]["checkout"] = "changed"
    with pytest.raises(ValueError, match="identical source"):
        run(tmp_path / "second", resume=tmp_path / "first", ctx=changed)
    assert not (tmp_path / "second").exists()


def test_tampering_or_gaps_do_not_become_a_silent_partial_prefix(tmp_path):
    run(tmp_path / "first")
    path = tmp_path / "first/records/000000.json"
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"prediction":"7"', b'"prediction":"8"'))
    with pytest.raises(ValueError):
        stream.read_prefix(tmp_path / "first", context(rows()), rows(), Tokenizer(), [2])
    path.write_bytes(original)
    (tmp_path / "first/records/000001.json").unlink()
    with pytest.raises(ValueError, match="missing or reordered"):
        stream.read_prefix(tmp_path / "first", context(rows()), rows(), Tokenizer(), [2])


@pytest.mark.parametrize("change", [
    {"canonical_training_target_match": False}, {"prediction": "8"},
    {"response_ce_targets": 99}, {"response_ce_sum": float("nan")},
    {"terminated": False, "truncated": True}, {"generated_ids_including_terminal": [2, 7]},
])
def test_completed_record_validates_tokens_text_scores_and_ce(change):
    with pytest.raises(ValueError):
        stream.validate_record({**record(rows()[0]), **change}, rows()[0], 8, Tokenizer(), [2])


def test_native_evaluator_commits_before_a_later_guard_refusal(tmp_path, monkeypatch):
    import torch
    from experiments.reasoning_posttraining_v1 import native_runtime
    from experiments.reasoning_posttraining_v1.tests.test_arm_integration import EOSModel, compiled
    from experiments.reasoning_posttraining_v1.tests.test_data_contract import Tokenizer as NativeFixtureTokenizer
    data = compiled()
    data += [{**data[0], "id": "other", "parent_id": "other"}]
    monkeypatch.setattr(native_runtime, "hidden_states", lambda *args: torch.zeros(1))
    monkeypatch.setattr(native_runtime, "chunked_sum", lambda *a, **k:
                        (torch.tensor(0.), int((a[2][:, 1:] != -100).sum())))
    count = 0
    def guard():
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("fixture RAM refusal")
        return {"fixture": True}
    ctx = context(data)
    with pytest.raises(RuntimeError, match="RAM refusal"):
        stream.evaluate_streamed(EOSModel(), data, NativeFixtureTokenizer(), [1], "fixture",
            lambda _: None, guard, directory=tmp_path / "native", context=ctx,
            evaluator=native_runtime.evaluate)
    assert len(stream.read_prefix(tmp_path / "native", ctx, data, NativeFixtureTokenizer(), [1])) == 1
