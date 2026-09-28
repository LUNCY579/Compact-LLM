import pytest
from experiments.reasoning_posttraining_v1.preflight import LIMITS, build_plan
from experiments.reasoning_posttraining_v1.overlap import keys


def small_input():
    rows = [{"id": str(i), "parent_id": str(i), "source_split": "train",
             "ids": [5, 6, 7, 2], "labels": [-100, -100, 7, 2], "prompt_ids": [5, 6]} for i in range(32)]
    return {"microfit_source": "forge_v1_primitives", "selected_worlds": 32,
            "selected_ids": [r["id"] for r in rows]}, rows


def test_budget_request_keeps_world_and_exposure_counts_separate():
    contract, rows = small_input()
    p = build_plan(contract, rows)
    assert p["unique_worlds"] == 32 and p["presentations"] == 256
    assert p["processed_tokens"] == 1024 and p["supervised_tokens"] == 512
    assert p["processed_tokens"] * 2 + 4096 < LIMITS["processed_training_tokens_all_attempts"]
    assert p["quality_claim"] is False and p["native_runner_status"].endswith("REQUIRED")


@pytest.mark.parametrize("bad", ["split", "order", "length", "world"])
def test_budget_request_refuses_changed_input(bad):
    contract, rows = small_input()
    if bad == "split": rows[0]["source_split"] = "validation"
    if bad == "order": rows.reverse()
    if bad == "length": rows[0]["ids"] = [1] * 2049
    if bad == "world": rows[0]["parent_id"] = rows[1]["parent_id"]
    with pytest.raises(ValueError): build_plan(contract, rows)


def test_overlap_keys_distinguish_normalization_and_digit_blinding():
    def row(s): return {"messages": [{"role": "user", "content": s}]}
    a, b, c = keys(row("Count\u00a0Ａ 12")), keys(row("count a 12")), keys(row("count a 34"))
    assert a["exact_text"] != b["exact_text"]
    assert a["normalized_text"] == b["normalized_text"] != c["normalized_text"]
    assert a["digit_blinded_text"] == c["digit_blinded_text"]
