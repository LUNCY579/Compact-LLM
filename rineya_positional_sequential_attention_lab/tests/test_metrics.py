"""Tests for ``rineya_psa.evaluation``: metrics, the batched runner, preservation suites.

Every model-facing test builds a randomly initialised 2-layer Qwen3 on the CPU. Nothing
here touches a real checkpoint or the GPU -- another process owns the card.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from rineya_psa.evaluation import metrics as M
from rineya_psa.evaluation import preservation as P
from rineya_psa.evaluation import runner as R
from rineya_psa.paths import REPO_ROOT
from rineya_psa.types import BenchmarkItem, TokenizedItem

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


# --------------------------------------------------------------------------------------
# Fixtures: a tiny model and a tokenizer stub with a byte-level vocabulary
# --------------------------------------------------------------------------------------

VOCAB = 64
EOS_ID = 1
PAD_ID = 0


class StubTokenizer:
    """Deterministic character-level tokenizer: ord(c) % (VOCAB - 3) + 3.

    Real tokenizers are avoided so the tests stay hermetic and fast; the runner only needs
    ``apply_chat_template``, ``encode``, ``decode``, and the two special ids.
    """

    chat_template = "stub"
    pad_token_id = PAD_ID
    eos_token_id = EOS_ID
    padding_side = "left"

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert tokenize is False
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(c) % (VOCAB - 3) + 3 for c in str(text)]

    @staticmethod
    def decode(ids, skip_special_tokens=True):
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in (PAD_ID, EOS_ID, 2):
                continue
            out.append(chr(ord("a") + (i - 3) % 26))
        return "".join(out)


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(20260828)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        attn_implementation="sdpa",
    )
    model = Qwen3ForCausalLM(config).eval()
    model.config.use_cache = False
    return model


@pytest.fixture(scope="module")
def tok():
    return StubTokenizer()


def make_item(idx: int, *, answer: str = "ab", n_user: int = 40) -> BenchmarkItem:
    return BenchmarkItem(
        item_id=f"item-{idx}",
        scenario_id=f"scen-{idx // 2}",
        family="exact_retrieval",
        fmt="natural",
        length_bucket="2k",
        target_positions=(0.5,),
        local_position="middle",
        split="calibration",
        system="sys",
        user="u" * n_user + f"{idx}",
        answer=answer,
        answer_aliases=(),
        distractors=(),
        evidence_sentences=(),
        seed=idx,
    )


# --------------------------------------------------------------------------------------
# accuracy / normalisation
# --------------------------------------------------------------------------------------


def test_accuracy_normalises_case_whitespace_and_trailing_punctuation():
    item = BenchmarkItem(
        item_id="i",
        scenario_id="s",
        family="exact_retrieval",
        fmt="natural",
        length_bucket="2k",
        target_positions=(0.5,),
        local_position="middle",
        split="calibration",
        system="",
        user="q",
        answer="Talia Frost",
        answer_aliases=("T. Frost",),
        distractors=(),
        evidence_sentences=(),
        seed=0,
    )
    assert M.accuracy("  talia   frost. ", item)
    assert M.accuracy("T. FROST", item)
    assert not M.accuracy("Talia Frostbite", item)  # no fuzzy matching
    assert not M.accuracy("", item)
    # Strict mode rejects a trailing justification; first_line mode accepts it.
    assert not M.accuracy("Talia Frost\nbecause the appendix says so", item)
    assert M.accuracy("Talia Frost\nbecause the appendix says so", item, first_line=True)


# --------------------------------------------------------------------------------------
# positional shape
# --------------------------------------------------------------------------------------


def test_edge_middle_gap_and_worst_position():
    curve = {0.02: 0.80, 0.50: 0.40, 0.98: 0.90}
    assert M.edge_middle_gap(curve) == pytest.approx((0.80 + 0.90) / 2 - 0.40)
    assert M.worst_position(curve) == pytest.approx(0.40)
    assert M.worst_position_at(curve) == pytest.approx(0.50)
    # Unordered input must give the same answer.
    assert M.edge_middle_gap({0.98: 0.90, 0.50: 0.40, 0.02: 0.80}) == pytest.approx(
        M.edge_middle_gap(curve)
    )


def test_edge_middle_gap_picks_the_grid_point_nearest_the_middle():
    # 0.45 is closer to 0.5 than 0.67 is, even though 0.67 comes later.
    curve = {0.05: 1.0, 0.45: 0.2, 0.67: 0.5, 0.95: 1.0}
    assert M.edge_middle_gap(curve) == pytest.approx(1.0 - 0.2)


def test_position_robustness_index_bounds():
    assert M.position_robustness_index({0.1: 0.5, 0.9: 0.5}) == pytest.approx(1.0)
    assert M.position_robustness_index({0.1: 0.0, 0.9: 1.0}) == pytest.approx(0.0, abs=1e-6)
    # Uniformly zero is flat, not undefined.
    assert M.position_robustness_index({0.1: 0.0, 0.9: 0.0}) == pytest.approx(1.0)
    # A log-probability curve must fail loudly rather than return a ratio above 1.
    with pytest.raises(ValueError, match=r"accuracies in \[0, 1\]"):
        M.position_robustness_index({0.1: -8.5, 0.9: -8.4})


def test_ushape_fit_recovers_a_known_quadratic_exactly():
    positions = np.linspace(0.02, 0.98, 9)
    c = positions - 0.5
    truth = (0.30, -0.10, 1.25)
    acc = truth[0] + truth[1] * c + truth[2] * c**2
    fit = M.ushape_fit(positions, acc)
    assert (fit.beta0, fit.beta1, fit.beta2) == pytest.approx(truth, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.n == 9
    assert fit.se0 == pytest.approx(0.0, abs=1e-7)


def test_ushape_fit_standard_errors_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(7)
    positions = np.linspace(0.02, 0.98, 11)
    c = positions - 0.5
    acc = 0.4 - 0.05 * c + 0.9 * c**2 + rng.normal(0, 0.03, size=c.size)

    fit = M.ushape_fit(positions, acc)
    design = np.column_stack([np.ones_like(c), c, c**2])
    reference = sm.OLS(acc, design).fit()

    assert [fit.beta0, fit.beta1, fit.beta2] == pytest.approx(list(reference.params), rel=1e-9)
    assert [fit.se0, fit.se1, fit.se2] == pytest.approx(list(reference.bse), rel=1e-9)
    assert fit.r_squared == pytest.approx(reference.rsquared, rel=1e-9)


def test_ushape_fit_saturated_design_reports_nan_standard_errors():
    fit = M.ushape_fit([0.1, 0.5, 0.9], [0.2, 0.1, 0.3])
    assert fit.n == 3
    assert all(math.isnan(v) for v in (fit.se0, fit.se1, fit.se2))


def test_ushape_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        M.ushape_fit([0.1, 0.9], [0.2, 0.3])


# --------------------------------------------------------------------------------------
# teacher-forced alignment: the off-by-one
# --------------------------------------------------------------------------------------


def _reference_two_step_nll(model, input_ids, answer_ids):
    """Independent reference: score each answer token with its own forward pass."""
    total = 0.0
    prefix = list(input_ids)
    for gold in answer_ids:
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([prefix])).logits[0, -1]
        total -= float(torch.log_softmax(logits.float(), dim=-1)[gold])
        prefix.append(int(gold))
    return total


def test_gold_nll_alignment_two_token_answer(tiny_model):
    """A one-position shift would still be a finite number; only the value gives it away."""
    input_ids = tuple(range(3, 15))
    answer_ids = (21, 34)
    tok_item = TokenizedItem(
        item_id="align",
        input_ids=input_ids,
        answer_ids=answer_ids,
        evidence_spans=(),
        actual_positions=(0.5,),
    )

    got = M.gold_nll(tiny_model, tok_item)
    expected = _reference_two_step_nll(tiny_model, input_ids, answer_ids)
    assert got == pytest.approx(expected, abs=1e-4)

    # The test is only meaningful if a shifted window would have failed it: score the
    # window that an off-by-one implementation would have used and check it differs.
    seq = torch.tensor([list(input_ids) + list(answer_ids)])
    with torch.inference_mode():
        logits = tiny_model(input_ids=seq).logits[0]
    start = len(input_ids)
    gold = torch.tensor(answer_ids)
    for shift in (-1, 1):
        lp = torch.log_softmax(logits[start - 1 + shift : start + 1 + shift].float(), dim=-1)
        wrong = float(-lp.gather(-1, gold[:, None]).squeeze(-1).sum())
        assert abs(wrong - expected) > 1e-2, f"shift {shift} is indistinguishable"


def test_answer_margin_sign_matches_the_argmax(tiny_model):
    input_ids = tuple(range(3, 20))
    with torch.inference_mode():
        logits = tiny_model(input_ids=torch.tensor([list(input_ids)])).logits[0, -1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    best = int(lp.argmax())
    runner_up = int(lp.topk(2).indices[1])

    def margin_for(gold: int) -> float:
        return M.answer_margin(
            tiny_model,
            TokenizedItem(
                item_id="m",
                input_ids=input_ids,
                answer_ids=(gold, 9),
                evidence_spans=(),
                actual_positions=(0.5,),
            ),
        )

    assert margin_for(best) == pytest.approx(float(lp[best] - lp[runner_up]), abs=1e-5)
    assert margin_for(best) > 0
    assert margin_for(runner_up) == pytest.approx(float(lp[runner_up] - lp[best]), abs=1e-5)
    assert margin_for(runner_up) < 0


def test_answer_logit_window_is_the_only_off_by_one():
    assert M.answer_logit_window(10, 2) == (7, 9)
    assert M.answer_logit_window(5, 4) == (0, 4)
    with pytest.raises(ValueError):
        M.answer_logit_window(3, 3)  # would need a logit for the very first token


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------


def test_plan_batches_caps_total_tokens_not_rows():
    # An 8K item runs alone at the default budget.
    assert R.plan_batches([8192], batch_tokens=R.DEFAULT_BATCH_TOKENS) == [[0]]
    lengths = [8192, 512, 512, 512]
    batches = R.plan_batches(lengths, batch_tokens=R.DEFAULT_BATCH_TOKENS)
    assert [8192] in [[lengths[i] for i in b] for b in batches]
    for batch in batches:
        assert len(batch) * max(lengths[i] for i in batch) <= R.DEFAULT_BATCH_TOKENS or len(
            batch
        ) == 1
    # Every item is scheduled exactly once.
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))


def test_plan_batches_buckets_by_length():
    """Short items must never be padded up to a long item's width."""
    lengths = [10, 1000, 11, 1001, 12]
    batches = R.plan_batches(lengths, batch_tokens=40, max_rows=8)
    grouped = [sorted(lengths[i] for i in b) for b in batches]
    assert grouped == [[10, 11, 12], [1000], [1001]]
    assert sorted(i for b in batches for i in b) == list(range(5))

    # Padding waste over a wide length spread stays small because of the sort.
    rng = np.random.default_rng(3)
    spread = [int(v) for v in rng.integers(64, 2048, size=200)]
    plan = R.plan_batches(spread, batch_tokens=4096, max_rows=32)
    padded = sum(len(b) * max(spread[i] for i in b) for b in plan)
    assert padded / sum(spread) < 1.10
    assert R.plan_batches([], batch_tokens=4096) == []


def test_left_padded_batch_matches_unpadded_rows(tiny_model):
    """The padding contract: a padded row's logits equal the same row run alone."""
    rows = [list(range(3, 3 + n)) for n in (6, 11, 20)]
    ids, mask, position_ids = R._pad_left(rows, PAD_ID, torch.device("cpu"))
    with torch.inference_mode():
        batched = tiny_model(
            input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False
        ).logits
        for slot, row in enumerate(rows):
            solo = tiny_model(input_ids=torch.tensor([row]), use_cache=False).logits[0]
            assert torch.allclose(batched[slot, -len(row) :], solo, atol=1e-4)
    assert torch.isfinite(batched).all()


def test_run_items_teacher_forced_matches_the_single_item_path(tiny_model, tok):
    items = [make_item(i, n_user=10 + 7 * i) for i in range(6)]
    records = R.run_items(tiny_model, tok, items, mode="teacher_forced", batch_tokens=256)

    assert [r.item_id for r in records] == [i.item_id for i in items]
    assert {r.mode for r in records} == {"teacher_forced"}
    assert all(r.scenario_id == it.scenario_id for r, it in zip(records, items, strict=True))
    assert all(r.n_tokens > 0 for r in records)
    assert all(r.peak_vram_bytes == 0 for r in records)  # CPU run
    assert all(r.latency_s is not None and r.latency_s >= 0 for r in records)

    # Batched scores reproduce the unbatched reference to float32 round-off.
    for record, item in zip(records, items, strict=True):
        tok_item = R.tokenize_item(tok, item)
        reference = M.teacher_forced_scores(tiny_model, tok_item)
        assert record.gold_nll == pytest.approx(reference.gold_nll, abs=1e-3)
        assert record.answer_margin == pytest.approx(reference.answer_margin, abs=1e-3)


def test_run_items_greedy_stops_at_newline_and_strips_the_template(tiny_model, tok):
    items = [make_item(i) for i in range(3)]
    records = R.run_items(
        tiny_model, tok, items, mode="greedy", batch_tokens=4096, max_new_tokens=4
    )
    assert all(r.mode == "greedy" for r in records)
    assert all(r.prediction is not None and "\n" not in r.prediction for r in records)
    assert all(isinstance(r.correct, bool) for r in records)
    assert all(r.gold_nll is not None for r in records)


def test_run_items_record_selection_leaves_unrequested_fields_none(tiny_model, tok):
    records = R.run_items(
        tiny_model, tok, [make_item(0)], mode="teacher_forced", record=("gold_nll",)
    )
    (record,) = records
    assert record.gold_nll is not None
    assert record.prediction is None
    assert record.correct is None
    assert record.answer_margin is None
    assert record.latency_s is None

    with pytest.raises(ValueError, match="unknown record fields"):
        R.run_items(tiny_model, tok, [make_item(0)], record=("nope",))


def test_run_items_rejects_unknown_mode(tiny_model, tok):
    with pytest.raises(ValueError, match="unknown mode"):
        R.run_items(tiny_model, tok, [make_item(0)], mode="beam")


def test_run_items_accepts_pretokenised_pairs(tiny_model, tok):
    item = make_item(0)
    tok_item = R.tokenize_item(tok, item)
    (record,) = R.run_items(tiny_model, tok, [(item, tok_item)], mode="teacher_forced")
    assert record.n_tokens == tok_item.n_tokens
    with pytest.raises(TypeError):
        R.run_items(tiny_model, tok, ["not an item"])


def test_run_items_returns_empty_for_empty_input(tiny_model, tok):
    assert R.run_items(tiny_model, tok, []) == []


def test_run_items_inside_a_disabled_psa_session_reproduces_the_baseline(tiny_model, tok):
    """The hooks package is owned by another module; skip cleanly if it is not ready."""
    pytest.importorskip("rineya_psa.hooks.registry")
    from rineya_psa.hooks.specs import PSAContext

    items = [make_item(i) for i in range(3)]
    baseline = R.run_items(tiny_model, tok, items, mode="teacher_forced")
    hooked = R.run_items(
        tiny_model, tok, items, mode="teacher_forced", ctx=PSAContext(enabled=False)
    )
    for left, right in zip(baseline, hooked, strict=True):
        assert left.item_id == right.item_id
        assert left.gold_nll == pytest.approx(right.gold_nll, abs=2e-3)
        assert left.prediction == right.prediction
    # The session must have been torn down: the model is back on its own kernel.
    assert tiny_model.config._attn_implementation != "psa"


def test_tokenize_item_measures_evidence_positions_when_offsets_exist(tok):
    item = make_item(0)
    tok_item = R.tokenize_item(tok, item)
    # The stub tokenizer has no offset mapping, so the runner must echo the request
    # rather than invent a measurement.
    assert tok_item.actual_positions == item.target_positions
    assert tok_item.evidence_spans == ()


@pytest.mark.model
def test_tokenize_item_with_the_real_qwen3_tokenizer():
    """The offset-mapping path only exists on a fast tokenizer; check it on the real one."""
    from rineya_psa.paths import MODELS

    if not (MODELS["qwen3"].path / "tokenizer.json").exists():
        pytest.skip("Qwen3-0.6B checkpoint not present")
    from rineya_psa.loading import load_tokenizer

    real_tok = load_tokenizer("qwen3")
    evidence = "Talia Frost was assigned the call-sign Mosaic."
    item = BenchmarkItem(
        item_id="real-0",
        scenario_id="real",
        family="exact_retrieval",
        fmt="natural",
        length_bucket="2k",
        target_positions=(0.5,),
        local_position="middle",
        split="calibration",
        system="You are answering a closed-context reading test.",
        user="CONTEXT\n"
        + "Filler sentence about the archive. " * 60
        + evidence
        + " More filler about the audit. " * 60
        + "\n\nQUESTION\nWho was assigned Mosaic?",
        answer="Talia Frost",
        answer_aliases=(),
        distractors=(),
        evidence_sentences=(evidence,),
        seed=0,
    )

    rendered = R.render_prompt(real_tok, item)
    assert "<|im_start|>assistant" in rendered
    assert "<think>" in rendered  # enable_thinking=False emits an empty, pre-closed block
    assert rendered.count("<|im_start|>") == 3

    tok_item = R.tokenize_item(real_tok, item)
    assert tok_item.n_tokens > 300
    assert tok_item.answer_ids
    assert len(tok_item.evidence_spans) == 1
    start, end = tok_item.evidence_spans[0]
    assert 0 < start < end < tok_item.n_tokens
    # The measured position must be a real measurement, not the echoed request.
    (actual,) = tok_item.actual_positions
    assert actual == pytest.approx(((start + end) / 2) / tok_item.n_tokens)
    assert actual != item.target_positions[0]
    decoded = real_tok.decode(list(tok_item.input_ids[start:end]))
    assert "Talia Frost" in decoded


def test_to_dataframe_and_write_parquet(tiny_model, tok, tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    records = R.run_items(tiny_model, tok, [make_item(i) for i in range(4)])
    frame = R.to_dataframe(records)
    assert len(frame) == 4
    assert {"item_id", "scenario_id", "gold_nll", "actual_positions"} <= set(frame.columns)

    out = R.write_parquet(records, tmp_path / "runs" / "records.parquet")
    assert out.exists()
    import pandas as pd

    assert len(pd.read_parquet(out)) == 4


def test_write_parquet_refuses_a_checkpoint_path(tiny_model, tok):
    from rineya_psa.paths import MODELS

    records = R.run_items(tiny_model, tok, [make_item(0)])
    with pytest.raises(PermissionError):
        R.write_parquet(records, MODELS["qwen3"].path / "records.parquet")


# --------------------------------------------------------------------------------------
# preservation
# --------------------------------------------------------------------------------------


def test_repetition_rate_and_degenerate_loop_detector():
    assert P.repetition_rate("the archive audit ran on tuesday morning") == pytest.approx(0.0)
    looped = "yes yes yes yes yes yes yes yes"
    assert P.repetition_rate(looped) > 0.5

    clean = P.detect_degenerate_loop("Talia Frost signed the transfer slip on Friday.")
    assert not clean.degenerate
    assert clean.longest_run == 1

    broken = P.detect_degenerate_loop(looped)
    assert broken.degenerate
    assert broken.longest_run >= 4
    assert "repeated_token_loop" in broken.labels
    assert broken.detector in {"observable_quality_v4.5", "psa_fallback"}


def test_detect_degenerate_loop_fallback_matches_project_thresholds(tmp_path: Path):
    # Point the loader at an empty tree so the vendored detector is unavailable.
    P._observable_quality.cache_clear()
    try:
        report = P.detect_degenerate_loop("a b a b a b a b a b", repo_root=tmp_path)
        assert report.detector == "psa_fallback"
        assert report.degenerate
        assert report.bigram_repeat >= 0.55
        clean = P.detect_degenerate_loop("one two three four five", repo_root=tmp_path)
        assert not clean.degenerate
    finally:
        P._observable_quality.cache_clear()


def test_nan_guard_finds_the_first_bad_score():
    assert P.nan_guard([1.0, 2.0, 3.0])["clean"] is True
    bad = P.nan_guard([1.0, float("nan"), float("inf")])
    assert bad["clean"] is False
    assert bad["n_nonfinite"] == 2
    assert bad["first_bad_index"] == 1
    assert bad["n_nan"] == 1


def test_generation_stability_report_shape():
    report = P.generation_stability_report(
        ["fine answer here", "no no no no no no no", ""], scores=[0.1, 0.2, float("nan")]
    )
    assert report["n"] == 3
    assert 0.0 < report["degenerate_rate"] <= 1.0
    assert report["empty_rate"] == pytest.approx(1 / 3)
    assert report["nan_guard"]["clean"] is False
    assert isinstance(report["labels"], dict)


def test_preservation_source_status_reports_every_axis():
    status = P.preservation_source_status()
    assert set(status) == set(P.PRESERVATION_SUITES)
    assert len(P.PRESERVATION_SUITES) == 7
    for suite, info in status.items():
        assert isinstance(info["available"], bool), suite
        assert info["kind"] in {"gen3_classes", "sft_jsonl", "demo_split"}


@pytest.mark.slow
def test_load_preservation_suites_from_the_real_corpora():
    status = P.preservation_source_status()
    if not all(info["available"] for info in status.values()):
        pytest.skip(f"preservation corpora not present: {status}")

    suites = P.load_preservation_suites(n_per_suite=8, seed=99)
    assert set(suites) == set(P.PRESERVATION_SUITES)
    for name, records in suites.items():
        assert 0 < len(records) <= 8, name
        for record in records:
            assert record.suite == name
            assert record.messages and record.messages[-1]["role"] in {"user", "system"}
            assert record.max_new_tokens >= 8
            assert (REPO_ROOT / record.source).exists()

    # Deterministic: the same (seed, n) draws the same ids.
    again = P.load_preservation_suites(n_per_suite=8, seed=99)
    assert [r.record_id for r in again["arithmetic"]] == [
        r.record_id for r in suites["arithmetic"]
    ]
    # A different seed draws a different sample from a 15 000-row pool.
    other = P.load_preservation_suites(n_per_suite=8, seed=100)
    assert [r.record_id for r in other["arithmetic"]] != [
        r.record_id for r in suites["arithmetic"]
    ]


def test_load_preservation_suites_warns_instead_of_substituting(tmp_path: Path):
    with pytest.warns(RuntimeWarning, match="could not be loaded"):
        suites = P.load_preservation_suites(
            suites=("arithmetic",), repo_root=tmp_path, n_per_suite=4
        )
    assert suites == {}
    with pytest.raises(FileNotFoundError):
        P.load_preservation_suites(
            suites=("arithmetic",), repo_root=tmp_path, n_per_suite=4, strict=True
        )
    with pytest.raises(KeyError):
        P.load_preservation_suites(suites=("not_a_suite",))


def _synthetic_preservation_records(n: int = 3) -> list[P.PreservationRecord]:
    return [
        P.PreservationRecord(
            record_id=f"r{i}",
            suite="arithmetic",
            source="synthetic",
            messages=(
                {"role": "system", "content": "Answer with a number."},
                {"role": "user", "content": f"What is {i} + 2?"},
            ),
            reference="4",
            accepted_answers=("4",),
            max_new_tokens=8,
        )
        for i in range(n)
    ]


def test_run_preservation_generates_and_scores(tiny_model, tok):
    records = _synthetic_preservation_records(5)
    out = P.run_preservation(tiny_model, tok, records, batch_tokens=256)
    assert set(out) == {"predictions", "scores"}
    assert set(out["predictions"]) == {r.record_id for r in records}
    assert all(isinstance(v, str) for v in out["predictions"].values())
    block = out["scores"]["arithmetic"]
    assert block["n"] == 5
    assert block["coverage"] == pytest.approx(1.0)
    assert 0.0 <= block["exact_match"] <= 1.0
    assert block["stability"]["n"] == 5
    assert P.run_preservation(tiny_model, tok, []) == {"predictions": {}, "scores": {}}


def test_score_preservation_reports_coverage_and_stability():
    records = [
        P.PreservationRecord(
            record_id=f"r{i}",
            suite="arithmetic",
            source="synthetic",
            messages=({"role": "user", "content": "2 + 2?"},),
            reference="4",
            accepted_answers=("4",),
            max_new_tokens=8,
        )
        for i in range(3)
    ]
    summary = P.score_preservation(records, {"r0": "4.", "r1": "five"})
    block = summary["arithmetic"]
    assert block["n"] == 3
    assert block["coverage"] == pytest.approx(2 / 3)
    assert block["exact_match"] == pytest.approx(1 / 3)  # "4." normalises to "4"
    assert block["stability"]["n"] == 3


# ---------------------------------------------------------------------------
# Regression: a padded batch must not move a position-binned intervention.
#
# The PSA kernel bins a source token by j / T. In a LEFT-padded row the first real token
# sits at padded index T - real_len, so without `ctx.valid_mask` it is binned as if it sat
# in the middle of the context. `run_items` never set valid_mask, so every batched
# position-binned intervention was applied at the wrong positions - silently, with
# plausible numbers. Found by audit while Phase 8 was running on that path.
# ---------------------------------------------------------------------------

def test_position_binned_context_is_not_corrupted_by_padding(tiny_qwen3_model, tiny_arch):
    """Same item, alone vs batched with a longer neighbour, must score identically."""
    import torch

    from rineya_psa.evaluation.runner import run_items
    from rineya_psa.hooks.binning import AbsoluteBinning
    from rineya_psa.hooks.registry import register_psa
    from rineya_psa.hooks.specs import LogitBiasSpec, PSAContext

    register_psa()
    model, arch = tiny_qwen3_model, tiny_arch
    binning = AbsoluteBinning(4)
    table = torch.zeros(arch.num_layers, arch.num_kv_heads, binning.n_bins)
    table[:, :, 0] = 4.0  # a bias that only bites if bin 0 is where we think it is

    def ctx() -> PSAContext:
        return PSAContext(
            enabled=True,
            logit_bias=LogitBiasSpec(table=table.clone(), lam=1.0, binning=binning),
            position_bins=binning,
        )

    tok = StubTokenizer()
    short = make_item(1, n_user=40)
    longer = make_item(2, n_user=400)   # forces left padding when batched together
    alone = run_items(model, tok, [short], ctx=ctx(), record=("gold_nll",))
    together = run_items(
        model, tok, [longer, short], ctx=ctx(), record=("gold_nll",),
        batch_tokens=100_000, max_batch_rows=8,   # deliberately try to batch them
    )
    by_id = {r.item_id: r for r in together}

    assert alone[0].gold_nll is not None
    assert by_id[short.item_id].gold_nll == pytest.approx(alone[0].gold_nll, abs=1e-4), (
        "the short item scored differently when batched with a longer one: the position "
        "bins moved with the padding"
    )


# ---------------------------------------------------------------------------
# SPEC 13.2's three accuracy columns. `contains` did not exist while README.md
# claimed all three were reported; an audit caught the mismatch.
# ---------------------------------------------------------------------------

def test_contains_gold_catches_the_two_real_scorer_artefacts():
    from rineya_psa.evaluation.metrics import accuracy, contains_gold

    item = make_item(0, answer="the loom overseer")
    # 1. format non-compliance: the model found it, strict refuses it
    verbose = "The answer is: **The loom overseer**."
    assert not accuracy(verbose, item)
    assert contains_gold(verbose, item)
    # 2. exact answer still counts
    assert contains_gold("the loom overseer", item)
    # and a genuinely wrong answer is still wrong
    assert not contains_gold("the kiln inspector", item)


def test_contains_is_a_ceiling_not_a_score():
    """A model that lists every candidate 'contains' the gold one. That is the point."""
    from rineya_psa.evaluation.metrics import accuracy_columns

    item = make_item(0, answer="the ferry reeve")
    shotgun = "maybe the loom overseer, the ferry reeve, or the kiln inspector"
    cols = accuracy_columns(shotgun, item)
    assert cols["contains"] is True
    assert cols["strict"] is False
    assert set(cols) == {"strict", "first_line", "contains"}


def test_accuracy_columns_uses_aliases():
    from rineya_psa.evaluation.metrics import accuracy_columns

    item = BenchmarkItem(
        item_id="a", scenario_id="s", family="semantic_single_hop", fmt="natural",
        length_bucket="2k", target_positions=(0.5,), local_position="middle",
        split="calibration", system="s", user="u", answer="the ferry reeve",
        answer_aliases=("ferry reeve",), distractors=(), evidence_sentences=(), seed=0,
    )
    assert accuracy_columns("it was the ferry reeve", item)["contains"] is True
    assert accuracy_columns("ferry reeve", item)["strict"] is True
