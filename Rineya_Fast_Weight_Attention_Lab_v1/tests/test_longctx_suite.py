"""Contracts for the P2b long-context suite.

These test the properties the P1/P2 suite silently violated: that a length is a
token count, that the needle is really where we say it is, that the answer space
is wide, and that train and eval never share an item.
"""

from __future__ import annotations

import pytest

from rineya_fwa.data_longctx import (
    FACT_TEMPLATE,
    INTERFERENCE_CATEGORIES,
    build_interference_battery_v2,
    build_position_suite_v2,
    build_train_rows,
    cc0_filler_sentences,
    single_token_codes,
)
from rineya_fwa.eval_longctx import mcnemar_exact, wilson
from rineya_fwa.loading import load_tokenizer

POSITIONS = (0.05, 0.50, 0.95)


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


def test_codes_are_single_tokens(tok):
    codes = single_token_codes()
    assert len(codes) > 256, "answer space must be much wider than the 10 colours of P1/P2"
    for c in codes[:64]:
        assert len(tok.encode(c, add_special_tokens=False)) == 1


def test_filler_is_varied(tok):
    sents = cc0_filler_sentences()
    assert len(sents) >= 200
    assert len(set(sents)) == len(sents), "filler must not repeat; P1/P2 used one sentence"
    # and it must not leak the probe vocabulary
    for s in sents:
        assert "vault" not in s.lower()
        assert "code" not in s.lower()


@pytest.mark.parametrize("length", [512, 1024])
def test_context_length_is_token_exact(tok, length):
    items = build_position_suite_v2(
        tok, lengths=[length], positions=POSITIONS, n_per_cell=1, n_decoys=8, seed=3
    )
    for item in items:
        n = item["n_context_tokens"]
        # the assembler fills with whichever sentence fits the remaining budget,
        # so the shortfall is bounded by the shortest sentence in the pool
        assert length * 0.97 <= n <= length, f"{item['id']}: {n} tokens for target {length}"


def test_needle_is_present_and_positioned(tok):
    items = build_position_suite_v2(
        tok, lengths=[1024], positions=POSITIONS, n_per_cell=2, n_decoys=8, seed=4
    )
    for item in items:
        ctx = item["messages"][1]["content"]
        fact = FACT_TEMPLATE.format(key=item["target_key"], code=item["answer"])
        assert fact in ctx, f"{item['id']}: needle missing"
        assert ctx.count(f"vault {item['target_key']}") + ctx.count(f"Vault {item['target_key']}") == 2, (
            "target key must appear exactly twice: once in the fact, once in the question"
        )
        # needle offset should track the requested relative position
        head = ctx[: ctx.index(fact)]
        frac = len(tok.encode(head, add_special_tokens=False)) / max(item["n_context_tokens"], 1)
        assert abs(frac - item["position"]) < 0.12, f"{item['id']}: needle at {frac:.3f}"


def test_gold_is_a_candidate_and_decoys_are_distinct(tok):
    items = build_position_suite_v2(
        tok, lengths=[1024], positions=POSITIONS, n_per_cell=2, n_decoys=16, n_candidates=32, seed=5
    )
    for item in items:
        assert item["gold_token_id"] in item["candidate_token_ids"]
        assert len(set(item["candidate_token_ids"])) == len(item["candidate_token_ids"])
        assert len(item["candidate_token_ids"]) == 32


def test_confusable_keys_share_a_prefix(tok):
    items = build_position_suite_v2(
        tok, lengths=[1024], positions=(0.5,), n_per_cell=1, n_decoys=16,
        seed=6, key_mode="confusable",
    )
    ctx = items[0]["messages"][1]["content"]
    import re

    keys = set(re.findall(r"RX-\d+", ctx))
    assert len(keys) >= 10
    prefixes = {k[:5] for k in keys}
    assert len(prefixes) <= 3, f"confusable keys should cluster, got prefixes {prefixes}"


def test_train_and_eval_are_disjoint(tok):
    train = build_train_rows(tok, n_rows=16, lengths=[512], n_decoys=8, seed=909)
    ev = build_position_suite_v2(
        tok, lengths=[512], positions=POSITIONS, n_per_cell=2, n_decoys=8, seed=101, split="eval"
    )
    assert not ({r["id"] for r in train} & {i["id"] for i in ev})
    train_ctx = {r["messages"][1]["content"] for r in train}
    eval_ctx = {i["messages"][1]["content"] for i in ev}
    assert not (train_ctx & eval_ctx), "no eval context may appear verbatim in training"
    for row in train:
        assert row["messages"][-1]["role"] == "assistant"


def test_interference_battery_covers_every_category(tok):
    items = build_interference_battery_v2(tok, target_tokens=512, n_per_category=2, seed=1)
    assert len(items) == 2 * len(INTERFERENCE_CATEGORIES)
    seen = {i["category"] for i in items}
    assert seen == set(INTERFERENCE_CATEGORIES)
    for item in items:
        assert item["gold_token_id"] in item["candidate_token_ids"]
        if item["category"] in ("D", "G"):
            assert "superseded_token_id" in item
            assert item["superseded_token_id"] != item["gold_token_id"], (
                "a correction item must have a distinct stale value to be overwritten"
            )
        else:
            assert "superseded_token_id" not in item


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson(7, 10)
    assert lo < 0.7 < hi
    lo, hi = wilson(10, 10)
    assert lo < 1.0 and hi == pytest.approx(1.0, abs=1e-9)
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi > 0.0


def test_mcnemar_is_paired_and_symmetric():
    a = [{"id": f"i{i}", "exact": i < 5} for i in range(10)]
    b = [{"id": f"i{i}", "exact": i < 8} for i in range(10)]
    r = mcnemar_exact(a, b)
    assert r["n_paired"] == 10
    assert r["b_better"] == 3 and r["a_better"] == 0
    assert r["p_two_sided"] == pytest.approx(0.25, abs=1e-9)  # 2 * (1/2)^3
    # identical inputs -> no evidence
    assert mcnemar_exact(a, a)["p_two_sided"] == 1.0
    # swapping arms swaps the direction, keeps the p-value
    rs = mcnemar_exact(b, a)
    assert rs["a_better"] == r["b_better"] and rs["p_two_sided"] == r["p_two_sided"]


def test_p2_suite_would_have_failed_these_contracts(tok):
    """Regression guard: the old suite's defects, stated as failures."""
    from rineya_fwa.data import COLOURS
    from rineya_fwa.eval_position import build_position_suite

    old = build_position_suite(lengths=[512], n_per_cell=1, seed=11)
    assert len(COLOURS) == 10  # answer space that made the ceiling possible
    ctx = old[0]["messages"][1]["content"]
    sentences = [s for s in ctx.split(". ") if s.strip()]
    assert len(set(sentences)) < len(sentences) / 2, (
        "the old filler is one sentence repeated; this is the ceiling"
    )
    n_tok = len(tok.encode(ctx, add_special_tokens=False))
    assert n_tok > 800, f"old 'length 512' item is really {n_tok} tokens"


def test_supervision_is_restricted_to_the_answer_token(tok):
    """The loss must measure retrieval, not template completion.

    With the default chat template the assistant turn is
    ``<code> <|im_end|> \n``: two of the three supervised tokens are free to
    predict, so the reported loss is ~3x lower than the retrieval
    cross-entropy it is read as. P1/P2's "final loss 2e-4" is diluted this way.
    """
    from rineya_fwa.train import encode_row

    rows = build_train_rows(tok, n_rows=2, lengths=[512], n_decoys=8, seed=909, hops=2)
    for row in rows:
        gold = row["messages"][-1]["content"]
        gold_ids = tok.encode(gold, add_special_tokens=False)

        default = encode_row(tok, row, 768)
        n_default = int((default["labels"] != -100).sum())

        answer_only = encode_row(tok, row, 768, supervise_answer_only=True)
        sup = [
            int(i)
            for i, t in zip(answer_only["input_ids"].tolist(), answer_only["labels"].tolist())
            if t != -100
        ]
        assert sup == gold_ids, f"expected exactly {gold!r}, got {tok.decode(sup)!r}"
        assert n_default > len(sup), "the default path should supervise more than the answer"
