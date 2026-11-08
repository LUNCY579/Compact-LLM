"""Regression tests for the fully-masked-row bug that produced eval_loss = NaN.

The original dataset right-truncated a chat row at ``max_length``. Because the
assistant turn sits at the end, any row whose prompt reached the limit lost its
answer, and the code then masked the whole row rather than failing. Such a row
contributes zero supervised tokens, so:

* in training it burns a forward/backward pass for no gradient, and
* in evaluation a set made entirely of such rows divides by zero -> NaN.

On the Stage-A mix at ``max_sequence_length=3072`` this hit 42% of dev rows and
39% of train rows -- and, because the data is bimodal, exactly the long-context
half the run was supposed to learn from.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_halo_engine as engine  # noqa: E402


class StubTokenizer:
    """One character per token, so lengths are trivial to reason about."""

    eos_token_id = 0
    chat_template = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        # "|" is the assistant header: a real template emits it in both the
        # prompt-only and the full rendering, so the prompt stays a prefix.
        rendered = "".join(
            ("|" + m["content"]) if m["role"] == "assistant" else m["content"]
            for m in messages
        )
        return rendered + ("|" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(c) for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def row(context_chars: int, answer: str = "AB"):
    return {
        "messages": [
            {"role": "user", "content": "c" * context_chars},
            {"role": "assistant", "content": answer},
        ]
    }


def supervised(item) -> int:
    return int((item["labels"] != -100).sum())


class OverflowPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = StubTokenizer()
        # prompt renders to context+"|" so a 20-char context is 21 prompt tokens
        self.short = row(10)
        self.long = row(100)

    # -- the bug itself ---------------------------------------------------
    def test_overflowing_row_is_never_emitted_fully_masked(self) -> None:
        for policy in ("drop", "left-context"):
            ds = engine.LazyChatSFTDataset(
                [self.short, self.long], self.tok, 32, overflow_policy=policy
            )
            for index in range(len(ds)):
                self.assertGreater(
                    supervised(ds[index]), 0, f"policy={policy} produced a fully-masked row"
                )

    def test_drop_excludes_the_overflowing_row_and_counts_it(self) -> None:
        ds = engine.LazyChatSFTDataset(
            [self.short, self.long], self.tok, 32, overflow_policy="drop"
        )
        self.assertEqual(len(ds), 1)
        audit = ds.audit()
        self.assertEqual(audit["rows_in_file"], 2)
        self.assertEqual(audit["rows_used"], 1)
        self.assertEqual(audit["rows_dropped_for_overflow"], 1)

    def test_left_context_keeps_every_row_and_the_whole_answer(self) -> None:
        ds = engine.LazyChatSFTDataset(
            [self.short, self.long], self.tok, 32, overflow_policy="left-context"
        )
        self.assertEqual(len(ds), 2)
        for index in range(len(ds)):
            item = ds[index]
            self.assertLessEqual(int(item["input_ids"].shape[0]), 32)
            # the answer is "AB": both tokens must survive, at the end
            self.assertEqual(supervised(item), 2)
            self.assertEqual(item["labels"][-2:].tolist(), [ord("A"), ord("B")])

    def test_error_policy_fails_fast(self) -> None:
        ds = engine.LazyChatSFTDataset(
            [self.long], self.tok, 32, overflow_policy="error"
        )
        with self.assertRaises(engine.RowOverflowError):
            ds[0]

    # -- guards -----------------------------------------------------------
    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            engine.LazyChatSFTDataset([self.short], self.tok, 32, overflow_policy="keep")

    def test_answer_longer_than_the_budget_is_an_error_not_a_silent_mask(self) -> None:
        ds = engine.LazyChatSFTDataset(
            [row(5, answer="A" * 40)], self.tok, 8, overflow_policy="left-context"
        )
        with self.assertRaises(engine.RowOverflowError):
            ds[0]

    def test_row_without_an_assistant_turn_is_an_error(self) -> None:
        ds = engine.LazyChatSFTDataset(
            [row(5, answer="")], self.tok, 64, overflow_policy="left-context"
        )
        with self.assertRaises(engine.RowOverflowError):
            ds[0]

    def test_template_that_is_not_a_prefix_is_rejected(self) -> None:
        class Reordering(StubTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
                if add_generation_prompt:
                    return "zzz"
                return "".join(m["content"] for m in messages)

        ds = engine.LazyChatSFTDataset([self.short], Reordering(), 64)
        with self.assertRaises(RuntimeError):
            ds[0]

    # -- labelling stays correct -----------------------------------------
    def test_prompt_is_masked_and_answer_is_supervised(self) -> None:
        ds = engine.LazyChatSFTDataset([self.short], self.tok, 64)
        item = ds[0]
        labels = item["labels"].tolist()
        self.assertEqual(labels.count(-100), 11)  # 10 context chars + generation marker
        self.assertEqual(labels[-2:], [ord("A"), ord("B")])
        self.assertEqual(item["input_ids"].shape, item["labels"].shape)
        self.assertEqual(int(item["attention_mask"].sum()), int(item["input_ids"].shape[0]))

    def test_a_fitting_row_is_untouched_by_the_policy(self) -> None:
        reference = engine.LazyChatSFTDataset([self.short], self.tok, 64)[0]
        for policy in ("drop", "left-context", "error"):
            item = engine.LazyChatSFTDataset(
                [self.short], self.tok, 64, overflow_policy=policy
            )[0]
            self.assertEqual(item["input_ids"].tolist(), reference["input_ids"].tolist())
            self.assertEqual(item["labels"].tolist(), reference["labels"].tolist())


if __name__ == "__main__":
    unittest.main()
