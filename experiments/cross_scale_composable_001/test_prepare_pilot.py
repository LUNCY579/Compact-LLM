import copy
import unittest

from . import prepare_pilot as p
from . import receipts as r


def row(i):
    return {"id": f"fixture-{i}", "parent_id": f"parent-{i}", "source_split": "train",
            "source_release": "forge_v2", "truncated": False, "mode": "direct",
            "reasoning_content_tokens": 0, "post_eos_tokens_supervised": False,
            "ids": [1, 2, 3, 151645, 198], "labels": [-100, -100, 3, 151645, -100],
            "prompt_ids": [1, 2], "answer_start": 2, "answer_end": 3,
            "processed_tokens": 5, "supervised_tokens": 2, "final_content_tokens": 1,
            "family": "ledger_replay"}


class PilotTests(unittest.TestCase):
    def test_selection_is_order_independent_and_one_exposure(self):
        data = [row(i) for i in range(384)]
        a = p.select(data, set())
        b = p.select(list(reversed(data)), set())
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 64)
        self.assertEqual(len(a[2]), 320)
        self.assertFalse(set(a[2]) & {x["id"] for x in a[0]})

    def test_never_replace_excluded_selected_parent(self):
        data = [row(i) for i in range(384)]
        selected = p.select(data, set())[0]
        with self.assertRaisesRegex(r.Blocked, "parent overlap"):
            p.select(data, {selected[0]["parent_id"]})

    def test_mask_corruption_prompt_completion_post_eos(self):
        for index, value in ((0, 1), (2, -100), (4, 198)):
            value_row = row(0)
            value_row["labels"][index] = value
            with self.subTest(index=index), self.assertRaises(r.Blocked):
                p.validate_row(value_row)

    def test_wrong_prefix_truncation_eos_split_budget_refused(self):
        mutations = (("prompt_ids", [1, 5]), ("truncated", True), ("source_split", "test"),
                     ("processed_tokens", 6), ("answer_start", 0), ("family", "unknown"))
        for key, value in mutations:
            with self.subTest(key=key), self.assertRaises(r.Blocked):
                p.validate_row({**row(0), key: value})
        bad = row(0)
        bad["ids"][3] = 151644
        with self.assertRaises(r.Blocked):
            p.validate_row(bad)

    def test_duplicate_source_parent_refused(self):
        data = [row(i) for i in range(384)]
        data[1]["parent_id"] = data[0]["parent_id"]
        with self.assertRaises(r.Blocked):
            p.select(data, set())


if __name__ == "__main__":
    unittest.main()
