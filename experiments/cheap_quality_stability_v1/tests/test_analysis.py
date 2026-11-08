"""Synthetic mechanics checks only. Run under the campaign CPU supervisor."""
import importlib.util
import sys
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cheap_stability_analysis", STUDY / "analyze.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def record(correct, terminated=True):
    return {"correct": correct, "terminated": terminated, "reason": None if correct else "value mismatch"}


class StabilityMechanics(unittest.TestCase):
    def test_stable_gain_requires_every_draw_and_source_failure(self):
        gain = analysis.item_metrics(record(False), [record(True)] * 4)
        self.assertTrue(gain["stable_gain"])
        self.assertEqual(gain["gain_draws"], 4)
        kept = analysis.item_metrics(record(True), [record(True)] * 4)
        self.assertFalse(kept["stable_gain"])
        mixed = analysis.item_metrics(record(False), [record(True)] * 3 + [record(False)])
        self.assertFalse(mixed["stable_gain"])
        self.assertTrue(mixed["mixed_final_outcomes"])

    def test_recurrent_loss_conditioned_on_source_correct(self):
        damaged = analysis.item_metrics(record(True), [record(False)] * 2 + [record(True)] * 2)
        self.assertTrue(damaged["recurrent_loss"])
        self.assertFalse(damaged["universal_loss"])
        already_wrong = analysis.item_metrics(record(False), [record(False)] * 4)
        self.assertEqual(already_wrong["loss_draws"], 0)
        self.assertFalse(already_wrong["recurrent_loss"])

    def test_transition_denominator_and_no_gain_loss_cancellation(self):
        source = {"kept": record(True), "lost": record(True), "gain": record(False), "wrong": record(False)}
        final = {"kept": record(True), "lost": record(False), "gain": record(True), "wrong": record(False, False)}
        counts = analysis.transition_counts(source, final)
        self.assertEqual((counts["lost_total"], counts["gained_total"], counts["net_total"]), (1, 1, 0))
        self.assertEqual(counts["conditional_loss_fraction"], "1/2")
        self.assertEqual(counts["conditional_loss_rate"], 0.5)
        self.assertEqual(counts["terminated_total"], 3)

    def test_jaccard_uses_sets_and_empty_overlap_is_unknown(self):
        self.assertEqual(analysis.overlap(["a", "a", "b"], ["b", "c"])["jaccard"], 1 / 3)
        self.assertIsNone(analysis.overlap([], [])["jaccard"])

    def test_stored_total_mismatch_invalidates_analysis(self):
        counts = analysis.transition_counts({"a": record(True)}, {"a": record(False)})
        stored = {key: counts[key] for key in ("reference_n", "correct_total", "retained_total", "lost_total", "gained_total", "terminated_total")}
        analysis.validate_stored(stored, counts, "synthetic")
        stored["lost_total"] = 0
        with self.assertRaisesRegex(analysis.InvalidEvidence, "lost_total"):
            analysis.validate_stored(stored, counts, "synthetic")

    def test_json_rejects_duplicate_ids_keys_and_nonfinite_values(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(analysis.InvalidEvidence):
                analysis.json_no_duplicates(text)
        with self.assertRaisesRegex(analysis.InvalidEvidence, "duplicate item"):
            analysis.unique_rows([{"id": "a"}, {"id": "a"}], "synthetic", 2)

    def test_full_monitor_prompt_binding_fails_closed(self):
        row = {"id": "a", "group_id": "g", "reply_class": "test", "split": "task",
               "messages": [{"role": "user", "content": "What is 2 plus 3?"}], "answer": "5."}
        audited = {"id": "a", "source_row_sha256": analysis.sha(analysis.canonical(row)),
                   "messages_sha256": analysis.sha(analysis.canonical(row["messages"])),
                   "operative_user_message_index_0based": 0, "operative_prompt": "What is 2 plus 3?",
                   "operative_prompt_sha256": analysis.sha(b"What is 2 plus 3?"), "original_answer": "5.",
                   "family_id": "g", "reply_class": "test"}
        analysis.validate_bindings({"a": row}, {"a": audited})
        altered = {**audited, "operative_prompt": "What is 2 plus 4?"}
        with self.assertRaisesRegex(analysis.InvalidEvidence, "operative prompt mismatch"):
            analysis.validate_bindings({"a": row}, {"a": altered})
        with self.assertRaisesRegex(analysis.InvalidEvidence, "IDs do not match"):
            analysis.validate_bindings({"a": row}, {})

    def test_scorer_rejects_model_import_and_outputs_require_real_eos_boolean(self):
        with self.assertRaisesRegex(analysis.InvalidEvidence, "undeclared import"):
            analysis.load_scorer("import torch\n")
        scorer = analysis.load_scorer((analysis.REPO / "experiments/architecture_benchmark_001/retention.py").read_text(encoding="utf-8"))
        target = {"kind": "number", "value": "5", "bare_allowed": True}
        outputs = {"a": {"text": "5.", "terminated": True}}
        self.assertTrue(analysis.score_outputs(outputs, {"a": target}, scorer)["a"]["correct"])
        outputs["a"]["terminated"] = False
        self.assertFalse(analysis.score_outputs(outputs, {"a": target}, scorer)["a"]["correct"])
        outputs["a"]["terminated"] = "true"
        with self.assertRaisesRegex(analysis.InvalidEvidence, "EOS types"):
            analysis.score_outputs(outputs, {"a": target}, scorer)
        with self.assertRaisesRegex(analysis.InvalidEvidence, "IDs differ"):
            analysis.score_outputs({}, {"a": target}, scorer)
        self.assertFalse({"torch", "numpy", "transformers"} & set(sys.modules))


if __name__ == "__main__":
    unittest.main()
