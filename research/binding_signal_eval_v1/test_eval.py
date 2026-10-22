"""Mechanical stdlib checks; no model, torch, tokenizer or sealed-data access."""
from contextlib import nullcontext, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if __package__:
    from . import common as c, probe, score
else:
    import common as c
    import probe
    import score


PROMPTS = {"person_answerable": "Who is assigned in the occupied fixture?",
           "person_missing": "Who is assigned in the missing fixture?",
           "person_vacant": "Who is assigned in the vacant fixture?",
           "role_answerable": "Which role is targeted in the role fixture?"}
ANSWERS = {"person_answerable": "Ada", "person_missing": "UNKNOWN", "person_vacant": "UNKNOWN", "role_answerable": "Steward"}


def fixtures():
    inputs, gold, predictions = [], [], []
    for world in range(32):
        parent = f"world-{world:02d}"
        for category in score.CATEGORIES:
            for view in range(3):
                key = f"opaque-{len(inputs):04d}"
                inputs.append({"id": key, "parent_id": parent, "prompt": [
                    {"role": "system", "content": "Use only the stated assignment facts."},
                    {"role": "user", "content": PROMPTS[category]}], "prompt_ids": [10, 11]})
                gold.append({"id": key, "parent_id": parent, "category": category, "reference": ANSWERS[category]})
                predictions.append({"id": key, "parent_id": parent, "prediction": ANSWERS[category], "terminated": True})
    return inputs, gold, predictions


def decode_fixture(text):
    category = next(category for category, prompt in PROMPTS.items() if prompt == text)
    return {"world": {"people": ["Ada", "Bea", "Cal", "Dax", "Eve", "Fox"]}, "answer": ANSWERS[category]}


class BaselineMechanicsTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.gold, self.predictions = fixtures()
        self.gold_map = score.validate_gold(self.inputs, self.gold)

    def scored(self, predictions):
        return score.score_rows(self.inputs, self.gold_map, predictions, decode_fixture)

    def lose_worlds(self, predictions, count, category="person_missing"):
        for row in predictions:
            target = self.gold_map[row["id"]]
            if target["category"] == category and int(row["parent_id"].split("-")[1]) < count:
                row["prediction"] = "Ada"

    def test_worker_inputs_reject_gold_category_and_assistant_fields(self):
        self.assertEqual(len(c.validate_inputs(self.inputs)), 384)
        for key in ("reference", "category", "gold_path", "world", "labels", "max_sequence"):
            changed = copy.deepcopy(self.inputs); changed[0][key] = "leaked"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "forbidden"):
                c.validate_inputs(changed)
        changed = copy.deepcopy(self.inputs); changed[0]["prompt"].append({"role": "assistant", "content": "Ada"})
        with self.assertRaisesRegex(ValueError, "only system and user"):
            c.validate_inputs(changed)

    def test_worker_plan_unknown_gold_path_rejected_before_any_file_read(self):
        plan = {key: None for key in c.PLAN_KEYS}; plan["gold_path"] = "never-open-this"
        with patch.object(c, "absolute", side_effect=AssertionError("must reject shape before filesystem access")):
            with self.assertRaisesRegex(ValueError, "gold paths are forbidden"):
                c.validate_plan(plan)

    def test_normalization_keeps_punctuation_and_separates_termination(self):
        predictions = copy.deepcopy(self.predictions)
        positive = next(row for row in predictions if self.gold_map[row["id"]]["category"] == "person_answerable")
        positive.update(prediction="  ADA\t", terminated=False)
        unknown = next(row for row in predictions if self.gold_map[row["id"]]["category"] == "person_missing")
        unknown["prediction"] = "UNKNOWN."
        result, maps, _ = self.scored(predictions)
        detail = result["rows"][positive["id"]]
        self.assertTrue(detail["canonical_text_exact"]); self.assertFalse(detail["strict_text_exact"])
        self.assertFalse(detail["canonical_complete_exact"])
        self.assertFalse(result["rows"][unknown["id"]]["canonical_exact_unknown"])
        self.assertEqual(score.normalize(" ＵＮＫＮＯＷＮ \n"), "unknown")
        self.assertFalse(maps["canonical_complete_exact"]["person_answerable"]["world-00"])

    def test_world_is_unit_and_all_three_views_are_required(self):
        predictions = copy.deepcopy(self.predictions)
        row = next(row for row in predictions if self.gold_map[row["id"]]["category"] == "person_missing")
        row["prediction"] = "Ada"
        result, maps, _ = self.scored(predictions)
        self.assertEqual(result["categories"]["person_missing"]["canonical_complete_exact"], 95)
        self.assertEqual(sum(maps["canonical_complete_exact"]["person_missing"].values()), 31)
        self.assertEqual(len(maps["canonical_complete_exact"]["person_missing"]), 32)

    def test_holm_two_rejects_six_directed_wins_but_accepts_seven(self):
        _, source, _ = self.scored(self.predictions)
        for count, expected in ((6, False), (7, True)):
            predictions = copy.deepcopy(self.predictions); self.lose_worlds(predictions, count)
            _, control, _ = self.scored(predictions)
            result = score.sensitivity(source, control)
            self.assertEqual(result["repair_separation_passed"], expected)
            cpair = result["world_contrasts_source_minus_lora"]["canonical_complete_exact"]["person_missing"]
            self.assertEqual((cpair["net_source_minus_lora"], cpair["nd"]), (count, count))
            self.assertEqual(cpair["exact_p"], 2 / 2**count)
            self.assertEqual(cpair["holm_adjusted_p"], 4 / 2**count)
            self.assertIs(result["training_permission"], False)

    def test_both_floor_supports_new_skill_headroom_without_repair_separation(self):
        predictions = copy.deepcopy(self.predictions)
        for category in score.UNKNOWN_CATEGORIES: self.lose_worlds(predictions, 32, category)
        _, maps, _ = self.scored(predictions)
        result = score.sensitivity(maps, maps)
        self.assertEqual(result["status"], "BASELINE_REPAIR_SEPARATION_NOT_ESTABLISHED")
        self.assertEqual(result["new_skill_headroom"], "NEW_SKILL_HEADROOM_PRESENT")
        self.assertEqual(result["lora_unknown_category_world_failures"], {category: 32 for category in score.UNKNOWN_CATEGORIES})
        self.assertFalse(result["training_permission"])

    def test_ceiling_does_not_invent_new_skill_headroom(self):
        _, maps, _ = self.scored(self.predictions)
        result = score.sensitivity(maps, maps)
        self.assertEqual(result["new_skill_headroom"], "INSUFFICIENT_HEADROOM_FOR_7_WORLD_HOLM2_GAIN")
        self.assertFalse(result["repair_separation_passed"])

    def test_false_unknown_and_roster_person_are_exact_content_diagnostics(self):
        predictions = copy.deepcopy(self.predictions)
        positive = next(row for row in predictions if self.gold_map[row["id"]]["category"] == "person_answerable")
        positive.update(prediction="unknown", terminated=False)
        negatives = [row for row in predictions if self.gold_map[row["id"]]["category"] == "person_missing"]
        negatives[0]["prediction"] = " ADA "
        negatives[1]["prediction"] = "Ada."
        negatives[2]["prediction"] = "I think Ada"
        result, maps, roster = self.scored(predictions)
        self.assertEqual(result["totals"]["false_unknown_on_positive"], 1)
        self.assertEqual(result["totals"]["exact_roster_person_on_unanswerable"], 1)
        self.assertFalse(maps["canonical_complete_exact"]["no_false_unknown_all6_positive_controls"]["world-00"])
        self.assertTrue(roster["world-00"])

    def test_zero_discordance_p_one_and_fixed_parity_threshold(self):
        rows = {str(i): True for i in range(32)}
        result = score.exact_pair(rows, rows)
        self.assertEqual((result["nd"], result["exact_p"]), (0, 1))
        self.assertIsNone(result["minimum_significant_abs_net_at_observed_nd"])

    def test_failed_supervisor_cannot_be_rescued_by_complete_terminal(self):
        terminal = {"status": "COMPLETE", "complete": True, "failure": None, "before": {}, "after": {}}
        for supervisor in ({"status": "EXITED", "returncode": 2, "wall_seconds": 1},
                           {"status": "TIMEOUT_OR_INTERRUPTED", "returncode": 0, "wall_seconds": 1}):
            with self.assertRaisesRegex(ValueError, "successful exit"):
                c.verify_terminal(supervisor, terminal)

    def test_offline_loader_refuses_failed_supervisor_before_model_or_score_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / "source"; out.mkdir()
            c.write_new(out / "supervisor.json", {"status": "EXITED", "returncode": 2, "wall_seconds": 1})
            c.write_new(out / "worker_terminal.json", {"status": "COMPLETE", "complete": True, "failure": None, "before": {}, "after": {}})
            c.write_new(out / "GENERATIONS.json", {})
            binding = {"status": "COMPLETE", "complete": True, "model_role": "source", "controller_returncode": 0}
            for stem, filename in (("supervisor", "supervisor.json"), ("worker_terminal", "worker_terminal.json"), ("generations", "GENERATIONS.json")):
                binding[stem + "_sha256"] = c.sha(out / filename)
            c.write_new(out / "BASELINE_BINDING.json", binding)
            with patch.object(c, "model_snapshot", side_effect=AssertionError("no model custody access on failed execution")):
                with self.assertRaisesRegex(ValueError, "successful exit"):
                    score.verify_worker({"output_root": str(root)}, root / "plan.json", root / "grant.json", "source")

    def test_generator_never_feeds_ids_or_gold_and_counts_each_forward(self):
        class Tensor:
            def __init__(self, values): self.values = values; self.shape = (1, len(values[0]))
        class Logits:
            def __init__(self, token): self.token = token
            def __getitem__(self, key): return self
            def argmax(self): return self.token
        calls = []
        def model(current, *, past, use_cache, last_position_only):
            calls.append(current.values)
            return SimpleNamespace(cache=True, logits=Logits(7 if past is None else 151645))
        tokenizer = SimpleNamespace(apply_chat_template=lambda prompt, **kwargs: [10, 11],
                                    decode=lambda ids, **kwargs: "decoded text")
        torch = SimpleNamespace(inference_mode=nullcontext, long="long", tensor=lambda values, **kwargs: Tensor(values))
        with redirect_stdout(io.StringIO()):
            result = probe.generate(model, tokenizer, self.inputs, {151645}, torch)
        self.assertEqual(len(calls), 768)
        self.assertEqual(sum(row["processed_input_tokens"] for row in result), 384 * 3)
        self.assertEqual(sum(row["forward_calls"] for row in result), 768)
        self.assertTrue(all(calls[i] == [[10, 11]] and calls[i + 1] == [[7]] for i in range(0, len(calls), 2)))
        self.assertTrue(all(row["generated_tokens"] == 2 and row["terminated"] for row in result))

    def test_registered_aggregate_limits_and_deadline_are_fixed(self):
        self.assertEqual((c.LIMITS["forward_calls"], c.LIMITS["processed_input_tokens"], c.LIMITS["reserved_generated_tokens"]),
                         (19200, 225540, 18432))
        with patch.object(c.time, "time", return_value=1000):
            with self.assertRaisesRegex(ValueError, "remaining baseline reservation"):
                c.phase_guard({"output_root": "unused"}, {"deadline_unix": 2000, "session_start_unix": 0}, seconds_needed=1800)

    def test_outputs_are_create_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"; c.write_new(path, {"keep": True})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError): c.write_new(path, {"changed": True})
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
