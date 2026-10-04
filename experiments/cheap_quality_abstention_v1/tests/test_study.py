"""Tiny generated panel only; never reads historical banks or loads a model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.cheap_quality_abstention_v1 import study as S


def tiny_inputs():
    prompts, keys = [], []
    answers = {"B": "Alpha", "P": "Alpha", "C": "Beta", "U": "UNKNOWN"}
    for suffix, variant in S.VARIANTS.items():
        metadata = {"id": "RLC-001-" + suffix, "scenario_id": "RLC-001", "variant": variant,
                    "answerable": suffix != "U", "domain": "fixture_domain", "split": "showcase",
                    "task_type": "fixture_task"}
        prompts.append({**metadata, "question": "Which fixture item is described?",
                        "context": "PRIVATE CONTEXT NOT FOR REVIEW", "prompt": "PRIVATE PROMPT"})
        keys.append({**metadata, "answer": answers[suffix], "accepted_answers": [answers[suffix]]})
    banks = {}
    for model, flags in (("supa", {"B": True, "P": True, "C": False, "U": True}),
                         ("opallo", {"B": True, "P": False, "C": True, "U": False})):
        predictions = {r["id"]: {"id": r["id"], "prediction": "UNKNOWN" if r["variant"] == "unanswerable"
                       and model == "supa" else "fixture response", "terminated": True, "status": "ok",
                       "generated_tokens": 2, "input_tokens": 12, "decoding": "greedy", "seed": 17}
                       for r in prompts}
        banks[model] = {"schema": 1, "label": model, "model_directory": S.EXPECTED_MODELS[model],
                       "implementation_digest": S.EXPECTED_IMPLEMENTATION, "numerics": S.EXPECTED_NUMERICS,
                       "transformers": "4.51.3", "device": "cuda", "max_context": 4096,
                       "max_new_tokens": 24, "rows_total": 4, "rows_skipped_context_overflow": 0,
                       "evidence_kind": "exploratory_long_context_retention_probe_only",
                       "licenses_efficacy_claim": False, "licenses_release": False,
                       "predictions": predictions, "per_row_correct": {"RLC-001-" + k: v for k, v in flags.items()},
                       "all256_hits": sum(flags.values()), "wall_seconds": 1.0}
    return prompts, keys, banks


class SmallStudyTests(unittest.TestCase):
    def setUp(self):
        test_root = S.PACKAGE / "runs" / "unit-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.tmp.name)
        prompts, keys, banks = tiny_inputs()
        data = {"prompts": prompts, "answer_key": keys, **banks}
        self.paths = {}
        for name, body in data.items():
            path = self.root / (name + (".jsonl" if isinstance(body, list) else ".json"))
            path.write_bytes(b"".join(map(S.encoded, body)) if isinstance(body, list) else S.encoded(body))
            self.paths[name] = path

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, name="packet"):
        path = self.root / name
        S.prepare(path, self.paths, expected_worlds=1)
        return path

    def reviews(self, packet):
        private = S.rows_from_bytes((packet / "private.join.jsonl").read_bytes())
        sha = S.sha_file(packet / "annotation.packet.jsonl")
        paths = []
        for side in ("a", "b"):
            labels = []
            for row in private:
                label = "ANSWER"
                if row["variant"] == "unanswerable" and row["model"] == "supa":
                    label = "ABSTAIN"
                if row["variant"] == "paraphrase" and row["model"] == "opallo":
                    label = "ABSTAIN"
                if row["variant"] == "unanswerable" and row["model"] == "opallo" and side == "b":
                    label = "OTHER"
                labels.append({"id": row["id"], "label": label, "rationale": "Synthetic test label."})
            body = {"schema": S.REVIEW_SCHEMA, "packet_sha256": sha, "reviewer_id": side,
                    "method": "codex_assisted", "labels": labels}
            path = self.root / ("review-" + side + ".json")
            path.write_bytes(S.encoded(body))
            paths.append(path)
        return paths

    def test_packet_blinding_and_deterministic_complete_population(self):
        first, second = self.prepare(), self.prepare("packet-repeat")
        self.assertEqual((first / "annotation.packet.jsonl").read_bytes(),
                         (second / "annotation.packet.jsonl").read_bytes())
        rows = S.rows_from_bytes((first / "annotation.packet.jsonl").read_bytes())
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(set(r) == {"id", "question", "response"} for r in rows))
        self.assertEqual([r["id"] for r in rows], sorted(r["id"] for r in rows))
        text = (first / "batches/batch-01.md").read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE CONTEXT", text)
        self.assertNotIn("PRIVATE PROMPT", text)
        self.assertNotIn("RLC-001", text)
        groups = S.rows_from_bytes((first / "response.groups.jsonl").read_bytes())
        self.assertEqual(sum(len(g["members"]) for g in groups), 8)
        self.assertEqual({m["id"] for g in groups for m in g["members"]}, {r["id"] for r in rows})
        self.assertTrue(all(set(g) == {"group_id", "response", "members"} for g in groups))
        self.assertTrue(all(set(m) == {"id", "question"} for g in groups for m in g["members"]))

    def test_failed_validation_preserves_prior_input_freeze(self):
        bank = S.loads(self.paths["supa"].read_bytes())
        bank["predictions"]["RLC-001-B"]["terminated"] = False
        self.paths["supa"].write_bytes(S.encoded(bank))
        with self.assertRaisesRegex(S.Blocked, "unterminated"):
            self.prepare()
        self.assertTrue((self.root / "packet/INPUT_FREEZE.json").exists())
        self.assertFalse((self.root / "packet/annotation.packet.jsonl").exists())

    def test_source_precision_drift_blocks(self):
        prompts, keys, banks = tiny_inputs()
        banks["supa"] = copy.deepcopy(banks["supa"])
        banks["supa"]["numerics"]["cuda_matmul_allow_tf32"] = True
        with self.assertRaisesRegex(S.Blocked, "precision"):
            S.validate_inputs(prompts, keys, banks, 1)

    def test_at_cap_output_blocks_even_if_termination_claims_true(self):
        prompts, keys, banks = tiny_inputs()
        banks["opallo"]["predictions"]["RLC-001-U"]["generated_tokens"] = 24
        with self.assertRaisesRegex(S.Blocked, "truncation"):
            S.validate_inputs(prompts, keys, banks, 1)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        for data in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(S.Blocked):
                S.loads(data)

    def test_incomplete_world_panel_blocks(self):
        prompts, keys, banks = tiny_inputs()
        with self.assertRaisesRegex(S.Blocked, "panel"):
            S.validate_inputs(prompts[:-1], keys, banks, 1)

    def test_full_unknown_diagnostic_is_not_substring_or_gold_rescoring(self):
        self.assertTrue(S.literal_flags("UNKNOWN")["unknown_raw_full_exact"])
        punct = S.literal_flags(' "UNKNOWN." ')
        self.assertFalse(punct["unknown_whitespace_full_exact"])
        self.assertTrue(punct["unknown_punctuation_diagnostic"])
        self.assertFalse(S.literal_flags("The answer is UNKNOWN.")["unknown_punctuation_diagnostic"])
        self.assertFalse(S.literal_flags("unknown")["unknown_punctuation_diagnostic"])

    def test_disagreement_and_ambiguous_are_not_adjudicated(self):
        self.assertEqual(S.label_bounds("ABSTAIN", "ABSTAIN", {"ABSTAIN"}), (1, 1))
        self.assertEqual(S.label_bounds("ANSWER", "ANSWER", {"ABSTAIN"}), (0, 0))
        self.assertEqual(S.label_bounds("ABSTAIN", "CLARIFY", {"ABSTAIN", "CLARIFY"}), (0, 1))
        self.assertEqual(S.label_bounds("AMBIGUOUS", "AMBIGUOUS", {"ABSTAIN"}), (0, 1))

    def test_complete_review_and_source_correct_pair_report(self):
        packet = self.prepare()
        a, b = self.reviews(packet)
        result = S.evaluate(packet, a, b, self.root / "report")
        paired = result["paired_overall"]
        self.assertEqual((paired["source_correct"], paired["candidate_correct"]), (3, 2))
        self.assertEqual((paired["gains_source_wrong"], paired["losses_source_correct"]), (1, 2))
        self.assertEqual(result["unresolved"], 1)
        opallo = result["by_model"]["opallo"]
        self.assertEqual((opallo["assertions_on_missing_evidence_lower"], opallo["assertions_on_missing_evidence_upper"]), (0, 1))
        self.assertEqual(opallo["semantic_false_abstention_on_answerables_lower"], 1)
        self.assertEqual(opallo["semantic_false_abstention_on_answerables_upper"], 1)
        self.assertEqual(paired["false_abstention_on_source_correct_answerables_pairs"], 2)
        self.assertEqual(S.loads((self.root / "report/LABEL_A_SNAPSHOT.json").read_bytes()), S.loads(a.read_bytes()))
        for name in ("report.json", "REPORT.html", "groups.csv", "paired.csv", "rows.private.csv", "COSTS.json"):
            self.assertTrue((self.root / "report" / name).is_file())

    def test_missing_or_duplicate_review_labels_fail_closed(self):
        packet = self.prepare()
        a, b = self.reviews(packet)
        full = S.loads(a.read_bytes())
        for labels in (full["labels"][:-1], full["labels"] + full["labels"][:1]):
            changed = {**full, "labels": labels}
            a.write_bytes(S.encoded(changed))
            with self.assertRaises(S.Blocked):
                S.load_review(a, {r["id"] for r in full["labels"]}, full["packet_sha256"])

    def test_review_binding_and_method_are_enforced(self):
        packet = self.prepare()
        a, b = self.reviews(packet)
        body = S.loads(a.read_bytes())
        for field, value in (("packet_sha256", "wrong"), ("method", "human_certified")):
            changed = {**body, field: value}
            a.write_bytes(S.encoded(changed))
            with self.assertRaises(S.Blocked):
                S.load_review(a, {r["id"] for r in body["labels"]}, body["packet_sha256"])

    def test_same_reviewer_is_not_two_reviews(self):
        packet = self.prepare()
        a, b = self.reviews(packet)
        body = S.loads(b.read_bytes())
        body["reviewer_id"] = "a"
        b.write_bytes(S.encoded(body))
        with self.assertRaisesRegex(S.Blocked, "distinct"):
            S.evaluate(packet, a, b, self.root / "report")

    def test_review_drift_after_evaluation_freeze_blocks(self):
        packet = self.prepare()
        a, b = self.reviews(packet)
        body = S.loads(a.read_bytes())
        sha = S.sha_file(a)
        a.write_bytes(a.read_bytes() + b"\n")
        with self.assertRaisesRegex(S.Blocked, "review changed"):
            S.load_review(a, {r["id"] for r in body["labels"]}, body["packet_sha256"], sha)

    def test_bootstrap_keeps_variants_in_scenario_clusters(self):
        pairs = []
        for world, delta in (("world-positive", 1), ("world-negative", -1)):
            for _ in range(20):
                pairs.append(({"world": world, "value": 0}, {"world": world, "value": delta}))
        interval = S.cluster_interval(pairs, lambda a, b: b["value"] - a["value"])
        self.assertEqual(interval, [-1.0, 1.0])

    def test_source_and_packet_drift_block_evaluation(self):
        packet = self.prepare()
        a, b = self.reviews(packet)
        self.paths["prompts"].write_bytes(self.paths["prompts"].read_bytes() + b"\n")
        with self.assertRaisesRegex(S.Blocked, "historical input drift"):
            S.evaluate(packet, a, b, self.root / "report-drift")
        self.paths["prompts"].write_bytes(self.paths["prompts"].read_bytes()[:-1])
        with (packet / "annotation.packet.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(S.Blocked, "prepared artifact drift"):
            S.evaluate(packet, a, b, self.root / "report-packet-drift")

    def test_existing_output_is_preserved(self):
        path = self.prepare()
        before = (path / "INPUT_FREEZE.json").read_bytes()
        with self.assertRaises(S.Blocked):
            S.prepare(path, self.paths, expected_worlds=1)
        self.assertEqual((path / "INPUT_FREEZE.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
