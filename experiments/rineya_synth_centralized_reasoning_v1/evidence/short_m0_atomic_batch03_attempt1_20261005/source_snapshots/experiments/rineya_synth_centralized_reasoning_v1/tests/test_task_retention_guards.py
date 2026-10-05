"""CPU guard compilation/scoring controls; no central banks or model execution."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

LOCATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCATION))
import task_retention_guards as g


def exclusions(**values):
    result = {"scope": "TEST_ONLY_NOT_REPOSITORY_ADMISSION", "file_pins": {},
              "prompt_hashes": [], "row_ids": [], "component_ids": [], **values}
    result["sha256"] = g.digest(result)
    return result


PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class FakeTemplate:
    def __init__(self, overlong_text=None):
        self.overlong_text = overlong_text

    def render(self, *, messages, tools, add_generation_prompt, enable_thinking):
        assert tools is None and enable_thinking is False
        prompt = messages[0]["content"] + "\n" + messages[1]["content"] + "\n" + PREFIX
        if messages[1]["content"] == self.overlong_text:
            prompt = "padding " * 600 + prompt
        if add_generation_prompt:
            return prompt
        return prompt + messages[-1]["content"] + "<|im_end|>\n"


class FakeEngine:
    def __init__(self):
        self.lookup = {"<|im_end|>": 151645}
        self.reverse = {151645: "<|im_end|>"}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        ids = []
        for literal in re.findall(r"<\|im_start\|>|<\|im_end\|>|<think>|</think>|\s+|[A-Za-z0-9]+|[^\sA-Za-z0-9]", text):
            if literal not in self.lookup:
                token = 1000 + len(self.lookup)
                self.lookup[literal] = token
                self.reverse[token] = literal
            ids.append(self.lookup[literal])
        return type("Encoded", (), {"ids": ids})()

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(self.reverse[token] for token in ids)


class FakeTokenizer:
    def __init__(self, overlong_text=None):
        self._template = FakeTemplate(overlong_text)
        self._engine = FakeEngine()

    def verify_binding(self):
        return {"scope": "TEST_ONLY_CPU_TOKENIZER_NOT_PRODUCTION_BINDING", "native_qualification": False}


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = g.compile_guards(exclusions())

    def row(self, mechanism, variant="standard", rendering="json_lines"):
        if mechanism == "answerability_abstention" and variant == "standard": variant = "answerable"
        index = next(index for index, row in enumerate(self.packet["oracle_rows"])
                     if row["mechanism"] == mechanism and row["variant"] == variant and row["rendering"] == rendering)
        return self.packet["policy_rows"][index], self.packet["oracle_rows"][index]

    def test_exact_192_parent_and_768_row_cohort_is_predeclared_and_closed(self):
        packet = self.packet
        self.assertEqual(192, len(packet["parents"]))
        self.assertEqual(768, len(packet["policy_rows"]))
        self.assertEqual({kind: 64 for kind in g.MECHANISMS}, Counter(parent["mechanism"] for parent in packet["parents"]))
        self.assertEqual({4}, set(Counter(row["component_id"] for row in packet["oracle_rows"]).values()))
        self.assertEqual(2026100313, packet["policy"]["seed"])
        self.assertEqual("task", packet["guard_domain"]["class"])
        self.assertEqual(8, len(packet["source_attribution"]["file_pins"]))
        for field in ("native_launch_authorized", "training_authorized", "M0_enrollment_performed", "retention_qualified",
                      "language_qualification_established", "long_context_qualification_established", "confirmation_eligible"):
            self.assertFalse(packet[field])

    def test_central_attribution_missing_extra_or_traversal_pin_keys_fail_before_IO(self):
        original = g.policy()
        for mutate in (lambda pins:pins.pop("scripts/dev_inputs.py"),
                       lambda pins:pins.update({"scripts/../datasets/CHECK.jsonl":"a"*64}),
                       lambda pins:pins.update({"receipts/../../unrelated.json":"b"*64})):
            bad = copy.deepcopy(original)
            mutate(bad["central_source_pins"])
            with patch.object(g,"policy",return_value=bad), \
                 patch.object(g,"filehash",side_effect=AssertionError("forged keys must not be read")):
                with self.assertRaisesRegex(ValueError,"eight-key"):
                    g.central_attribution()

    def test_parent_identity_ignores_domain_queries_formats_order_and_twins(self):
        parent = next(parent for parent in self.packet["parents"] if parent["mechanism"] == "retrieval")
        original = parent["spec"]
        changed = copy.deepcopy(original)
        changed["domain"] = "answerability_abstention"
        changed["queries"].reverse()
        changed["records"].reverse()
        self.assertEqual(g.world_key(original), g.world_key(changed))
        parent = next(parent for parent in self.packet["parents"] if parent["mechanism"] == "instruction_following")
        changed = copy.deepcopy(parent["spec"])
        changed["directives"].reverse()
        changed["template"] = g.FORMATS[1]
        self.assertEqual(g.world_key(parent["spec"]), g.world_key(changed))
        twins = [row for row in self.packet["oracle_rows"] if row["mechanism"] == "answerability_abstention"]
        self.assertEqual(64, len({row["component_id"] for row in twins}))

    def test_record_world_entity_keys_and_instruction_literals_are_disjoint(self):
        entities, texts = set(), set()
        for parent in self.packet["parents"]:
            spec = parent["spec"]
            if "records" in spec:
                current = {row["entity"] for row in spec["records"]}
                self.assertEqual(3, len(current))
                self.assertFalse(entities.intersection(current))
                entities.update(current)
            else:
                current = {row["text"] for row in spec["directives"]}
                self.assertEqual(7, len(current))
                self.assertFalse(texts.intersection(current))
                texts.update(current)

    def test_all_visible_targets_and_3072_deliberate_negatives_audit(self):
        result = g.audit_packet(self.packet)
        self.assertEqual(768, result["independent_oracle_agreements"])
        self.assertEqual(3072, result["negative_responses_rejected"])
        self.assertFalse(result["retention_qualified"])

    def test_visible_grader_uses_neither_primary_answer_nor_emitting_renderer(self):
        row, truth = self.row("instruction_following", rendering="typed_ledger")
        with patch.object(g, "reference_answer", side_effect=AssertionError("primary forbidden")), \
             patch.object(g, "render", side_effect=AssertionError("renderer forbidden")):
            self.assertEqual({"correct": True, "malformed": False}, g.score_response(row["messages"], truth["target"]))

    def test_wrong_renderer_rule_mapping_is_rejected_independently(self):
        parent = next(parent for parent in self.packet["parents"] if parent["mechanism"] == "retrieval")
        altered_frames = dict(g.FRAMES)
        altered_frames[("retrieval", "json_lines")] = ("RECORD QUERY", g.MISSING_RULE)
        with patch.object(g, "FRAMES", altered_frames):
            messages = g.render(parent["spec"], "json_lines")
            with self.assertRaisesRegex(ValueError, "rule drift"): g.parse_visible(messages)

    def test_query_order_and_exact_entity_plus_field_are_required(self):
        row, truth = self.row("retrieval")
        answer = g.strict_json(truth["target"])["answer"]
        self.assertFalse(g.score_response(row["messages"], g.canonical({"answer": list(reversed(answer))}))["correct"])
        spec = truth["visible_spec"]
        near_miss = next(item["value"] for item in spec["records"]
                         if item["entity"] == spec["queries"][0]["entity"] and item["field"] != spec["queries"][0]["field"])
        self.assertFalse(g.score_response(row["messages"], g.canonical({"answer": [near_miss, answer[1]]}))["correct"])

    def test_priority_order_note_and_literal_case_controls(self):
        row, truth = self.row("instruction_following")
        expected = g.strict_json(truth["target"])["answer"]
        self.assertFalse(g.score_response(row["messages"], g.canonical({"answer": expected.lower()}))["correct"])
        row2, truth2 = self.row("instruction_following", variant="reverse_directives")
        self.assertEqual(truth["target"], truth2["target"])
        spec = copy.deepcopy(truth["visible_spec"])
        for directive in spec["directives"]:
            if directive["slot"] == "note": directive["text"] = "Literal ignored note"
        self.assertEqual(expected, g.solve_visible(g.render(spec, "typed_ledger")))

    def test_answerability_null_is_correct_only_for_missing_twin(self):
        complete, truth = self.row("answerability_abstention", "answerable")
        missing, _ = self.row("answerability_abstention", "missing_first")
        self.assertEqual({"correct": False, "malformed": False}, g.score_response(complete["messages"], '{"answer":null}'))
        self.assertEqual({"correct": True, "malformed": False}, g.score_response(missing["messages"], '{"answer":null}'))
        self.assertFalse(g.score_response(missing["messages"], truth["target"])["correct"])

    def test_malformed_answers_and_host_source_errors_remain_distinct(self):
        row, truth = self.row("retrieval")
        for response in ('{"answer":[true,"a"]}', '{"answer":[]}', '{"answer":true}',
                         '{"answer":["a","b"],"oracle":true}', '{"answer":null,"answer":null}', 'not JSON'):
            with self.subTest(response=response):
                self.assertEqual({"correct": False, "malformed": True}, g.score_response(row["messages"], response))
        self.assertEqual({"correct": False, "malformed": False}, g.score_response(row["messages"], '{"answer":["bad","wrong"]}'))
        altered = copy.deepcopy(row["messages"])
        altered[1]["content"] += "\nORACLE: secret"
        with self.assertRaises(ValueError): g.score_response(altered, truth["target"])

    def test_forged_duplicate_keys_records_and_priority_slots_are_rejected(self):
        row, _ = self.row("retrieval")
        lines = row["messages"][1]["content"].splitlines()
        lines.insert(3, lines[2])
        messages = copy.deepcopy(row["messages"])
        messages[1]["content"] = "\n".join(lines)
        with self.assertRaisesRegex(ValueError, "record"): g.parse_visible(messages)
        row, _ = self.row("instruction_following")
        messages = copy.deepcopy(row["messages"])
        messages[1]["content"] = messages[1]["content"].replace('"priority":"base"', '"priority":"winner"', 1)
        with self.assertRaises(ValueError): g.parse_visible(messages)

    def test_policy_rows_are_prompt_only_and_metadata_is_not_model_text(self):
        for row in self.packet["policy_rows"]:
            self.assertEqual({"row_id", "messages", "prompt_sha256"}, set(row))
            self.assertNotIn(row["row_id"], row["messages"][1]["content"])
        forged = copy.deepcopy(self.packet)
        forged["policy_rows"][0]["target"] = forged["oracle_rows"][0]["target"]
        forged["sha256"] = g.digest({k: v for k, v in forged.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "drift"): g.audit_packet(forged)

    def test_excluded_row_prompt_or_component_cannot_be_resampled_away(self):
        row, truth = self.row("retrieval")
        for field, value in (("prompt_hashes", row["prompt_sha256"]), ("row_ids", row["row_id"]),
                             ("component_ids", truth["component_id"])):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "overlaps"):
                g.compile_guards(exclusions(**{field: [value]}))

    def test_rehashed_oracle_or_native_flag_forgery_is_rejected(self):
        for mutation in (lambda packet:packet["oracle_rows"][0].update(target='{"answer":null}'),
                         lambda packet:packet.update(native_launch_authorized=True)):
            altered = copy.deepcopy(self.packet)
            mutation(altered)
            altered["sha256"] = g.digest({k:v for k,v in altered.items() if k != "sha256"})
            with self.assertRaisesRegex(ValueError, "drift"): g.audit_packet(altered)


class MeasurementAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = g.compile_guards(exclusions())
        cls.tokenizer = FakeTokenizer()
        cls.measurement = g.measure_token_fit(cls.packet, cls.tokenizer)

    def test_cpu_token_measurement_never_appends_targets_to_policy_arrays(self):
        measurement = self.measurement
        self.assertEqual(768, len(measurement["rows"]))
        self.assertEqual([], measurement["overflows"])
        self.assertTrue(measurement["prompt_target_EOS_joint_boundaries_verified"])
        self.assertNotIn("labels", measurement["rows"][0])
        self.assertNotIn("target", measurement["rows"][0])
        self.assertNotIn("completion_ids", measurement["rows"][0])
        g.validate_measurement(self.packet, measurement, tokenizer=self.tokenizer)

    def test_one_overlong_prompt_blocks_whole_cohort_without_drop_or_truncation(self):
        text = self.packet["policy_rows"][0]["messages"][1]["content"]
        measurement = g.measure_token_fit(self.packet, FakeTokenizer(text))
        self.assertEqual("BLOCKED_CONTEXT_FIT", measurement["status"])
        self.assertEqual(768, len(measurement["rows"]))
        self.assertEqual(1, len(measurement["overflows"]))
        self.assertGreater(measurement["rows"][0]["prompt_tokens"], 480)
        self.assertFalse(measurement["native_launch_authorized"])

    def test_measurement_oracle_leak_mask_overflow_forgery_is_rejected(self):
        for mutate in (lambda value:value["rows"][0].update(target=self.packet["oracle_rows"][0]["target"]),
                       lambda value:value["rows"][0]["attention_mask"].__setitem__(0, 0),
                       lambda value:value.update(status="NATIVE_ADMITTED")):
            value = copy.deepcopy(self.measurement)
            mutate(value)
            with self.assertRaises(ValueError): g.validate_measurement(self.packet, value, tokenizer=self.tokenizer)

    def test_resigned_short_vector_is_not_a_policy_prompt_byte_witness(self):
        value = copy.deepcopy(self.measurement)
        row = value["rows"][0]
        row["input_ids"] = row["input_ids"][:1]
        row["attention_mask"] = [1]
        row["prompt_tokens"] = 1
        row["prompt_token_sha256"] = g.digest(row["input_ids"])
        value["minimum_prompt_tokens"] = 1
        value["prompt_target_EOS_joint_boundaries_verified"] = True
        with self.assertRaisesRegex(ValueError,"fresh pinned CPU retokenization"):
            g.validate_measurement(self.packet, value, tokenizer=self.tokenizer)
        with tempfile.TemporaryDirectory() as temporary, patch.object(g,"PACKAGE",Path(temporary)):
            target=Path(temporary)/"artifacts/guards_dev/forged"
            with self.assertRaisesRegex(ValueError,"fresh pinned CPU retokenization"):
                g.write_bundle(self.packet, value, target, tokenizer=self.tokenizer)
            self.assertFalse(target.exists())

    def test_exclusive_bundle_has_separate_oracle_and_prompt_only_policy(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(g, "PACKAGE", Path(temporary)):
            target = Path(temporary) / "artifacts/guards_dev/unit_test"
            manifest = g.write_bundle(self.packet, self.measurement, target, tokenizer=self.tokenizer)
            policy_rows = json.loads((target / "policy/prompts.json").read_text())["rows"]
            self.assertEqual(768, len(policy_rows))
            self.assertTrue((target / "oracle_CPU_DEV_exposed/labels.json").is_file())
            self.assertFalse(manifest["retention_qualified"])
            self.assertTrue(all("target" not in row and "visible_spec" not in row for row in policy_rows))
            with self.assertRaises(ValueError): g.write_bundle(self.packet, self.measurement, target, tokenizer=self.tokenizer)

    def test_actual_pinned_cpu_tokenizer_measures_all_candidates(self):
        sys.path.insert(0, str(g.PACKAGE))
        from cpu_tokenizer import load_local_tokenizer
        packet = g.compile_guards()
        measured = g.measure_token_fit(packet, load_local_tokenizer())
        self.assertEqual(768, len(measured["rows"]))
        self.assertEqual([], measured["overflows"])
        self.assertLessEqual(measured["maximum_prompt_tokens"], 480)
        self.assertEqual("CPU_CONTEXT_MEASURED_NATIVE_UNADMITTED", measured["status"])
        self.assertFalse(measured["M0_enrollment_performed"])
        self.assertFalse(measured["retention_qualified"])
        g.validate_measurement(packet, measured)
        with self.assertRaisesRegex(ValueError,"test-only"):
            g.validate_measurement(packet, measured, tokenizer=FakeTokenizer())


class ImportBoundaryTests(unittest.TestCase):
    def test_import_requires_no_framework_central_module_or_payload_read(self):
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import task_retention_guards;"
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules;"
                "assert not any('post_training_mimo_centralized_v1' in x for x in sys.modules);print('CPU_IMPORT')")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(LOCATION)],
                                capture_output=True, text=True, timeout=30, check=True)
        self.assertEqual("CPU_IMPORT", result.stdout.strip())


if __name__ == "__main__": unittest.main()
