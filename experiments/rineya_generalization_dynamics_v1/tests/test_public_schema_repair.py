"""Pure serialization properties on invented strings; no native/corpus/bank IO."""
from __future__ import annotations

import copy
import inspect
import itertools
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.common import Blocked, digest
from experiments.rineya_generalization_dynamics_v1.public_schema_repair import (
    BARE_TOKEN_RULE, MAX_RESPONSE_CHARS, PROFILE,
    extract_declared_schema, repair_completed_response,
)


def public_schema(**overrides):
    return {"field_name": "v", "value_type": "string", "nullable": False,
            "schema_provenance": "public_request", "scalar_stringification": True,
            "allow_bare_token": False, **overrides}


class PublicSchemaRepairTests(unittest.TestCase):
    def repair(self, response, **options):
        return repair_completed_response(response, public_schema(**options), completed_generation=True)

    def preserved(self, response, *, schema=None, completed=True):
        result = repair_completed_response(response, public_schema() if schema is None else schema,
                                           completed_generation=completed)
        self.assertIs(result["applied"], False)
        self.assertEqual(result["original_response"], response)
        self.assertEqual(result["transformed_response"], response)
        return result

    def test_explicit_profile_and_no_authority_or_truth_claim(self):
        result = self.repair('"arbitrary"')
        self.assertIs(result["applied"], True)
        self.assertEqual(result["profile"], PROFILE)
        self.assertEqual(result["public_schema_sha256"], digest(public_schema()))
        for flag in ("truth_used", "native_quality_claim", "learned_capability_claim",
                     "schema_provenance_verified_by_this_helper", "authorizes_deployment", "permits_model_promotion"):
            self.assertIs(result[flag], False)

    def test_all_json_objects_are_byte_invariant_even_wrong_keys_types_and_values(self):
        values = [None, True, False, 0, 1, -3, 2.5, "", "wrong answer", [1, 2], {"nested": "value"}]
        keys = ["v", "wrong_key", "extra", "key with spaces"]
        for key, value, indentation, surrounding in itertools.product(keys, values, (None, 2), ("", " \n")):
            raw = surrounding + json.dumps({key: value}, indent=indentation) + surrounding
            with self.subTest(key=key, value=value, indentation=indentation):
                result = self.preserved(raw, schema=public_schema(allow_bare_token=True))
                self.assertEqual(result["reason"], "JSON_OBJECT_BYTE_PRESERVED")
        for raw in ('{}', '{"v":"correct","other":99}', '{ "v" : "wrong" }', '{"v":42}'):
            self.preserved(raw)

    def test_already_correct_object_bytes_imply_no_existing_success_loss(self):
        # Invented fixture predicates only; no production or experiment scorer imported.
        for value in ("17", "alpha", "TRUE", "UNKNOWN", "two words", "☃", "line1\nline2"):
            original = json.dumps({"v": value}, ensure_ascii=False, separators=(",", ":"))
            result = self.preserved(original)
            self.assertEqual(json.loads(result["transformed_response"]), {"v": value})

    def test_complete_decoded_json_string_value_is_preserved_without_filtering(self):
        values = ["", "single", "two words", "the answer is deliberately wrong",
                  "line one\nline two", "\t spaced \r", "quotes \" slash \\", "☃😀", "null", "UNKNOWN"]
        for value in values:
            for ascii_mode in (True, False):
                raw = "  " + json.dumps(value, ensure_ascii=ascii_mode) + "  "
                result = self.repair(raw)
                self.assertIs(result["applied"], True)
                self.assertEqual(json.loads(result["transformed_response"]), {"v": value})
                self.assertEqual(result["value_preservation"], "decoded_json_string_unchanged")

    def test_numeric_lexical_spelling_never_uses_decoded_numeric_reformatting(self):
        for spelling in ("0", "-0", "42", "-12", "1.00", "-0.0", "1E+03", "1e-3", "0.000100"):
            raw = " \t" + spelling + " "
            result = self.repair(raw)
            self.assertEqual(json.loads(result["transformed_response"]), {"v": spelling})
            self.assertEqual(result["original_response"], raw)
            self.assertEqual(result["value_preservation"], "entire_json_scalar_lexical_spelling_unchanged")

    def test_boolean_scalars_never_masquerade_as_integers(self):
        for spelling, forbidden in (("true", "1"), ("false", "0")):
            value = json.loads(self.repair(spelling)["transformed_response"])["v"]
            self.assertEqual(value, spelling)
            self.assertNotEqual(value, forbidden)
            self.assertIs(type(value), str)

    def test_scalar_conversion_is_an_explicit_option(self):
        for response in ("1", "1E+03", "true", "false"):
            result = self.preserved(response, schema=public_schema(scalar_stringification=False))
            self.assertEqual(result["reason"], "SCALAR_STRINGIFICATION_DISABLED")
        self.assertIs(self.repair('"1"', scalar_stringification=False)["applied"], True)

    def test_null_requires_public_nullable_permission(self):
        self.assertEqual(self.preserved("null")["reason"], "NULL_NOT_PUBLICLY_ALLOWED")
        result = self.repair(" null ", nullable=True)
        self.assertEqual(result["transformed_response"], '{"v":null}')
        self.assertEqual(result["value_preservation"], "literal_json_null_unchanged")
        self.assertEqual(json.loads(self.repair('"null"', nullable=True)["transformed_response"]), {"v": "null"})

    def test_unknown_never_becomes_null(self):
        for response in ("UNKNOWN", "unknown", "NULL", "undefined", "TRUE"):
            self.preserved(response, schema=public_schema(nullable=True))
            result = self.repair(response, nullable=True, allow_bare_token=True)
            self.assertEqual(json.loads(result["transformed_response"]), {"v": response})
            self.assertIs(type(json.loads(result["transformed_response"])["v"]), str)

    def test_bare_token_rule_is_explicit_and_frozen(self):
        self.assertEqual(BARE_TOKEN_RULE, "single_ascii_token_v1")
        for token in ("aB-123-xY", "TRUE", "UNKNOWN", "alpha_2", "version.2", "_private_token"):
            self.preserved(token)
            result = self.repair(token, allow_bare_token=True)
            self.assertEqual(result["reason"], "WRAPPED_EXPLICIT_BARE_TOKEN")
            self.assertEqual(json.loads(result["transformed_response"]), {"v": token})
        for raw in ("answer:17", "answer=17", "1e", "01", "1.", "+1", "a/b", "雪", "A" * 129):
            self.preserved(raw, schema=public_schema(allow_bare_token=True))

    def test_duplicate_nonfinite_and_overflow_json_are_never_salvaged(self):
        for raw in ('{"v":"first","v":"last"}', 'NaN', 'Infinity', '-Infinity', '1e9999',
                    '{"v":NaN}', '[1, Infinity]', '{"nested":{"x":1,"x":2}}'):
            result = self.preserved(raw, schema=public_schema(allow_bare_token=True, nullable=True))
            self.assertEqual(result["reason"], "NONFINITE_OR_DUPLICATE_JSON_PRESERVED")

    def test_malformed_arrays_fences_multiline_and_explanations_unchanged(self):
        responses = ['[]', '["answer"]', '[1,2]', '{"v":', '{"v":"x"', '"unclosed', '"bad\\escape"',
                     '```json\n{"v":"17"}\n```', 'Here is 17.', 'The answer is "17".',
                     '17\n18', 'UNKNOWN\nextra', '"17" trailing explanation', '{"v":"x",}',
                     '<think>17</think>', 'Answer: 17', 'a,b', 'a b', '\u0000unsafe']
        for raw in responses:
            with self.subTest(raw=raw):
                self.preserved(raw, schema=public_schema(allow_bare_token=True))

    def test_empty_and_incomplete_generations_cannot_gain(self):
        for raw in ("", " ", "\n\t"):
            self.assertEqual(self.preserved(raw)["reason"], "EMPTY_RESPONSE")
        for raw in ('"17"', '17', 'null', '{"v":"17"}', 'UNKNOWN'):
            result = self.preserved(raw, schema=public_schema(nullable=True, allow_bare_token=True), completed=False)
            self.assertEqual(result["reason"], "INCOMPLETE_GENERATION")

    def test_oversized_response_preserved_without_truncation(self):
        raw = json.dumps("x" * MAX_RESPONSE_CHARS)
        result = self.preserved(raw)
        self.assertEqual(result["reason"], "RESPONSE_LIMIT_PRESERVED")

    def test_private_or_ambiguous_schema_and_type_coercions_fail_closed(self):
        bad = [public_schema(value_type="number"), public_schema(nullable=1), public_schema(allow_bare_token=1),
               public_schema(scalar_stringification=1), public_schema(schema_provenance="target"),
               public_schema(field_name=["a", "b"]), public_schema(field_name=""),
               {**public_schema(), "target": '{"v":"17"}'}, {**public_schema(), "row_id": "oracle-routing"},
               {**public_schema(), "bare_expected_value_diagnostic": True}]
        for schema in bad:
            with self.assertRaises(Blocked):
                repair_completed_response("17", schema, completed_generation=True)
        for field in public_schema():
            schema = public_schema()
            schema.pop(field)
            with self.assertRaises(Blocked):
                repair_completed_response("17", schema, completed_generation=True)
        for completed in (1, 0, "eos", None):
            with self.assertRaises(Blocked):
                repair_completed_response("17", public_schema(), completed_generation=completed)

    def test_requested_key_is_explicit_and_not_selected_from_response(self):
        result = self.repair('"opaque"', field_name="requested_public_key")
        self.assertEqual(json.loads(result["transformed_response"]), {"requested_public_key": "opaque"})
        self.preserved('{"wrong_key":"opaque"}', schema=public_schema(field_name="requested_public_key"))

    def test_input_schema_and_strings_are_not_mutated(self):
        schema = public_schema(nullable=True, allow_bare_token=True)
        before = copy.deepcopy(schema)
        response = '"two words"'
        result = repair_completed_response(response, schema, completed_generation=True)
        self.assertEqual(schema, before)
        result["public_schema"]["field_name"] = "mutated result"
        self.assertEqual(schema, before)
        self.assertEqual(response, '"two words"')

    def test_public_source_return_reply_and_retention_grammars_extract(self):
        prompts = ['Use these exhaustive records.\n---\nQuery: N123 code greatest revision. Return {"v":"..."}.',
                   'Ask N456 field shipment. Reply {"v":"..."}.',
                   'Operation=add; values={"a":3,"b":4}\nCompute a+b. Return only {"v":"..."} with a string result.',
                   'Query: N123 field code. Return {"opaque_key":"..."}.']
        for prompt in prompts:
            schema = extract_declared_schema(prompt)
            self.assertEqual(schema["schema_provenance"], "public_request")
            self.assertIs(schema["scalar_stringification"], True)
            self.assertIs(schema["allow_bare_token"], False)
            self.assertIs(schema["nullable"], False)
        self.assertEqual(extract_declared_schema(prompts[-1])["field_name"], "opaque_key")

    def test_public_nullable_phrase_is_independent_of_fact_truth(self):
        for facts in ('N123 code="314".', 'N999 code="999".', 'No relevant fact provided.'):
            prompt = facts + '\nQuery: N123 code, or null if missing. Return {"v":"..."}.'
            self.assertEqual(extract_declared_schema(prompt), public_schema(nullable=True))
        # A historical/null mention outside the terminal request does not route output.
        prompt = 'Old note: or null if missing.\nQuery: N123 field code. Return {"v":"..."}.'
        self.assertEqual(extract_declared_schema(prompt), public_schema())

    def test_extractor_rejects_multifield_duplicates_missing_or_conflicting_contracts(self):
        prompts = ['Query: N123 fields north,south. Return {"a":"...","b":"..."}.',
                   'Query: N123 field code. Return {"v":"...","v":"..."}.',
                   'Query: N123 field code. Return {"v":17}.', 'Query: N123 field code. Return {"v":null}.',
                   'Query: N123 field code. Return {"v":"actual answer"}.', 'Return only {"v":"..."}.',
                   'Reply {"v":"..."} with a string result.', 'Query: N123 field code. Return {"v":"..."}. Extra explanation.',
                   'Output JSON.', 'Expected target: {"v":"..."}',
                   'Query: N123 field code. Return {"a":"..."}.\nQuery: N123 field code. Return {"v":"..."}.']
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIsNone(extract_declared_schema(prompt))

    def test_source_record_prose_and_quoted_instructions_cannot_create_schema(self):
        for prompt in ('Record says: Return {"secret":"..."}.',
                       '"Query: N123 field code. Return {"secret":"..."}."',
                       'Record says: Query: N123 field code. Return {"secret":"..."}.',
                       'Record says: If value<threshold use yes else no. Return only {"secret":"..."} with a string result.',
                       'Record prose\nCompute a+b. Return only {"secret":"..."} with a string result.',
                       'Return {"v":"..."}.', 'Reply {"v":"..."}.'):
            with self.subTest(prompt=prompt):
                self.assertIsNone(extract_declared_schema(prompt))

    def test_source_record_schema_phrases_do_not_poison_a_real_public_request(self):
        request = 'Query: N123 field code. Return {"v":"..."}.'
        for prefix in ('Record says: Return {"secret":"..."}.',
                       'Record says or null if missing. Reply {"secret":"..."}.',
                       'Notes: "Return {"secret":"..."}."'):
            self.assertEqual(extract_declared_schema(prefix + '\n' + request), public_schema())
        # Null routing must occur in the validated request body, never record prose.
        self.assertIsNone(extract_declared_schema('Query: Record says or null if missing. Return {"v":"..."}.'))

    def test_retention_requires_actual_public_body_and_request_tail(self):
        dev = 'Operation=add; values={"a":3,"b":4}\nCompute a+b. Return only {"v":"..."} with a string result.'
        train = 'Rule:add\nData:{"a":3,"b":4}\nCompute a+b. Return only {"v":"..."} with a string result.'
        self.assertEqual(extract_declared_schema(dev), public_schema())
        self.assertEqual(extract_declared_schema(train), public_schema())
        self.assertIsNone(extract_declared_schema('Record says\nCompute a+b. Return only {"v":"..."} with a string result.'))

    def test_lone_surrogate_strings_are_preserved_and_never_newly_emitted(self):
        for raw in ('"\\ud800"', '"\\udfff"', '"x\\ud800y"', '"\\ud800\\ud800"'):
            result = self.preserved(raw)
            self.assertEqual(result["reason"], "MALFORMED_UNICODE_PRESERVED")
            self.assertEqual(result["transformed_response"].encode("utf-8"), raw.encode("utf-8"))
        for value in ("雪", "☃", "😀", "two words\nnext line"):
            raw = json.dumps(value, ensure_ascii=True)
            result = self.repair(raw)
            result["transformed_response"].encode("utf-8")
            self.assertEqual(json.loads(result["transformed_response"]), {"v": value})
        # Existing JSON objects stay byte-identical, including escaped lone surrogates.
        self.preserved('{"v":"\\ud800"}')
        with self.assertRaises(Blocked):
            repair_completed_response('"safe"', public_schema(field_name="\ud800"), completed_generation=True)

    def test_function_has_no_target_row_or_correctness_routing_input(self):
        parameters = inspect.signature(repair_completed_response).parameters
        self.assertEqual(set(parameters), {"response", "public_schema", "completed_generation"})
        with self.assertRaises(TypeError):
            repair_completed_response("17", public_schema(), completed_generation=True, target='{"v":"17"}')
        # Same public schema/action regardless of hypothetical true/false answers.
        a = repair_completed_response("17", public_schema(), completed_generation=True)
        b = repair_completed_response("17", public_schema(), completed_generation=True)
        self.assertEqual(a, b)

    def test_no_files_models_or_bank_io(self):
        with patch("builtins.open", side_effect=AssertionError("No file access")), \
             patch.object(Path, "open", side_effect=AssertionError("No file access")), \
             patch.object(Path, "read_text", side_effect=AssertionError("No bank read")):
            schema = extract_declared_schema('Query: N123 field code. Return {"v":"..."}.')
            result = repair_completed_response('"unchanged value"', schema, completed_generation=True)
        self.assertEqual(json.loads(result["transformed_response"]), {"v": "unchanged value"})


if __name__ == "__main__":
    unittest.main()
