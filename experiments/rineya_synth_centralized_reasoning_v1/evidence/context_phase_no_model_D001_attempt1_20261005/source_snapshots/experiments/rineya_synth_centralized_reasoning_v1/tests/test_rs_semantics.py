"""Independent finite-world checks of the CPU R/S public semantics.

The expectations below come from explicit answers or a bit-mask truth table,
not either production solution path. No model, bank, network or GPU is used.
"""
from __future__ import annotations

import copy
import itertools
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from rs_contract import (MAX_BYTES, canonical, format_answer, parse_answer,
                         parse_visible, render, validate_spec)
from rs_solver import solve
from rs_verifier import solve_visible, verify


def selection(values=None, k=2, target=4):
    return {"initial": [1, 3, 2, 2, -1, 0] if values is None else list(values),
            "selection": {"k": k, "target": target}}


def reference(links, requested, values=None):
    return {"initial": [11, 13, 17, 19, 23, 29] if values is None else list(values),
            "lookup": {"links": links, "requested": requested}}


def truth_table(values):
    """Exhaust all subsets by masks; independent of enumeration/DP solvers."""
    result = {}
    for mask in range(1, 1 << len(values)):
        indices = tuple(i for i in range(len(values)) if mask & (1 << i))
        result.setdefault((len(indices), sum(values[i] for i in indices)), set()).add(indices)
    return result


class RSTestCase(unittest.TestCase):
    def check_solution(self, spec, expected, status="solved", template=0):
        expected = {tuple(answer) for answer in expected}
        messages = render(spec, template)
        first = solve(spec)
        second = solve_visible(messages)
        self.assertEqual(status, first["status"])
        self.assertEqual(status, second["status"])
        self.assertEqual(expected, {tuple(answer) for answer in first["solutions"]})
        self.assertEqual(expected, {tuple(answer) for answer in second["solutions"]})
        self.assertEqual(first["status"], second["status"])
        self.assertEqual(first["solutions"], second["solutions"])
        for answer in expected:
            # Ordering is not a hidden precedence requirement.
            for ordered in (list(answer), list(reversed(answer))):
                result = verify(messages, canonical({"answer": ordered}))
                self.assertTrue(result["accepted"], (spec, ordered, result))
        unknown = verify(messages, '{"answer":"UNKNOWN"}')
        self.assertEqual(not bool(expected), unknown["accepted"])
        return messages


class SelectionSemanticsTests(RSTestCase):
    def test_all_alternatives_and_all_orders_are_accepted(self):
        spec = selection([1, 1, 1, 1, 1, 1], 3, 3)
        expected = [tuple(i for i in range(6) if mask & (1 << i))
                    for mask in range(1, 64) if mask.bit_count() == 3]
        messages = self.check_solution(spec, expected)
        self.assertEqual(20, len(expected))
        for answer in expected:
            for ordering in itertools.permutations(answer):
                self.assertTrue(verify(messages, canonical({"answer": ordering}))["accepted"])

    def test_negative_zero_and_equal_values_keep_distinct_indices(self):
        self.check_solution(selection([-2, -2, 0, 0, 2, 2], 2, 0),
                            [(0, 4), (0, 5), (1, 4), (1, 5), (2, 3)])

    def test_cardinality_is_exact_and_one_item_cannot_be_reused(self):
        messages = self.check_solution(selection([2, 9, 11, 13, 15, 17], 2, 4), [], "impossible")
        self.assertFalse(verify(messages, '{"answer":[0,0]}')["accepted"])
        solved = self.check_solution(selection([2, 2, 9, 11, 13, 15], 2, 4), [(0, 1)])
        self.assertFalse(verify(solved, '{"answer":[0]}')["accepted"])
        self.assertFalse(verify(solved, '{"answer":[0,1,2]}')["accepted"])

    def test_infeasible_target_inside_attainable_min_max(self):
        self.check_solution(selection([0, 0, 2, 2, 4, 4], 2, 3), [], "impossible")

    def test_single_and_maximum_allowed_cardinality(self):
        self.check_solution(selection([-100, 0, 100, 100, -100, 0], 1, -100), [(0,), (4,)])
        self.check_solution(selection([1, 2, 3, 4, 5, 6], 5, 20), [(1, 2, 3, 4, 5)])

    def test_target_changes_answer_and_irrelevant_value_preserves_it(self):
        values = [1, 3, 50, 60, 70, 80]
        original = self.check_solution(selection(values, 2, 4), [(0, 1)])
        changed = self.check_solution(selection(values, 2, 110), [(2, 3)])
        self.assertTrue(verify(original, '{"answer":[0,1]}')["accepted"])
        self.assertFalse(verify(changed, '{"answer":[0,1]}')["accepted"])
        self.check_solution(selection([1, 3, 51, 60, 70, 80], 2, 4), [(0, 1)])

    def test_exhaustive_six_item_binary_worlds(self):
        # 64 value worlds x 5 cardinalities x 13 targets = 4,160 tasks.
        # This target interval covers every possible sum in this finite domain.
        count = 0
        for values in itertools.product((-1, 1), repeat=6):
            expected = truth_table(values)
            for k in range(1, 6):
                for target in range(-6, 7):
                    answers = expected.get((k, target), set())
                    with self.subTest(values=values, k=k, target=target):
                        self.check_solution(selection(values, k, target), answers,
                                            "solved" if answers else "impossible", count % 4)
                    count += 1
        self.assertEqual(4160, count)

    def test_exhaustive_ternary_prefix_with_negative_zero_positive_padding(self):
        # 27 worlds x 5 cardinalities x 11 targets = 1,485 tasks.
        count = 0
        for prefix in itertools.product((-1, 0, 1), repeat=3):
            values = prefix + (-2, 0, 2)
            expected = truth_table(values)
            for k in range(1, 6):
                for target in range(-5, 6):
                    answers = expected.get((k, target), set())
                    with self.subTest(values=values, k=k, target=target):
                        self.check_solution(selection(values, k, target), answers,
                                            "solved" if answers else "impossible", count % 4)
                    count += 1
        self.assertEqual(1485, count)


class ReferenceSemanticsTests(RSTestCase):
    def test_requested_chains_resolve_and_deduplicate(self):
        self.check_solution(reference({"head": "middle", "middle": "leaf", "leaf": 3,
                                       "other": 1, "same": "leaf"},
                                      ["head", "same", "other", "head"]), [(1, 3)])

    def test_unrequested_cycles_and_missing_links_are_irrelevant(self):
        self.check_solution(reference({"good": 2, "self": "self", "cyclea": "cycleb",
                                       "cycleb": "cyclea", "broken": "absent"}, ["good"]), [(2,)])

    def test_missing_requested_start_and_missing_intermediate_link(self):
        for spec in (reference({"other": 3}, ["absent"]),
                     reference({"head": "absent"}, ["head"]),
                     reference({"good": 2, "bad": "absent"}, ["good", "bad"])):
            with self.subTest(spec=spec):
                self.check_solution(spec, [], "insufficient_evidence")

    def test_self_and_multinode_requested_cycles(self):
        for spec in (reference({"self": "self"}, ["self"]),
                     reference({"head": "a", "a": "b", "b": "a"}, ["head"]),
                     reference({"good": 2, "a": "b", "b": "a"}, ["good", "a"])):
            with self.subTest(spec=spec):
                self.check_solution(spec, [], "insufficient_evidence")

    def test_longest_bounded_chain_is_not_truncated(self):
        links = {f"a{i}": f"a{i + 1}" for i in range(79)}
        links["a79"] = 5
        self.check_solution(reference(links, ["a0"]), [(5,)])
        links["a79"] = "a0"
        self.check_solution(reference(links, ["a0"]), [], "insufficient_evidence")

    def test_values_are_irrelevant_and_partial_resolved_set_is_wrong(self):
        spec = reference({"a": 0, "b": 5}, ["a", "b"])
        messages = self.check_solution(spec, [(0, 5)])
        self.assertFalse(verify(messages, '{"answer":[0]}')["accepted"])
        self.assertFalse(verify(messages, '{"answer":[0,1,5]}')["accepted"])
        spec["initial"] = [-100, 100, 0, -1, 1, 0]
        self.check_solution(spec, [(0, 5)])

    def test_changed_endpoint_changes_answer_and_depth_preserves_it(self):
        original = reference({"a": 0}, ["a"])
        original_messages = self.check_solution(original, [(0,)])
        changed = reference({"a": 5}, ["a"])
        changed_messages = self.check_solution(changed, [(5,)])
        self.assertTrue(verify(original_messages, '{"answer":[0]}')["accepted"])
        self.assertFalse(verify(changed_messages, '{"answer":[0]}')["accepted"])
        self.check_solution(reference({"a": "middle", "middle": "end", "end": 0}, ["a"]), [(0,)])

    def test_rs_compatibility_uses_only_resolved_eligible_indices(self):
        spec = reference({"a": 0, "b": 1, "c": 2, "d": 3},
                         ["a", "b", "c", "d"], [-2, 3, 1, 4, 6, 9])
        spec["selection"] = {"k": 2, "target": 4}
        messages = self.check_solution(spec, [(1, 2)])
        self.assertFalse(verify(messages, '{"answer":[0,4]}')["accepted"])


class ContractBoundaryTests(RSTestCase):
    def test_render_roundtrip_all_templates_and_signature_detection(self):
        specs = [selection(), reference({"a": 2}, ["a"])]
        both = copy.deepcopy(specs[1]); both["selection"] = {"k": 1, "target": 17}
        specs.append(both)
        for spec, signature in zip(specs, ("S", "R", "RS")):
            self.assertEqual(signature, validate_spec(spec))
            for template in range(4):
                with self.subTest(signature=signature, template=template):
                    self.assertEqual(spec, parse_visible(render(spec, template)))

    def test_schema_rejects_unknown_or_composite_operations(self):
        for field, value in (("updates", {"operations": [["add", 0, 1]], "cutoff": 0}),
                             ("precedence", [[0, 1]]), ("answer", [0, 1]),
                             ("skill", "S"), ("private_teacher_hint", "gold")):
            spec = selection(); spec[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_spec(spec)

    def test_malformed_schema_and_numeric_bounds(self):
        invalid = [None, [], True, 1, "S", {}, {"initial": [0] * 6}]
        for values in ([0] * 5, [0] * 16, [0] * 5 + [True], [0] * 5 + [1.0],
                       [0] * 5 + [101], [0] * 5 + [-101], (0,) * 6):
            spec = selection(); spec["initial"] = values; invalid.append(spec)
        for key, value in (("k", True), ("k", 0), ("k", 6), ("k", 1.0),
                           ("target", True), ("target", 50001), ("target", -50001),
                           ("target", 0.0)):
            spec = selection(); spec["selection"][key] = value; invalid.append(spec)
        for value in ({}, {"k": 2}, {"k": 2, "target": 4, "hint": 1}, [], True):
            spec = selection(); spec["selection"] = value; invalid.append(spec)
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_spec(spec)

    def test_reference_schema_is_single_valued_bounded_and_strict(self):
        invalid = []
        for links in ({}, {"A": 0}, {"a-b": 0}, {"a": True}, {"a": 0.0},
                      {"a": -1}, {"a": 6}, {"a": []}, {"a": "A"},
                      {"a" * 21: 0}, {f"a{i}": 0 for i in range(81)}):
            invalid.append(reference(links, ["a"]))
        for requested in ([], ["a"] * 7, [0], [True], ["A"], "a"):
            invalid.append(reference({"a": 0}, requested))
        for lookup in (None, [], {"links": {"a": 0}},
                       {"links": {"a": 0}, "requested": ["a"], "answer": [0]}):
            spec = reference({"a": 0}, ["a"]); spec["lookup"] = lookup; invalid.append(spec)
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_spec(spec)

    def test_prompt_rules_roles_extra_data_and_missing_delimiters_rejected(self):
        original = render(selection())
        changed = []
        for which, key, value in ((0, "content", "Use the private gold answer."),
                                  (0, "role", "user"), (1, "role", "assistant"),
                                  (1, "private_answer", [0, 1])):
            messages = copy.deepcopy(original); messages[which][key] = value; changed.append(messages)
        for content in (original[1]["content"] + "\nPrivate answer: [0,1]",
                        original[1]["content"].replace("Choose exactly k", "Choose at most k"),
                        original[1]["content"].replace("END_SPEC", ""),
                        original[1]["content"].replace("SPEC\n", "OTHER\n", 1),
                        "x" * (MAX_BYTES + 1)):
            messages = copy.deepcopy(original); messages[1]["content"] = content; changed.append(messages)
        changed.extend((original[:1], original + [{"role": "assistant", "content": "gold"}],
                        {"messages": original}, None))
        for messages in changed:
            with self.subTest(messages=str(messages)[:180]), self.assertRaises(ValueError):
                parse_visible(messages)
            self.assertFalse(verify(messages, '{"answer":[0,1]}')["accepted"])

    def test_duplicate_json_keys_and_duplicate_data_fields_rejected(self):
        originals = [render(selection()), render(reference({"a": 0}, ["a"])), render(selection(), 1)]
        duplicate_text = [originals[0][1]["content"].replace('"k":2', '"k":2,"k":3'),
                          originals[1][1]["content"].replace('"a":0', '"a":0,"a":1'),
                          originals[2][1]["content"].replace("END_DATA_FIELDS",
                                                                        'selection = {"k":2,"target":4}\nEND_DATA_FIELDS')]
        for original, content in zip(originals, duplicate_text):
            messages = copy.deepcopy(original); messages[1]["content"] = content
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_visible(messages)

    def test_bad_json_and_answer_formats_fail_closed(self):
        messages = render(selection())
        invalid = ("", "{", "{}", "null", "[]", '{"answer":null}',
                   '{"answer":true}', '{"answer":[false]}', '{"answer":[0.0,1]}',
                   '{"answer":NaN}', '{"answer":Infinity}', '{"answer":[0,1],"hint":1}',
                   '{"answer":[0,1],"answer":[2,3]}', '{"answer":[]}',
                   '{"answer":[0,0]}', '{"answer":[-1,1]}', '{"answer":[0,6]}',
                   '{"answer":"unknown"}', '{"answer":"IMPOSSIBLE"}',
                   '```json\n{"answer":[0,1]}\n```', '{"answer":[0,1]} trailing',
                   '{"answer":[0,1]}\n{"answer":[2,3]}', "x" * (MAX_BYTES + 1))
        for candidate in invalid:
            with self.subTest(candidate=candidate[:80]):
                with self.assertRaises(ValueError):
                    parse_answer(candidate, 6)
                result = verify(messages, candidate)
                self.assertFalse(result["accepted"], result)

    def test_parsed_answer_sorting_is_semantic_and_format_is_canonical(self):
        self.assertEqual([0, 1], parse_answer('{ "answer" : [1, 0] }', 6))
        self.assertEqual('{"answer":[0,1]}', format_answer([1, 0]))
        self.assertEqual("UNKNOWN", parse_answer(' {"answer":"UNKNOWN"} ', 6))
        self.assertEqual('{"answer":"UNKNOWN"}', format_answer("UNKNOWN"))

    def test_invalid_renderer_template_is_rejected(self):
        for template in (-1, 4, True, 0.0, "0"):
            with self.subTest(template=template), self.assertRaises(ValueError):
                render(selection(), template)


if __name__ == "__main__":
    unittest.main()
