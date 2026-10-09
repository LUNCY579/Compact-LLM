"""Authored arithmetic compilation/scoring controls, no central population/model."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import itertools
import json
import os
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
READS = []; IMPORTS = []


def audit(event, args):
    if event == "import" and args and args[0].split(".", 1)[0] in (
            "torch", "transformers", "bridge_data", "dev_scorer", "tokenizers", "psutil", "ctypes", "safetensors"):
        IMPORTS.append(args[0]); raise AssertionError("central/native module import forbidden")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = str(args[0]).replace("\\", "/").lower()
        if any(word in name for word in ("/data/", "token_pool", "candidate_records", "/weights", "/model", "/check/", "/final/")):
            raise AssertionError("existing corpus/model/bank read forbidden")
        READS.append(str(args[0]))


sys.addaudithook(audit); sys.path.insert(0, str(HERE))
import arithmetic_guard as G


class ArithmeticControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = G.policy(); cls.population = G.compile_population(cls.policy)

    def test_all128parents512rows_independent_answers_and_exact_crossed_grouping(self):
        value = self.population
        self.assertEqual(128, len(value["parents"])); self.assertEqual(512, len(value["rows"]))
        self.assertEqual(512, len({row["row_id"] for row in value["rows"]}))
        for height in (1, 2, 3, 4): self.assertEqual(32, sum(parent["depth"] == height for parent in value["parents"]))
        for offset, parent in enumerate(value["parents"]):
            siblings = value["rows"][offset * 4:offset * 4 + 4]
            self.assertEqual({(c, r) for c in (0, 1) for r in self.policy["renderings"]},
                             {(row["condition"], row["rendering"]) for row in siblings})
            self.assertEqual({parent["parent_id"]}, {row["parent_id"] for row in siblings})
            self.assertEqual(1, len({row["answer"] for row in siblings}))
            for row in siblings:
                self.assertEqual({"correct": True, "malformed": False},
                                 G.score(row, json.dumps({"answer": row["answer"]}), self.policy))
                self.assertEqual({"correct": False, "malformed": False},
                                 G.score(row, json.dumps({"answer": row["answer"] + 1}), self.policy))
        self.assertIsNone(value["M0_eligible_parent_count"]); self.assertFalse(value["precision_established"])

    def test_allocation_is_target_independent_reproducible_and_frozen_before_solvers(self):
        def forbidden(*args): raise AssertionError("allocation called answer solver")
        with patch.object(G, "recursive", forbidden), patch.object(G, "postfix", forbidden):
            parents, audit = G.allocate(self.policy)
        self.assertEqual(parents, self.population["parents"])
        self.assertEqual(audit, self.population["allocation_audit"])
        self.assertEqual(self.population, G.compile_population(self.policy))

    def test_small_tree_enumeration_independent_paths_and_relevant_condition_change(self):
        count = 0
        for op, left, right in itertools.product(("add", "sub", "mul"), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)):
            tree = {"op": op, "args": [left, right]}
            expected = left + right if op == "add" else left - right if op == "sub" else left * right
            self.assertEqual(expected, G.recursive(tree)); self.assertEqual(expected, G.postfix(tree)); count += 1
        self.assertEqual(75, count)
        add = {"op": "add", "args": [7, 2]}; sub = {"op": "sub", "args": [7, 2]}
        self.assertNotEqual(G.postfix(add), G.postfix(sub))
        with patch.object(G, "recursive", lambda *args: (_ for _ in ()).throw(AssertionError("recursive path used"))):
            self.assertEqual(5, G.postfix(sub))

    def test_strict_visible_JSON_wrong_type_extra_duplicate_trailing_and_length(self):
        row = self.population["rows"][0]
        for response in ('{"answer":true}', '{"answer":1.0}', '{"answer":"1"}', '{"answer":null}',
                         '{"answer":0,"extra":0}', '{"answer":0,"answer":0}', '{}', '[]',
                         '{"answer":NaN}', '{"answer":0} trailing', 'x' * 4097, True):
            with self.subTest(response=str(response)[:40]):
                self.assertEqual({"correct": False, "malformed": True}, G.score(row, response, self.policy))

    def test_schema_depth_leaf_and_intermediate_overflow_refuse_on_both_paths(self):
        deep = 1
        for _ in range(5): deep = {"op": "add", "args": [deep, 1]}
        big = {"op": "mul", "args": [{"op": "mul", "args": [50, 50]},
                                       {"op": "mul", "args": [50, 50]}]}
        cancelled = {"op": "sub", "args": [big, deepcopy(big)]}
        for tree in (True, 51, -51, deep, big, cancelled, {"op": "div", "args": [1, 2]},
                     {"op": "add", "args": [1]}, {"op": "add", "args": [1, 2], "extra": 0}):
            for solve in (G.recursive, G.postfix):
                with self.subTest(tree=str(tree)[:50], solve=solve.__name__), self.assertRaises(ValueError): solve(tree)

    def test_host_content_hash_target_and_unregistered_parent_mutations_raise(self):
        original = self.population["rows"][0]
        for mutate in (
                lambda row: row.update(answer=row["answer"] + 1),
                lambda row: row.update(prompt=row["prompt"] + "changed"),
                lambda row: row.update(row_id="0" * 64),
                lambda row: row["parent"].update(expression_sha256="0" * 64),
                lambda row: row.update(condition=True)):
            row = deepcopy(original); mutate(row)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): G.score(row, '{"answer":0}', self.policy)
        parent = deepcopy(original["parent"]); tree = {"op": "add", "args": [-50, 50]}
        parent.update(canonical_expression=tree, expression_sha256=G.digest(tree), depth=1,
            parent_id=G.digest({"source_id": self.policy["source_id"], "canonical_expression": tree}))
        if parent not in self.population["parents"]:
            with self.assertRaisesRegex(ValueError, "outside registered finite allocation"): G.validate_parent(parent, self.policy)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "malformed_policy.json"
            for mutate in (lambda p: p.update(native_scoring_authorized=True),
                           lambda p: p.update(depths=[True, 2, 3, 4]),
                           lambda p: p["semantics"].update(maximum_depth=5),
                           lambda p: p["selection"]["candidate_grammar"].update(root_op="mul")):
                value = deepcopy(self.policy); mutate(value)
                value["sha256"] = G.digest({k: v for k, v in value.items() if k != "sha256"})
                path.write_bytes((G.canonical(value) + "\n").encode())
                with patch.object(G, "POLICY", path), self.subTest(policy=mutate), self.assertRaises(ValueError): G.policy()


if __name__ == "__main__":
    files = ("arithmetic_guard.py", "SYNTH_ARITHMETIC_GUARD_POLICY001.json", "test_arithmetic_guard.py")
    before = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files}
    start = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ArithmeticControls))
    receipt = {"kind": "SYNTH_ARITHMETIC_GUARD_CPU_CONTROLS001", "tests": result.testsRun,
        "success": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
        "seconds": time.monotonic() - start, "source_before": before,
        "source_after": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files},
        "read_paths": sorted(set(READS)), "forbidden_import_attempts": IMPORTS,
        "central_module_or_population_imported": False, "model_GPU_tokenizer_or_M0_performed": False,
        "native_scoring_authorized": False, "actual_model_quality_established": False}
    receipt["sha256"] = G.digest(receipt)
    (HERE / "CPU_CONTROLS001.json").write_bytes((G.canonical(receipt) + "\n").encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)
