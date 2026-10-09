"""Complete SOURCE/FILTER/UNUSED CPU ledger controls; no corpus or target reads."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == "rineya_synth_centralized_reasoning_v1")
READS = []; IMPORTS = []


def audit(event, args):
    if event == "import" and args and args[0].split(".", 1)[0] in (
            "torch", "transformers", "bridge_data", "dev_scorer", "tokenizers", "psutil", "ctypes", "safetensors"):
        IMPORTS.append(args[0]); raise AssertionError("native/central import forbidden")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = str(args[0]).replace("\\", "/").lower()
        if any(word in name for word in ("/data/", "token_pool", "candidate_records", "/weights", "/model", "/check/", "/final/",
                                        "arithmetic_guard_population", "negative_examples", "labels.json")):
            raise AssertionError("corpus/target/model/bank read forbidden")
        READS.append(str(args[0]))


sys.addaudithook(audit); sys.path.insert(0, str(HERE))
import arithmetic_source_ledger as L


class LedgerControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = L.reference(PACKAGE / "arithmetic_guard.py")
        cls.policy = L.reference(PACKAGE / "SYNTH_ARITHMETIC_GUARD_POLICY001.json")
        cls.ledger = L.build(cls.source, cls.policy)

    def reseal(self, value):
        value["sha256"] = L.digest({k: v for k, v in value.items() if k != "sha256"})
        return value

    def test_all512_source_filter_unused_closure_and_128_original_parent_order(self):
        value = self.ledger
        self.assertEqual(512, len(value["candidates"]))
        self.assertEqual([(depth, index) for depth in (1, 2, 3, 4) for index in range(128)],
                         [(row["depth"], row["candidate_index"]) for row in value["candidates"]])
        self.assertEqual(list(range(1, 513)), [row["candidate_ordinal"] for row in value["candidates"]])
        self.assertEqual(128, value["counts"]["SELECTED_PARENT"])
        self.assertEqual(512, sum(value["counts"].values()))
        guard, frozen = L.load_guard(self.source, self.policy)
        parents, aggregate = guard.allocate(frozen)
        self.assertEqual(parents, value["selected_parents"])
        self.assertEqual(aggregate, value["original_allocation_audit"])
        self.assertEqual(guard.digest([p["parent_id"] for p in parents]), value["selected_parent_order_sha256"])
        for row in value["candidates"]:
            self.assertEqual(guard.normalize(row["raw_expression"]), row["canonical_expression"])
            self.assertEqual(guard.digest(row["raw_expression"]), row["raw_expression_sha256"])
            self.assertEqual(guard.digest(row["canonical_expression"]), row["canonical_expression_sha256"])
            self.assertEqual({"native_model_occurrences": 0, "native_training_occurrences": 0,
                              "future_occurrence_reference": None}, row["consume"])
        self.assertTrue(L.validate(value, self.source, self.policy)["original_selected_population_unchanged"])

    def test_no_reference_verifier_scorer_or_population_targets_used_and_prompt_inputs_unchanged(self):
        guard, frozen = L.load_guard(self.source, self.policy)
        def forbidden(*args): raise AssertionError("target or model based selection forbidden")
        with patch.object(guard, "recursive", forbidden), patch.object(guard, "postfix", forbidden), \
             patch.object(guard, "compile_population", forbidden), patch.object(guard, "score", forbidden):
            replay = L.reconstruct(guard, frozen, self.source, self.policy)
            self.assertEqual(self.ledger, replay)
            original, _ = guard.allocate(frozen)
            for before, after in zip(original, replay["selected_parents"]):
                for condition in (0, 1):
                    for rendering in frozen["renderings"]:
                        self.assertEqual(guard.render(before, condition, rendering, frozen),
                                         guard.render(after, condition, rendering, frozen))
        self.assertEqual(0, replay["consume"]["CPU_solver_calls"])

    def test_dropped_duplicated_or_reordered_candidates_rejected_after_resealing(self):
        for mutate in (lambda v: v["candidates"].pop(),
                       lambda v: v["candidates"].__setitem__(1, deepcopy(v["candidates"][0])),
                       lambda v: v["candidates"].reverse()):
            value = deepcopy(self.ledger); mutate(value); self.reseal(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): L.validate(value, self.source, self.policy)

    def test_candidate_hash_filter_selection_parent_and_consume_tampering_rejected(self):
        selected = next(i for i, row in enumerate(self.ledger["candidates"]) if row["selection_state"] == "SELECTED_PARENT")
        unused = next(i for i, row in enumerate(self.ledger["candidates"]) if row["selection_state"] == "ELIGIBLE_UNUSED")
        changes = (
            lambda v: v["candidates"][0].update(raw_expression_sha256="0" * 64),
            lambda v: v["candidates"][0].update(canonical_expression_sha256="0" * 64),
            lambda v: v["candidates"][unused].update(selection_state="SELECTED_PARENT"),
            lambda v: v["candidates"][selected].update(filter_reason="duplicate_canonical_content"),
            lambda v: v["candidates"][selected]["selected_parent"].update(parent_id="0" * 64),
            lambda v: v["candidates"][selected]["consume"].update(native_model_occurrences=1),
            lambda v: v.update(selected_parent_order_sha256="0" * 64))
        for change in changes:
            value = deepcopy(self.ledger); change(value); self.reseal(value)
            with self.subTest(change=change), self.assertRaises(ValueError): L.validate(value, self.source, self.policy)

    def test_changed_source_policy_or_retyped_schema_cannot_claim_native_consume(self):
        for ref_name in ("source", "policy"):
            source, policy = deepcopy(self.source), deepcopy(self.policy)
            (source if ref_name == "source" else policy)["sha256"] = "0" * 64
            with self.subTest(ref_name=ref_name), self.assertRaises(ValueError): L.build(source, policy)
        for version in (True, 1.0, "1", None, 2):
            value = deepcopy(self.ledger); value["schema_version"] = version; self.reseal(value)
            with self.subTest(version=version), self.assertRaises(ValueError): L.validate(value, self.source, self.policy)
        self.assertEqual(0, self.ledger["consume"]["native_model_occurrences"])
        self.assertIsNone(self.ledger["consume"]["future_occurrence_reference"])
        self.assertFalse(self.ledger["native_scoring_authorized"])


if __name__ == "__main__":
    before = {name: L.reference(HERE / name) for name in ("arithmetic_source_ledger.py", "test_arithmetic_source_ledger.py")}
    start = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LedgerControls))
    value = {"kind": "SYNTH_ARITHMETIC_SOURCE_LEDGER_CPU_CONTROLS001", "tests": result.testsRun,
        "success": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
        "seconds": time.monotonic() - start, "source_before": before,
        "source_after": {name: L.reference(HERE / name) for name in before},
        "selected_original_sources": {"generator": LedgerControls.source, "policy": LedgerControls.policy},
        "read_paths": sorted(set(READS)), "forbidden_import_attempts": IMPORTS,
        "model_GPU_tokenizer_M0_or_existing_target_population_read": False, "native_consumption_performed": False,
        "native_scoring_authorized": False}
    value["sha256"] = L.digest(value)
    (HERE / "CPU_CONTROLS001.json").write_bytes((L.canonical(value) + "\n").encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)
