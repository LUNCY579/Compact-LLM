"""Registration metadata controls only; never TRAIN arrays, target labels or models."""
from copy import deepcopy
from pathlib import Path
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
            "torch", "transformers", "psutil", "ctypes", "tokenizers", "safetensors", "jax", "tensorflow"):
        IMPORTS.append(args[0]); raise AssertionError("framework/native import forbidden")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        value = str(args[0]).replace("\\", "/").lower()
        if any(fragment in value for fragment in ("/data/", "/weights", "/model", "/check/", "/final/", "token_pool", "candidate_records", "labels.json")):
            raise AssertionError("actual corpus/model/bank read forbidden")
        READS.append(str(args[0]))


sys.addaudithook(audit)
sys.path.insert(0, str(HERE))
import retained_schedule_binding as binding
import checkpoint_retention as structural


class ScheduleControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration = binding.read(HERE / "STUDY_DEV_REGISTRATION_DRAFT.json")
        cls.lr = binding.read(HERE / "FROZEN_LR_METADATA_001.json")
        cls.paths = {name: PACKAGE / name for name in (
            "study_native_chunk.py", "study_torch_backend.py", "final_adapter_inference.py")}

    def reseal(self, value):
        value["sha256"] = binding.digest({k: v for k, v in value.items() if k != "sha256"})
        return value

    def test_real27_registered_groups_producer_api_and_exact_independent_prefixes(self):
        arms, rates = binding.validate_metadata(self.registration, self.lr)
        binding.producer_api_join(self.registration, arms, self.paths["study_native_chunk.py"])
        expected = {"M1": [(42592, 1916), (85422, 3747), (130562, 5614)],
                    "M2": [(42476, 1882), (85309, 3759), (130574, 5615)]}
        for arm in ("M1", "M2"):
            ids = []; input_total = target_total = 0
            for ordinal, row in enumerate(self.registration["arms"][arm]["updates"], 1):
                ids += row["occurrence_row_ids"]; input_total += row["input_tokens"]; target_total += row["supervised_tokens"]
                value = arms[arm]["counters"][ordinal - 1]
                self.assertEqual((ordinal * 20, input_total, target_total),
                                 (value["consumed_occurrences"], value["input_tokens"], value["supervised_tokens"]))
                self.assertEqual(hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False).encode()).hexdigest(), value["consumed_order_sha256"])
            self.assertEqual(expected[arm], [(arms[arm]["counters"][n - 1]["input_tokens"],
                arms[arm]["counters"][n - 1]["supervised_tokens"]) for n in (9, 18, 27)])
        self.assertEqual(54, sum(len(spec["counters"]) for spec in arms.values()))
        self.assertEqual(0.0, rates[-1]["float_value"])

    def test_missing_duplicate_and_reordered_update_or_occurrence_groups_reject(self):
        changes = (
            lambda updates: updates.pop(),
            lambda updates: updates.__setitem__(1, deepcopy(updates[0])),
            lambda updates: updates.__setitem__(slice(0, 2), [updates[1], updates[0]]),
            lambda updates: updates[0]["occurrence_row_ids"].__setitem__(1, updates[0]["occurrence_row_ids"][0]),
            lambda updates: updates[0]["occurrence_row_ids"].reverse())
        for change in changes:
            value = deepcopy(self.registration); change(value["arms"]["M1"]["updates"]); self.reseal(value)
            with self.subTest(change=change), self.assertRaises(ValueError): binding.validate_metadata(value, self.lr)

    def test_token_and_exposure_marginal_drift_reject_even_with_resealed_document(self):
        for field in ("input_tokens", "supervised_tokens", "parent_exposures"):
            value = deepcopy(self.registration); row = value["arms"]["M2"]["updates"][5]
            if field == "parent_exposures": row[field][next(iter(row[field]))] += 1
            else: row[field] += 1
            self.reseal(value)
            with self.subTest(field=field), self.assertRaises(ValueError): binding.validate_metadata(value, self.lr)

    def test_per_update_LR_and_registered_chunk_join_drift_reject(self):
        for mutation in (
                lambda v: v["arms"]["M1"]["updates"][8].update(learning_rate=0.0),
                lambda v: v["arms"]["M1"]["resume_chunks"][1].update(occurrence_order_sha256="0" * 64),
                lambda v: v["arms"]["M2"]["totals"].update(input_tokens=130562)):
            value = deepcopy(self.registration); mutation(value); self.reseal(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): binding.validate_metadata(value, self.lr)
        lr = deepcopy(self.lr); lr["learning_rates"][26]["float_value"] = 1e-4
        with self.assertRaises(ValueError): binding.validate_metadata(self.registration, lr)

    def test_closed_partial_has54_rows_and_cannot_be_native_schedule_or_admission(self):
        value = binding.build(HERE / "STUDY_DEV_REGISTRATION_DRAFT.json", HERE / "FROZEN_LR_METADATA_001.json", self.paths, structural)
        candidate = value["schedule_candidate"]
        self.assertEqual(54, sum(len(a["counters"]) for a in candidate["arms"].values()))
        self.assertFalse(value["native_launch_authorized"]); self.assertFalse(value["structural_validation"]["accepted"])
        for replacement in (None, "PENDING", "UNKNOWN"):
            altered = deepcopy(candidate)
            for arm in ("M1", "M2"):
                altered["arms"][arm]["identity"]["trajectory_id"] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(structural.ValidationError):
                structural.validate_schedule(altered, structural.digest(altered))
        for arm in ("M1", "M2"):
            identity = candidate["arms"][arm]["identity"]
            self.assertIsNone(identity["runtime_sha256"]); self.assertIsNone(identity["recipe_sha256"])
            self.assertIsNone(identity["base_content_sha256"]); self.assertIsNone(identity["tokenizer_content_sha256"])


if __name__ == "__main__":
    sources = ("retained_schedule_binding.py", "test_retained_schedule_binding.py", "checkpoint_retention.py",
               "STUDY_DEV_REGISTRATION_DRAFT.json", "FROZEN_LR_METADATA_001.json")
    before = {name: binding.reference(HERE / name) for name in sources}
    start = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScheduleControls))
    value = {"kind": "RETAINED_SCHEDULE_METADATA_CPU_CONTROLS001", "tests": result.testsRun,
             "success": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
             "seconds": time.monotonic() - start, "source_before": before,
             "source_after": {name: binding.reference(HERE / name) for name in sources},
             "read_paths": sorted(set(READS)), "forbidden_import_attempts": IMPORTS,
             "actual_corpus_arrays_labels_model_or_native_action": False, "native_launch_authorized": False,
             "actual_export_parity_established": False}
    value["sha256"] = binding.digest(value)
    (HERE / "CPU_CONTROLS001.json").write_bytes((binding.canonical(value) + "\n").encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)
