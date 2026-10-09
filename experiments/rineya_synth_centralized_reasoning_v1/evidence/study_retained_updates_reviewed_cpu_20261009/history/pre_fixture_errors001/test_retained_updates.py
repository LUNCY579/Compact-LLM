"""Authored stdlib-only retention controls; no corpus, framework or model reads."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import ast
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
READS = []
FORBIDDEN_IMPORTS = []


def audit(event, args):
    if event == "import" and args and args[0].split(".", 1)[0] in (
            "torch", "transformers", "jax", "tensorflow", "psutil"):
        FORBIDDEN_IMPORTS.append(args[0]); raise AssertionError("forbidden framework/native import")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        value = str(args[0]).replace("\\", "/").lower()
        if any(x in value for x in ("/data/", "/model", "/weights", "/check/", "/final/")):
            raise AssertionError("forbidden real data/model read")
        READS.append(str(args[0]))


sys.addaudithook(audit)
sys.path.insert(0, str(PACKAGE))
import study_schedule as schedule


def module(path, name):
    value = ModuleType(name); value.__file__ = str(path); sys.modules[name] = value
    exec(compile(path.read_bytes(), str(path), "exec"), value.__dict__)
    return value


chunk = module(HERE / "study_native_chunk.py", "authored_retained_chunk")
backend = module(HERE / "study_torch_backend.py", "authored_retained_backend")
training = module(HERE / "study_training.py", "authored_retained_training")
SOURCE = PACKAGE / "tests/test_study_native_chunk.py"
tree = ast.parse(SOURCE.read_bytes())
definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QuadraticAdam")
namespace = {"deepcopy": deepcopy, "Path": Path, "hashlib": hashlib, "json": json,
             "math": math, "os": os, "random": random, "chunk": chunk, "schedule": schedule}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)


class ScalarBackend(namespace["QuadraticAdam"]):
    def save_retained_update(self, path, binding, *, producer_policy_sha256):
        if self.phase != "complete" or self.gradient != 0.0 or binding != chunk.retained_binding(REGISTRATION, self.arm, self.completed_updates):
            raise ValueError("retained full update/empty gradient/cursor")
        metadata = self.metadata(binding)
        metadata.update(kind=chunk.RETAINED_BACKEND_KIND, resumable=False,
                        producer_policy_sha256=producer_policy_sha256)
        raw = schedule.canonical({"metadata": metadata, "state": self.state(),
                                  "occurrence_order": list(self.order)}).encode()
        def write():
            with Path(path).open("xb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        self.invoke("retained_write", write)
        return {"schema_version": 1, "kind": chunk.RETAINED_BACKEND_KIND, "path": str(path),
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "metadata": metadata,
                "metadata_sha256": schedule.digest(metadata)}


def authored():
    pools, arms = {}, {}
    for arm, token in (("M1", 20), ("M2", 21)):
        identity = "AUTHORED_INERT_" + arm
        row = {"row_id": identity, "prompt_ids": [10], "completion_ids": [token, schedule.EOS_ID],
               "input_ids": [10, token, schedule.EOS_ID], "labels": [-100, token, schedule.EOS_ID],
               "attention_mask": [1, 1, 1], "prompt_tokens": 1, "total_tokens": 3, "supervised_tokens": 2}
        schedule.validate_token_row(row)
        pools[arm] = {identity: row}
        updates = [{"update_ordinal": n, "learning_rate": schedule.learning_rates()[n - 1],
                    "occurrence_row_ids": [identity] * 20, "input_tokens": 60, "supervised_tokens": 40}
                   for n in range(1, 28)]
        arms[arm] = {"updates": updates, "occurrence_order_sha256": schedule.digest([identity] * 540)}
    registration = {"CPU_authored": True, "arms": arms, "source_and_decision_pins":
                    {"AUTHORED_INERT_NO_NATIVE_DECISIONS": "0" * 64}}
    registration["sha256"] = schedule.digest(registration)
    return registration, pools


REGISTRATION, POOLS = authored()


class Preserve:
    def __init__(self, directory):
        self.directory = Path(directory); self.events = []; self.fail = None; self.corrupt = None
    def __call__(self, event):
        self.events.append(deepcopy(event)); name = event["event"]
        if self.fail == name: raise RuntimeError("injected " + name)
        if name == "checkpoint_path": return str(self.directory / ("C%d.archive" % (event["chunk_ordinal"] * 9)))
        if name == "retained_update_path": return str(self.directory / ("retained_%02d.archive" % event["update_ordinal"]))
        payload = event["payload"]; path = self.directory / ("%04d_%s.json" % (len(self.events), name))
        raw = (schedule.canonical(payload) + "\n").encode()
        with path.open("xb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        ref = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
               "payload_sha256": schedule.digest(payload), "durable": True}
        if self.corrupt == name: ref["sha256"] = "0" * 64
        return ref


class RetentionControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.journal = lambda name, fn: fn(); self.guard = lambda stage: None
        self.addCleanup(patch.stopall)
        patch.object(chunk.schedule, "validate_registration", lambda r: chunk.require(r == REGISTRATION, "authored registration only")).start()
        patch.object(chunk.schedule, "_inputs", lambda: ({}, {}, deepcopy(POOLS), {})).start()

    def fixture(self, arm="M1", *, policy=True, name="one", scope=chunk.FAKE_SCOPE):
        path = self.root / name; path.mkdir()
        declared = chunk.producer_policy() if policy else None
        recipe = chunk.recipe(REGISTRATION, arm, scope, retained_update_policy=declared)
        return ScalarBackend(arm, recipe, self.journal, self.guard, scope=scope), Preserve(path), declared

    def run_part(self, obj, preserve, policy, *, arm="M1", ordinal=1, prior=None, **kwargs):
        return chunk.run_chunk(REGISTRATION, arm, ordinal, obj, POOLS[arm].__getitem__,
            self.journal, self.guard, preserve, retained_update_policy=policy,
            resume_descriptor=prior, latest_terminal_costs=None if prior is None else prior["terminal_costs"], **kwargs)

    def test_all27_both_arms_exact_cumulative_state_costs_and_original_resume(self):
        for arm in ("M1", "M2"):
            prior = None; all_entries = []
            for ordinal in (1, 2, 3):
                obj, preserve, policy = self.fixture(arm, name=arm + str(ordinal))
                value = self.run_part(obj, preserve, policy, arm=arm, ordinal=ordinal, prior=prior)
                self.assertEqual("COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED", value["status"], value.get("error"))
                all_entries = value["retained_updates"]; prior = value["resume_descriptor"]
                chunk._validate_resume(prior, REGISTRATION, arm, ordinal + 1, value["recipe"], value["costs"])
                self.assertEqual(ordinal * 9, prior["envelope"]["binding"]["completed_updates"])
                self.assertEqual(chunk.BACKEND_KIND, prior["envelope"]["backend_descriptor"]["kind"])
            self.assertEqual(list(range(1, 28)), [e["record"]["binding"]["completed_updates"] for e in all_entries])
            self.assertEqual(27, value["costs"]["protocol"]["optimizer_step"]["returned"])
            self.assertEqual(540, value["costs"]["protocol"]["backward"]["returned"])
            self.assertEqual((1620, 1080), (obj.input_tokens, obj.supervised_tokens))
            for entry in all_entries:
                record = entry["record"]; payload = json.loads(Path(record["backend_descriptor"]["path"]).read_bytes())
                self.assertEqual(payload["state"]["order"], payload["occurrence_order"])
                self.assertEqual(0.0, payload["state"]["gradient"])
                self.assertFalse(record["GD_export_parity_qualified"])
                self.assertEqual(record["binding"]["completed_updates"] in (9, 27), entry["GD_export_parity_request_publication"] is not None)
            self.assertEqual(2, value["costs"]["primitive_operations"]["GD_export_parity_request_write"]["returned"])

    def test_default_has_original_recipe_cost_schema_and_only_scientific_checkpoints(self):
        obj, preserve, policy = self.fixture(policy=False)
        value = self.run_part(obj, preserve, policy)
        self.assertEqual("COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED", value["status"], value.get("error"))
        self.assertNotIn("retained_updates", value); self.assertNotIn("retained_update_policy", value["recipe"])
        self.assertEqual(set(chunk.OPERATIONS), set(value["costs"]["protocol"]))
        self.assertEqual(1, value["costs"]["protocol"]["checkpoint_save"]["returned"])

    def test_save_and_publication_failure_block_next_update_preserve_actual_costs(self):
        for mode in ("save", "publish", "verify", "GD_request"):
            obj, preserve, policy = self.fixture(name=mode)
            if mode == "save":
                original = obj.save_retained_update
                def fail(path, binding, **kwargs):
                    if binding["completed_updates"] == 3: raise RuntimeError("injected retained save")
                    return original(path, binding, **kwargs)
                obj.save_retained_update = fail
            elif mode == "GD_request": preserve.fail = "GD_export_parity_request"
            elif mode == "publish": preserve.fail = "retained_update_publish"
            else: preserve.corrupt = "retained_update_publish"
            value = self.run_part(obj, preserve, policy)
            expected = 3 if mode == "save" else 9 if mode == "GD_request" else 1
            self.assertEqual("FAILED_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED", value["status"])
            self.assertEqual(expected, value["completed_updates"])
            self.assertEqual(expected, value["costs"]["protocol"]["gradient_reset"]["returned"])
            self.assertEqual(expected * 20, value["costs"]["protocol"]["backward"]["returned"])
            self.assertIsNone(value["resume_descriptor"])
            attempt = value["retained_update_attempts"][-1]
            if mode != "save":
                self.assertTrue(Path(attempt["path"]).exists()); self.assertIsNotNone(attempt["backend_descriptor"])
            if mode == "verify": self.assertIsNotNone(attempt["publication"])
            self.assertTrue(any(n["failed"] for n in value["costs"]["primitive_operations"].values()))

    def test_missing_or_reordered_retained_refs_or_cost_erasure_forbid_resume(self):
        obj, preserve, policy = self.fixture(); value = self.run_part(obj, preserve, policy)
        original = value["resume_descriptor"]
        for mode in ("missing", "reordered", "cost"):
            altered = deepcopy(original); envelope = altered["envelope"]
            if mode == "missing": envelope["retained_updates"].pop()
            elif mode == "reordered": envelope["retained_updates"][0], envelope["retained_updates"][1] = envelope["retained_updates"][1], envelope["retained_updates"][0]
            else:
                for costs in (envelope["posted_save_costs"], altered["durable_terminal"]["costs"], altered["terminal_costs"], altered["promotion"]["terminal_costs"]):
                    costs["primitive_operations"].pop("retained_update_save")
                altered["durable_terminal"]["costs_sha256"] = schedule.digest(altered["durable_terminal"]["costs"])
            sink = Preserve(self.root / mode); sink.directory.mkdir()
            envelope["sha256"] = schedule.digest({k: v for k, v in envelope.items() if k != "sha256"})
            altered["publication"] = sink({"event": "checkpoint_publish", "payload": envelope})
            altered["durable_terminal"]["envelope_sha256"] = envelope["sha256"]
            altered["terminal_publication"] = sink({"event": "terminal_costs_publish", "payload": altered["durable_terminal"]})
            altered["promotion"].update(envelope_sha256=envelope["sha256"], durable_terminal_sha256=schedule.digest(altered["durable_terminal"]))
            altered["promotion_publication"] = sink({"event": "checkpoint_promote", "payload": altered["promotion"]})
            altered["sha256"] = schedule.digest({k: v for k, v in altered.items() if k != "sha256"})
            with self.assertRaises(ValueError): chunk._validate_resume(altered, REGISTRATION, "M1", 2, value["recipe"], altered["terminal_costs"])

    def test_wrong_policy_denied_before_any_row_or_backend_and_factory_forces_policy(self):
        reads = []; obj, preserve, policy = self.fixture()
        obj.required_retained_update_policy = policy
        for candidate in (None, {**policy, "retain_update_ordinals": [9, 18, 27]}):
            with self.assertRaises(ValueError): chunk.run_chunk(REGISTRATION, "M1", 1, obj,
                lambda row: reads.append(row), self.journal, self.guard, preserve, retained_update_policy=candidate)
        self.assertEqual([], reads); self.assertEqual([], preserve.events); self.assertEqual({}, obj.primitives)
        factory = training._NativeFactory({"config": {"retained_update_policy": policy},
            "modules": {"study_native_chunk": chunk}}, {})
        self.assertEqual(policy, factory.required_retained_update_policy)
        self.assertIn("retained_update_policy", training.FIELDS & training.APPROVAL & training.REVIEW)

    def test_real_backend_retained_snapshot_serializes_complete_state_and_refuses_resume_kind(self):
        obj = object.__new__(backend.StudyTorchBackend)
        obj._failed = False; obj._phase = "complete"; obj.completed_updates = 1
        obj._pending = obj._loss = None; obj.require_cuda = True
        obj.order = ["AUTHORED"] * 20; obj.input_tokens = 60; obj.supervised_tokens = 40
        obj.arm = "M1"; obj.seed = 101; obj.recipe_sha256 = "1" * 64; obj.cohort_sha256 = "2" * 64
        obj.backend_source_sha256 = "3" * 64; obj._fresh_attestation = {"sha256": "4" * 64}
        obj.entries = [("AUTHORED_adapter", object())]; obj.adapter_inventory_sha256 = "5" * 64
        obj.model = obj.optimizer = object(); obj.primitive_counts = {}
        obj.call = self.journal; obj.guard = self.guard; obj._verify_base = lambda **kwargs: None
        tensor = SimpleNamespace(device=SimpleNamespace(type="cpu"))
        state = {"schema_version": 1, "adapters": {"AUTHORED_adapter": tensor},
                 "optimizer": {"state": {0: {key: tensor for key in ("step", "exp_avg", "exp_avg_sq")}},
                               "param_groups": [{"lr": backend.learning_rate(1)}]},
                 "gradients": {"AUTHORED_adapter": None}, "rng": {"python": random.Random(101).getstate(),
                   "cpu": "AUTHORED_CPU_RNG", "cuda0": "AUTHORED_CUDA0_RNG"}}
        obj._validate_optimizer = lambda s, n, lr: self.assertEqual((1, backend.learning_rate(1)), (n, lr))
        payloads = []
        def save(payload, stream):
            payloads.append(payload); stream.write(b"AUTHORED_ARCHIVE_COMPLETE_STATE")
        obj.torch = SimpleNamespace(save=save)
        patch.object(backend.mechanics, "snapshot_training_state", lambda *a, **k: deepcopy(state)).start()
        patch.object(backend.mechanics, "_clone_tree", lambda torch, value, **k: deepcopy(value)).start()
        patch.object(backend.mechanics, "_validate_state", lambda *a: self.assertEqual(set(a[3]), set(state))).start()
        cursor = {"completed_updates": 1, "consumed_occurrences": 20, "input_tokens": 60, "supervised_tokens": 40,
                  "consumed_order_sha256": backend.digest(obj.order), "next_update_ordinal": 2,
                  "learning_rate": backend.learning_rate(1)}
        descriptor = obj.save_retained_update(str(self.root / "one.archive"), cursor, producer_policy_sha256="6" * 64)
        self.assertEqual(backend.RETAINED_KIND, descriptor["kind"])
        self.assertEqual(set(state), set(payloads[0]["state"]))
        self.assertEqual(obj.order, payloads[0]["occurrence_order"])
        self.assertEqual({"attempted": 1, "returned": 1, "failed": 0, "pending": 0}, obj.primitive_counts["retained_write"])
        with self.assertRaisesRegex(ValueError, "only at9/18/27"): obj.snapshot_boundary(cursor)
        before = deepcopy(obj.primitive_counts)
        with self.assertRaisesRegex(ValueError, "checkpoint kind"): obj._read(descriptor, cursor)
        self.assertEqual(before, obj.primitive_counts)
        state["gradients"]["AUTHORED_adapter"] = tensor
        fresh = object.__new__(backend.StudyTorchBackend); fresh.__dict__.update(obj.__dict__); fresh.primitive_counts = {}
        with self.assertRaisesRegex(ValueError, "gradients None"):
            fresh.save_retained_update(str(self.root / "bad.archive"), cursor, producer_policy_sha256="6" * 64)
        self.assertFalse((self.root / "bad.archive").exists()); self.assertTrue(fresh._failed)


if __name__ == "__main__":
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob("*.py")}
    start = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RetentionControls))
    receipt = {"kind": "AUTHORED_RETAINED_UPDATE_CPU_CONTROLS_V1", "tests": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors), "success": result.wasSuccessful(),
               "seconds": time.monotonic() - start, "source_before": before,
               "source_after": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob("*.py")},
               "read_paths": sorted(set(READS)), "forbidden_import_attempts": FORBIDDEN_IMPORTS,
               "framework_imported": "torch" in sys.modules, "native_model_or_optimizer_performed": False,
               "real_corpus_or_bank_read": False, "actual_export_parity_established": False,
               "limitations": ["Scalar backend is authored inert math, not native Torch parity.",
                   "Backend snapshot slice retains real method control flow with inert serialization and mechanics seams."]}
    receipt["sha256"] = schedule.digest(receipt)
    (HERE / "CPU_CONTROLS001.json").write_bytes((schedule.canonical(receipt) + "\n").encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)
