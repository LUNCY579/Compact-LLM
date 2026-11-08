"""Model-free checks of changed data ordering/custody and unchanged train maths."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
def load(name):
    spec = importlib.util.spec_from_file_location("mixture_fixture_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module
trainer, exporter = load("train"), load("materialize")


class MechanicsTests(unittest.TestCase):
    def test_uniform_order_preserves_every_native_and_binding_view_once(self):
        order = trainer.expected_membership(9, 4)
        self.assertEqual(order[0], ("native_dual899", 0))
        self.assertEqual([i for source, i in order if source == "native_dual899"], list(range(9)))
        self.assertEqual(sorted(i for source, i in order if source == "binding_v2"), list(range(4)))
        self.assertEqual([i for i, (source, _) in enumerate(order) if source == "binding_v2"], [3, 6, 9, 12])
        self.assertEqual(order, trainer.expected_membership(9, 4))

    def test_budget_includes_all_qualification_and_both_small_checkpoints(self):
        self.assertEqual(577417 + 2203 + 2203 + 2 * (31 + 116 + 2048), 586213)
        self.assertLessEqual(586213, trainer.LIMITS["processed_training_tokens_all_attempts"])
        self.assertLessEqual(2 * trainer.CHECKPOINT_BYTES + 4 * 2**20, trainer.LIMITS["new_disk_bytes"])

    def test_old_parameterization_checkpoint_is_not_a_mixture_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            checkpoint = root / "checkpoint"; checkpoint.mkdir()
            body = {"schema": 1, "record": "DOSELORA_TRAINING_CHECKPOINT", "experiment": "DOSELORA-001",
                    "status": "COMPLETE", "binding": {}, "progress": {}, "rng": {}, "adapter_spec": {}, "entries": []}
            (checkpoint / "manifest.json").write_text(json.dumps({**body, "sha256": exporter.digest(body)}))
            with self.assertRaisesRegex(ValueError, "binding-mixture"):
                exporter.inspect_checkpoint(root / "source", checkpoint, "lora")
            with self.assertRaisesRegex(ValueError, "single mixture"):
                exporter.inspect_checkpoint(root / "source", checkpoint, "native")

    def test_existing_export_refused_before_any_lease_or_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                exporter.materialize_candidate(Path("absent-source"), Path("absent-checkpoint"), Path(directory))

    def test_qualified_adapter_and_numerical_functions_are_unchanged(self):
        old = HERE.parent / "retention_highdose_lora_resume_v1"
        self.assertEqual((HERE / "adapters.py").read_bytes(), (old / "adapters.py").read_bytes())
        def functions(path):
            return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        before, after = functions(old / "train.py"), functions(HERE / "train.py")
        class NormalizeRecordLabels(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    node.value = node.value.replace("BINDING_MIXTURE_", "DOSELORA_")
                return node
        for name in ("qualify_and_inject", "qualify_final_merge", "progress_at"):
            self.assertEqual(ast.dump(before[name], include_attributes=False),
                             ast.dump(NormalizeRecordLabels().visit(deepcopy(after[name])), include_attributes=False))
        # The complete optimization loop is reused verbatim; only final geometry/receipt labels differ.
        old_loop = next(node for node in ast.walk(before["train_groups"]) if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple))
        new_loop = next(node for node in ast.walk(after["train_groups"]) if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple))
        self.assertEqual(ast.dump(old_loop, include_attributes=False), ast.dump(new_loop, include_attributes=False))
        self.assertNotIn("torch", sys.modules)


class SupervisorCustodyTests(unittest.TestCase):
    """Replay actual export admission against a tiny completed journal."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.journal = self.root / "journal"
        self.run = self.journal / "runs" / "train-001"
        self.events = self.journal / "events"
        self.run.mkdir(parents=True)
        self.events.mkdir()
        self.walls = ("total_owned_work_wall_seconds", "model_worker_wall_seconds_and_conservative_gpu_seconds")
        self.grant = {"experiment": trainer.EXPERIMENT,
                      "limits": {self.walls[0]: 100, self.walls[1]: 100, "model_worker_launches": 1}}
        self.binding = {"plan_path": str(self.root / "plan.json"), "prepared": str(self.root / "prepared"),
                        "grant_sha256": exporter.digest(self.grant)}
        self.put(self.journal / "grant.json", self.grant)
        launch = {"schema": 1, "out": str(self.run), "supervisor_pid": 7,
                  "argv": ["python", "train.py", "--journal", str(self.journal), "--out", str(self.run),
                           "--run-id", self.run.name, "--plan", self.binding["plan_path"],
                           "--prepared", self.binding["prepared"], "--grant", str(self.journal / "grant.json"),
                           "--_worker"]}
        self.supervisor = {"status": "EXITED", "returncode": 0, "child_pid": 11,
                           "cleanup": None, "wall_seconds": 1.0}
        records = {"launch.json": launch, "supervisor.json": self.supervisor,
                   "spawn.json": {"child_pid": 11, "supervisor_pid": 7, "launch_sha256": exporter.digest(launch)},
                   "worker_claim.json": {"pid": 11, "parent_pid": 7, "supervisor_pid": 7,
                                         "launcher_redirected": False, "launch_sha256": exporter.digest(launch)}}
        for filename, value in records.items():
            self.put(self.run / filename, value)
        start = exporter.seal({"index": 0, "previous": None, "kind": "start", "run_id": self.run.name,
                              "phase": "train", "charges": {self.walls[0]: 10, self.walls[1]: 10,
                                                             "model_worker_launches": 1}})
        self.settlement = {"index": 1, "previous": start["sha256"], "kind": "settle", "run_id": self.run.name,
                           "receipt_sha256": exporter.file_hash(self.run / "supervisor.json"), "charges": {},
                           "actual_wall": {self.walls[0]: 1, self.walls[1]: 1}}
        self.put(self.events / "000000.json", start)
        self.put(self.events / "000001.json", exporter.seal(self.settlement))
        used = {self.walls[0]: 1, self.walls[1]: 1, "model_worker_launches": 1}
        self.summary = {"experiment": trainer.EXPERIMENT, "run_id": self.run.name, "returncode": 0,
                        "worker_failed": False, "supervisor": self.supervisor,
                        "budget_after": {"used_or_reserved": used, "active_reservations": [],
                                         "budget_breached": False,
                                         "remaining": {k: self.grant["limits"][k] - v for k, v in used.items()}}}
        self.put(self.run / "run_summary.json", self.summary)

    @staticmethod
    def put(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_settled_success_binds_all_supervisor_and_journal_bytes(self):
        receipt = exporter._supervised_success(self.run, self.binding)
        self.assertEqual(receipt["training_supervisor_sha256"], exporter.file_hash(self.run / "supervisor.json"))
        self.assertEqual(set(receipt["training_journal_event_sha256"]), {"000000.json", "000001.json"})

    def test_late_worker_failure_overrules_success(self):
        self.put(self.run / "failure.json", {"status": "FAILED"})
        with self.assertRaisesRegex(ValueError, "failure receipt"):
            exporter._supervised_success(self.run, self.binding)

    def test_failed_supervisor_cannot_be_rescued_by_matching_summary(self):
        self.supervisor["returncode"] = 2
        self.put(self.run / "supervisor.json", self.supervisor)
        self.put(self.run / "run_summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "clean successful exit"):
            exporter._supervised_success(self.run, self.binding)

    def test_unsettled_launch_overrules_claimed_success(self):
        event = dict(self.settlement, kind="charge")
        self.put(self.events / "000001.json", exporter.seal(event))
        with self.assertRaisesRegex(ValueError, "unsettled launch"):
            exporter._supervised_success(self.run, self.binding)

    def test_budget_breach_overrules_claimed_success(self):
        self.settlement["actual_wall"][self.walls[0]] = 101
        self.put(self.events / "000001.json", exporter.seal(self.settlement))
        with self.assertRaisesRegex(ValueError, "budget is breached"):
            exporter._supervised_success(self.run, self.binding)

    def test_settlement_requires_actual_supervisor_hash(self):
        self.settlement["receipt_sha256"] = "0" * 64
        self.put(self.events / "000001.json", exporter.seal(self.settlement))
        with self.assertRaisesRegex(ValueError, "different supervisor receipt"):
            exporter._supervised_success(self.run, self.binding)


if __name__ == "__main__":
    unittest.main()
