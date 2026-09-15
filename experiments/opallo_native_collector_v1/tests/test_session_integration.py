"""Synthetic orchestration / real local Git / real process tests, never model results.

run_job is replaced by explicit scripted captures. Temporary outputs are deleted;
no synthetic calibration result, approved reviewer or native receipt is published.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.opallo_native_collector_v1 import contract as c
from experiments.opallo_native_collector_v1 import native as n
from experiments.opallo_native_collector_v1 import session as s
from experiments.opallo_native_collector_v1 import supervisor as v
from test_collector import Scripted, Tokenizer, identity, recipe
import torch


def new_git(root: Path) -> str:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    c.git(root, "config", "user.name", "Mechanical Fixture Only")
    c.git(root, "config", "user.email", "fixture@example.invalid")
    (root / "README").write_text("Synthetic orchestration fixture; not scientific approval.\n")
    c.git(root, "add", "."); c.git(root, "commit", "-qm", "mechanical fixture")
    return c.git(root, "rev-parse", "HEAD").decode().strip()


class LargeVocabularyTests(unittest.TestCase):
    def test_full_size_reduction_and_unrounded_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, scale in enumerate((0.0, 0.1, 1.0, 15.0, 80.0)):
                x = torch.linspace(-scale, scale, 151936, dtype=torch.float64)
                value, blob = n.signal_from_logits(x, 513, len(x))
                path = root / f"{index}.f64"; path.write_bytes(blob); value["path"] = path.name
                n.validate_signal(value, root, 513, len(x))
                self.assertEqual(value["unknown_probability"], math.exp(float(torch.log_softmax(x, 0)[513])))
    def test_replay_tolerance_never_changes_gate_boundary(self):
        frozen = c.screen()
        tau = 0.1
        rows = [{"prediction": "original", "unknown_probability": math.nextafter(tau, 0.0)},
                {"prediction": "original", "unknown_probability": tau},
                {"prediction": "original", "unknown_probability": math.nextafter(tau, 1.0)}]
        self.assertEqual(frozen.gated(rows, tau), ["original", "UNKNOWN", "UNKNOWN"])
    def test_tiny_probability_tamper_is_rejected_by_exact_exp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); value, blob = n.signal_from_logits(torch.zeros(20), 3, 20)
            (root / "x.f64").write_bytes(blob); value["path"] = "x.f64"
            value["unknown_probability"] = math.nextafter(value["unknown_probability"], 1.0)
            with self.assertRaises(c.Blocked): n.validate_signal(value, root, 3, 20)
    def test_job_request_pins_cover_every_registered_job(self):
        for phase, plans in s.PLANS.items():
            for plan in plans:
                self.assertEqual(c.digest(s.requests_for(plan)), c.pins()["job_prompts_sha256"][phase][plan[0]])
    def test_review_id_pins_have_no_gold_or_prompts(self):
        rows = c.pins()["review_rows"]
        self.assertEqual(len(rows), 192)
        self.assertTrue(all(set(row) == {"id", "scenario_id"} for row in rows))


class SessionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"; self.head = new_git(self.repo)
        self.root = self.base / "session"; self.root.mkdir()
        self.manifest = {"repo": str(self.repo), "authorization_commit": self.head,
                         "run_id": "SYNTHETIC_ONLY", "recipe": recipe(), **c.CLOSED}
        c.write_new(self.root / "SESSION.json", self.manifest)
        self.calls = []
        for mocker in (patch.object(s, "load_session", return_value=self.manifest),
                       patch.object(s.Path, "home", return_value=self.base / "home"),
                       patch.object(s, "run_job", side_effect=self.fake_job)):
            mocker.start(); self.addCleanup(mocker.stop)
    def fake_job(self, root, manifest, phase, plan):
        label, _, _, _, _, confidence = plan
        out = root / phase / label; out.mkdir()
        requests = s.requests_for(plan)
        self.calls.extend((phase, label, r["id"]) for r in requests)
        # One synthetic tensor trace per job is cloned as an orchestration fixture.
        # This is not 400 real generations, and no such scientific count is claimed.
        row = n.capture_one(Scripted(), requests[0], root, f"{phase}/{label}/fixture", confidence)
        rows = []
        for request in requests:
            item = copy.deepcopy(row)
            item["id"] = request["id"]; item["prompt_sha256"] = c.digest(request)
            item["rendered_prompt"], item["input_ids"] = n.render_input(Tokenizer(), request, Scripted.vocabulary)
            rows.append(item)
        ident = identity(); ident["runtime_sha256"] = manifest["recipe"]["runtime_sha256"]
        ident["recipe_sha256"] = c.digest(manifest["recipe"])
        value = {"kind": "native_capture", "status": "PASS_NATIVE_JOB_MECHANICS",
                 "requests_sha256": c.digest(requests), "identity": ident, "rows": rows}
        c.write_new(out / "CAPTURE.json", value)
        return value
    def calibration(self):
        s.execute_phase(self.root, "qualify"); s.execute_phase(self.root, "calibrate")
    def commit_freeze(self):
        packet = self.root / "calibrate" / "FREEZE_PACKET.json"
        (self.repo / "freeze.json").write_bytes(packet.read_bytes())
        c.git(self.repo, "add", "freeze.json"); c.git(self.repo, "commit", "-qm", "SYNTHETIC freeze fixture")
        head = c.git(self.repo, "rev-parse", "HEAD").decode().strip()
        s.freeze(self.root, head, "freeze.json")
        return head
    def test_full_mock_orchestration_reuse_git_interlock_and_one_shot_score(self):
        self.calibration()
        self.assertEqual(len(self.calls), 144)
        self.assertTrue(all(not row[2].startswith("OB1-check") for row in self.calls))
        on = c.read(self.root / "qualify/on8/CAPTURE.json")["rows"]
        baseline = c.read(self.root / "calibrate/baseline.json")["rows"]
        self.assertEqual(baseline[:8], on)
        self.assertEqual(len(baseline), 64)
        with self.assertRaises((c.Blocked, FileNotFoundError)): s.execute_phase(self.root, "check")
        self.assertFalse((self.root / "check").exists())
        self.assertEqual(len(self.calls), 144)
        self.commit_freeze(); s.execute_phase(self.root, "check")
        self.assertEqual(len(self.calls), 400)
        self.assertEqual(len({x[2] for x in self.calls if x[:2] == ("check", "baseline128")}), 128)
        result = s.score(self.root)
        self.assertEqual(result["status"], "OFFLINE_REPLAY_ONLY")
        self.assertTrue(all(result[k] is False for k in c.CLOSED))
        self.assertIsNone(result["gate_threshold"])  # All-UNKNOWN fixture => off wins ties.
        with self.assertRaises(FileExistsError): s.score(self.root)
        with self.assertRaises(c.Blocked): s.execute_phase(self.root, "check")
    def test_uncommitted_packet_cannot_open_check(self):
        self.calibration()
        (self.repo / "freeze.json").write_bytes((self.root / "calibrate/FREEZE_PACKET.json").read_bytes())
        with self.assertRaises(c.Blocked): s.freeze(self.root, self.head, "freeze.json")
        self.assertFalse((self.root / "GIT_FREEZE.json").exists())
    def test_postfreeze_calibration_tamper_blocks_check(self):
        self.calibration(); self.commit_freeze()
        p = self.root / "calibrate/baseline.json"; p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaises(c.Blocked): s.execute_phase(self.root, "check")
        self.assertEqual(len(self.calls), 144)
    def test_phase_inventory_cannot_drop_a_hash(self):
        s.execute_phase(self.root, "qualify")
        path = self.root / "qualify/RESULT.json"; value = c.read(path)
        del value["artifact_sha256"][next(iter(value["artifact_sha256"]))]
        path.write_text(json.dumps(value))
        with self.assertRaises(c.Blocked): s.verify_phase(self.root, "qualify")
    def test_failed_job_retains_reservation_and_no_retry(self):
        with patch.object(s, "run_job", side_effect=c.Blocked("scripted failure before model")):
            with self.assertRaises(c.Blocked): s.execute_phase(self.root, "qualify")
        self.assertEqual(c.read(self.root / "qualify/START.json")["generations_reserved"], 24)
        self.assertEqual(c.read(self.root / "qualify/BLOCKED.json")["status"], "BLOCKED")
        with self.assertRaises(c.Blocked): s.execute_phase(self.root, "qualify")
        self.assertFalse((self.root / "qualify/RESULT.json").exists())
    def test_identity_drift_blocks_qualification(self):
        original = self.fake_job
        def drift(root, manifest, phase, plan):
            value = original(root, manifest, phase, plan)
            if plan[0] == "off8": value["identity"]["runtime_sha256"] = "b"*64
            return value
        with patch.object(s, "run_job", side_effect=drift):
            with self.assertRaises(c.Blocked): s.execute_phase(self.root, "qualify")
        self.assertFalse((self.root / "qualify/RESULT.json").exists())
    def test_worker_cannot_run_without_phase_start(self):
        with self.assertRaises(FileNotFoundError): s.verify_active_worker(self.root, self.manifest, "qualify", "on8")
    def test_worker_cannot_jump_to_check_without_prior_receipts(self):
        c.write_new(self.root / "check/START.json", {"phase": "check", "session_sha256": c.digest(self.manifest), "generations_reserved":256})
        with self.assertRaises(FileNotFoundError): s.verify_active_worker(self.root, self.manifest, "check", "baseline128")


class ClaimAndProcessTests(unittest.TestCase):
    def test_new_output_path_does_not_reset_subject_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); repo = base / "repo"; head = new_git(repo)
            model = base / "model"; model.mkdir()
            auth = {"run_id": "SYNTHETIC_ONLY"}
            with patch.object(s, "reviewed_inputs", return_value=(auth, recipe())), patch.object(c, "asset_manifest", return_value={}):
                s.initialize(repo, base / "first", model, head, "fake.json")
                with self.assertRaises(c.Blocked): s.initialize(repo, base / "second", model, head, "fake.json")
            self.assertFalse((base / "second").exists())
    def test_owned_descendant_killed_unrelated_process_untouched(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            try:
                code = "import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path('pid').write_text(str(p.pid)); time.sleep(60)"
                with self.assertRaises(c.Blocked):
                    v.run_bounded([sys.executable, "-c", code], root, root / "job", seconds=2.0, min_free_gib=0, max_rss_gib=1)
                self.assertTrue((root / "pid").exists(), str(c.read(root / "job/SUPERVISOR.json")) + (root / "job/stderr.log").read_text())
                child = int((root / "pid").read_text())
                try:
                    self.assertIn(psutil.Process(child).status(), (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD))
                except psutil.NoSuchProcess:
                    pass
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.kill(); unrelated.wait(timeout=5)
    def test_supervisor_preserves_stderr_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(c.Blocked):
                v.run_bounded([sys.executable, "-c", "import sys; print('failure fixture',file=sys.stderr); raise SystemExit(3)"],
                              root, root / "job", seconds=5, min_free_gib=0, max_rss_gib=1)
            self.assertIn("failure fixture", (root / "job/stderr.log").read_text())



class NativeBoundaryTests(unittest.TestCase):
    def test_state_audit_covers_nonpersistent_buffers(self):
        model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        model.register_buffer("nonpersistent", torch.ones(2), persistent=False)
        before = n.tensor_identity(model)
        with torch.no_grad(): model.nonpersistent[0] += 1
        after = n.tensor_identity(model)
        self.assertEqual(before["tensor_manifest_sha256"], after["tensor_manifest_sha256"])
        self.assertNotEqual(before["buffer_manifest_sha256"], after["buffer_manifest_sha256"])
    def test_meta_buffer_blocks_state_audit(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer("nonpersistent", torch.empty(2, device="meta"), persistent=False)
        with self.assertRaises(c.Blocked): n.tensor_identity(model)
    def test_nonfinite_buffer_blocks_state_audit(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer("nonpersistent", torch.tensor([float("nan")]), persistent=False)
        with self.assertRaises(c.Blocked): n.tensor_identity(model)
    def test_checkpoint_bytes_are_compared_not_just_shape(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
            save_file(model.state_dict(), str(root / "model.safetensors"))
            n.tensor_identity(model, root)
            with torch.no_grad(): model.weight[0,0] += 1
            with self.assertRaises(c.Blocked): n.tensor_identity(model, root)
    def test_pending_authorization_blocks_before_asset_load_and_run_claim(self):
        from experiments.opallo_native_collector_v1 import cli
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); repo = base / "repo"; new_git(repo)
            c.write_new(repo / "pending.json", {"protocol": c.pins()["protocol"],
                "subject_commit": c.pins()["subject_commit"], "collector": c.COLLECTOR, "status": "PENDING"})
            c.git(repo, "add", "pending.json"); c.git(repo, "commit", "-qm", "pending fixture")
            head = c.git(repo, "rev-parse", "HEAD").decode().strip()
            with patch.object(c, "asset_manifest") as assets, patch.object(n, "NativeEngine") as engine:
                rc = cli.main(["init", "--repo", str(repo), "--session", str(base / "session"),
                               "--model-dir", str(base / "model"), "--authorization-commit", head,
                               "--authorization-path", "pending.json"])
                self.assertEqual(rc, 2); assets.assert_not_called(); engine.assert_not_called()
            self.assertFalse((base / "session").exists())
            self.assertFalse((repo / ".git/opallo-native-claims").exists())
    def test_worker_pinned_prompt_check_precedes_model_load(self):
        from experiments.opallo_native_collector_v1 import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); phase, label = "qualify", "on8"
            manifest = {"test_fixture": True}
            requests = s.requests_for(s.PLANS[phase][0]); requests[0]["messages"][1]["content"] += " unauthorized mutation"
            task = {"phase": phase, "label": label, "split": "calibration", "formatted": False,
                    "confidence": True, "requests": requests, "requests_sha256": c.digest(requests),
                    "session_sha256": c.digest(manifest)}
            out = root / phase / label
            c.write_new(out / "REQUEST.json", task)
            c.write_new(out / "GO.json", {"parent_pid": os.getppid(), "task_sha256": c.digest(task),
                                          "session_sha256": c.digest(manifest)})
            with patch.object(v, "contain_worker"), patch.object(s, "load_session", return_value=manifest), \
                 patch.object(s, "verify_active_worker"), patch.object(n, "NativeEngine") as engine:
                with self.assertRaises(c.Blocked): cli.worker(root, phase, label, os.getppid())
                engine.assert_not_called()
            self.assertFalse((out / "WORKER_STARTED.json").exists())


if __name__ == "__main__": unittest.main()
