"""CPU tests of the PARENTQ worker: decode equivalence, lease rule, freeze refusals, and
end-to-end runs of the REAL parentq_worker._run_bound with a tiny random Qwen3.

As in GEO's test_native_e2e.py, only the environment is faked: the GEO binding (the real
frozen GEO worker and benchmark modules are used), preflight, CUDA calls and device_map.
Source weights are tiny safetensors with the real 151,936-token vocabulary, and the pinned
tokenizer/generation_config sidecars are copied from Supa (no model weights are read).
The real bind_geo against the real GEO freeze runs in a clean subprocess.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
import numpy as np
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

import geo_frozen
import parentq_decide
import parentq_rows
import parentq_supervise
import parentq_launch_wmi
import parentq_worker
from test_parentq_worker_mechanics import passing_receipt, fixture_grant

geo = geo_frozen.load("worker")
geometry = geo_frozen.load("geometry")
benchmark = geo_frozen.load("benchmark")
SUPA = geometry.SOURCES["BASE"][0].parent
SIDECARS = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Unlimited:
    def __init__(self):
        self.forwards = self.completed_forwards = self.token_positions = 0

    def forward(self, positions):
        self.forwards += 1
        self.token_positions += positions

    def check(self):
        pass


def tiny_model(seed=0, vocab=512):
    config = Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=15,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                         tie_word_embeddings=True)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(config).eval()


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.state = geo.HookState(torch, torch.as_tensor(np.linalg.qr(np.random.default_rng(1).standard_normal((32, 8)))[0],
                                                          dtype=torch.float32))
        handle = self.model.model.layers[geometry.LAYER].register_forward_hook(geo.make_hook(self.state))
        self.addCleanup(handle.remove)

    def test_tokens_equal_geo_fixed_slot_decode_for_selected_eos_positions(self):
        rng = np.random.default_rng(3)
        for trial in range(6):
            ids = [int(x) for x in rng.integers(0, 500, int(rng.integers(20, 200)))]
            free = geo.decode(self.model, torch, "cpu", ids, pad_id=510, eos_ids=(510, 511), state=self.state,
                              budget=Unlimited())
            for stop in (None, 0, 3, 17, geo.OUTPUT_SLOTS - 1):
                eos = (510, 511) if stop is None else (free["raw_tokens"][stop], 511)
                fixed = geo.decode(self.model, torch, "cpu", ids, pad_id=510, eos_ids=eos, state=self.state, budget=Unlimited())
                budget = Unlimited()
                early = parentq_worker.decode_until_eos(geo, self.model, torch, "cpu", ids, pad_id=510, eos_ids=eos,
                                                        state=self.state, budget=budget)
                self.assertEqual(early["tokens"], fixed["tokens"], (trial, stop))
                self.assertEqual(early["scored_tokens"], fixed["scored_tokens"], (trial, stop))
                self.assertEqual(early["ended"], fixed["ended"])
                self.assertEqual(early["margins"], fixed["margins"][:len(early["margins"])])
                self.assertEqual(budget.forwards, len(fixed["scored_tokens"]))


class UnitTests(unittest.TestCase):
    def test_allocation_and_schedule(self):
        bank = parentq_rows.build_bank(benchmark)
        canary = json.loads((HERE / parentq_worker.CANARY_JSON).read_text(encoding="utf-8"))["rows"]
        units = parentq_worker.schedule_units(bank, canary)
        self.assertEqual([(arm, block, len(ids)) for arm, block, ids in units],
                         [("BASE", "CANARY", 8), ("BASE", "T", 24), ("BASE", "H_ON", 32), ("BASE", "H_OFF", 32),
                          ("H", "CANARY", 8), ("H", "H_ON", 32), ("H", "H_OFF", 32), ("T", "CANARY", 8), ("T", "T", 24)])
        caps = parentq_worker.allocation(200, geo.INPUT_SLOTS, geo.OUTPUT_SLOTS)
        self.assertEqual((caps["generations"], caps["forwards"], caps["token_positions"]), (200, 6416, 64964))

    def test_preflight_lease_rule_uses_this_lane_label(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / "gpu.lock"
            kwargs = dict(processes=[], gpu_query=lambda: ("GPU, 8188, 8000, 0", ""), ancestors={4242})
            lease_blocker = "shared GPU lease not held by this worker's supervisor"
            self.assertIn(lease_blocker, parentq_worker.preflight(geo, "tok", lock=lock, **kwargs)["blockers"])
            lock.write_text(json.dumps({"pid": 4242, "token": "tok", "label": parentq_worker.LEASE_LABEL}), encoding="utf-8")
            record = parentq_worker.preflight(geo, "tok", lock=lock, **kwargs)
            self.assertTrue(record["lease_owned_by_supervisor"])
            self.assertNotIn(lease_blocker, record["blockers"])
            for lease in ({"pid": 4242, "token": "tok", "label": geo.LEASE_LABEL},
                          {"pid": 1, "token": "tok", "label": parentq_worker.LEASE_LABEL},
                          {"pid": 4242, "token": "other", "label": parentq_worker.LEASE_LABEL}):
                lock.write_text(json.dumps(lease), encoding="utf-8")
                self.assertIn(lease_blocker, parentq_worker.preflight(geo, "tok", lock=lock, **kwargs)["blockers"])
            self.assertFalse(parentq_worker._NO_LOCK.exists())

    def test_freeze_refusals(self):
        with tempfile.TemporaryDirectory() as folder:
            here = Path(folder)
            for name in parentq_worker.REQUIRED_FILES:
                (here / name).write_text(name, encoding="utf-8")
            (here / "CPU_VALIDATION_PARENTQ.json").write_text(json.dumps(passing_receipt()), encoding="utf-8")
            payload = {"files": {name: sha(here / name) for name in parentq_worker.REQUIRED_FILES},
                       "runtime_root": str(parentq_worker.RUNTIME_ROOT.resolve()),
                       "geo_freeze": {"sha256": parentq_worker.GEO_FREEZE_SHA256},
                       "supervisor_dependencies": json.loads((parentq_worker.GEO_DIR / parentq_worker.GEO_FREEZE_NAME).read_bytes())["supervisor_dependencies"],
                       "cpu_validation": {"file": "CPU_VALIDATION_PARENTQ.json", "passed": True,
                                          "sha256": sha(here / "CPU_VALIDATION_PARENTQ.json")}}
            def write(data):
                path = here / "FREEZE.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                return path, sha(path)
            path, digest = write(payload)
            self.assertEqual(parentq_worker.validate_freeze(path, digest, here)["files"], payload["files"])
            with self.assertRaisesRegex(ValueError, "STOP_FREEZE_DIGEST_MISMATCH"):
                parentq_worker.validate_freeze(path, "0" * 64, here)
            for broken, error in (({**payload, "geo_freeze": {"sha256": "0" * 64}}, "STOP_GEO_FREEZE_NOT_REGISTERED"),
                                  ({**payload, "cpu_validation": {**payload["cpu_validation"], "passed": False}},
                                   "STOP_NO_PASSING_PINNED_CPU_VALIDATION"),
                                  ({**payload, "files": {k: v for k, v in payload["files"].items() if k != "parentq_rows.py"}},
                                   "STOP_INCOMPLETE_FREEZE_MEMBERSHIP")):
                path, digest = write(broken)
                with self.assertRaisesRegex(ValueError, error):
                    parentq_worker.validate_freeze(path, digest, here)
            path, digest = write(payload)
            (here / "parentq_rows.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "STOP_FREEZE_FILE_MISMATCH:parentq_rows.py"):
                parentq_worker.validate_freeze(path, digest, here)

    def test_real_bind_geo_validates_the_frozen_geo_lane(self):
        """In a clean interpreter: GEO's binding, its full freeze validation and its modules."""
        code = ("import parentq_worker as w; g, b, geo = w.bind_geo(); bench = b.load_local('benchmark'); "
                "print(len(bench.build_screen()), g.LEASE_LABEL, geo.LAYER)")
        completed = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, timeout=300,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertEqual(completed.stdout.split(), ["18", "triskill-geo001-sectionh", "14"])

    def test_names_agree_with_supervisor_and_launcher(self):
        for name in ("LEASE_LABEL", "START_NAME", "WORKER_PID_NAME", "LOAD_BEGUN_NAME", "OUT_NAME", "INCOMPLETE_NAME", "FREEZE_NAME"):
            self.assertEqual(getattr(parentq_supervise, name), getattr(parentq_worker, name), name)
        self.assertEqual(parentq_launch_wmi.START_NAME, parentq_worker.START_NAME)
        self.assertEqual(parentq_supervise.MAX_LOADED_ATTEMPTS, 3)
        self.assertNotEqual(parentq_worker.LEASE_LABEL, geo.LEASE_LABEL)


class Fixture:
    """Tiny sources with the real vocabulary, tiny GEO geometry, built once per class."""

    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.geo_dir, src = root / "geo", root / "src"
        self.geo_dir.mkdir()
        for name in ("BASE", "T", "TH"):
            (src / name).mkdir(parents=True)
        config = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=15,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=1024,
                             tie_word_embeddings=True, eos_token_id=151645, bos_token_id=151643)
        torch.manual_seed(0)
        Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(src / "BASE", safe_serialization=True)
        for name in SIDECARS:
            shutil.copy(SUPA / name, src / "BASE" / name)
        from safetensors.torch import save_file
        base = {k: v.float() for k, v in Qwen3ForCausalLM.from_pretrained(str(src / "BASE"), dtype=torch.float32)
                .state_dict().items() if k != "lm_head.weight"}
        generator = torch.Generator().manual_seed(1)
        t = {k: (v + 0.02 * torch.randn(v.shape, generator=generator)).contiguous() for k, v in base.items()}
        th = {k: (v + 0.02 * torch.randn(v.shape, generator=generator)).contiguous() for k, v in t.items()}
        save_file(t, str(src / "T" / "model.safetensors"), metadata={"format": "pt"})
        save_file(th, str(src / "TH" / "model.safetensors"), metadata={"format": "pt"})
        self.sources = {name: (src / name / "model.safetensors", sha(src / name / "model.safetensors"))
                        for name in ("BASE", "T", "TH")}
        basis = np.linalg.qr(np.random.default_rng(2).standard_normal((32, 8)))[0]
        with open(self.geo_dir / geo.GEOMETRY_NPZ, "xb") as stream:
            np.savez(stream, basis=basis, rotation=np.eye(8), random_basis=basis)
        (self.geo_dir / geo.GEOMETRY_JSON).write_text(json.dumps(
            {"status": "PREPARED", "geometry_sha256": sha(self.geo_dir / geo.GEOMETRY_NPZ)}), encoding="utf-8")
        self.runs = 0

    def run(self, *, force_canary=False, fail_load_call=None, block_second_preflight=False,
            block_first_preflight=False, fail_initial_model_load=False, force_decision_error=False,
            supervisor_dies_after=None):
        self.runs += 1
        w = parentq_worker._exec_pinned(HERE / "parentq_worker.py", sha(HERE / "parentq_worker.py"), f"e2e_{self.runs}")
        bank = parentq_rows.build_bank(benchmark)
        canary = json.loads((HERE / parentq_worker.CANARY_JSON).read_text(encoding="utf-8"))["rows"]
        n = sum(len(ids) for _, _, ids in parentq_worker.schedule_units(bank, canary))
        w.validate_freeze = lambda path, digest, here=HERE: {
            "files": {name: sha(HERE / name) for name in ("parentq_worker.py", "parentq_rows.py", "parentq_decide.py", "parentq_safety.py")},
            "runtime_root": str(Path(self.temp.name).resolve()),
            "native_allocation": parentq_worker.allocation(n, geo.INPUT_SLOTS, geo.OUTPUT_SLOTS)}
        w.GEO_DIR = self.geo_dir
        sys.path.insert(0, str(HERE.parent / "fresh_dual_skill_instrument_v1"))
        from score_arms import render

        class Renderer:
            def template_selftest(self): pass
            def render(self, system, user): return render(system, user)

        class Bound:
            def verify_current(self): pass
            def load_local(self, role): return benchmark
            def load_renderer(self): return Renderer()
            def receipt(self): return {"fake": True}

        geo_view = types.SimpleNamespace(**{name: getattr(geo, name) for name in dir(geo) if not name.startswith("__")})
        if fail_load_call is not None:
            loads = []
            def load(*args, **kwargs):
                loads.append(args[1])
                if len(loads) == fail_load_call:
                    raise RuntimeError("STOP_HOST_ACTIVE_RESERVE")
                return geo.load_arm_weights(*args, **kwargs)
            geo_view.load_arm_weights = load
        fake_geometry = types.SimpleNamespace(SOURCES=self.sources, LAYER=geometry.LAYER, TensorFile=geometry.TensorFile)
        calls = []
        def preflight(geo_module, token, **kwargs):
            calls.append(token)
            blockers = ["fixture provisional RAM blocker"] if ((block_second_preflight and len(calls) == 2)
                        or (block_first_preflight and len(calls) == 1)) else []
            return {"blockers": blockers, "lease": {"pid": 4242, "token": token}}
        w.preflight = preflight
        real_exec = w._exec_pinned
        def exec_pinned(path, expected, name):
            module = real_exec(path, expected, name)
            if name == "safety":
                module.RUNTIME_ROOT = Path(self.temp.name).resolve()
                module.validate_readiness = lambda *a: {"fixture_only": True}
                module.validate_runtime_root = lambda path: path.resolve()
                module.check_runtime_ownership = lambda *a: None
            if name == "decide":
                if force_canary:
                    original = module.canary_check
                    def forced(canary_rows, outputs):
                        result = original(canary_rows, outputs)
                        for record in result.values():
                            record["passed"] = record["complete"]
                        return result
                    module.canary_check = forced
                if force_decision_error:
                    def fail(*a, **k): raise ValueError("fixture decision failure")
                    module.decide = fail
            return module
        w._exec_pinned = exec_pinned
        import psutil
        checks, real_exists = [], psutil.pid_exists
        def pid_exists(pid):
            if pid == 4242:
                checks.append(pid)
                return supervisor_dies_after is None or len(checks) <= supervisor_dies_after
            return real_exists(pid)
        attempt = Path(tempfile.mkdtemp(dir=self.temp.name))
        (attempt / w.START_NAME).write_text(json.dumps({"lease_token": "token", "operator_grant": fixture_grant("x" * 64, self.runs),
            "experiment": "TRISKILL-GEO-001-PARENTQ", "freeze_sha256": "x" * 64, "attempt_number": self.runs,
            "supervisor_pid": 4242, "supervisor_create_time": 123.0, "runtime_root": self.temp.name}), encoding="utf-8")
        out = attempt / "native_parentq.json"
        saved = (torch.cuda.init, torch.cuda.set_per_process_memory_fraction, torch.cuda.mem_get_info,
                 transformers.AutoModelForCausalLM.from_pretrained, psutil.pid_exists, psutil.Process)
        original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
        original_process = psutil.Process
        def from_pretrained(path, **kwargs):
            self.assert_marker_before_load = (attempt / w.LOAD_BEGUN_NAME).exists()
            if fail_initial_model_load:
                raise RuntimeError("fixture initial model load failure")
            kwargs["device_map"] = {"": "cpu"}
            return original_from_pretrained(str(self.sources["BASE"][0].parent), **kwargs)
        torch.cuda.init = lambda: None
        torch.cuda.set_per_process_memory_fraction = lambda *a, **k: None
        torch.cuda.mem_get_info = lambda *a, **k: (8 << 30, 8 << 30)
        transformers.AutoModelForCausalLM.from_pretrained = from_pretrained
        psutil.pid_exists = pid_exists
        psutil.Process = lambda pid=None: types.SimpleNamespace(create_time=lambda: 123.0) if pid == 4242 else original_process(pid)
        try:
            result = w._run_bound(out, attempt / "FREEZE.json", "x" * 64, "token", attempt, _started=time.monotonic(),
                                  _bind_geo=lambda: (geo_view, Bound(), fake_geometry))
        finally:
            (torch.cuda.init, torch.cuda.set_per_process_memory_fraction, torch.cuda.mem_get_info,
             transformers.AutoModelForCausalLM.from_pretrained, psutil.pid_exists, psutil.Process) = saved
        return result, out, attempt, w


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = Fixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.temp.cleanup()

    def progress(self, attempt):
        return [json.loads(line) for line in (attempt / parentq_worker.PROGRESS_NAME).read_text(encoding="utf-8").splitlines()]

    def test_full_schedule_with_canary_forced_open(self):
        result, out, attempt, w = self.fixture.run(force_canary=True)
        self.assertTrue(result["completed_schedule"], result["status"])
        self.assertTrue(out.exists())
        self.assertTrue(result["status"].startswith("PARENTS_"), result["status"])
        self.assertEqual(result["verdict"]["status"], result["status"])
        self.assertEqual(result["development_padding_parity"], [True, True])
        self.assertEqual(result["weight_loads"], 2)   # H then T after the initial BASE load
        self.assertEqual(result["budget"]["completed_generations"], 200)
        self.assertLessEqual(result["budget"]["model_forwards"], result["allocation"]["forwards"])
        self.assertEqual({arm: len(texts) for arm, texts in result["outputs"].items()}, {"BASE": 96, "H": 72, "T": 32})
        events = self.progress(attempt)
        self.assertTrue(all("utc" in e and "elapsed_seconds" in e for e in events))
        self.assertTrue(self.fixture.assert_marker_before_load)
        self.assertTrue((attempt / w.LOAD_BEGUN_NAME).exists())
        self.assertEqual(sum(e["event"] == "generation" for e in events), 200)
        self.assertEqual([e["arm"] for e in events if e["event"] == "weights_loaded"], ["H", "T"])
        self.assertEqual([e["arm"] for e in events if e["event"] == "canary"], ["BASE", "H", "T"])
        self.assertNotEqual(result["verdict"]["h_contract"]["label"], "Q2_INCOMPLETE")

    def test_real_canary_stops_a_wrong_arm_and_is_terminal(self):
        result, out, attempt, w = self.fixture.run()   # a random tiny model cannot reproduce Supa's texts
        self.assertEqual(result["status"], "STOP_ARM_CANARY:BASE")
        self.assertEqual(result["verdict"]["status"], "STOP_ARM_CANARY:BASE")
        self.assertTrue(out.exists())
        self.assertEqual(result["budget"]["completed_generations"], 8)
        self.assertTrue(all("canary_divergence" in r for r in result["records"]["BASE"].values()))

    def test_first_preflight_failure_also_writes_blocked_record(self):
        result, out, attempt, w = self.fixture.run(block_first_preflight=True)
        self.assertTrue((attempt / w.BLOCKED_NAME).exists())
        self.assertFalse((attempt / w.LOAD_BEGUN_NAME).exists())
        self.assertFalse(out.exists())

    def test_initial_model_load_failure_is_counted_even_without_resident_model(self):
        result, out, attempt, w = self.fixture.run(fail_initial_model_load=True)
        self.assertEqual(result["model_loads"], 0)
        self.assertTrue(self.fixture.assert_marker_before_load)
        self.assertTrue((attempt / w.LOAD_BEGUN_NAME).exists())
        self.assertTrue((attempt / w.INCOMPLETE_NAME).exists())
        self.assertFalse((attempt / w.BLOCKED_NAME).exists())
        self.assertFalse(out.exists())

    def test_decision_failure_is_persisted_with_original_worker_records(self):
        result, out, attempt, w = self.fixture.run(force_decision_error=True)
        self.assertEqual(result["status"], "STOP_DECISION_ERROR")
        self.assertEqual(result["decision_error"]["type"], "ValueError")
        self.assertEqual(len(result["outputs"]["BASE"]), 8)
        self.assertTrue((attempt / w.INCOMPLETE_NAME).exists())

    def test_block_before_model_load_writes_only_a_blocked_record(self):
        result, out, attempt, w = self.fixture.run(block_second_preflight=True)
        self.assertEqual(result["model_loads"], 0)
        self.assertFalse(out.exists())
        self.assertTrue((attempt / parentq_worker.BLOCKED_NAME).exists())
        self.assertEqual(result["verdict"]["status"], "STOPPED_BEFORE:padding")

    def test_stop_at_the_h_weight_load_is_incomplete_and_retryable(self):
        result, out, attempt, w = self.fixture.run(force_canary=True, fail_load_call=1)
        self.assertEqual(result["status"], "STOP_HOST_ACTIVE_RESERVE")
        self.assertEqual(result["verdict"]["status"], "STOPPED_BEFORE:canary:H")
        self.assertFalse(out.exists())
        self.assertTrue((attempt / parentq_worker.INCOMPLETE_NAME).exists())
        self.assertEqual(len(result["outputs"]["BASE"]), 96)

    def test_lost_supervisor_stops_the_worker(self):
        result, out, attempt, w = self.fixture.run(force_canary=True, supervisor_dies_after=3)
        self.assertEqual(result["status"], "STOP_SUPERVISOR_LOST")
        self.assertTrue(result["verdict"]["status"].startswith("STOPPED_BEFORE"))
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
