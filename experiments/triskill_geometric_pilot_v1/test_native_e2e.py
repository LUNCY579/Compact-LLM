"""End-to-end CPU runs of the REAL worker._run_bound_screen with a tiny random Qwen3.

Adapted from the section H review's native-path probe. Only the environment is faked:
freeze/binding validation, preflight, CUDA calls and device_map. Source weights are tiny
safetensors (BASE BF16, T/TH FP32) with the real 151,936-token vocabulary, and the pinned
tokenizer/generation_config sidecars are copied from the Supa directory (no model weights
are read). Nothing touches a GPU or the shared lease.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fresh_dual_skill_instrument_v1"))
import numpy as np
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

import benchmark
import geometry

SUPA = geometry.SOURCES["BASE"][0].parent
SIDECARS = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
GRANT = "operator grant for the CPU end-to-end test fixture"


class E2EFixture:
    """Builds tiny sources, geometry and a fake pilot directory once per test class."""

    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.pilot, src = root / "pilot", root / "src"
        self.pilot.mkdir()
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
        digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.sources = {name: (src / name / "model.safetensors", digest(src / name / "model.safetensors"))
                        for name in ("BASE", "T", "TH")}
        key = f"model.layers.{geometry.LAYER}.self_attn.o_proj.weight"
        dt = t[key].double().numpy() - base[key].double().numpy()
        dh = th[key].double().numpy() - t[key].double().numpy()
        u, _ = geometry.left_subspace(dt, 4)
        v, _ = geometry.left_subspace(dh, 4)
        b, r, k = geometry.construct(u, v)
        with open(self.pilot / "geometry_sectionh.npz", "xb") as stream:
            np.savez(stream, basis=b, rotation=r, random_basis=geometry.haar_frame(b.shape, 5), commutator=k)
        (self.pilot / "geometry_sectionh.json").write_text(json.dumps(
            {"status": "PREPARED", "geometry_sha256": digest(self.pilot / "geometry_sectionh.npz")}), encoding="utf-8")
        for name in ("screen_sectionh.json", "development_sectionh.json", "worker.py"):
            shutil.copy(HERE / name, self.pilot / name)
        self.runs = 0

    def run(self, *, force_gate=False, fail_load_call=None, block_second_preflight=False, supervisor_dies_after=None):
        """One worker run in a fresh attempt directory; returns (result, out_path, attempt_dir)."""
        self.runs += 1
        spec = importlib.util.spec_from_file_location(f"worker_e2e_{self.runs}", self.pilot / "worker.py")
        w = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = w
        spec.loader.exec_module(w)
        w._triskill_source_binding = (str(w.HERE / "worker.py"), w.sha256(w.HERE / "worker.py"))
        fake_geometry = types.SimpleNamespace(SOURCES=self.sources, LAYER=geometry.LAYER, TensorFile=geometry.TensorFile)
        bench = types.SimpleNamespace(**{name: getattr(benchmark, name) for name in dir(benchmark) if not name.startswith("__")})
        real_gate = benchmark.parent_gate
        if force_gate:
            # Forced in both places: the worker's stop and decide_screen's re-derivation.
            def gate(rows, scores):
                result = real_gate(rows, scores)
                result["passed"], result["stop"] = True, None
                return result
            bench.parent_gate = gate

        class Renderer:
            def template_selftest(self): pass
            def render(self, system, user):
                from score_arms import render
                return render(system, user)

        class Bound:
            def verify_worker_execution(self, module): pass
            def verify_current(self): pass
            def load_local(self, role): return bench
            def load_renderer(self): return Renderer()
            def receipt(self): return {"fake": True}

        w._validate_freeze = lambda path, sha: ({}, Bound(), fake_geometry)
        calls = []
        def preflight(token):
            calls.append(token)
            blockers = ["RAM below 5 GiB preload reserve"] if block_second_preflight and len(calls) == 2 else []
            return {"blockers": blockers, "lease": {"pid": 4242, "token": token}}
        w.preflight = preflight
        if fail_load_call is not None:
            real_load, loads = w.load_arm_weights, []
            def load(*args, **kwargs):
                loads.append(args[1])
                if len(loads) == fail_load_call:
                    raise RuntimeError("STOP_HOST_ACTIVE_RESERVE")
                return real_load(*args, **kwargs)
            w.load_arm_weights = load
        import psutil
        checks = []
        real_exists = psutil.pid_exists
        def pid_exists(pid):
            if pid == 4242:
                checks.append(pid)
                return supervisor_dies_after is None or len(checks) <= supervisor_dies_after
            return real_exists(pid)
        attempt = Path(tempfile.mkdtemp(dir=self.temp.name))
        (attempt / w.START_NAME).write_text(json.dumps({"lease_token": "token", "operator_grant": GRANT}), encoding="utf-8")
        freeze = attempt / "FREEZE.json"
        freeze.write_text("{}", encoding="utf-8")
        out = attempt / "native_screen_sectionh.json"
        saved = (torch.cuda.init, torch.cuda.set_per_process_memory_fraction, torch.cuda.mem_get_info,
                 transformers.AutoModelForCausalLM.from_pretrained, psutil.pid_exists)
        original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
        def from_pretrained(path, **kwargs):
            kwargs["device_map"] = {"": "cpu"}
            return original_from_pretrained(str(self.sources["BASE"][0].parent), **kwargs)
        torch.cuda.init = lambda: None
        torch.cuda.set_per_process_memory_fraction = lambda *a, **k: None
        torch.cuda.mem_get_info = lambda *a, **k: (8 << 30, 8 << 30)
        transformers.AutoModelForCausalLM.from_pretrained = from_pretrained
        psutil.pid_exists = pid_exists
        if force_gate:
            benchmark.parent_gate = bench.parent_gate
        try:
            result = w._run_bound_screen(out, freeze, "x" * 64, "token", attempt_dir=attempt, _started=time.monotonic())
        finally:
            (torch.cuda.init, torch.cuda.set_per_process_memory_fraction, torch.cuda.mem_get_info,
             transformers.AutoModelForCausalLM.from_pretrained, psutil.pid_exists) = saved
            benchmark.parent_gate = real_gate
        return result, out, attempt, w


class NativeEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = E2EFixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.temp.cleanup()

    def progress(self, attempt, w):
        return [json.loads(line) for line in (attempt / w.PROGRESS_NAME).read_text(encoding="utf-8").splitlines()]

    def test_full_schedule_with_gate_forced_open(self):
        result, out, attempt, w = self.fixture.run(force_gate=True)
        self.assertTrue(result["completed_schedule"], result["status"])
        self.assertTrue(out.exists())
        self.assertTrue(result["hook_liveness"]["ok"])
        self.assertEqual(result["development_padding_parity"], [True, True])
        self.assertEqual(result["budget"]["completed_screen_generations"], w.MAX_GENERATIONS)
        self.assertLessEqual(result["budget"]["model_forwards"], result["allocation"]["forwards"])
        self.assertLessEqual(result["budget"]["token_positions_including_padding"], result["allocation"]["token_positions"])
        self.assertEqual(result["weight_loads"], 5)  # T, H, T, BASE, TH after the initial BASE load
        readout = result["teacher_forced_readout"]
        for row_id, metrics in readout.items():
            if metrics:
                self.assertEqual(metrics["TH"]["argmax_agrees_with_TH_token"], metrics["TH"]["steps"], row_id)
        self.assertEqual(result["verdict"]["status"], result["status"])
        self.assertIn(result["status"], ("UNDETECTABLE_ON_J", "NO_SIGNAL", "SCREEN_SIGNAL_EXPLORATORY"))
        generations = [entry for entry in self.progress(attempt, w) if entry["event"] == "generation"]
        self.assertEqual(len(generations), w.MAX_GENERATIONS)

    def test_mid_run_stop_keeps_completed_rows_and_is_not_a_mechanics_failure(self):
        result, out, attempt, w = self.fixture.run(force_gate=True, fail_load_call=3)  # T, H, then the T reload
        self.assertEqual(result["status"], "STOP_HOST_ACTIVE_RESERVE")
        self.assertEqual(result["verdict"]["status"], "STOPPED_BEFORE:liveness")
        self.assertEqual(result["verdict"]["stopped_by"], "STOP_HOST_ACTIVE_RESERVE")
        self.assertTrue(out.exists())  # the model loaded, so the screen is spent
        self.assertEqual({arm: len(record["outputs"]) for arm, record in result["arms"].items()},
                         {"BASE": 8, "T": 8, "H": 18})
        self.assertEqual(sum(entry["event"] == "generation" for entry in self.progress(attempt, w)), 34)

    def test_block_before_model_load_leaves_the_screen_unspent(self):
        result, out, attempt, w = self.fixture.run(block_second_preflight=True)
        self.assertEqual(result["model_loads"], 0)
        self.assertFalse(out.exists())
        self.assertTrue((attempt / w.BLOCKED_NAME).exists())
        self.assertEqual(result["verdict"]["status"], "STOPPED_BEFORE:padding")

    def test_lost_supervisor_stops_the_worker(self):
        result, out, attempt, w = self.fixture.run(force_gate=True, supervisor_dies_after=5)
        self.assertEqual(result["status"], "STOP_SUPERVISOR_LOST")
        self.assertTrue(result["verdict"]["status"].startswith("STOPPED_BEFORE"))


if __name__ == "__main__":
    unittest.main()
