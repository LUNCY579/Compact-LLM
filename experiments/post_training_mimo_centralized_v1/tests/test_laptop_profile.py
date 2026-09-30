"""Stdlib prospective-profile controls; no Torch imports or native jobs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import laptop_profile
from experiments.post_training_mimo_centralized_v1.scripts import qualify_native
from experiments.post_training_mimo_centralized_v1.scripts import supervise_native


class LaptopProfileTests(unittest.TestCase):
    def test_historical_config_remains_byte_pinned_and_accepted(self):
        raw = laptop_profile.LEGACY_CONFIG_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), laptop_profile.LEGACY_CONFIG_SHA256)
        legacy = json.loads(raw)
        qualify_native.validate_config(legacy)
        self.assertFalse(laptop_profile.metadata_present(legacy))
        self.assertNotIn("worker_rss_limit_gib", legacy)
        for field, value in (("host_free_gib", 7.3), ("worker_free_gib", 5.9),
                             ("gpu_free_gib", 3.9), ("gpu_worker_free_gib", .49)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                qualify_native.validate_config({**legacy, field: value})

    def test_both_exact_stage_registrations_are_accepted(self):
        for stage in ("load_forward", "full"):
            cfg = laptop_profile.expected_config(stage)
            self.assertTrue(laptop_profile.metadata_present(cfg))
            laptop_profile.validate_config(cfg)
            qualify_native.validate_config(cfg)
            self.assertEqual(cfg["resource_profile"], "LAPTOP_GPU_RESIDENT_V1")
            self.assertEqual(cfg["host_free_gib"], 2.0)
            self.assertEqual(cfg["worker_free_gib"], 2.0)
            self.assertEqual(cfg["worker_rss_limit_gib"], 2.5)
            self.assertEqual(cfg["gpu_free_gib"], 4.0)
            self.assertEqual(cfg["gpu_worker_free_gib"], .5)

    def test_probe_has_zero_update_generation_and_parity_budget(self):
        probe = laptop_profile.expected_config("load_forward")
        self.assertEqual(probe["maximum_optimizer_calls"], 0)
        self.assertEqual(probe["generation_cap"], 0)
        self.assertEqual(probe["parity_prompts"], 0)
        self.assertEqual(probe["contexts"], [256])
        self.assertEqual(probe["wall_seconds"], 600)
        self.assertEqual(probe["update_semantics"], "no adapters, backward, optimizer updates, generation or benchmarks; strict load and one frozen 256-token forward only")
        self.assertEqual(laptop_profile.PROBE_PASS_STATUS, "PASS_E0_LOAD_FORWARD_RESOURCE_ONLY")
        self.assertNotEqual(laptop_profile.PROBE_PASS_STATUS, laptop_profile.FULL_PASS_STATUS)
        journal = qualify_native.CallAccounting(lambda *args, **kwargs: None, probe["maximum_optimizer_calls"])
        calls = []
        with self.assertRaisesRegex(RuntimeError, "OPTIMIZER_CALL_BUDGET"):
            journal.call("optimizer_step", lambda: calls.append("executed"), stage="forbidden_probe_step")
        self.assertEqual(calls, [])
        self.assertEqual(journal.snapshot()["operations"]["optimizer_step"]["attempted"], 0)

    def test_full_preserves_every_historical_nonresource_workload_value(self):
        historical = json.loads(laptop_profile.LEGACY_CONFIG_PATH.read_bytes())
        full = laptop_profile.expected_config("full")
        overrides = {"status", "host_free_gib", "worker_free_gib"} | laptop_profile.PROFILE_FIELDS
        for field, value in historical.items():
            if field not in overrides:
                self.assertEqual(json.dumps(full[field], sort_keys=True), json.dumps(value, sort_keys=True), field)
        self.assertEqual(full["contexts"], [256, 512, 1024])
        self.assertEqual(full["maximum_optimizer_calls"], 5)

    def test_profile_rejects_missing_extra_and_cross_stage_fields(self):
        for stage in ("load_forward", "full"):
            registered = laptop_profile.expected_config(stage)
            for field in registered:
                cfg = deepcopy(registered)
                del cfg[field]
                with self.subTest(stage=stage, missing=field), self.assertRaises(RuntimeError):
                    laptop_profile.validate_config(cfg)
            cfg = {**registered, "unregistered_override": True}
            with self.assertRaises(RuntimeError):
                qualify_native.validate_config(cfg)
            cfg = {**registered, "qualification_stage": "full" if stage == "load_forward" else "load_forward"}
            with self.assertRaises(RuntimeError):
                qualify_native.validate_config(cfg)

    def test_weakened_numeric_model_and_identity_values_fail_closed(self):
        cases = (("host_free_gib", 1.9), ("host_free_gib", 3.0), ("host_free_gib", 2),
                 ("worker_free_gib", 1.9), ("worker_rss_limit_gib", 2.6),
                 ("gpu_free_gib", 3.9), ("gpu_worker_free_gib", .49), ("disk_free_gib", 3.9),
                 ("worker_rss_limit_gib", True), ("host_free_gib", float("nan")),
                 ("host_free_gib", float("inf")), ("wall_seconds", 601),
                 ("resource_profile", "LAPTOP"), ("status", "FROZEN_E0_ONLY"),
                 ("admission_budget", "available_only"), ("qualification_stage", "other"),
                 ("weight_sha256", "0" * 64), ("model_dir", "unbound-model"),
                 ("attention", "sdpa"), ("base_dtype", "float32"),
                 ("zero_adapter_atol", .01), ("resume", "approximate"),
                 ("maximum_optimizer_calls", 1), ("generation_cap", 1), ("parity_prompts", 1),
                 ("contexts", [128]))
        cfg = laptop_profile.expected_config("load_forward")
        for field, value in cases:
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                qualify_native.validate_config({**cfg, field: value})

    def test_any_prospective_metadata_forces_exact_validation(self):
        legacy = json.loads(laptop_profile.LEGACY_CONFIG_PATH.read_bytes())
        for field in laptop_profile.PROFILE_FIELDS:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                qualify_native.validate_config({**legacy, field: "invalid"})

    def test_historical_config_drift_rejected_before_derivation(self):
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "altered.json"
            altered.write_bytes(laptop_profile.LEGACY_CONFIG_PATH.read_bytes() + b" ")
            with patch.object(laptop_profile, "LEGACY_CONFIG_PATH", altered), self.assertRaisesRegex(RuntimeError, "historical E0 config drift"):
                laptop_profile.expected_config("full")

    def test_admission_reserves_all_remaining_worker_growth(self):
        cfg = laptop_profile.expected_config("load_forward")
        for rss, required in ((0, 4.5), (.75, 3.75), (1.25, 3.25), (2.5, 2.0)):
            with self.subTest(rss=rss):
                self.assertEqual(laptop_profile.admission_required_gib(cfg, rss), required)
                self.assertEqual(required + rss, 4.5)
        # A worker beyond the cap still has the system reserve; the worker guard
        # separately rejects RSS > cap after preserving that resource sample.
        self.assertEqual(laptop_profile.admission_required_gib(cfg, 2.6), 2.0)
        for rss in (-1, True, float("inf"), float("nan"), "0.5", None):
            with self.subTest(rss=rss), self.assertRaises(RuntimeError):
                laptop_profile.admission_required_gib(cfg, rss)

    def test_containment_limits_are_explicit_and_not_native_qualification(self):
        for stage in ("load_forward", "full"):
            cfg = laptop_profile.expected_config(stage)
            limits = laptop_profile.resource_limits(cfg)
            self.assertEqual(limits["worker_free_gib"], 2.0)
            self.assertEqual(limits["worker_rss_limit_gib"], 2.5)
            self.assertEqual(limits["admission_budget"], laptop_profile.ADMISSION_BUDGET)
            self.assertEqual(limits["wall_seconds"], 600 if stage == "load_forward" else 1800)

    def test_supervisor_dispatches_exact_limits_and_records_them(self):
        # Replace every external query and the watchdog: this launches no worker.
        for stage in (None, "load_forward", "full"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                cfg = (json.loads(laptop_profile.LEGACY_CONFIG_PATH.read_bytes()) if stage is None
                       else laptop_profile.expected_config(stage))
                config_path = base / "config.json"
                config_path.write_text(json.dumps(cfg), encoding="utf-8")
                output = base / "result"
                calls = []
                def fake_query(command, **kwargs):
                    if command[0] == "nvidia-smi":
                        return ""
                    if command[-1] == "--git-common-dir":
                        return str(base)
                    return "a" * 40
                def fake_watchdog(command, cwd, out, **kwargs):
                    calls.append((command, kwargs))
                argv = ["supervise_native", "--config", str(config_path), "--output", str(output)]
                fake_os = SimpleNamespace(name="nt", getpid=lambda: 12345)
                with patch.object(sys, "argv", argv), patch.object(supervise_native, "os", fake_os), \
                     patch.object(supervise_native.subprocess, "check_output", side_effect=fake_query), \
                     patch.object(supervise_native, "run_bounded", side_effect=fake_watchdog):
                    self.assertEqual(supervise_native.main(), 0)
                self.assertEqual(len(calls), 1)
                limits = calls[0][1]
                self.assertEqual(limits["seconds"], cfg["wall_seconds"])
                self.assertEqual(limits["min_free_gib"], 6.0 if stage is None else 2.0)
                self.assertEqual(limits["max_rss_gib"], 4.0 if stage is None else 2.5)
                launch = json.loads((base / "result_supervisor/LAUNCH.json").read_text())
                self.assertEqual(limits["go"]["config_sha256"], launch["config_sha256"])
                if stage is None:
                    self.assertNotIn("resource_limits", launch)
                    self.assertNotIn("resource_profile", launch)
                else:
                    self.assertEqual(launch["resource_limits"], laptop_profile.resource_limits(cfg))
                    self.assertEqual(launch["qualification_stage"], stage)
                    self.assertEqual(launch["resource_profile"], laptop_profile.RESOURCE_PROFILE)
                self.assertFalse((base / "rineya-central-gpu0.lock").exists())

    def test_imports_do_not_load_model_frameworks(self):
        code = "import sys; import experiments.post_training_mimo_centralized_v1.scripts.laptop_profile; import experiments.post_training_mimo_centralized_v1.scripts.qualify_native; assert not any(n == 'torch' or n.startswith('torch.') or n == 'transformers' or n.startswith('transformers.') for n in sys.modules)"
        run = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
