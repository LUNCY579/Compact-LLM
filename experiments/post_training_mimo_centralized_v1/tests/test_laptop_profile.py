"""Stdlib prospective-profile controls; no Torch imports or native jobs."""
from copy import deepcopy
import ast
import hashlib
from itertools import product
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


class FakeData:
    """Shape and scalar-mask predicates only; no tensor/framework allocation."""
    def __init__(self, shape, values=None):
        self.shape = shape
        self.values = values

    def __eq__(self, value):
        return SimpleNamespace(all=lambda: all(item == value for item in self.values))


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

    def test_all_exact_stage_registrations_are_accepted(self):
        for stage in ("load_forward", "full", "backward_resume"):
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

    def test_streamed_profile_is_exact_and_preserves_stage_workload(self):
        for stage in ("load_forward", "full", "backward_resume"):
            original = laptop_profile.expected_config(stage)
            streamed = laptop_profile.expected_config(stage, profile=laptop_profile.STREAMED_PROFILE)
            qualify_native.validate_config(streamed)
            self.assertEqual(streamed["resource_profile"], "LAPTOP_STREAMED_V1")
            self.assertEqual(streamed["loader_mode"], "stream_read_2MiB_v1")
            self.assertEqual(streamed["host_free_gib"], 1.25)
            self.assertEqual(streamed["worker_free_gib"], 1.25)
            self.assertEqual(streamed["worker_rss_limit_gib"], 1.75)
            differences = {"resource_profile", "host_free_gib", "worker_free_gib", "worker_rss_limit_gib", "loader_mode"}
            for field in set(original) | set(streamed):
                if field not in differences:
                    self.assertEqual(original[field], streamed[field], field)
            self.assertNotIn("loader_mode", original)

    def test_streamed_identity_loader_and_resource_mutations_fail_closed(self):
        for stage in ("load_forward", "full", "backward_resume"):
            cfg = laptop_profile.expected_config(stage, profile=laptop_profile.STREAMED_PROFILE)
            for field, value in (("loader_mode", "mmap"), ("loader_mode", "stream_read_1MiB_v1"),
                                 ("resource_profile", laptop_profile.RESOURCE_PROFILE),
                                 ("resource_profile", "unknown"), ("host_free_gib", 1.0),
                                 ("worker_free_gib", 1.0), ("worker_rss_limit_gib", 2.0),
                                 ("worker_rss_limit_gib", True), ("gpu_free_gib", 3.9)):
                with self.subTest(stage=stage, field=field), self.assertRaises(RuntimeError):
                    qualify_native.validate_config({**cfg, field: value})
            for field in cfg:
                mutated = deepcopy(cfg)
                del mutated[field]
                with self.subTest(stage=stage, missing=field), self.assertRaises(RuntimeError):
                    laptop_profile.validate_config(mutated)
            with self.assertRaises(RuntimeError):
                laptop_profile.validate_config({**cfg, "extra_loader_option": "auto"})
            original = laptop_profile.expected_config(stage)
            with self.assertRaises(RuntimeError):
                qualify_native.validate_config({**original, "loader_mode": laptop_profile.STREAMED_LOADER_MODE})

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

    def test_backward_resume_has_fixed_partial_scope_and_distinct_status(self):
        cfg = laptop_profile.expected_config("backward_resume")
        self.assertEqual(cfg["contexts"], [256, 512, 1024])
        self.assertEqual(cfg["parity_prompts"], 32)
        self.assertEqual(cfg["maximum_optimizer_calls"], 5)
        self.assertEqual(cfg["generation_cap"], 0)
        self.assertEqual(cfg["wall_seconds"], 1800)
        self.assertEqual(cfg["update_semantics"], "disposable adapter backward/update/resume resource qualification only; padding and sampled-likelihood qualification remain separate and unresolved")
        statuses = {laptop_profile.PROBE_PASS_STATUS, laptop_profile.BACKWARD_PASS_STATUS, laptop_profile.FULL_PASS_STATUS}
        self.assertEqual(len(statuses), 3)
        self.assertEqual(laptop_profile.BACKWARD_PASS_STATUS, "PASS_E0_BACKWARD_RESUME_RESOURCE_ONLY")
        for field, value in (("generation_cap", 32), ("maximum_optimizer_calls", 0),
                             ("parity_prompts", 0), ("contexts", [256]),
                             ("qualification_stage", "full"), ("update_semantics", "fully qualified")):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                qualify_native.validate_config({**cfg, field: value})

    def test_profile_rejects_missing_extra_and_cross_stage_fields(self):
        for stage in ("load_forward", "full", "backward_resume"):
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

    def test_streamed_admission_reserves_registered_remaining_growth(self):
        cfg = laptop_profile.expected_config("load_forward", profile=laptop_profile.STREAMED_PROFILE)
        for rss, required in ((0, 3.0), (.75, 2.25), (1.35, 1.65), (1.75, 1.25)):
            with self.subTest(rss=rss):
                self.assertAlmostEqual(laptop_profile.admission_required_gib(cfg, rss), required)
                self.assertAlmostEqual(required + rss, 3.0)
        self.assertEqual(laptop_profile.admission_required_gib(cfg, 1.8), 1.25)

    def test_unpadded_scope_has_exact_streamed_only_stage_and_budget(self):
        cfg = laptop_profile.expected_config("unpadded_ce", profile=laptop_profile.STREAMED_PROFILE)
        qualify_native.validate_config(cfg)
        self.assertEqual(cfg["runtime_scope"], "UNPADDED_SINGLE_SEQUENCE_CE_V1")
        self.assertEqual(cfg["generation_cap"], 0)
        self.assertEqual(cfg["parity_prompts"], 32)
        self.assertEqual(cfg["contexts"], [256, 512, 1024])
        self.assertEqual(cfg["maximum_optimizer_calls"], 5)
        self.assertEqual(cfg["wall_seconds"], 1800)
        self.assertEqual(cfg["host_free_gib"], 1.25)
        self.assertEqual(cfg["worker_rss_limit_gib"], 1.75)
        self.assertEqual(cfg["update_semantics"], "single-sequence unpadded completion CE qualification; padding, packing, batched inference and sampled-policy likelihood remain unsupported")
        self.assertEqual(laptop_profile.UNPADDED_PASS_STATUS, "PASS_E0_UNPADDED_CE_SCOPED")
        self.assertNotIn(laptop_profile.UNPADDED_PASS_STATUS,
                         {laptop_profile.FULL_PASS_STATUS, laptop_profile.PROBE_PASS_STATUS, laptop_profile.BACKWARD_PASS_STATUS})
        with self.assertRaises(RuntimeError):
            laptop_profile.expected_config("unpadded_ce")
        for field, value in (("runtime_scope", "full"), ("generation_cap", 32),
                             ("qualification_stage", "backward_resume"), ("resource_profile", laptop_profile.RESOURCE_PROFILE),
                             ("logprob_max_atol", .1), ("logprob_mean_atol", .1)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                qualify_native.validate_config({**cfg, field: value})
        for field in cfg:
            changed = deepcopy(cfg)
            del changed[field]
            with self.subTest(missing=field), self.assertRaises(RuntimeError):
                laptop_profile.validate_config(changed)
        for stage in ("load_forward", "full", "backward_resume"):
            with self.subTest(old_stage=stage), self.assertRaises(RuntimeError):
                qualify_native.validate_config({**laptop_profile.expected_config(stage, profile=laptop_profile.STREAMED_PROFILE),
                                                "runtime_scope": laptop_profile.UNPADDED_SCOPE})

    def test_unpadded_forward_omits_only_valid_redundant_masks(self):
        ids = FakeData((1, 80))
        for mask in (None, FakeData((1, 80), [1] * 80), FakeData((1, 80), [True] * 80)):
            original = dict(use_cache=False, logits_to_keep=1, attention_mask=mask)
            normalized = laptop_profile.prepare_unpadded_forward(ids, original)
            self.assertEqual(normalized, {"use_cache": False, "logits_to_keep": 1})
            self.assertIn("attention_mask", original)
            self.assertIs(original["attention_mask"], mask)
        self.assertEqual(laptop_profile.prepare_unpadded_forward(ids, {"use_cache": False}), {"use_cache": False})

    def test_unpadded_forward_rejects_batches_zero_masks_packing_and_cache(self):
        for shape in ((2, 80), (80,), (1, 80, 1), (0, 80), (True, 80), (1, 0), (1, 1025)):
            with self.subTest(shape=shape), self.assertRaises(RuntimeError):
                laptop_profile.prepare_unpadded_forward(FakeData(shape), {"use_cache": False})
        ids = FakeData((1, 80))
        for mask in (FakeData((1, 80), [1] * 79 + [0]), FakeData((1, 80), [0] * 80),
                     FakeData((1, 80), [1] * 79 + [float("nan")]), FakeData((1, 79), [1] * 79),
                     FakeData((1, 1, 80, 80), [1] * 6400)):
            with self.subTest(mask_shape=mask.shape), self.assertRaises(RuntimeError):
                laptop_profile.prepare_unpadded_forward(ids, {"use_cache": False, "attention_mask": mask})
        for field, value in (("packing", True), ("packed", True), ("cu_seqlens", [0, 40, 80]),
                             ("position_ids", FakeData((1, 80))), ("past_key_values", object()),
                             ("cache_position", [0]), ("inputs_embeds", object()), ("runtime_scope_bypass", True)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                laptop_profile.prepare_unpadded_forward(ids, {"use_cache": False, field: value})
        for kwargs in ({}, {"use_cache": True}, {"use_cache": 0}, {"use_cache": False, "logits_to_keep": True},
                       {"use_cache": False, "logits_to_keep": 81}, {"use_cache": False, "logits_to_keep": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                laptop_profile.prepare_unpadded_forward(ids, kwargs)

    def test_unpadded_diagnostic_exception_is_private_fixed_and_validated_first(self):
        tree = ast.parse(Path(qualify_native.__file__).read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        forward = next(node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "forward")
        guard = forward.body[0]
        self.assertIsInstance(guard, ast.If)
        self.assertEqual(ast.unparse(guard.test), "unpadded_ce")
        self.assertEqual(ast.unparse(guard.body[0].value.func), "laptop_profile.prepare_unpadded_forward")
        diagnostic = guard.body[1]
        self.assertEqual(ast.unparse(diagnostic.test), "stage == 'unpad_allones_mask'")
        self.assertEqual(ast.unparse(diagnostic.body[0].value.func), "require")
        self.assertEqual(ast.unparse(diagnostic.body[0].value.args[0]), "tuple(ids.shape) == (1, 80) and training is False")
        self.assertEqual(ast.unparse(diagnostic.body[1].value.func), "require")
        self.assertEqual(ast.unparse(diagnostic.body[2].targets[0]), "validated['attention_mask']")
        self.assertEqual(ast.unparse(diagnostic.body[2].value), "kwargs['attention_mask']")
        self.assertLess(guard.lineno, forward.body[-1].lineno)

    def test_unpadded_gate_requires_bitwise_causality_and_registered_allones_tolerance(self):
        tree = ast.parse(Path(qualify_native.__file__).read_text(encoding="utf-8"))
        stage = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "unpadded_ce" and any(isinstance(child, ast.With) for child in node.body))
        source = ast.unparse(stage)
        self.assertIn("torch.equal(short_logits[:, :-3], altered[:, :-3])", source)
        self.assertIn("torch.tensor([[17, 18, 19]], device='cuda')", source)
        self.assertIn("attention_mask=torch.ones_like(short_ids)", source)
        self.assertIn("metrics['max_abs'] <= cfg['logprob_max_atol']", source)
        self.assertIn("metrics['mean_abs'] <= cfg['logprob_mean_atol']", source)
        self.assertIn("require(metrics['qualified'], 'explicit all-ones likelihood disagreement')", source)
        self.assertIn("result['checks']['native_unpadded_causality']", source)
        self.assertIn("result['checks']['explicit_allones_mask_equivalence']", source)

    def test_containment_limits_are_explicit_and_not_native_qualification(self):
        for stage in ("load_forward", "full", "backward_resume"):
            cfg = laptop_profile.expected_config(stage)
            limits = laptop_profile.resource_limits(cfg)
            self.assertEqual(limits["worker_free_gib"], 2.0)
            self.assertEqual(limits["worker_rss_limit_gib"], 2.5)
            self.assertEqual(limits["admission_budget"], laptop_profile.ADMISSION_BUDGET)
            self.assertEqual(limits["wall_seconds"], 600 if stage == "load_forward" else 1800)

    def test_supervisor_dispatches_exact_limits_and_records_them(self):
        # Replace every external query and the watchdog: this launches no worker.
        profiles = (laptop_profile.RESOURCE_PROFILE, laptop_profile.STREAMED_PROFILE)
        stages = ("load_forward", "full", "backward_resume")
        registrations = [(None, None)] + list(product(profiles, stages)) + [(laptop_profile.STREAMED_PROFILE, "unpadded_ce")]
        for profile, stage in registrations:
            with self.subTest(profile=profile, stage=stage), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                cfg = (json.loads(laptop_profile.LEGACY_CONFIG_PATH.read_bytes()) if stage is None
                       else laptop_profile.expected_config(stage, profile=profile))
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
                self.assertEqual(limits["min_free_gib"], cfg["worker_free_gib"])
                self.assertEqual(limits["max_rss_gib"], 4.0 if stage is None else cfg["worker_rss_limit_gib"])
                launch = json.loads((base / "result_supervisor/LAUNCH.json").read_text())
                self.assertEqual(limits["go"]["config_sha256"], launch["config_sha256"])
                if stage is None:
                    self.assertNotIn("resource_limits", launch)
                    self.assertNotIn("resource_profile", launch)
                else:
                    self.assertEqual(launch["resource_limits"], laptop_profile.resource_limits(cfg))
                    self.assertEqual(launch["qualification_stage"], stage)
                    self.assertEqual(launch["resource_profile"], profile)
                    if profile == laptop_profile.STREAMED_PROFILE:
                        self.assertEqual(launch["loader_mode"], laptop_profile.STREAMED_LOADER_MODE)
                    else:
                        self.assertNotIn("loader_mode", launch)
                self.assertFalse((base / "rineya-central-gpu0.lock").exists())

    def test_imports_do_not_load_model_frameworks(self):
        code = "import sys; import experiments.post_training_mimo_centralized_v1.scripts.laptop_profile; import experiments.post_training_mimo_centralized_v1.scripts.qualify_native; assert not any(n == 'torch' or n.startswith('torch.') or n == 'transformers' or n.startswith('transformers.') for n in sys.modules)"
        run = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_core_math_and_complete_update_resume_body_remain_historical(self):
        # This compares committed numerical source, not a synthetic efficacy run.
        relative = "experiments/post_training_mimo_centralized_v1/scripts/qualify_native.py"
        historical = subprocess.run(["git", "show", "ba683ac0fcadbf04e4e179200d035b6a980e388f:" + relative],
                                    capture_output=True, text=True, check=True).stdout
        current = Path(qualify_native.__file__).read_text(encoding="utf-8")
        old_tree, new_tree = ast.parse(historical), ast.parse(current)
        for name in ("completion_loss", "install_lora", "tensor_digest"):
            old = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            new = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(new, include_attributes=False), ast.dump(old, include_attributes=False), name)
        def update_resume_body(tree):
            run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
            body = next(node for node in run.body if isinstance(node, ast.Try)).body
            start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "optimizer" for target in node.targets))
            end = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                       and any(ast.unparse(target) == "result['checks']['finite_scoped_backward']" for target in node.targets))
            return ast.dump(ast.Module(body=body[start:end + 1], type_ignores=[]), include_attributes=False)
        self.assertEqual(update_resume_body(new_tree), update_resume_body(old_tree))

    def test_padding_failure_telemetry_precedes_unchanged_failure_gate(self):
        tree = ast.parse(Path(qualify_native.__file__).read_text(encoding="utf-8"))
        conditional = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                           and ast.unparse(node.test) == "not backward_resume_only and (not unpadded_ce)")
        require_calls = [node for node in ast.walk(conditional) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "require"
                         and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                         and node.args[1].value == "native padding likelihood disagreement"]
        self.assertEqual(len(require_calls), 1)
        self.assertEqual(ast.unparse(require_calls[0].args[0]), "padding_ok")
        telemetry = next(node for node in ast.walk(conditional) if isinstance(node, ast.If)
                         and ast.unparse(node.test) == "not padding_ok")
        assignment = telemetry.body[0]
        self.assertIsInstance(assignment, ast.Assign)
        self.assertEqual(ast.unparse(assignment.targets[0]), "result['failed_check_measurement']")
        self.assertEqual({keyword.arg for keyword in assignment.value.keywords},
                         {"left", "right", "max_abs", "mean_abs", "reference_logprobs",
                          "observed_logprobs", "changed_argmax_positions"})
        self.assertEqual(ast.unparse(assignment.value.func), "padding_failure_measurement")
        self.assertLess(telemetry.lineno, require_calls[0].lineno)

    def test_failed_padding_measurements_preserve_finite_values_and_nonfinite_failure(self):
        finite = qualify_native.padding_failure_measurement(left=2, right=0, max_abs=.05, mean_abs=.02,
                    reference_logprobs=[-1.0, -2.0], observed_logprobs=[-1.05, -2.01], changed_argmax_positions=[1])
        self.assertEqual(finite, {"check": "native_padding_likelihood", "left": 2, "right": 0,
                    "max_abs": .05, "mean_abs": .02, "reference_logprobs": [-1.0, -2.0],
                    "observed_logprobs": [-1.05, -2.01], "changed_argmax_positions": [1]})
        failing = qualify_native.padding_failure_measurement(left=2, right=0, max_abs=float("inf"),
                    mean_abs=float("nan"), reference_logprobs=[-1.0, float("-inf")],
                    observed_logprobs=[float("nan"), -2.01], changed_argmax_positions=[])
        self.assertIsNone(failing["max_abs"])
        self.assertIsNone(failing["mean_abs"])
        self.assertEqual(failing["reference_logprobs"], [-1.0, None])
        self.assertEqual(failing["observed_logprobs"], [None, -2.01])
        json.dumps(failing, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
