"""Preparation integration controls: synthetic bindings, no native model or tokenizer."""
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import prepare_e1 as prepare


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[1]


def synthetic_binding():
    rows = []
    for component, count in (("acquisition", 4096), ("deterministic_replay", 768), ("language_replay", 256)):
        for index in range(count):
            rows.append({"key": f"{component}:{index}", "component": component, "role": "TRAIN",
                         "parent_id": f"{component}:parent:{index}",
                         "nonpadding_tokens": 100 + index % 13, "completion_tokens": 3 + index % 7})
    return {"schema_version": 1, "status": "SYNTHETIC_CONTROL_NOT_NATIVE_EVIDENCE",
            "parent_sha256": "a" * 64, "model_sha256": "b" * 64,
            "data_sha256": prepare.e1_data.digest(rows), "rows": rows}


def blocked_gates(reason="SYNTHETIC_E0_NOT_PASSED"):
    return {"schema_version": 1, "status": "BLOCKED_E1_DIAGNOSTIC_ONLY",
            "gates": {"e0_native": {"status": "BLOCKED", "reasons": [{"code": reason}]}},
            "native_launch_authorized": False, "e1_launch_ready": False,
            "no_native_executor": True, "training_updates": 0, "model_loads": 0,
            "teacher_calls": 0, "held_out_access": {"CHECK": False, "FINAL": False}}


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((PACKAGE / "configs/E1_PLAN_V1.json").read_text())
        cls.binding = synthetic_binding()

    @contextmanager
    def sandbox(self, *, gates=None, config=None, binding=None):
        """Keep real scheduler, serialization and file hashes; replace data/gate/git providers."""
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory) / "repo"
            package = root / "experiments" / "synthetic_e1"
            scripts = package / "scripts"
            scripts.mkdir(parents=True)
            for module, filename in ((prepare, "prepare_e1.py"), (prepare.e1_data, "e1_data.py"),
                                     (prepare.e1_schedule, "e1_schedule.py"), (prepare.e1_gates, "e1_gates.py")):
                path = scripts / filename
                path.write_text("# Synthetic source-identity fixture for " + filename + "\n")
                stack.enter_context(patch.object(module, "__file__", str(path)))
            config_path = package / "configs/E1_PLAN_V1.json"
            config_path.parent.mkdir()
            config_path.write_bytes(prepare.e1_data.canonical(self.config if config is None else config) + b"\n")

            def git_output(command, **kwargs):
                self.assertEqual(command[:3], ["git", "-C", str(root)])
                if command[3:] == ["rev-parse", "HEAD"]:
                    return "c" * 40 + "\n"
                if command[3:] == ["status", "--porcelain"]:
                    return "?? unrelated-control.tmp\n"
                if command[3:6] == ["status", "--porcelain", "--"]:
                    return ""
                raise AssertionError("Unexpected Git command: " + repr(command))

            git_run = stack.enter_context(patch.object(prepare.subprocess, "run", return_value=
                                                      subprocess.CompletedProcess([], 0, b"", b"")))
            git_read = stack.enter_context(patch.object(prepare.subprocess, "check_output", side_effect=git_output))
            bind = stack.enter_context(patch.object(prepare.e1_data, "bind", return_value=
                                                   deepcopy(self.binding if binding is None else binding)))
            preflight = stack.enter_context(patch.object(prepare.e1_gates, "preflight", return_value=
                                                        deepcopy(blocked_gates() if gates is None else gates)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield {"root": root, "package": package, "config": config_path, "output": package / "outputs/plan",
                   "bind": bind, "preflight": preflight, "git_run": git_run, "git_read": git_read}

    def test_current_frozen_config_validates(self):
        self.assertIsNone(prepare.validate_config(deepcopy(self.config)))

    def test_frozen_contract_changes_rejected(self):
        changes = [
            (("status",), "DRAFT"), (("seed",), 211), (("seed",), 307), (("native_execution",), True),
            (("optimizer", "learning_rate"), 0.0002), (("lora", "targets"), ["q_proj", "k_proj", "v_proj"]),
            (("sequence", "acquisition_per_step"), 15), (("sequence", "gradient_accumulation"), 19),
            (("schedule", "warmup_steps"), 12), (("schedule", "steps"), 255),
            (("schedule", "selection"), "best_dev"), (("schedule", "checkpoint_steps"), [0, 128, 256]),
            (("budget", "training_token_ceiling"), 5_242_881), (("budget", "teacher_call_ceiling"), 1),
            (("schema_version",), 2), (("loss",), "mean token CE across the entire batch"),
            (("resource_minima", "host_free_gib"), 7.0), (("resource_minima", "worker_free_gib"), 5.9),
        ]
        for path, value in changes:
            config = deepcopy(self.config)
            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                prepare.validate_config(config)

    def test_boolean_and_float_scalar_substitutions_rejected(self):
        changes = [(("seed",), 101.0), (("native_execution",), 0),
                   (("sequence", "microbatch"), True), (("sequence", "acquisition_per_step"), 16.0),
                   (("lora", "base_frozen"), 1), (("budget", "teacher_call_ceiling"), False),
                   (("schedule", "steps"), 256.0)]
        for path, value in changes:
            config = deepcopy(self.config)
            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                prepare.validate_config(config)

    def test_fresh_import_has_no_environment_or_backend_side_effects(self):
        code = (
            "import os,sys; before=dict(os.environ); "
            "from experiments.post_training_mimo_centralized_v1.scripts import prepare_e1; "
            "assert dict(os.environ)==before; "
            "assert not any(name in sys.modules for name in ('torch','transformers','tokenizers','safetensors')); "
            "assert not any('tokenize_bridge' in name or 'tokenize_retention' in name for name in sys.modules); "
            "print('IMPORT_SAFE')"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPOSITORY,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")

    def test_write_refuses_overwrite_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            prepare.write(path, {"original": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare.write(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)

    def test_prepare_writes_bound_plan_cursor_and_separate_blocked_gates(self):
        with self.sandbox() as fixture:
            receipt = prepare.prepare(fixture["config"], fixture["output"])
            files = {path.name: path for path in fixture["output"].iterdir()}
            self.assertEqual(set(files), {"DATA_BINDING.json", "PLAN.json", "GATES.json", "CURSOR_STEP_0.json",
                                          "CONFIG.json", "PLAN_RECEIPT.json"})
            read = lambda name: json.loads(files[name].read_bytes())
            plan, cursor, gates = read("PLAN.json"), read("CURSOR_STEP_0.json"), read("GATES.json")
            self.assertEqual(read("PLAN_RECEIPT.json"), receipt)
            self.assertEqual(read("DATA_BINDING.json"), self.binding)
            self.assertEqual(read("CONFIG.json"), self.config)
            self.assertEqual(set(receipt["files"]), set(files) - {"PLAN_RECEIPT.json"})
            for name, identity in receipt["files"].items():
                self.assertEqual(identity, prepare.e1_data.identity(files[name]))
            self.assertEqual(receipt["identities"]["config_sha256"], prepare.e1_data.identity(fixture["config"])["sha256"])
            self.assertEqual(receipt["identities"]["source_sha256"], prepare.e1_data.digest(receipt["source_files"]))
            self.assertEqual(len(receipt["source_files"]), 5)
            self.assertEqual(receipt["code_commit"], "c" * 40)
            self.assertEqual(receipt["overall_dirty_status"], "?? unrelated-control.tmp\n")
            self.assertEqual(receipt["plan_sha256"], plan["plan_sha256"])
            self.assertEqual(receipt["rows"], 5120)
            self.assertEqual(receipt["steps"], 256)
            self.assertEqual(receipt["nonpadding_tokens"], sum(row["nonpadding_tokens"] for row in self.binding["rows"]))
            self.assertEqual(receipt["supervised_tokens"], sum(row["completion_tokens"] for row in self.binding["rows"]))
            self.assertEqual(prepare.e1_schedule.validate_resume_cursor(
                cursor, plan, rows=self.binding["rows"], identities=receipt["identities"]), 1)
            self.assertEqual(cursor["consumed_rows"], 0)
            self.assertEqual(cursor["consumed_nonpadding_tokens"], 0)
            self.assertFalse(receipt["native_launch_authorized"])
            self.assertEqual(receipt["model_calls"], 0)
            self.assertEqual(receipt["training_updates"], 0)
            self.assertFalse(gates["native_launch_authorized"])
            self.assertFalse(gates["e1_launch_ready"])
            self.assertNotIn("gates", plan)
            fixture["bind"].assert_called_once_with(fixture["package"], self.config)
            fixture["preflight"].assert_called_once_with(fixture["package"])
            self.assertEqual(fixture["git_run"].call_count, 5)

    def test_gate_diagnostics_do_not_change_deterministic_plan_hash(self):
        with self.sandbox() as fixture:
            first = prepare.prepare(fixture["config"], fixture["output"])
            other_gates = blocked_gates("SYNTHETIC_DIFFERENT_NATIVE_DIAGNOSTIC")
            other_gates["gates"]["e0_native"]["status"] = "PASS_E0_EVIDENCE_CONSISTENCY_ONLY"
            fixture["preflight"].return_value = other_gates
            second_output = fixture["output"].with_name("plan_second")
            second = prepare.prepare(fixture["config"], second_output)
            self.assertEqual(first["plan_sha256"], second["plan_sha256"])
            self.assertEqual((fixture["output"] / "PLAN.json").read_bytes(), (second_output / "PLAN.json").read_bytes())
            self.assertNotEqual(first["files"]["GATES.json"]["sha256"], second["files"]["GATES.json"]["sha256"])
            self.assertFalse(second["native_launch_authorized"])

    def test_existing_output_stops_before_data_or_git_reads(self):
        with self.sandbox() as fixture:
            fixture["output"].mkdir(parents=True)
            marker = fixture["output"] / "preserved.txt"
            marker.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "write-once"):
                prepare.prepare(fixture["config"], fixture["output"])
            self.assertEqual(marker.read_text(), "preserve")
            fixture["git_run"].assert_not_called()
            fixture["git_read"].assert_not_called()
            fixture["bind"].assert_not_called()
            fixture["preflight"].assert_not_called()

    def test_untracked_or_changed_source_blocks_before_data_binding(self):
        for problem in ("untracked", "changed"):
            with self.subTest(problem=problem), self.sandbox() as fixture:
                if problem == "untracked":
                    fixture["git_run"].side_effect = subprocess.CalledProcessError(1, ["git", "ls-files"])
                    error = subprocess.CalledProcessError
                else:
                    fixture["git_read"].side_effect = None
                    fixture["git_read"].return_value = " M scripts/prepare_e1.py\n"
                    error = ValueError
                with self.assertRaises(error):
                    prepare.prepare(fixture["config"], fixture["output"])
                self.assertFalse(fixture["output"].exists())
                fixture["bind"].assert_not_called()
                fixture["preflight"].assert_not_called()

    def test_a_gate_flag_cannot_grant_native_launch(self):
        gates = blocked_gates()
        gates["native_launch_authorized"] = True
        with self.sandbox(gates=gates) as fixture:
            with self.assertRaisesRegex(ValueError, "cannot authorize native"):
                prepare.prepare(fixture["config"], fixture["output"])
            self.assertFalse(fixture["output"].exists())

    def test_invalid_config_stops_before_git_or_data_binding(self):
        config = deepcopy(self.config)
        config["seed"] = 307
        with self.sandbox(config=config) as fixture:
            with self.assertRaises(ValueError):
                prepare.prepare(fixture["config"], fixture["output"])
            self.assertFalse(fixture["output"].exists())
            fixture["git_run"].assert_not_called()
            fixture["bind"].assert_not_called()
            fixture["preflight"].assert_not_called()

    def test_binding_held_out_role_cannot_reach_output_or_gates(self):
        binding = deepcopy(self.binding)
        binding["rows"][0]["role"] = "CHECK"
        with self.sandbox(binding=binding) as fixture:
            with self.assertRaisesRegex(ValueError, "TRAIN only"):
                prepare.prepare(fixture["config"], fixture["output"])
            self.assertFalse(fixture["output"].exists())
            fixture["preflight"].assert_not_called()

    def test_config_or_source_edit_during_binding_or_preflight_blocks_publication(self):
        for phase in ("bind", "preflight"):
            for target in ("config", "source"):
                with self.subTest(phase=phase, target=target), self.sandbox() as fixture:
                    changed = fixture["config"] if target == "config" else fixture["package"] / "scripts/e1_schedule.py"
                    original = changed.read_bytes()

                    def mutate(*args, **kwargs):
                        # Config whitespace preserves parsed semantics but changes its byte identity.
                        # The source mutation preserves size to exercise the digest check itself.
                        changed.write_bytes(original + b"\n" if target == "config" else b"!" + original[1:])
                        return deepcopy(self.binding) if phase == "bind" else blocked_gates()

                    fixture[phase].side_effect = mutate
                    with self.assertRaisesRegex(ValueError, "planning source/config changed"):
                        prepare.prepare(fixture["config"], fixture["output"])
                    self.assertFalse(fixture["output"].exists())
                    self.assertNotEqual(changed.read_bytes(), original)

    def test_config_changed_after_parse_before_source_inventory_is_rejected(self):
        with self.sandbox() as fixture:
            original_read = prepare.e1_data.read_json

            def read_then_change(path):
                result = original_read(path)
                path.write_bytes(path.read_bytes() + b"\n")
                return result

            with patch.object(prepare.e1_data, "read_json", side_effect=read_then_change):
                with self.assertRaisesRegex(ValueError, "config changed while parsing"):
                    prepare.prepare(fixture["config"], fixture["output"])
            self.assertFalse(fixture["output"].exists())
            fixture["bind"].assert_not_called()
            fixture["preflight"].assert_not_called()

    def test_head_change_during_binding_or_preflight_blocks_publication(self):
        for phase in ("bind", "preflight"):
            with self.subTest(phase=phase), self.sandbox() as fixture:
                original_git_read = fixture["git_read"].side_effect
                revision = {"value": "c" * 40}

                def git_read(command, **kwargs):
                    if command[3:] == ["rev-parse", "HEAD"]:
                        return revision["value"] + "\n"
                    return original_git_read(command, **kwargs)

                def change_head(*args, **kwargs):
                    revision["value"] = "d" * 40
                    return deepcopy(self.binding) if phase == "bind" else blocked_gates()

                fixture["git_read"].side_effect = git_read
                fixture[phase].side_effect = change_head
                with self.assertRaisesRegex(ValueError, "source commit changed while planning"):
                    prepare.prepare(fixture["config"], fixture["output"])
                self.assertFalse(fixture["output"].exists())
                head_reads = [call for call in fixture["git_read"].call_args_list
                              if call.args[0][3:] == ["rev-parse", "HEAD"]]
                self.assertEqual(len(head_reads), 2)

    def test_source_becoming_dirty_without_byte_change_blocks_publication(self):
        with self.sandbox() as fixture:
            original_git_read = fixture["git_read"].side_effect
            state = {"dirty": False}

            def git_read(command, **kwargs):
                if state["dirty"] and command[3:6] == ["status", "--porcelain", "--"]:
                    return "M  " + command[-1] + "\n"
                return original_git_read(command, **kwargs)

            def become_dirty(*args, **kwargs):
                state["dirty"] = True
                return blocked_gates()

            fixture["git_read"].side_effect = git_read
            fixture["preflight"].side_effect = become_dirty
            with self.assertRaisesRegex(ValueError, "planning source/config became uncommitted"):
                prepare.prepare(fixture["config"], fixture["output"])
            self.assertFalse(fixture["output"].exists())


if __name__ == "__main__":
    unittest.main()
