from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    transformers_stub.StoppingCriteria = object
    transformers_stub.StoppingCriteriaList = list
    sys.modules["transformers"] = transformers_stub

from scripts.common import experiment_utils
from scripts.common.incremental_results import DurableJsonlStore
from scripts.common.quality_evaluation import build_corpus_profile, load_quality_config
from scripts.phase_1_dense_sft import evaluate_model as generation_evaluator
from scripts.phase_1_dense_sft.evaluate_model import (
    evaluation_manifest_mismatches,
    format_optional_number,
)
from scripts.phase_2_layer_selection.compare_outputs import compare_rows
from scripts.phase_2_layer_selection import evaluate_teacher_forcing as teacher_forcing
from scripts.phase_2_layer_selection.evaluate_teacher_forcing import manifest_mismatches
from scripts.phase_2c_targeted_recovery import targeted_recovery


class ReliabilityV147Tests(unittest.TestCase):
    def test_atomic_replace_retries_transient_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            destination = Path(temp) / "destination"
            source.write_text("complete", encoding="utf-8")
            real_replace = experiment_utils.os.replace
            calls = 0

            def flaky_replace(src: Path, dst: Path) -> None:
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError(13, "temporarily locked", str(dst))
                real_replace(src, dst)

            with mock.patch.object(experiment_utils.os, "replace", side_effect=flaky_replace), mock.patch.object(
                experiment_utils.time, "sleep", return_value=None
            ):
                experiment_utils.replace_path_with_retry(source, destination)

            self.assertEqual(calls, 3)
            self.assertEqual(destination.read_text(encoding="utf-8"), "complete")

    def test_best_effort_progress_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            experiment_utils, "write_json", side_effect=PermissionError("locked")
        ):
            self.assertFalse(
                experiment_utils.write_json_best_effort(Path(temp) / "progress.json", {"ok": True})
            )

    def test_optional_number_handles_none_and_nonfinite(self) -> None:
        self.assertEqual(format_optional_number(None), "N/A")
        self.assertEqual(format_optional_number(float("nan")), "N/A")
        self.assertEqual(format_optional_number(0.75), "0.750")

    def test_generation_resume_rejects_manifestless_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark = root / "benchmark.jsonl"
            output = root / "evaluation"
            output.mkdir()
            benchmark.write_text(
                json.dumps({
                    "id": "a",
                    "category": "test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "reference": "Hi",
                }) + "\n",
                encoding="utf-8",
            )
            (output / "results.jsonl").write_text(
                json.dumps({"id": "a", "reply": "Hi"}) + "\n",
                encoding="utf-8",
            )
            argv = [
                "evaluate_model.py",
                "--model-id", "model",
                "--benchmark", str(benchmark),
                "--output-dir", str(output),
                "--resume",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(RuntimeError, "no evaluation manifest"):
                    generation_evaluator.main()

    def test_generation_manifest_normalizes_equivalent_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            adapter = root / "adapter"
            model.mkdir()
            adapter.mkdir()
            requested = {
                "schema_version": 2,
                "model_id": str(model),
                "adapter_path": str(adapter),
                "benchmark": str(root / "benchmark.jsonl"),
                "benchmark_sha256": "x",
                "quality_config_sha256": "y",
                "quality_evaluation_version": "4.5",
                "max_new_tokens": 32,
                "response_control": "eos_only",
                "min_new_tokens_before_boundary": 1,
                "do_sample": False,
                "system_prompt": "prompt",
                "requested_limit": 180,
                "case_count": 180,
                "record_ids_sha256": "ids",
                "temperature": 0.7,
                "top_p": 0.9,
            }
            existing = dict(requested)
            existing["model_id"] = str(model / ".." / "model")
            existing["adapter_path"] = str(adapter / ".." / "adapter")
            self.assertEqual(evaluation_manifest_mismatches(existing, requested), [])
            existing["max_new_tokens"] = 31
            self.assertTrue(any("max_new_tokens" in item for item in evaluation_manifest_mismatches(existing, requested)))

    def test_teacher_forcing_manifest_normalizes_equivalent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            input_path = root / "input.jsonl"
            input_path.write_text("", encoding="utf-8")
            requested = {
                "schema_version": 1,
                "model_id": str(model),
                "adapter_path": None,
                "input": str(input_path),
                "input_sha256": "x",
                "requested_limit": 10,
                "input_case_count": 10,
                "record_ids_sha256": "ids",
                "max_sequence_length": 1024,
            }
            existing = dict(requested)
            existing["model_id"] = str(model / ".." / "model")
            self.assertEqual(manifest_mismatches(existing, requested), [])

    def test_output_comparison_rejects_partial_case_coverage(self) -> None:
        reference = [{"id": "a", "reply": "A"}, {"id": "b", "reply": "B"}]
        candidate = [{"id": "a", "reply": "A"}]
        with self.assertRaisesRegex(ValueError, "same stable IDs"):
            compare_rows(reference, candidate)

    def test_output_comparison_rejects_duplicate_ids(self) -> None:
        reference = [{"id": "a", "reply": "A"}, {"id": "a", "reply": "A2"}]
        candidate = [{"id": "a", "reply": "A"}]
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            compare_rows(reference, candidate)

    def test_phase2c_resume_is_forwarded_to_generation_and_nll(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generation = root / "generation"
            nll = root / "teacher_forcing"
            generation.mkdir(parents=True)
            nll.mkdir(parents=True)
            (root / "dev.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
            (generation / "summary.json").write_text(
                json.dumps({"case_count": 1, "mean_quality_score": 1.0}), encoding="utf-8"
            )
            (nll / "summary.json").write_text(
                json.dumps({"case_count": 1, "mean_target_nll": 1.0}), encoding="utf-8"
            )
            (nll / "evaluation_manifest.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            (generation / "results.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
            args = SimpleNamespace(
                dev_file=root / "dev.jsonl",
                max_new_tokens=32,
                eval_limit=1,
                resume=True,
            )
            commands: list[list[str]] = []

            def capture(command: list[str], sentinel: Path, resume: bool, **_: object) -> None:
                commands.append(command)

            with mock.patch.object(targeted_recovery, "run_if_needed", side_effect=capture):
                targeted_recovery.evaluate_model("model", None, root, args)

            self.assertEqual(len(commands), 2)
            self.assertIn("--resume", commands[0])
            self.assertIn("--resume", commands[1])

    def test_phase2c_rejects_manifestless_legacy_nll_without_authorized_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generation = root / "generation"
            nll = root / "teacher_forcing"
            generation.mkdir(parents=True)
            nll.mkdir(parents=True)
            (root / "dev.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
            (generation / "summary.json").write_text(
                json.dumps({"case_count": 1, "mean_quality_score": 1.0}), encoding="utf-8"
            )
            (generation / "results.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
            (nll / "summary.json").write_text(
                json.dumps({"case_count": 1, "mean_target_nll": 1.0}), encoding="utf-8"
            )
            (nll / "results.jsonl").write_text('{"id":"a","target_tokens":1,"mean_target_nll":1.0}\n', encoding="utf-8")
            args = SimpleNamespace(
                dev_file=root / "dev.jsonl",
                max_new_tokens=32,
                eval_limit=1,
                resume=True,
            )
            with mock.patch.object(targeted_recovery, "run_if_needed"):
                with self.assertRaisesRegex(RuntimeError, "no approved dataset migration"):
                    targeted_recovery.evaluate_model("model", None, root, args)

    def test_phase2c_routes_authorized_manifestless_legacy_nll_through_child_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generation = root / "generation"
            nll = root / "teacher_forcing"
            generation.mkdir(parents=True)
            nll.mkdir(parents=True)
            (root / "dev.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
            (generation / "summary.json").write_text(
                json.dumps({"case_count": 1, "mean_quality_score": 1.0}), encoding="utf-8"
            )
            (generation / "results.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
            (nll / "summary.json").write_text(
                json.dumps({"case_count": 1, "mean_target_nll": 1.0}), encoding="utf-8"
            )
            (nll / "results.jsonl").write_text('{"id":"a","target_tokens":1,"mean_target_nll":1.0}\n', encoding="utf-8")
            args = SimpleNamespace(
                dev_file=root / "dev.jsonl",
                max_new_tokens=32,
                eval_limit=1,
                resume=True,
                legacy_dev_input_sha256="old-hash",
            )
            with mock.patch.object(targeted_recovery, "run_if_needed"), mock.patch.object(
                targeted_recovery, "run"
            ) as run_mock, mock.patch.object(targeted_recovery, "read_json", side_effect=[
                {"case_count": 1, "mean_quality_score": 1.0},
                {"case_count": 1, "mean_target_nll": 1.0},
            ]):
                targeted_recovery.evaluate_model("model", None, root, args)
            command = run_mock.call_args.args[0]
            self.assertIn("--legacy-input-sha256", command)
            self.assertIn("old-hash", command)

    def test_phase2c_wrapper_builds_the_selected_dataset_directory(self) -> None:
        wrapper = Path("scripts/phase_2c_targeted_recovery/run_phase_2c.ps1").read_text(encoding="utf-8")
        self.assertIn('"--output-dir", $DatasetOutputDir', wrapper)
        self.assertIn('must be sibling train.jsonl/dev.jsonl files', wrapper)

    def test_invalid_json_sentinel_is_quarantined_before_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "summary.json"
            sentinel.write_text("{broken", encoding="utf-8")
            with mock.patch.object(targeted_recovery, "run") as run_mock:
                targeted_recovery.run_if_needed(
                    ["python", "step.py"], sentinel, True, required_json_keys=("case_count",)
                )
            run_mock.assert_called_once()
            self.assertFalse(sentinel.exists())
            self.assertTrue(list(Path(temp).glob("summary.json.invalid-*")))

    def test_known_phase2c_contract_migration_is_authorized(self) -> None:
        data_path = Path("data/phase_2c_targeted_recovery/dev.jsonl")
        migration = experiment_utils.approved_content_only_migration(
            data_path,
            previous_sha256="8ab54fd2e4cf25509a95fe7d6bc8a84fa5a230f0d57fdae5a50eb0e559063095",
        )
        self.assertIsNotNone(migration)
        assert migration is not None
        self.assertTrue(migration["stable_ids_messages_targets_unchanged"])
        self.assertTrue(migration["requires_generation_rescore"])

    def test_content_only_migration_rejects_semantic_digest_mismatch(self) -> None:
        source_data = Path("data/phase_2c_targeted_recovery/dev.jsonl")
        source_manifest = Path("data/phase_2c_targeted_recovery/dataset_manifest.json")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "dev.jsonl"
            manifest = root / "dataset_manifest.json"
            data.write_bytes(source_data.read_bytes())
            payload = json.loads(source_manifest.read_text(encoding="utf-8"))
            payload["content_only_migrations"]["dev.jsonl"]["id_messages_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(
                experiment_utils.approved_content_only_migration(
                    data,
                    previous_sha256="8ab54fd2e4cf25509a95fe7d6bc8a84fa5a230f0d57fdae5a50eb0e559063095",
                )
            )

    def test_teacher_forcing_authorizes_and_manifests_legacy_contract_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "dev.jsonl"
            output = root / "evidence"
            output.mkdir()
            data.write_text(
                json.dumps({
                    "id": "a",
                    "messages": [
                        {"role": "user", "content": "What remains?"},
                        {"role": "assistant", "content": "Two."},
                    ],
                }) + "\n",
                encoding="utf-8",
            )
            current_hash = experiment_utils.sha256(data)
            semantic_hash = experiment_utils._id_messages_sha256_for_jsonl(data)
            (root / "dataset_manifest.json").write_text(
                json.dumps({
                    "content_only_migrations": {
                        "dev.jsonl": {
                            "compatible_predecessor_sha256": ["old-hash"],
                            "current_sha256": current_hash,
                            "id_messages_sha256": semantic_hash,
                            "stable_ids_messages_targets_unchanged": True,
                        }
                    }
                }),
                encoding="utf-8",
            )
            (output / "results.jsonl").write_text(
                json.dumps({
                    "id": "a",
                    "target_tokens": 1,
                    "mean_target_nll": 1.25,
                    "target_perplexity": 3.49,
                }) + "\n",
                encoding="utf-8",
            )
            (output / "summary.json").write_text(
                json.dumps({
                    "model_id": "model",
                    "adapter_path": None,
                    "input": str(data),
                    "case_count": 1,
                    "mean_target_nll": 1.25,
                }),
                encoding="utf-8",
            )
            argv = [
                "evaluate_teacher_forcing.py",
                "--model-id", "model",
                "--input", str(data),
                "--output-dir", str(output),
                "--limit", "1",
                "--resume",
                "--legacy-input-sha256", "old-hash",
            ]
            with mock.patch.object(sys, "argv", argv):
                teacher_forcing.main()
            self.assertTrue((output / "evaluation_manifest.json").exists())
            migration = json.loads((output / "INPUT_CONTRACT_MIGRATION.json").read_text(encoding="utf-8"))
            self.assertTrue(migration["legacy_rows_validated_by_stable_id"])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["case_count"], 1)
            self.assertAlmostEqual(summary["mean_target_nll"], 1.25)

    def test_phase2c_run_config_accepts_only_known_contract_data_change(self) -> None:
        train = Path("data/phase_2c_targeted_recovery/train.jsonl")
        dev = Path("data/phase_2c_targeted_recovery/dev.jsonl")
        current = {
            "source_model": "same",
            "train_file": {
                "path": str(train), "bytes": train.stat().st_size,
                "sha256": experiment_utils.sha256(train),
            },
            "dev_file": {
                "path": str(dev), "bytes": dev.stat().st_size,
                "sha256": experiment_utils.sha256(dev),
            },
        }
        previous = json.loads(json.dumps(current))
        previous["train_file"].update(
            bytes=4256355,
            sha256="b8440e3f501727669c1833220f66aa77cf9a2abace3bb7de28b0fef5b63b7400",
        )
        previous["dev_file"].update(
            bytes=745849,
            sha256="8ab54fd2e4cf25509a95fe7d6bc8a84fa5a230f0d57fdae5a50eb0e559063095",
        )
        args = SimpleNamespace(train_file=train, dev_file=dev)
        migration = targeted_recovery.approved_run_config_contract_migration(previous, current, args)
        self.assertIsNotNone(migration)
        changed = json.loads(json.dumps(previous))
        changed["source_model"] = "different"
        self.assertIsNone(
            targeted_recovery.approved_run_config_contract_migration(changed, current, args)
        )

    def test_contract_migration_rescores_existing_reply_without_generation(self) -> None:
        from scripts.phase_1_dense_sft.evaluate_model import (
            normalize_benchmark_record,
            rescore_migrated_rows,
        )
        from scripts.phase_2c_targeted_recovery.build_phase_2c_dataset import generated_rows

        record = normalize_benchmark_record(
            next(row for row in generated_rows("p2c_dev", train=False) if row["id"] == "p2c_dev_arithmetic_03")
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = DurableJsonlStore.open(root / "results.jsonl", resume=False)
            store.append(
                {
                    "id": record["id"],
                    "category": record["category"],
                    "messages": record["messages"],
                    "reference": record["reference"],
                    "reply": "I need a full tray first.",
                    "visible_generated_tokens": 6,
                    "quality_score": 1.0,
                }
            )
            rescore_migrated_rows(
                records=[record],
                store=store,
                quality_config=load_quality_config(),
                corpus_profile=build_corpus_profile([record]),
                migration={
                    "previous_sha256": "old",
                    "current_sha256": "new",
                    "manifest_path": "dataset_manifest.json",
                },
                output_dir=root,
            )
            row = store.rows_by_id[str(record["id"])]
            self.assertLess(row["quality_score"], 0.50)
            self.assertIn("explicit_required_content_missing", row["major_issues"])
            self.assertTrue((root / "QUALITY_CONTRACT_MIGRATION.json").exists())

    def test_completed_phase2c_summary_is_reopened_for_contract_migration(self) -> None:
        data_path = Path("data/phase_2c_targeted_recovery/dev.jsonl")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "evaluation_manifest.json"
            summary = root / "summary.json"
            manifest.write_text(
                json.dumps({
                    "benchmark_sha256": "8ab54fd2e4cf25509a95fe7d6bc8a84fa5a230f0d57fdae5a50eb0e559063095"
                }),
                encoding="utf-8",
            )
            summary.write_text(json.dumps({"case_count": 1}), encoding="utf-8")
            targeted_recovery.prepare_contract_migration_resume(
                data_path=data_path,
                manifest_path=manifest,
                summary_path=summary,
                hash_field="benchmark_sha256",
            )
            self.assertFalse(summary.exists())
            self.assertTrue(list(root.glob("summary.json.invalid-*")))


if __name__ == "__main__":
    unittest.main()
