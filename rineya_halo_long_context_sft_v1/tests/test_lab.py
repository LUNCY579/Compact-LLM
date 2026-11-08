from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_stage_mix  # noqa: E402
import common  # noqa: E402
import score_main  # noqa: E402
import train_halo_qlora  # noqa: E402


class YamlAndStratumTests(unittest.TestCase):
    def test_simple_yaml_and_config(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "stage_a_qlora.yaml"
        config = common.load_config(path)
        self.assertEqual(config["stage"], "a")
        self.assertEqual(config["seed"], 11)
        self.assertEqual(config["training"]["lora_rank"], 16)
        self.assertEqual(config["training"]["max_sequence_length"], 4096)
        self.assertEqual(config["strata"], ["2k", "4k"])
        self.assertTrue(config["training"]["qlora"])

    def test_infer_stratum_from_metadata_and_proxy(self) -> None:
        row = {
            "id": "x",
            "messages": [{"role": "user", "content": "short"}, {"role": "assistant", "content": "ok"}],
            "metadata": {"context_stratum": "4K"},
        }
        self.assertEqual(common.infer_stratum(row), "4k")
        lineage = {"y": {"stratum": "2k"}}
        lined = {
            "id": "y",
            "messages": [{"role": "user", "content": "short"}, {"role": "assistant", "content": "ok"}],
            "metadata": {},
        }
        self.assertEqual(common.infer_stratum(lined, lineage), "2k")
        filler = " token" * 2000
        proxy_row = {
            "id": "z",
            "messages": [{"role": "user", "content": filler}, {"role": "assistant", "content": "ok"}],
            "metadata": {},
        }
        self.assertEqual(common.infer_stratum(proxy_row), "2k")


class MixAndScoreTests(unittest.TestCase):
    def test_replay_fraction_and_filter(self) -> None:
        long_rows = [{"id": f"l{i}"} for i in range(8)]
        short_rows = [{"id": f"s{i}"} for i in range(20)]
        import random

        replay = build_stage_mix.sample_replay(short_rows, len(long_rows), 0.20, random.Random(11))
        self.assertEqual(len(replay), 2)
        capped = build_stage_mix.sample_replay(short_rows[:1], 8, 0.20, random.Random(11))
        self.assertEqual(len(capped), 1)

    def test_score_exact_and_trim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            preds = root / "preds.jsonl"
            records.write_text(
                json.dumps({
                    "id": "a",
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "UNKNOWN"},
                    ],
                    "metadata": {"context_stratum": "2k"},
                })
                + "\n"
                + json.dumps({
                    "id": "b",
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "file"},
                    ],
                    "metadata": {"context_stratum": "4k"},
                })
                + "\n",
                encoding="utf-8",
            )
            preds.write_text(
                json.dumps({"id": "a", "response": "UNKNOWN"})
                + "\n"
                + json.dumps({"id": "b", "response": " file "})
                + "\n",
                encoding="utf-8",
            )
            report = score_main.score(records, preds, "adapter")
            self.assertEqual(report["exact"]["correct"], 1)
            self.assertEqual(report["trim"]["correct"], 2)
            self.assertEqual(report["trim"]["by_stratum"]["4k"]["correct"], 1)


class ResumeAndAdapterTests(unittest.TestCase):
    def test_latest_checkpoint_and_adapter_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint-50").mkdir()
            (root / "checkpoint-100").mkdir()
            (root / "checkpoint-not-a-step").mkdir()
            latest = common.latest_checkpoint(root)
            self.assertIsNotNone(latest)
            self.assertEqual(latest.name, "checkpoint-100")
            adapter = root / "adapter"
            adapter.mkdir()
            with self.assertRaises(RuntimeError):
                common.validate_adapter_directory(adapter)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.bin").write_bytes(b"weights")
            common.validate_adapter_directory(adapter)

    def test_resume_latest_requires_checkpoint(self) -> None:
        config = common.load_config(Path(__file__).resolve().parents[1] / "configs" / "stage_a_qlora.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mix = root / "mix"
            mix.mkdir()
            (mix / "train_stage_a.jsonl").write_text(
                json.dumps({
                    "id": "t1",
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ],
                })
                + "\n",
                encoding="utf-8",
            )
            (mix / "val_stage_a_trainer.jsonl").write_text(
                (mix / "train_stage_a.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            config = dict(config)
            config["paths"] = dict(config["paths"])
            config["paths"]["output_dir"] = str(root)
            ns = type("NS", (), {
                "train_file": mix / "train_stage_a.jsonl",
                "dev_file": mix / "val_stage_a_trainer.jsonl",
                "output_dir": root,
                "max_steps": 2,
                "save_steps": 1,
                "eval_steps": 2,
                "max_sequence_length": 256,
                "resume_from_checkpoint": "latest",
                "allow_cpu": False,
            })()
            with self.assertRaises(FileNotFoundError):
                train_halo_qlora.build_command(config, ns)
            (root / "checkpoint-1").mkdir()
            command = train_halo_qlora.build_command(config, ns)
            self.assertIn("--resume-from-checkpoint", command)
            self.assertIn("latest", command)
            self.assertIn("--qlora", command)


if __name__ == "__main__":
    unittest.main()
