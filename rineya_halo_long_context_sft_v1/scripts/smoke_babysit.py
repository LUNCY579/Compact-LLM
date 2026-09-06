#!/usr/bin/env python3
"""Smoke the Stage A mix, resume wiring, adapter publish check, and optional 2-step GPU train."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (  # noqa: E402
    REPO_ROOT,
    latest_checkpoint,
    list_checkpoints,
    python_executable,
    validate_adapter_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Babysit smoke for Halo long-context QLoRA lab.")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--max-long-rows", type=int, default=4)
    return parser.parse_args()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print(" ".join(command), flush=True)
    return subprocess.run(command, cwd=str(REPO_ROOT), text=True)


def run_unit_tests() -> None:
    tests = LAB_ROOT / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def write_tiny_chat_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n):
        rows.append({
            "id": f"smoke_{index:03d}",
            "messages": [
                {"role": "system", "content": "Reply with the shortest exact answer."},
                {"role": "user", "content": f"What is {index}+1?"},
                {"role": "assistant", "content": str(index + 1)},
            ],
            "metadata": {"context_stratum": "short", "mix_source": "smoke"},
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def maybe_gpu_resume_smoke(output_dir: Path, skip_gpu: bool) -> dict:
    report = {"gpu": False, "resumed": False, "adapter_valid": False, "skipped": None}
    if skip_gpu:
        report["skipped"] = "user requested --skip-gpu"
        return report
    try:
        import torch
    except ImportError:
        report["skipped"] = "torch not importable"
        return report
    if not torch.cuda.is_available():
        report["skipped"] = "CUDA unavailable"
        return report
    halo = REPO_ROOT / "Rineya_Published_Cores" / "Rineya-Halo-1.2.0-devfreeze"
    if not halo.exists():
        report["skipped"] = f"Halo missing at {halo}"
        return report
    if not (halo / "model.safetensors").exists() and not (halo / "pytorch_model.bin").exists():
        report["skipped"] = "Halo weight file model.safetensors is not present in the vault directory"
        return report
    train_file = output_dir / "mix" / "smoke_train_short.jsonl"
    dev_file = output_dir / "mix" / "smoke_dev_short.jsonl"
    write_tiny_chat_jsonl(train_file, 8)
    write_tiny_chat_jsonl(dev_file, 4)
    config = LAB_ROOT / "configs" / "stage_a_qlora.yaml"
    first = run([
        python_executable(),
        str(SCRIPTS / "train_halo_qlora.py"),
        "--config",
        str(config),
        "--train-file",
        str(train_file),
        "--dev-file",
        str(dev_file),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        "2",
        "--save-steps",
        "1",
        "--eval-steps",
        "2",
        "--max-sequence-length",
        "256",
    ])
    if first.returncode != 0:
        raise RuntimeError(f"first smoke train exited {first.returncode}")
    report["gpu"] = True
    if latest_checkpoint(output_dir) is None:
        raise RuntimeError("expected checkpoint-* after 2-step smoke train")
    second = run([
        python_executable(),
        str(SCRIPTS / "train_halo_qlora.py"),
        "--config",
        str(config),
        "--train-file",
        str(train_file),
        "--dev-file",
        str(dev_file),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        "3",
        "--save-steps",
        "1",
        "--eval-steps",
        "3",
        "--max-sequence-length",
        "256",
        "--resume-from-checkpoint",
        "latest",
    ])
    if second.returncode != 0:
        raise RuntimeError(f"resume train exited {second.returncode}")
    report["resumed"] = True
    adapter = output_dir / "adapter"
    validate_adapter_directory(adapter)
    report["adapter_valid"] = True
    report["latest_checkpoint"] = latest_checkpoint(output_dir).name if latest_checkpoint(output_dir) else None
    report["checkpoint_count"] = len(list_checkpoints(output_dir))
    return report


def main() -> None:
    args = parse_args()
    run_unit_tests()
    smoke_dir = REPO_ROOT / "artifacts" / "halo_long_context_sft_v1" / "smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir, ignore_errors=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    mix = run([
        python_executable(),
        str(SCRIPTS / "build_stage_mix.py"),
        "--max-long-rows",
        str(args.max_long_rows),
        "--output-dir",
        str(smoke_dir),
        "--force",
    ])
    if mix.returncode != 0:
        raise SystemExit(mix.returncode)
    dry = run([
        python_executable(),
        str(SCRIPTS / "train_halo_qlora.py"),
        "--output-dir",
        str(smoke_dir),
        "--train-file",
        str(smoke_dir / "mix" / "train_stage_a.jsonl"),
        "--dev-file",
        str(smoke_dir / "mix" / "val_stage_a_trainer.jsonl"),
        "--dry-run",
        "--max-steps",
        "2",
    ])
    if dry.returncode != 0:
        raise SystemExit(dry.returncode)

    with tempfile.TemporaryDirectory() as tmp:
        adapter = Path(tmp) / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
        (adapter / "adapter_model.bin").write_bytes(b"ok")
        validate_adapter_directory(adapter)

    gpu_report = maybe_gpu_resume_smoke(smoke_dir, args.skip_gpu)
    summary = {
        "unit_tests": "PASS",
        "mix_dir": str(smoke_dir / "mix"),
        "gpu_resume": gpu_report,
    }
    out = smoke_dir / "smoke_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Smoke complete.")


if __name__ == "__main__":
    main()
