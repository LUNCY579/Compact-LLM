from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import accelerate
import numpy
import peft
import psutil
import safetensors
import scipy
import torch
import transformers


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "phase2_mechanism" / "environment.json"
REPORT = ROOT / "reports" / "PHASE2_ENVIRONMENT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        nvidia = command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        driver = None
        reported_memory_mib = None
        if nvidia:
            parts = [part.strip() for part in nvidia.splitlines()[0].split(",")]
            if len(parts) == 3:
                reported_memory_mib = int(parts[1])
                driver = parts[2]
        gpu = {
            "name": properties.name,
            "total_memory_bytes_torch": properties.total_memory,
            "total_memory_mib_nvidia_smi": reported_memory_mib,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "driver_version": driver,
        }
    cpu_name = command_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()",
        ]
    )
    packet = {
        "schema_version": "1.0",
        "status": "ENVIRONMENT_CAPTURE_COMPLETE",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "cpu": cpu_name or platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_bytes": psutil.virtual_memory().total,
        },
        "gpu": gpu,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "safetensors": safetensors.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "psutil": psutil.__version__,
        },
        "precision": {
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "paper_inference_resolved_dtype": "torch.bfloat16",
        },
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
    }
    gpu_row = packet["gpu"] or {}
    report = "\n".join(
        [
            "# Phase 2 Execution Environment",
            "",
            f'Captured UTC: `{packet["captured_at_utc"]}` by `experiments/phase2_mechanism/capture_environment.py`.',
            "",
            "| Component | Detected value |",
            "|---|---|",
            f'| OS | {packet["host"]["platform"]} |',
            f'| CPU | {packet["host"]["cpu"]}; {packet["host"]["physical_cores"]} physical / {packet["host"]["logical_cores"]} logical cores |',
            f'| RAM | {packet["host"]["ram_bytes"]:,} bytes |',
            f'| GPU | {gpu_row.get("name")}; {gpu_row.get("total_memory_mib_nvidia_smi"):,} MiB; compute capability {gpu_row.get("compute_capability")} |',
            f'| NVIDIA driver | {gpu_row.get("driver_version")} |',
            f'| Python | {packet["software"]["python"]} |',
            f'| PyTorch / CUDA / cuDNN | {packet["software"]["torch"]} / {packet["software"]["cuda_runtime"]} / {packet["software"]["cudnn"]} |',
            f'| Transformers / PEFT / Accelerate | {packet["software"]["transformers"]} / {packet["software"]["peft"]} / {packet["software"]["accelerate"]} |',
            f'| Safetensors / NumPy / SciPy | {packet["software"]["safetensors"]} / {packet["software"]["numpy"]} / {packet["software"]["scipy"]} |',
            f'| Paper inference dtype | {packet["precision"]["paper_inference_resolved_dtype"]}; BF16 supported={packet["precision"]["bf16_supported"]} |',
            "",
            "Latency, TTFT, VRAM, and RSS are reported per condition because they are decoder- and trajectory-dependent. This environment record does not make latency portable to other hardware.",
            "",
        ]
    )
    REPORT.write_text(report, encoding="utf-8")
    packet["report"] = str(REPORT.relative_to(ROOT)).replace("\\", "/")
    packet["report_sha256"] = sha256(REPORT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": packet["status"], "artifact": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
