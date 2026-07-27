"""Smoke test for the broad short-response-model environment."""

from __future__ import annotations

import importlib
import sys

REQUIRED_IMPORTS = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "trl",
    "bitsandbytes",
    "safetensors",
    "sentence_transformers",
    "numpy",
    "pandas",
    "sklearn",
    "zvec",
    "fastapi",
    "pytest",
]

failures: list[tuple[str, str]] = []

print(f"Python: {sys.version.split()[0]}")

for module_name in REQUIRED_IMPORTS:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version not exposed")
        print(f"[OK] {module_name}: {version}")
    except Exception as exc:  # Import failures should be reported together.
        failures.append((module_name, repr(exc)))
        print(f"[FAIL] {module_name}: {exc}")

try:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"GPU memory: {free_bytes / 2**30:.2f} GiB free / {total_bytes / 2**30:.2f} GiB total")
except Exception as exc:
    failures.append(("torch CUDA verification", repr(exc)))

if failures:
    print("\nEnvironment verification failed:")
    for name, error in failures:
        print(f"  - {name}: {error}")
    raise SystemExit(1)

print("\nEnvironment verification passed.")
