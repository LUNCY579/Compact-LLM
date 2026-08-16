from __future__ import annotations

import importlib
import json
import platform
import sys


def main() -> int:
    required = ["torch", "safetensors", "numpy", "yaml", "rapidfuzz", "matplotlib"]
    optional = ["transformers", "accelerate", "peft"]
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "required": {},
        "optional": {},
    }
    failed = False
    for name in required:
        try:
            module = importlib.import_module(name)
            report["required"][name] = getattr(module, "__version__", "available")
        except Exception as exc:
            report["required"][name] = f"MISSING: {type(exc).__name__}: {exc}"
            failed = True
    for name in optional:
        try:
            module = importlib.import_module(name)
            report["optional"][name] = getattr(module, "__version__", "available")
        except Exception as exc:
            report["optional"][name] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        report["gpus"] = (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
    except Exception:
        pass
    try:
        import rineya_soup_lab

        report["rineya_soup_lab"] = rineya_soup_lab.__version__
    except Exception as exc:
        report["rineya_soup_lab"] = f"MISSING: {type(exc).__name__}: {exc}"
        failed = True
    print(json.dumps(report, indent=2))
    if report["optional"].get("transformers", "").startswith("unavailable"):
        print("\nNOTE: transformers is required for real model evaluation.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
