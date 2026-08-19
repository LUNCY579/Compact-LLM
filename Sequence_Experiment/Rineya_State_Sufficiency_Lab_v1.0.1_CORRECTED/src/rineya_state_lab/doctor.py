from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .utils import atomic_write_json, environment_manifest, package_version, utc_now_iso


def run_doctor(
    *,
    model_path: str | None = None,
    data_root: str | None = None,
    output: str | Path | None = None,
    require_cuda: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any, required: bool = True) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail, "required": required})

    check("python_version", sys.version_info >= (3, 10), sys.version)
    for package, required in [
        ("PyYAML", True),
        ("numpy", True),
        ("torch", False),
        ("transformers", False),
        ("peft", False),
        ("accelerate", False),
        ("bitsandbytes", False),
    ]:
        version = package_version(package)
        check(f"package:{package}", version is not None, version or "not installed", required=required)
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        check("cuda_available", cuda or not require_cuda, {"available": cuda, "version": torch.version.cuda}, required=require_cuda)
        if cuda:
            check(
                "cuda_devices",
                torch.cuda.device_count() > 0,
                [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                required=require_cuda,
            )
            check("bf16_support", True, bool(torch.cuda.is_bf16_supported()), required=False)
    except Exception as exc:
        check("torch_runtime", not require_cuda, f"{type(exc).__name__}: {exc}", required=require_cuda)

    selected_model = model_path or os.environ.get("RINEYA_BASE_MODEL")
    if selected_model:
        path = Path(selected_model).expanduser()
        looks_remote = "/" in selected_model and not path.exists() and not Path(selected_model).is_absolute()
        check("base_model", path.exists() or looks_remote, selected_model, required=False)
    else:
        check("base_model", False, "RINEYA_BASE_MODEL is unset", required=False)

    if data_root:
        try:
            from .data.layout import discover_dataset_root

            root = discover_dataset_root(data_root)
            check("dataset_root", True, str(root))
        except Exception as exc:
            check("dataset_root", False, f"{type(exc).__name__}: {exc}")

    required_failures = [item for item in checks if item["required"] and not item["passed"]]
    report = {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "passed": not required_failures,
        "required_failures": len(required_failures),
        "checks": checks,
        "environment": environment_manifest(),
    }
    if output is not None:
        atomic_write_json(output, report)
    return report
