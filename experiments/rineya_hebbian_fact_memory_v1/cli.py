"""Run isolated RHF-001 studies from the repository root."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import torch
from safetensors.torch import load_file
from .artifacts import json_write, new_directory, sha256, save_memory, load_memory
from .bench import study, ssfr
from .memory import BuildConfig, build, swap_readout


ROOT = Path(__file__).resolve().parent


def provenance() -> dict:
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(),
            "source_sha256": {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))},
            "dependencies": {n: importlib.metadata.version(n) for n in ("numpy", "safetensors")}}


def read_json(path: str, max_bytes: int = 4 * 1024**2):
    p = Path(path)
    if p.stat().st_size > max_bytes:
        raise ValueError("JSON input exceeds size budget")
    return json.loads(p.read_text(encoding="utf-8"), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def tensor_data(path: str) -> dict:
    if Path(path).suffix != ".safetensors" or Path(path).stat().st_size > 256 * 1024**2:
        raise ValueError("a bounded safetensors input is required")
    data = load_file(path, device="cpu")
    if set(data) != {"keys", "targets"}:
        raise ValueError("data must contain exactly keys and targets")
    return data


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=1)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("smoke", "ssfr", "sweep", "fit", "edit", "native-screen"):
        sub = commands.add_parser(name)
        sub.add_argument("--out", required=True, help="new directory, never overwritten")
        if name in {"smoke", "ssfr"}:
            sub.add_argument("--seed", type=int, default=17)
        if name == "ssfr":
            sub.add_argument("--steps", type=int, default=400)
        if name in {"sweep", "fit", "edit", "native-screen"}:
            sub.add_argument("--config", required=True)
        if name in {"fit", "edit"}:
            sub.add_argument("--data", required=True)
        if name == "edit":
            sub.add_argument("--memory", required=True)
        if name == "native-screen":
            sub.add_argument("--execute-native", action="store_true")
            sub.add_argument("--model-dir", required=True)
            sub.add_argument("--manifest", required=True)
            sub.add_argument("--rows", required=True)
            sub.add_argument("--module", required=True)
            sub.add_argument("--alpha", required=True, type=float)
            sub.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
            sub.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return p


def execute(args) -> dict:
    if args.command == "native-screen" and not args.execute_native:
        raise ValueError("native loading requires --execute-native; no model was loaded")
    if args.command in {"fit", "edit"}:
        data = tensor_data(args.data)
        config = BuildConfig(**read_json(args.config))
        if args.command == "fit":
            memory, result = build(data["keys"], data["targets"], config)
        else:
            old, manifest = load_memory(args.memory)
            memory, result = swap_readout(old, data["keys"], data["targets"], config)
            result["source_memory_sha256"] = manifest["weights_sha256"]
            result["full_retained_bank_rebuild"] = True
        result.update({"provenance": provenance(), "data_sha256": sha256(args.data),
                       "config_sha256": sha256(args.config), "native_qualified": False})
        save_memory(args.out, memory, result)
        return {"status": "CONSTRUCTED_NOT_NATIVE_QUALIFIED", "out": args.out}
    output = new_directory(args.out)
    json_write(output/"provenance.json", provenance())
    if args.command == "smoke":
        result = study(n=16, d=12, width=32, seed=args.seed, gd_steps=100)
    elif args.command == "ssfr":
        if not 1 <= args.steps <= 5000:
            raise ValueError("SSFR steps must be in [1,5000]")
        result = ssfr(seed=args.seed, steps=args.steps)
    elif args.command == "sweep":
        config = read_json(args.config)
        if set(config) != {"phase", "runs"} or config["phase"] not in {"development", "confirmation"}:
            raise ValueError("sweep requires explicit development/confirmation phase")
        if not isinstance(config["runs"], list) or not 1 <= len(config["runs"]) <= 16:
            raise ValueError("sweep requires 1..16 runs")
        signatures, reports = set(), []
        for index, run in enumerate(config["runs"]):
            if set(run) != {"n", "d", "width", "seed", "anisotropy", "gd_steps"}:
                raise ValueError("unexpected sweep parameters")
            for name, low, high in (("n", 4, 256), ("d", 4, 128), ("width", 4, 512), ("gd_steps", 0, 5000)):
                if type(run[name]) is not int or not low <= run[name] <= high:
                    raise ValueError(f"invalid bounded sweep field: {name}")
            signature = json.dumps(run, sort_keys=True)
            if signature in signatures:
                raise ValueError("duplicate sweep arm")
            signatures.add(signature)
            report = study(**run)
            name = f"run_{index:03d}.json"
            json_write(output/name, report)
            reports.append({"file": name, "sha256": sha256(output/name), "settings": run,
                            "strict_clean_recall": {v: a["noise"]["0.0"]["strict_accuracy"]
                                                     if a["status"] == "MEASURED" else None
                                                     for v, a in report["arms"].items()}})
        result = {"kind": "SYNTHETIC_GRID_NOT_ASYMPTOTIC_CAPACITY_PROOF", "native_qualified": False,
                  "phase": config["phase"], "config_sha256": sha256(args.config), "runs": reports,
                  "selection_note": "No monotonicity assumption, winner selection, or pooled seed inference"}
    else:
        from .native import load_qwen_local, screen
        config = BuildConfig(**read_json(args.config))
        manifest, rows = read_json(args.manifest), read_json(args.rows)
        model = load_qwen_local(Path(args.model_dir), manifest, args.device, args.dtype)
        memory, result = screen(model, rows, args.module, config, args.alpha,
                                cache_atol=.03 if args.dtype == "bfloat16" else 1e-4,
                                cache_rtol=.01 if args.dtype == "bfloat16" else 1e-4)
        result["runtime"] = {"transformers": importlib.metadata.version("transformers"),
                             "device": args.device, "dtype": args.dtype,
                             "checkpoint_manifest_sha256": sha256(args.manifest),
                             "rows_sha256": sha256(args.rows), "config_sha256": sha256(args.config)}
        save_memory(output/"constructed_memory", memory, result["construction"])
    json_write(output/"report.json", result)
    status = ("SCREEN_PASS" if result["screen_pass"] else "SCREEN_FAIL") if args.command == "native-screen" else "MEASURED"
    return {"status": status, "kind": result["kind"], "out": str(output),
            "native_qualified": False}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if not 1 <= args.threads <= 8:
            raise ValueError("threads must be 1..8")
        torch.set_num_threads(args.threads)
        result = execute(args)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 3 if result["status"] == "SCREEN_FAIL" else 0
    except (ValueError, RuntimeError, MemoryError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
