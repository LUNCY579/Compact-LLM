from __future__ import annotations

"""Create a measured Phase 11 dense or quantized deployment artifact."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import environment, sha256, sha256_tree, write_json
from scripts.phase_9_student_distillation.phase9_common import tree_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize or stage a Phase 11 deployment student.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("dense_copy", "bitsandbytes_nf4", "bitsandbytes_int8", "llamacpp_gguf"),
        default="dense_copy",
    )
    parser.add_argument("--compute-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--llama-cpp-convert-script", type=Path, default=None)
    parser.add_argument("--llama-cpp-quantize-exe", type=Path, default=None)
    parser.add_argument("--gguf-type", default="Q4_K_M")
    parser.add_argument("--smoke-text", default="Hello!")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; use --overwrite intentionally: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def transformers_quantize(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    args.output.mkdir(parents=True, exist_ok=True)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.compute_dtype]
    if args.backend == "bitsandbytes_nf4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    encoded = tokenizer(args.smoke_text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        generated = model.generate(**encoded, do_sample=False, max_new_tokens=4)
    if generated.shape[-1] <= encoded["input_ids"].shape[-1]:
        raise RuntimeError("Quantized smoke generation produced no continuation")
    try:
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
    except Exception as exc:
        raise RuntimeError(
            "The installed Transformers/bitsandbytes combination could load but not serialize this quantized model. "
            "Use a supported version pair or the llama.cpp GGUF backend; the dense source remains unchanged."
        ) from exc
    if not (args.output / "config.json").exists():
        raise RuntimeError("Quantized save is missing config.json")
    if not any(args.output.glob("*.safetensors")) and not any(args.output.glob("*.bin")):
        raise RuntimeError("Quantized save contains no model weight files")
    # Validate fresh loading with the serialized quantization config.
    reloaded = AutoModelForCausalLM.from_pretrained(
        args.output,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    reload_device = next(reloaded.parameters()).device
    reload_encoded = {key: value.to(reload_device) for key, value in encoded.items()}
    with torch.no_grad():
        reloaded.generate(**reload_encoded, do_sample=False, max_new_tokens=2)
    return {
        "artifact_type": "transformers_directory",
        "model_path": str(args.output),
        "quantization_backend": args.backend,
        "compute_dtype": args.compute_dtype,
        "artifact_bytes": tree_bytes(args.output),
        "reload_check": True,
    }


def llama_cpp_quantize(args: argparse.Namespace) -> dict[str, Any]:
    if args.llama_cpp_convert_script is None or args.llama_cpp_quantize_exe is None:
        raise ValueError("llamacpp_gguf requires --llama-cpp-convert-script and --llama-cpp-quantize-exe")
    if not args.llama_cpp_convert_script.exists():
        raise FileNotFoundError(args.llama_cpp_convert_script)
    if not args.llama_cpp_quantize_exe.exists():
        raise FileNotFoundError(args.llama_cpp_quantize_exe)
    args.output.mkdir(parents=True, exist_ok=True)
    f16_path = args.output / "model-f16.gguf"
    quantized_path = args.output / f"model-{args.gguf_type}.gguf"
    run(
        [
            sys.executable,
            str(args.llama_cpp_convert_script),
            str(args.model),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
    )
    if not f16_path.exists() or f16_path.stat().st_size == 0:
        raise RuntimeError("llama.cpp conversion did not create a valid F16 GGUF")
    run([str(args.llama_cpp_quantize_exe), str(f16_path), str(quantized_path), args.gguf_type])
    if not quantized_path.exists() or quantized_path.stat().st_size == 0:
        raise RuntimeError("llama.cpp quantization did not create a valid GGUF")
    f16_path.unlink(missing_ok=True)
    return {
        "artifact_type": "gguf_file",
        "model_path": str(quantized_path),
        "quantization_backend": args.backend,
        "gguf_type": args.gguf_type,
        "artifact_bytes": quantized_path.stat().st_size,
        "reload_check": False,
        "runtime_requirement": "Benchmark and deploy with a compatible llama.cpp executable; Transformers cannot load this GGUF directly.",
    }


def main() -> None:
    args = parse_args()
    prepare_output(args.output, args.overwrite)
    source = Path(args.model)
    if args.backend == "dense_copy":
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError("dense_copy requires a local checkpoint directory")
        shutil.copytree(source, args.output)
        result = {
            "artifact_type": "transformers_directory",
            "model_path": str(args.output),
            "quantization_backend": "dense_copy",
            "artifact_bytes": tree_bytes(args.output),
            "reload_check": False,
        }
    elif args.backend in {"bitsandbytes_nf4", "bitsandbytes_int8"}:
        result = transformers_quantize(args)
    else:
        result = llama_cpp_quantize(args)

    manifest = {
        "schema_version": "1.0",
        "phase": "11_production_rag",
        "source_model": args.model,
        "source_tree_sha256": sha256_tree(source) if source.exists() and source.is_dir() else None,
        **result,
        "environment": environment(),
        "measurement_boundary": (
            "artifact_bytes is serialized model storage. Runtime RSS/VRAM, KV cache, tokenizer, index, and application overhead "
            "are measured separately by benchmark_production_runtime.py."
        ),
    }
    manifest_path = args.output / "phase_11_quantization_manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
