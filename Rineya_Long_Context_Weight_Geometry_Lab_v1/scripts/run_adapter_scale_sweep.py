#!/usr/bin/env python3
"""Evaluate several LoRA scales while loading the base model and adapter once.

The generation contract is intentionally fixed to the public Rineya Long-Context
Demo protocol: BF16 weights, greedy decoding, Qwen thinking disabled, seed 17,
24 generated tokens, a 4096-token context ceiling, and no truncation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
CANONICAL_RUNNER_PATH = (
    REPOSITORY_ROOT / "Rineya_Long_Context_Demo_v1" / "scripts" / "run_local_hf.py"
)

SEED = 17
MAX_NEW_TOKENS = 24
MAX_CONTEXT = 4096
PROTOCOL_ID = "rineya_long_context_demo_v1_bf16_greedy_no_think_24_4096"
PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _load_canonical_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "rineya_long_context_demo_run_local_hf", CANONICAL_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import canonical runner: {CANONICAL_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANONICAL = _load_canonical_runner()


@dataclass(frozen=True)
class CandidatePlan:
    scale: float
    candidate_id: str
    output_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class LoraBinding:
    module_name: str
    module: Any
    active_adapters: tuple[str, ...]


@dataclass(frozen=True)
class PreparedRow:
    row_id: str
    input_tokens: int
    encoded: dict[str, Any]


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(row_ids: Sequence[str]) -> str:
    payload = "".join(
        json.dumps(row_id, ensure_ascii=False) + "\n" for row_id in row_ids
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_scales(values: Sequence[str | float]) -> tuple[float, ...]:
    """Return finite, numerically unique scales in the requested order."""

    if not values:
        raise ValueError("At least one adapter scale is required")
    scales: list[float] = []
    first_position: dict[float, int] = {}
    for position, raw in enumerate(values, 1):
        if isinstance(raw, bool):
            raise TypeError(f"Scale {position} must be a finite number")
        try:
            scale = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Scale {position} must be a finite number: {raw!r}"
            ) from exc
        if not math.isfinite(scale):
            raise ValueError(f"Scale {position} must be finite: {raw!r}")
        if scale == 0.0:
            scale = 0.0
        if scale in first_position:
            raise ValueError(
                f"Duplicate adapter scale {scale!r} at positions "
                f"{first_position[scale]} and {position}"
            )
        first_position[scale] = position
        scales.append(scale)
    return tuple(scales)


def scale_slug(scale: float) -> str:
    if not math.isfinite(scale):
        raise ValueError("Scale slug requires a finite number")
    if scale == 0.0:
        return "0"
    text = format(scale, ".15g").lower()
    return text.replace("-", "neg").replace("+", "plus").replace(".", "p")


def validate_candidate_prefix(value: str) -> str:
    if not PREFIX_RE.fullmatch(value):
        raise ValueError(
            "--candidate-prefix must start with an alphanumeric character and "
            "contain only letters, digits, '.', '_', or '-'"
        )
    return value


def plan_candidates(
    output_dir: Path, candidate_prefix: str, scales: Sequence[float]
) -> tuple[CandidatePlan, ...]:
    prefix = validate_candidate_prefix(candidate_prefix)
    root = output_dir.expanduser().resolve()
    plans: list[CandidatePlan] = []
    seen_paths: set[Path] = set()
    for scale in scales:
        candidate_id = f"{prefix}_scale{scale_slug(scale)}"
        output_path = root / f"{candidate_id}.jsonl"
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        for path in (output_path, manifest_path):
            if path in seen_paths:
                raise ValueError(f"Candidate output collision: {path}")
            seen_paths.add(path)
        plans.append(
            CandidatePlan(
                scale=scale,
                candidate_id=candidate_id,
                output_path=output_path,
                manifest_path=manifest_path,
            )
        )
    return tuple(plans)


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def refuse_existing_outputs(plans: Sequence[CandidatePlan]) -> None:
    collisions: list[Path] = []
    for plan in plans:
        for path in (
            plan.output_path,
            plan.manifest_path,
            temporary_path(plan.output_path),
            temporary_path(plan.manifest_path),
        ):
            if path.exists():
                collisions.append(path)
    if collisions:
        formatted = "\n".join(f"- {path}" for path in collisions)
        raise FileExistsError(
            f"Refusing to overwrite existing sweep outputs:\n{formatted}"
        )


def _publish_without_overwrite(temporary: Path, destination: Path) -> None:
    """Publish a same-directory temporary file without replacing a destination."""

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"Refusing to overwrite {destination}") from None
    temporary.unlink()


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows in iteration order and atomically publish without overwriting."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary output {temporary}")
    count = 0
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        _publish_without_overwrite(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return count


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    write_jsonl_exclusive(path, [value])


def discover_lora_layers(model: Any, lora_layer_type: type) -> tuple[LoraBinding, ...]:
    bindings: list[LoraBinding] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, lora_layer_type):
            continue
        raw_active = module.active_adapters
        active = (raw_active,) if isinstance(raw_active, str) else tuple(raw_active)
        if not active:
            raise RuntimeError(f"LoRA layer {module_name} has no active adapter")
        bindings.append(
            LoraBinding(
                module_name=module_name,
                module=module,
                active_adapters=tuple(str(value) for value in active),
            )
        )
    if not bindings:
        raise RuntimeError("The loaded adapter contains no LoRA layers to scale")
    return tuple(bindings)


def set_all_lora_scales(bindings: Sequence[LoraBinding], scale: float) -> int:
    if not math.isfinite(scale):
        raise ValueError("LoRA scale must be finite")
    calls = 0
    for binding in bindings:
        for adapter_name in binding.active_adapters:
            binding.module.set_scale(adapter_name, scale)
            calls += 1
    if calls == 0:
        raise RuntimeError("No active LoRA adapters were scaled")
    return calls


def prepare_rows(
    rows: Sequence[dict[str, Any]], tokenizer: Any
) -> tuple[PreparedRow, ...]:
    prepared: list[PreparedRow] = []
    for row in rows:
        row_id = str(row["id"])
        rendered = CANONICAL.render_chat(tokenizer, row, enable_thinking=False)
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens + MAX_NEW_TOKENS > MAX_CONTEXT:
            raise RuntimeError(
                f"{row_id}: {input_tokens} input tokens + {MAX_NEW_TOKENS} "
                f"generated tokens exceed context limit {MAX_CONTEXT}; truncation "
                "is forbidden"
            )
        prepared.append(
            PreparedRow(
                row_id=row_id,
                input_tokens=input_tokens,
                encoded=dict(encoded),
            )
        )
    return tuple(prepared)


def generate_candidate_rows(
    *,
    prepared_rows: Sequence[PreparedRow],
    model: Any,
    tokenizer: Any,
    torch: Any,
    plan: CandidatePlan,
    lora_layer_count: int,
) -> Iterable[dict[str, Any]]:
    device = CANONICAL.input_device(model)
    for index, prepared in enumerate(prepared_rows, 1):
        encoded = {name: tensor.to(device) for name, tensor in prepared.encoded.items()}
        CANONICAL.synchronize_if_needed(torch)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        CANONICAL.synchronize_if_needed(torch)
        latency = time.perf_counter() - started

        prompt_length = int(encoded["input_ids"].shape[-1])
        continuation = generated[0, prompt_length:]
        prediction = tokenizer.decode(
            continuation,
            skip_special_tokens=True,
        ).strip()
        result = {
            "id": prepared.row_id,
            "status": "ok",
            "prediction": prediction,
            "input_tokens": prepared.input_tokens,
            "generated_tokens": int(continuation.shape[-1]),
            "latency_seconds": round(latency, 6),
            "seed": SEED,
            "decoding": "greedy",
            "candidate_id": plan.candidate_id,
            "protocol_id": PROTOCOL_ID,
            "adapter_scale": plan.scale,
            "adapter_effective_scale_min": plan.scale,
            "adapter_effective_scale_max": plan.scale,
            "scaled_lora_layers": lora_layer_count,
            "provenance_manifest": str(plan.manifest_path),
        }
        print(
            f"[{plan.candidate_id}] [{index}/{len(prepared_rows)}] "
            f"{prepared.row_id}: ok"
        )
        yield result


def _json_sha256_if_present(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def model_provenance(model_reference: str) -> dict[str, Any]:
    candidate = Path(model_reference).expanduser()
    if not candidate.exists():
        return {"kind": "configured_identifier", "value": model_reference}
    resolved = candidate.resolve()
    result: dict[str, Any] = {
        "kind": "local_path",
        "path": str(resolved),
        "config_sha256": _json_sha256_if_present(resolved / "config.json"),
        "core_sha256": _json_sha256_if_present(resolved / "CORE.json"),
    }
    core_path = resolved / "CORE.json"
    if core_path.is_file():
        try:
            core = json.loads(core_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            core = None
        if isinstance(core, dict):
            checks = core.get("checks")
            if isinstance(checks, dict) and isinstance(
                checks.get("weight_sha256"), str
            ):
                result["declared_weight_sha256"] = checks["weight_sha256"]
    return result


def adapter_provenance(adapter: Path) -> dict[str, Any]:
    weight_candidates = sorted(adapter.glob("adapter_model.*"))
    if not weight_candidates:
        raise FileNotFoundError(f"Missing adapter_model weights in {adapter}")
    return {
        "path": str(adapter),
        "config_sha256": sha256_file(adapter / "adapter_config.json"),
        "weights": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weight_candidates
        ],
    }


def candidate_manifest(
    *,
    plan: CandidatePlan,
    all_scales: Sequence[float],
    candidate_index: int,
    model: dict[str, Any],
    adapter: dict[str, Any],
    input_path: Path,
    prepared_rows: Sequence[PreparedRow],
    output_sha256: str,
    lora_layer_count: int,
    scale_set_calls: int,
) -> dict[str, Any]:
    row_ids = [row.row_id for row in prepared_rows]
    return {
        "schema_version": "1.0",
        "kind": "adapter_scale_sweep_candidate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": plan.candidate_id,
        "candidate_index": candidate_index,
        "single_model_and_adapter_load": True,
        "model": model,
        "adapter": adapter,
        "adapter_scale": plan.scale,
        "requested_scales": list(all_scales),
        "lora_layer_count": lora_layer_count,
        "scale_set_calls": scale_set_calls,
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "rows": len(prepared_rows),
            "ordered_ids_sha256": ordered_ids_sha256(row_ids),
        },
        "protocol": {
            "id": PROTOCOL_ID,
            "torch_dtype": "bfloat16",
            "seed": SEED,
            "decoding": "greedy",
            "do_sample": False,
            "thinking": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_context": MAX_CONTEXT,
            "truncation": "forbidden",
        },
        "implementation": {
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "canonical_runner": str(CANONICAL_RUNNER_PATH),
            "canonical_runner_sha256": sha256_file(CANONICAL_RUNNER_PATH),
        },
        "output": {
            "path": str(plan.output_path),
            "sha256": output_sha256,
            "rows": len(prepared_rows),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Hugging Face base model")
    parser.add_argument("--adapter", type=Path, required=True, help="PEFT LoRA adapter")
    parser.add_argument("--input", type=Path, required=True, help="Prompt-only JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-prefix", required=True)
    parser.add_argument(
        "--scales",
        nargs="+",
        required=True,
        help="Finite, numerically unique LoRA multipliers in evaluation order",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Engineering/parity check only: evaluate the first N rows",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-gpu-memory", help='Example: "2GiB"')
    parser.add_argument("--max-cpu-memory", default="32GiB")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        scales = validate_scales(args.scales)
        plans = plan_candidates(args.output_dir, args.candidate_prefix, scales)
        refuse_existing_outputs(plans)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    input_path = args.input.expanduser().resolve(strict=True)
    adapter_path = args.adapter.expanduser().resolve(strict=True)
    if not (adapter_path / "adapter_config.json").is_file():
        parser.error(f"Missing adapter_config.json in {adapter_path}")
    if (
        plans[0].output_path.parent.exists()
        and not plans[0].output_path.parent.is_dir()
    ):
        parser.error(f"--output-dir is not a directory: {plans[0].output_path.parent}")
    plans[0].output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from peft import PeftModel
        from peft.tuners.lora.layer import LoraLayer
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Install the repository requirements, including transformers and peft"
        ) from exc

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch.bfloat16,
    }
    if args.max_gpu_memory:
        model_kwargs["max_memory"] = {
            0: args.max_gpu_memory,
            "cpu": args.max_cpu_memory,
        }
    try:
        base_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except TypeError:
        model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
        base_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    bindings = discover_lora_layers(model, LoraLayer)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows = CANONICAL.load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    prepared_rows = prepare_rows(rows, tokenizer)
    if not prepared_rows:
        raise SystemExit(f"No prompt rows selected from {input_path}")

    shared_model_provenance = model_provenance(args.model)
    shared_adapter_provenance = adapter_provenance(adapter_path)
    for candidate_index, plan in enumerate(plans, 1):
        scale_set_calls = set_all_lora_scales(bindings, plan.scale)
        set_seed(SEED)
        written = write_jsonl_exclusive(
            plan.output_path,
            generate_candidate_rows(
                prepared_rows=prepared_rows,
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                plan=plan,
                lora_layer_count=len(bindings),
            ),
        )
        if written != len(prepared_rows):
            raise RuntimeError(
                f"{plan.candidate_id}: wrote {written} rows, expected "
                f"{len(prepared_rows)}"
            )
        manifest = candidate_manifest(
            plan=plan,
            all_scales=scales,
            candidate_index=candidate_index,
            model=shared_model_provenance,
            adapter=shared_adapter_provenance,
            input_path=input_path,
            prepared_rows=prepared_rows,
            output_sha256=sha256_file(plan.output_path),
            lora_layer_count=len(bindings),
            scale_set_calls=scale_set_calls,
        )
        write_json_exclusive(plan.manifest_path, manifest)
        print(
            json.dumps(
                {
                    "candidate_id": plan.candidate_id,
                    "scale": plan.scale,
                    "rows": written,
                    "output": str(plan.output_path),
                    "manifest": str(plan.manifest_path),
                }
            )
        )


if __name__ == "__main__":
    main()
