#!/usr/bin/env python
"""Exact-host native checkpoint qualification for FMA-QUAL-001.

This runner is deliberately data-free: it uses a short synthetic prompt, never
sealed/confirmation evaluation data. It verifies the transition mechanics on an
actual local Qwen3 checkpoint and emits a receipt; it does not measure efficacy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.factorized_memory_attention_v1.fma.budget import parameter_budget
from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, load_shape
from experiments.factorized_memory_attention_v1.fma.memory import FactorizedMemoryBank
from experiments.factorized_memory_attention_v1.fma.qwen3 import convert_qwen3_model
from experiments.factorized_memory_attention_v1.fma.resources import require_host_ram


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a potentially multi-GB checkpoint without reading it into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def critical_source_snapshot(model_dir: str | Path) -> dict[str, dict[str, int | str]]:
    root = Path(model_dir)
    names = ("model.safetensors", "config.json", "tokenizer.json")
    out: dict[str, dict[str, int | str]] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required checkpoint asset is missing: {path}")
        stat = path.stat()
        out[name] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    return out


def clone_bank(shape, cfg: FMAConfig, source: FactorizedMemoryBank) -> FactorizedMemoryBank:
    restored = FactorizedMemoryBank(shape, cfg).to(
        device=source.token_factors.weight.device,
        dtype=source.token_factors.weight.dtype,
    )
    payload = io.BytesIO()
    torch.save(source.state_dict(), payload)
    payload.seek(0)
    restored.load_state_dict(torch.load(payload, map_location="cpu", weights_only=True))
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description="FMA-QUAL-001 native checkpoint qualifier")
    parser.add_argument("--model", required=True, help="local Supa/Qwen3 checkpoint directory")
    parser.add_argument(
        "--model-config",
        default=str(ROOT / "Rineya_Published_Cores" / "Rineya-Supa-1.7.0-devfreeze" / "config.json"),
    )
    parser.add_argument(
        "--core",
        default=str(ROOT / "Rineya_Published_Cores" / "Rineya-Supa-1.7.0-devfreeze" / "CORE.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="Memory attention qualification prompt.")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    guard = require_host_ram()

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(f"BLOCKED: Transformers/Qwen3 runtime unavailable: {exc}") from exc

    model_dir = Path(args.model).resolve()
    shape = load_shape(args.model_config)
    core = json.loads(Path(args.core).read_text(encoding="utf-8"))
    expected_weight_sha = str(core["checks"]["weight_sha256"])
    baseline_params = int(core["checks"]["parameter_count_serialized"])
    cfg = FMAConfig(rank=args.rank, gate=0.0)
    budget = parameter_budget(shape, cfg, baseline_params=baseline_params)

    before = critical_source_snapshot(model_dir)
    if before["model.safetensors"]["sha256"] != expected_weight_sha:
        raise SystemExit(
            "BLOCKED: checkpoint identity mismatch: "
            f"{before['model.safetensors']['sha256']} != {expected_weight_sha}"
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("BLOCKED: CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        baseline = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.detach().float().cpu()

    bank = FactorizedMemoryBank(shape, cfg)
    conversion = convert_qwen3_model(model, bank, layers=cfg.resolved_layers(shape))
    model.eval()

    with torch.inference_mode():
        g0 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.detach().float().cpu()

    g0_exact = bool(torch.equal(baseline, g0))
    g0_max_delta = float((baseline - g0).abs().max())
    g0_close = bool(torch.allclose(baseline, g0, atol=args.atol, rtol=args.rtol))

    # Serialization/reload of the compact bank must not change indexed memory.
    bank_probe_ids = input_ids[:, : min(input_ids.shape[1], 4)]
    restored = clone_bank(shape, cfg, bank)
    with torch.inference_mode():
        bank_reload_exact = bool(
            torch.equal(bank.memory(bank_probe_ids, 0), restored.memory(bank_probe_ids, 0))
        )
    del restored

    # Hard-memory structural bypass: make any native W_V execution fail loudly.
    bank.set_all_gates(1.0)
    saved_forwards = []
    for layer_idx in cfg.resolved_layers(shape):
        projection = model.model.layers[layer_idx].self_attn.v_proj
        saved_forwards.append((projection, projection.forward))

        def forbidden(*_args, **_kwargs):
            raise AssertionError("standard v_proj executed with g=1")

        projection.forward = forbidden

    cache_match = False
    finite = False
    hard_shape = None
    try:
        with torch.inference_mode():
            hard = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            finite = bool(torch.isfinite(hard.logits).all())
            hard_shape = list(hard.logits.shape)

            next_token = hard.logits[:, -1].argmax(-1, keepdim=True)
            cached = model(
                input_ids=next_token,
                past_key_values=hard.past_key_values,
                use_cache=True,
            ).logits[:, -1].detach().float().cpu()
            full_ids = torch.cat((input_ids, next_token), dim=1)
            full_mask = torch.ones_like(full_ids)
            full = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                use_cache=False,
            ).logits[:, -1].detach().float().cpu()
            cache_match = bool(torch.allclose(cached, full, atol=args.atol, rtol=args.rtol))
            cache_max_delta = float((cached - full).abs().max())
    finally:
        for projection, forward in saved_forwards:
            projection.forward = forward

    after = critical_source_snapshot(model_dir)
    source_unmodified = before == after

    cuda = {"available": torch.cuda.is_available()}
    if device.type == "cuda":
        cuda.update(
            device=torch.cuda.get_device_name(device),
            total_bytes=torch.cuda.get_device_properties(device).total_memory,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )

    passed = all(
        (
            g0_close,
            bank_reload_exact,
            finite,
            cache_match,
            source_unmodified,
        )
    )
    receipt = {
        "qualification_id": "FMA-QUAL-001-NATIVE",
        "status": "PASS" if passed else "FAIL",
        "evidence_classification": "native mechanical qualification only; no architecture efficacy result",
        "base_sha": "7d9b4fd450bc56405f6cec41459a7e83df7958ae",
        "model": {
            "path": str(model_dir),
            "checkpoint_weight_sha256": expected_weight_sha,
            "config_sha256": core["checks"]["preserved_asset_sha256"]["config.json"],
            "tokenizer_sha256": core["checks"]["preserved_asset_sha256"]["tokenizer.json"],
            "shape": shape.to_dict(),
            "dtype": str(dtype),
        },
        "architecture": {
            "rank": args.rank,
            "modified_layers": list(cfg.resolved_layers(shape)),
            "added_parameter_count": budget.added_fma_params,
            "gate_g0": 0.0,
            "gate_g1": 1.0,
        },
        "checks": {
            "g0_exact_equal": g0_exact,
            "g0_allclose": g0_close,
            "g0_max_abs_logit_delta": g0_max_delta,
            "g1_standard_value_projection_bypassed_structurally": True,
            "g1_logits_finite": finite,
            "g1_logit_shape": hard_shape,
            "cache_continuation_matches_full_reforward": cache_match,
            "cache_max_abs_logit_delta": cache_max_delta,
            "bank_serialization_reload_exact": bank_reload_exact,
            "source_checkpoint_unmodified": source_unmodified,
        },
        "conversion": {
            key: value for key, value in conversion.items() if key != "hook_handles"
        },
        "source_snapshot_before": before,
        "source_snapshot_after": after,
        "seed": args.seed,
        "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": package_version("safetensors"),
        },
        "resources": {
            "host_ram_guard": guard,
            "cuda": cuda,
        },
        "training_performed": False,
        "calibration_performed": False,
        "data_ids": [],
        "sealed_data_access": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
