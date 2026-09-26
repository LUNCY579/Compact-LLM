"""Read saved LoRA safetensors directly; never load a model or use a GPU.

This verifies serialized names/shapes/values/norms, NOT prediction roundtrip.
Only this audit's JSON output is written. Run from repository root:
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/results/audit_20260905/verify_saved_adapter_values.py
Use --require-complete after all requested stages have finished.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open

LAB = Path(__file__).resolve().parents[2]
REPO = LAB.parents[1]
AUDIT = Path(__file__).resolve().parent
LAYERS = (17, 18, 19)
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
RANK = 4


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_files():
    vault = REPO.parent / "Chat-Bubble" / "Rineya_Published_Cores"
    names = {"a11": "Rineya-Supa-A1.1-1.1.0-devfreeze", "a1": "Rineya-Supa-A1-1.0.0-devfreeze"}
    required = ("model.safetensors", "config.json", "generation_config.json", "tokenizer_config.json",
                "tokenizer.json", "vocab.json", "merges.txt", "CORE.json", "FILES.json")
    return {key: {name: vault / directory / name for name in (*required, f"modeling_supa_{key}.py")}
            for key, directory in names.items()}


def snapshot(files):
    return {key: {name: sha256(path) for name, path in entries.items()} for key, entries in files.items()}


def verify_arm(stage, arm, core_configs):
    directory = LAB / "results" / stage / f"arm_{arm}"
    manifest_path = directory / "run_manifest.json"
    info_path = directory / f"arm_{arm}_info.json"
    config_path = directory / "adapter" / "adapter_config.json"
    weights_path = directory / "adapter" / "adapter_model.safetensors"
    report = {"stage": stage, "arm": arm, "directory": str(directory), "status": "INCOMPLETE",
              "scope": "saved tensor values and norm only; no model load or prediction roundtrip"}
    required = (manifest_path, info_path, config_path, weights_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        report["missing_files"] = missing
        return report
    manifest = read_json(manifest_path)
    report["manifest_status"] = manifest.get("status")
    if manifest.get("status") != "COMPLETE":
        report["reason"] = "arm is not marked COMPLETE; do not inspect a writer's active files"
        return report
    hashes_before = {str(path): sha256(path) for path in required}
    info = read_json(info_path)
    config = read_json(config_path)
    core_name = manifest["config"]["core"]
    core = core_configs[core_name]
    hidden = core["hidden_size"]
    head_dim = core.get("head_dim", hidden // core["num_attention_heads"])
    q_width = head_dim * core["num_attention_heads"]
    kv_width = head_dim * core["num_key_value_heads"]
    dims = {"q_proj": (hidden, q_width), "k_proj": (hidden, kv_width),
            "v_proj": (hidden, kv_width), "o_proj": (q_width, hidden)}
    expected = {}
    for layer in LAYERS:
        for projection in PROJECTIONS:
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
            input_width, output_width = dims[projection]
            expected[f"{prefix}.lora_A.weight"] = [RANK, input_width]
            expected[f"{prefix}.lora_B.weight"] = [output_width, RANK]
    tensors = []
    sum_squares64 = 0.0
    sum_squares32 = 0.0
    with safe_open(weights_path, framework="numpy") as reader:
        keys = list(reader.keys())
        for name in keys:
            value = reader.get_tensor(name)
            finite = bool(np.isfinite(value).all())
            norm = None
            if finite:
                squared64 = float(np.square(value.astype(np.float64)).sum(dtype=np.float64))
                squared32 = float(np.square(value.astype(np.float32)).sum(dtype=np.float32))
                sum_squares64 += squared64
                sum_squares32 += squared32
                norm = math.sqrt(squared64)
            tensors.append({"name": name, "shape": list(value.shape), "expected_shape": expected.get(name),
                            "dtype": str(value.dtype), "parameters": int(value.size), "all_finite": finite,
                            "nonzero_values": int(np.count_nonzero(value)), "l2_norm_float64": norm})
    all_finite = all(t["all_finite"] for t in tensors)
    norm64 = math.sqrt(sum_squares64) if all_finite else None
    norm32 = math.sqrt(sum_squares32) if all_finite else None
    training = info.get("training") or {}
    recorded_norm = training.get("final_adapter_norm")
    norm_is_number = isinstance(recorded_norm, (int, float)) and math.isfinite(recorded_norm)
    delta = abs(norm64 - recorded_norm) if norm64 is not None and norm_is_number else None
    # The receipt rounds six decimals, while training reduces float32 squared
    # values on CUDA. NumPy float64 is an independent higher-precision check;
    # CPU/CUDA reduction order can differ slightly beyond decimal rounding.
    rounding_tolerance = 0.5e-6
    reduction_tolerance = 2 * float(np.finfo(np.float32).eps) * max(1.0, norm64 or 0.0)
    tolerance = rounding_tolerance + reduction_tolerance
    b_tensors = [t for t in tensors if t["name"].endswith(".lora_B.weight")]
    checks = {
        "exactly_24_expected_lora_tensors": set(keys) == set(expected) and len(keys) == 24,
        "all_shapes_match_core_attention_and_rank4": all(t["shape"] == t["expected_shape"] for t in tensors),
        "no_backbone_or_unexpected_tensors": not (set(keys) - set(expected)),
        "all_values_finite": all_finite,
        "all_saved_values_float32": all(t["dtype"] == "float32" for t in tensors),
        "all_12_B_matrices_nonzero": len(b_tensors) == 12 and all(t["nonzero_values"] > 0 for t in b_tensors),
        "expected_parameter_count": sum(t["parameters"] for t in tensors) == 122880,
        "parameter_count_matches_arm_info": sum(t["parameters"] for t in tensors) == info["adapter"]["trainable_params"],
        "saved_config_matches_attention_lora_only": (
            config.get("peft_type") == "LORA" and config.get("r") == RANK
            and set(config.get("layers_to_transform") or []) == set(LAYERS)
            and set(config.get("target_modules") or []) == set(PROJECTIONS)
            and config.get("bias") == "none" and not config.get("modules_to_save")
            and not config.get("target_parameters") and not config.get("use_dora")),
        "at_least_one_reported_applied_update": training.get("steps_applied", 0) > 0,
        "total_norm_matches_reported_with_explicit_tolerance": delta is not None and delta <= tolerance,
    }
    hashes_after = {str(path): sha256(path) for path in required}
    checks["saved_files_unchanged_during_read"] = hashes_before == hashes_after
    report.update(status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                  core=core_name, tensor_count=len(tensors), total_parameters=sum(t["parameters"] for t in tensors),
                  tensor_details=tensors, total_norm_float64=norm64, total_norm_float32_numpy_reduction=norm32,
                  training_final_adapter_norm=recorded_norm, absolute_norm_difference=delta,
                  norm_comparison={"decimal_rounding_absolute_tolerance": rounding_tolerance,
                                   "float32_reduction_absolute_tolerance": reduction_tolerance,
                                   "combined_absolute_tolerance": tolerance,
                                   "matches_decimal_rounding_alone": delta is not None and delta <= rounding_tolerance,
                                   "explanation": "Independent float64 norm versus six-decimal CUDA float32 sum-of-squares receipt; no prediction parity claim."},
                  saved_files_sha256_before=hashes_before, saved_files_sha256_after=hashes_after)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=["phase1_smoke", "phase1", "phase1_a1"])
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, default=AUDIT / "saved_adapter_values.json")
    args = parser.parse_args()
    if any(Path(stage).name != stage or stage in (".", "..") for stage in args.stages):
        parser.error("stages must be direct names under results")
    output = args.output.resolve()
    if not output.is_relative_to(AUDIT):
        parser.error("audit output must remain under results/audit_20260905")
    files = canonical_files()
    before = snapshot(files)
    configs = {name: read_json(paths["config.json"]) for name, paths in files.items()}
    records = []
    for stage in args.stages:
        for arm in ("B", "C"):
            try:
                records.append(verify_arm(stage, arm, configs))
            except Exception as error:
                records.append({"stage": stage, "arm": arm, "status": "FAIL", "error": f"{type(error).__name__}: {error}"})
    after = snapshot(files)
    unchanged = before == after
    statuses = [r["status"] for r in records]
    status = "FAIL" if "FAIL" in statuses or not unchanged else "INCOMPLETE" if "INCOMPLETE" in statuses else "PASS"
    report = {"status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.executable, "numpy": np.__version__, "model_loads": 0, "gpu_used": False,
              "prediction_roundtrip_verified": False, "scope": "Direct serialized adapter values, names, shapes, and norm only.",
              "stages": args.stages, "arms": records, "source_sha256": sha256(Path(__file__)),
              "checkpoint_sha256_before": before, "checkpoint_sha256_after": after,
              "canonical_checkpoint_unchanged": unchanged}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(output), "canonical_checkpoint_unchanged": unchanged,
                      "arms": [{k: r.get(k) for k in ("stage", "arm", "status", "tensor_count", "total_parameters", "absolute_norm_difference")} for r in records]}, indent=2))
    return 1 if status == "FAIL" else 2 if status == "INCOMPLETE" and args.require_complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
