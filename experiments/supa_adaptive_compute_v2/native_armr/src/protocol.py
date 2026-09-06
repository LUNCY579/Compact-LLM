"""Validate Track A scientific choices before delegating to shared execution."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math
import re

from .operators import REGISTERED_ARMS


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/protocol.json"
GPU_CONFIG = DEFAULT_CONFIG.with_name("protocol_gpu_v2_2.json")
CALIBRATED_GPU_CONFIG = DEFAULT_CONFIG.with_name("protocol_gpu_v2_3.json")
CALIBRATED_PROTOCOL_VERSION = "native_armr_v2.3_gpu_fp32_calibrated_control"
CALIBRATED_CONTROL_SELECTION = "calibration_only_scenario_balanced_ordinary_residual_energy"
CALIBRATED_RUNTIME_QUALIFICATION = "native_gpu_fp32_v2_3"
LEARNED_GPU_CONFIG = DEFAULT_CONFIG.with_name("protocol_gpu_v2_4.json")
LEARNED_PROTOCOL_VERSION = "native_armr_v2.4_gpu_fp32_pretrained_parallel_ffn_control"
LEARNED_RUNTIME_QUALIFICATION = "native_gpu_fp32_v2_4"
LEARNED_CONTROL_METADATA = {
    "kind": "pretrained_parallel_ffn", "insertion_layer": 13,
    "source_layers": [12, 13, 14], "summation_order": "(F12+F13)+F14",
    "local_frozen_copies": True,
}
GPU_RUNTIME_AMENDMENT = {
    "protocol_version": "native_armr_v2.2_gpu_fp32",
    "device": "cuda:0",
    "min_available_gib": 0.5,
    "min_available_vram_gib": 3.0,
    "max_cuda_memory_gib": 6.5,
    "tf32": False,
    "runtime_qualification": "native_gpu_fp32_v1",
    "qualification": {
        "examples_per_family": 1,
        "max_abs_logit_error": 0.002,
        "max_relative_logit_l2": 0.0002,
        "minimum_logit_cosine": 0.999999,
        "max_abs_nll_error": 0.002,
        "max_relative_hidden_l2": 0.0002,
        "minimum_hidden_cosine": 0.999999,
        "require_greedy_identity": True,
        "require_gpu_repeat_identity": True,
    },
}


def require_config_path(path: Path) -> Path:
    """Execution and audit use a frozen named protocol, never an ad-hoc copy."""
    from shared.guards import guard_non_sealed_path
    path = guard_non_sealed_path(path)
    if path not in (DEFAULT_CONFIG.resolve(), GPU_CONFIG.resolve(), CALIBRATED_GPU_CONFIG.resolve(), LEARNED_GPU_CONFIG.resolve()):
        raise ValueError("native execution requires the registered CPU or GPU protocol path")
    return path


def calibrated_gpu_config(binding: dict, alpha: float) -> dict:
    """Construct the sole prospective change, without licensing an input alpha.

    The freeze caller and loader independently verify the artifact; this pure
    constructor only defines the exact allowed difference from the old science.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("calibrated control alpha must be finite and positive")
    reference = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return deepcopy({**reference, **GPU_RUNTIME_AMENDMENT,
            "protocol_version": CALIBRATED_PROTOCOL_VERSION,
            "runtime_qualification": CALIBRATED_RUNTIME_QUALIFICATION,
            "control": {**reference["control"], "alpha": alpha,
                        "selection": CALIBRATED_CONTROL_SELECTION},
            "control_calibration": dict(binding)})


def learned_gpu_config(binding: dict, alpha: float) -> dict:
    """Define v2.4 without granting admission to an arbitrary supplied alpha."""
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("pretrained control alpha must be finite and positive")
    reference = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config = deepcopy({**reference, **GPU_RUNTIME_AMENDMENT})
    config.update(protocol_version=LEARNED_PROTOCOL_VERSION,
                  runtime_qualification=LEARNED_RUNTIME_QUALIFICATION,
                  control_calibration=dict(binding))
    config["arms"]["AM"] = {"operator": "pretrained_parallel_ffn", "insertion_layer": 13}
    config["control"] = {
        **deepcopy(LEARNED_CONTROL_METADATA), "alpha": alpha,
        "hidden_size": 1024, "intermediate_size": 3072,
        "normalization": "canonical_post_attention_layernorm_per_branch",
        "activation": "canonical_silu", "branch_coefficients": [1, 1, 1],
        "additional_matrix_parameters": 28311552,
        "additional_normalization_parameters": 3072,
        "additional_parameters": 28314624,
        "additional_active_parameters": 28314624,
        "newly_learned_parameters": 0,
        "additional_matmul_flops_per_token": 56623104,
        "a3_reference_matmul_flops_per_token": 62914560,
        "dominant_matmul_ratio_to_a3": .9,
        "training": False,
        "selection": CALIBRATED_CONTROL_SELECTION,
    }
    return config


def verify_learned_control_calibration(config: dict) -> dict:
    """Verify the committed v2.4 scale and canonical copied-weight provenance.

    The qualifier's CPU runtime copy retains the same version, control and
    artifact. This function validates the scientific control independently of
    device; load_protocol separately requires the exact final GPU settings.
    """
    from shared.guards import require_within
    from shared.legacy import REPO, ROOT
    from .learned_control_calibration import verify_calibration_artifact
    if config.get("protocol_version") != LEARNED_PROTOCOL_VERSION:
        raise ValueError("pretrained calibration binding requires the registered v2.4 protocol")
    binding = config.get("control_calibration")
    if not isinstance(binding, dict) or set(binding) != {"artifact_path", "artifact_sha256"}:
        raise ValueError("pretrained control requires its exact artifact path/hash binding")
    relative, digest = binding["artifact_path"], binding["artifact_sha256"]
    if not isinstance(relative, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid pretrained control artifact path/hash")
    path = require_within(REPO / relative, ROOT / "native_armr/evidence/00_learned_control_calibration")
    if path.name != "report.json" or path.relative_to(REPO).as_posix() != relative:
        raise ValueError("pretrained calibration artifact must use its canonical repository-relative report path")
    report = verify_calibration_artifact(path, digest, require_committed=True)
    alpha = config.get("control", {}).get("alpha")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or
            not math.isfinite(alpha) or alpha <= 0 or alpha != report["alpha"]):
        raise ValueError("control alpha differs from the exact pretrained calibration artifact")
    expected = learned_gpu_config(binding, report["alpha"])
    if config.get("control") != expected["control"] or config.get("arms", {}).get("AM") != expected["arms"]["AM"]:
        raise ValueError("pretrained control differs from the registered operator")
    if report.get("control") != LEARNED_CONTROL_METADATA:
        raise ValueError("pretrained calibration artifact control differs from the registered operator")
    if report["checkpoint_files_before"].get("model.safetensors") != config["checkpoint_sha256"]:
        raise ValueError("pretrained calibration artifact used a different canonical checkpoint")
    return report


def verify_control_calibration(config: dict) -> dict:
    """Recompute and verify the committed label-free alpha before inference."""
    from shared.guards import require_within
    from shared.legacy import REPO, ROOT
    from .control_calibration import verify_calibration_artifact
    if config.get("protocol_version") != CALIBRATED_PROTOCOL_VERSION:
        raise ValueError("calibration binding requires the registered v2.3 protocol")
    binding = config.get("control_calibration")
    if not isinstance(binding, dict) or set(binding) != {"artifact_path", "artifact_sha256"}:
        raise ValueError("calibrated control requires its exact artifact path/hash binding")
    relative, digest = binding["artifact_path"], binding["artifact_sha256"]
    if not isinstance(relative, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid calibrated control artifact path/hash")
    path = require_within(REPO / relative, ROOT / "native_armr/evidence/00_control_calibration")
    if path.name != "report.json" or path.relative_to(REPO).as_posix() != relative:
        raise ValueError("calibration artifact must use its canonical repository-relative report path")
    report = verify_calibration_artifact(path, digest, require_committed=True)
    alpha = config.get("control", {}).get("alpha")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or
            not math.isfinite(alpha) or alpha <= 0 or alpha != report["alpha"]):
        raise ValueError("control alpha differs from the exact fitted calibration artifact")
    expected_control = {key: config["control"][key] for key in
                        ("seed", "std", "intermediate_size", "insertion_layer")}
    if any(report["control"].get(key) != value for key, value in expected_control.items()):
        raise ValueError("calibration artifact control differs from the registered operator")
    if report["checkpoint_files_before"].get("model.safetensors") != config["checkpoint_sha256"]:
        raise ValueError("calibration artifact used a different canonical checkpoint")
    return report


def load_protocol(path: Path) -> dict:
    from shared.guards import guard_non_sealed_path, require_runtime
    config = json.loads(guard_non_sealed_path(path).read_text(encoding="utf-8"))
    require_runtime(config)
    calibrated = config.get("protocol_version") == CALIBRATED_PROTOCOL_VERSION
    learned = config.get("protocol_version") == LEARNED_PROTOCOL_VERSION
    if learned:
        expected = learned_gpu_config(config.get("control_calibration", {}), config.get("control", {}).get("alpha"))
        if config != expected:
            raise ValueError("v2.4 may change only the registered pretrained control and calibration binding; native science must match the reference")
        verify_learned_control_calibration(config)
    elif calibrated:
        expected = calibrated_gpu_config(config.get("control_calibration", {}), config.get("control", {}).get("alpha"))
        if config != expected:
            raise ValueError("v2.3 may change only the registered calibration binding and control scale; native science must match the reference")
        verify_control_calibration(config)
    elif config.get("device") == "cuda:0":
        reference = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        if config != {**reference, **GPU_RUNTIME_AMENDMENT}:
            raise ValueError("GPU amendment may change only registered runtime settings; native science must match CPU reference")
    if config.get("track") != "native_armr" or tuple(config.get("arms", {})) != REGISTERED_ARMS:
        raise ValueError("native protocol must contain exactly the preregistered ordered arms")
    expected = {
        "A0": {"operator": "intact", "extra_layer_applications": 0},
        "A1": {"operator": "native_single", "layer": 12, "applications": 2},
        "A3": {"operator": "native_block", "layers": [12, 13], "applications": 2},
        "A16": {"operator": "native_single", "layer": 16, "applications": 2},
        "AM": {"operator": "independent_frozen_swiglu", "insertion_layer": 13},
    }
    if learned:
        expected["AM"] = {"operator": "pretrained_parallel_ffn", "insertion_layer": 13}
    if config["arms"] != expected:
        raise ValueError("candidate operators differ from preregistration")
    control = config["control"]
    if not learned and any(control.get(key) != value for key, value in {
        "insertion_layer": 13, "seed": 7301, "std": .02,
        "alpha": control["alpha"] if calibrated else 1.,
        "hidden_size": 1024, "intermediate_size": 10240,
        "additional_matrix_parameters": 31457280,
        "additional_matmul_flops_per_token": 62914560, "training": False,
    }.items()):
        raise ValueError("compute control differs from preregistration")
    if config["analysis"]["primary_comparisons_hierarchical"] != [["A3", "A0"], ["A3", "AM"]]:
        raise ValueError("primary hierarchy changed")
    return config


def preflight(config_path: Path, manifest_path: Path, role: str, arm: str,
              shard_index: int, shard_count: int) -> dict:
    """Verify frozen inputs and committed registration without loading a model."""
    from shared.datasets import load_frozen_dataset
    from shared.hashing import sha256_file
    from shared.manifests import verify_registration
    from shared.legacy import REPO
    from shared.evaluation import shard_cases
    config = load_protocol(config_path)
    if arm not in config["arms"] or role not in config["data"]["allowed_roles"]:
        raise ValueError("unregistered native arm or dataset role")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    rows = load_frozen_dataset(manifest_path, track="native_armr", role=role)
    registration = verify_registration("native_armr", config_path)
    relative = manifest_path.resolve().relative_to(REPO).as_posix()
    if registration["dataset_manifests"].get(relative) != sha256_file(manifest_path):
        raise ValueError("preflight dataset is not bound by this track registration")
    cases = shard_cases(rows, shard_index, shard_count)
    if not cases:
        raise ValueError("empty shard; choose a smaller shard count")
    from shared.qualifications import verify_runtime_qualification
    qualification = verify_runtime_qualification(config, manifest_path, registration)
    return {"status": "PREFLIGHT_PASS", "track": "native_armr", "arm": arm,
            "role": role, "examples_in_role": len(rows), "shard_index": shard_index,
            "shard_count": shard_count, "examples_in_shard": len(cases),
            "effective_generation_token_cap_max": max(min(config["max_new_tokens"], c.get("max_new_tokens", config["max_new_tokens"])) for c in cases),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "config_sha256": sha256_file(config_path), "registration_verified": bool(registration),
            "runtime_qualification": qualification,
            "model_loaded": False,
            "baseline_note": "execution separately requires a complete matching A0 shard for candidates",
            "architecture_note": "confirmation execution separately requires all 128 architecture cases across all five arms"}
