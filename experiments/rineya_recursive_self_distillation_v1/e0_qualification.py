"""Read-only E0 evidence checks; never an execution permit or model loader."""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
from pathlib import Path

from .contracts import file_hash, read_json, require

PACKAGES = {"torch": "2.10.0+cu126", "transformers": "4.57.1",
            "safetensors": "0.7.0", "tokenizers": "0.22.2"}
CENTRAL = "experiments/post_training_mimo_centralized_v1"


def validate_e0_chain(config_path, result_path, root):
    """Consume the unchanged E0 producer's result and companion receipts.

    Historical HEAD may differ from RSD HEAD. Relevant producer bytes, runtime,
    config and containment must match; process-only PASS alone never suffices.
    Receipts are trusted local evidence, not proof against deliberate forgery.
    """
    config_path, result_path, root = map(Path, (config_path, result_path, root))
    config, result = read_json(config_path), read_json(result_path)
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import validate_config
    validate_config(config)
    require(result.get("experiment_id") == "RINEYA-0P6B-E0"
            and result.get("status") == "PASS_E0_SCOPED_NO_EFFICACY_CLAIM"
            and result.get("config_sha256") == file_hash(config_path), "genuine matching E0 PASS required")
    require(result.get("dirty_at_launch") is False and type(result.get("pretrained_loads")) is int
            and result["pretrained_loads"] == 1
            and type(result.get("optimizer_steps")) is int
            and result["optimizer_steps"] == config["maximum_optimizer_calls"], "E0 native execution evidence")
    checks = result.get("checks", {})
    for key in ("exact_native_load", "save_reload_resume_bitwise", "frozen_base_unchanged",
                "finite_scoped_backward", "eos_pad_mask", "native_padding_and_causality"):
        require(checks.get(key) is True, "missing E0 check: " + key)
    parity = checks.get("zero_adapter_parity", {})
    require(parity.get("prompts") == config["parity_prompts"]
            and parity.get("max_abs") == config["zero_adapter_atol"], "E0 zero-adapter parity evidence")
    likelihood = checks.get("sampled_logprobs", {})
    for name, limit in (("max_abs", "logprob_max_atol"), ("mean_abs", "logprob_mean_atol")):
        value = likelihood.get(name)
        require(type(value) in (int, float) and math.isfinite(value)
                and 0 <= value <= config[limit], "E0 sampled-logprob evidence")
    require(likelihood.get("backend") == "full_prefix_no_cache_temperature1_topk0_topp1"
            and bool(likelihood.get("tokens")), "E0 full-prefix sampling evidence")
    directory = result_path.parent
    supervision = directory.with_name(directory.name + "_supervisor")
    paths = {"source": directory / "SOURCE_RECEIPT.json", "runtime": directory / "RUNTIME_RECEIPT.json",
             "launch": supervision / "LAUNCH.json", "supervisor": supervision / "SUPERVISOR.json",
             "go": supervision / "GO.json"}
    source, runtime, launch, supervisor, go = (read_json(paths[k]) for k in paths)
    require(source.get("dirty") is False and source.get("code_sha") == result.get("code_sha")
            and source.get("script_sha256") == file_hash(root / CENTRAL / "scripts/qualify_native.py"),
            "E0 relevant producer source drift")
    require(type(result.get("code_sha")) is str and re.fullmatch(r"[0-9a-f]{40}", result["code_sha"]) is not None,
            "E0 recorded source commit required")
    transitive = {}
    for name in ("experiments/opallo_native_collector_v1/contract.py",
                 "experiments/maxrl_comp002_pretraining_v1/mechanics.py"):
        try:
            historical = subprocess.check_output(["git", "show", result["code_sha"] + ":" + name], cwd=root,
                                                  stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise ValueError("E0 historical dependency bytes unavailable: " + name) from exc
        transitive[name] = hashlib.sha256(historical).hexdigest()
        require(transitive[name] == file_hash(root / name), "E0 transitive producer source drift: " + name)
    require(launch.get("source_sha") == result["code_sha"]
            and launch.get("config_sha256") == file_hash(config_path)
            and launch.get("supervisor_sha256") == file_hash(root / CENTRAL / "scripts/supervise_native.py")
            and launch.get("watchdog_sha256") == file_hash(root / "experiments/opallo_native_collector_v1/supervisor.py")
            and go.get("config_sha256") == file_hash(config_path), "E0 containment/source binding drift")
    require(supervisor.get("status") == "PASS_PROCESS_ONLY" and supervisor.get("exit_code") == 0
            and supervisor.get("reason") is None and supervisor.get("limit_seconds") == config["wall_seconds"]
            and 0 <= supervisor.get("peak_observed_process_tree_rss_bytes", math.inf) <= 4 * 2**30,
            "matching E0 process containment PASS required")
    require(str(runtime.get("python", "")).startswith("3.11.15 ")
            and {k: runtime.get("packages", {}).get(k) for k in PACKAGES} == PACKAGES
            and runtime.get("cuda") == "12.6" and runtime.get("attention") == "eager"
            and runtime.get("base_dtype") == "bfloat16" and runtime.get("adapter_dtype") == "float32",
            "E0 registered runtime evidence")
    return {"result_sha256": file_hash(result_path), "historical_code_sha": result["code_sha"],
            "companion_sha256": {k: file_hash(p) for k, p in paths.items()}, "transitive_source_sha256": transitive,
            "runtime": runtime, "execution_authorized": False, "capability_qualified": False}
