"""Validate Track A scientific choices before delegating to shared execution."""
from __future__ import annotations

from pathlib import Path
import json

from .operators import REGISTERED_ARMS


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/protocol.json"


def load_protocol(path: Path) -> dict:
    from shared.guards import guard_non_sealed_path, require_runtime
    config = json.loads(guard_non_sealed_path(path).read_text(encoding="utf-8"))
    require_runtime(config)
    if config.get("track") != "native_armr" or tuple(config.get("arms", {})) != REGISTERED_ARMS:
        raise ValueError("native protocol must contain exactly the preregistered ordered arms")
    expected = {
        "A0": {"operator": "intact", "extra_layer_applications": 0},
        "A1": {"operator": "native_single", "layer": 12, "applications": 2},
        "A3": {"operator": "native_block", "layers": [12, 13], "applications": 2},
        "A16": {"operator": "native_single", "layer": 16, "applications": 2},
        "AM": {"operator": "independent_frozen_swiglu", "insertion_layer": 13},
    }
    if config["arms"] != expected:
        raise ValueError("candidate operators differ from preregistration")
    control = config["control"]
    if any(control.get(key) != value for key, value in {
        "insertion_layer": 13, "seed": 7301, "std": .02, "alpha": 1.,
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
    return {"status": "PREFLIGHT_PASS", "track": "native_armr", "arm": arm,
            "role": role, "examples_in_role": len(rows), "shard_index": shard_index,
            "shard_count": shard_count, "examples_in_shard": len(cases),
            "effective_generation_token_cap_max": max(min(config["max_new_tokens"], c.get("max_new_tokens", config["max_new_tokens"])) for c in cases),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "config_sha256": sha256_file(config_path), "registration_verified": bool(registration),
            "model_loaded": False,
            "baseline_note": "execution separately requires a complete matching A0 shard for candidates",
            "architecture_note": "confirmation execution separately requires all 128 architecture cases across all five arms"}
