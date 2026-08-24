#!/usr/bin/env python3
"""Verify true-count callback identity on the real Halo checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402
from rineya_significance_lab.runtime import (  # noqa: E402
    CountControlSpec,
    RuntimeCountController,
)


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper: {path}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("count-control config must be a mapping")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _positions_sha256(values: list[Any]) -> str:
    canonical = json.dumps(
        [value.tolist() for value in values],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def verify(config_path: Path, output_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mini = _module("halo_identity_mini", LAB_ROOT / "scripts" / "run_published_core_eval.py")
    capability = _module("halo_identity_capability", LAB_ROOT / "scripts" / "run_halo_capability.py")
    count_runner = _module("halo_identity_count", LAB_ROOT / "scripts" / "run_halo_count_controls.py")
    config = _load(config_path.resolve())
    paths = count_runner._resolve_paths(config)
    selected = count_runner.select_case_ids(config, _jsonl(paths["capability_cases"]))
    case_id = selected[0]
    inference = {str(row["case_id"]): row for row in _jsonl(paths["inference"])}
    row = inference[case_id]
    tokenizer = AutoTokenizer.from_pretrained(paths["model"], trust_remote_code=False)
    input_ids = tokenizer(str(row["prompt"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
    query_ids = tokenizer(str(row["question"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
    if capability._token_hash(input_ids) != row["token_ids_sha256"]:
        raise RuntimeError("frozen Halo prompt token hash mismatch")
    checkpoint = mini._checkpoint_hashes(paths["model"], config["model"])
    model: Any = AutoModelForCausalLM.from_pretrained(
        paths["model"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model = model.to("cpu").eval()
    mini._check_geometry(model.config, config["model"])
    torch.manual_seed(int(config["runtime"]["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    eos = tokenizer.eos_token_id
    eos_ids = {int(value) for value in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())
    prepared = mini.PreparedCase(
        case_id=case_id,
        scenario_id=str(row["cluster_id"]),
        metadata={"family": row["family"]},
        input_ids=input_ids,
        context_ids=input_ids,
        query_ids=query_ids,
        prompt_text_sha256=str(row["prompt_sha256"]),
        input_token_sha256=mini._token_sha256(input_ids),
        context_token_sha256=mini._token_sha256(input_ids),
        query_token_sha256=mini._token_sha256(query_ids),
    )
    condition = mini.Condition("B3", "significance", 1.0, True)
    kwargs = {
        "cache_config": config["cache"],
        "runtime_config": {
            "max_new_tokens": config["runtime"]["max_new_tokens"],
            "verify_query_rollback": config["runtime"]["verify_query_rollback"],
        },
        "eos_token_ids": eos_ids,
        "return_resident_positions": True,
    }
    standard = mini.execute_condition(model, prepared, condition, **kwargs)
    controller = RuntimeCountController(
        CountControlSpec(
            condition_id="true_realized_count",
            transform="true",
            seed=int(config["runtime"]["seed"]),
        ),
        case_id=case_id,
        chunk_size=int(config["cache"]["compression_chunk_tokens"]),
    )
    controlled = mini.execute_condition(
        model,
        prepared,
        condition,
        compression_callback=controller.compress,
        **kwargs,
    )
    standard_positions = standard.pop("_resident_positions")
    controlled_positions = controlled.pop("_resident_positions")
    standard_tokens = standard.pop("generated_token_ids")
    controlled_tokens = controlled.pop("generated_token_ids")
    checks = {
        "generated_token_ids_exact": standard_tokens == controlled_tokens,
        "resident_positions_exact": all(
            np.array_equal(left, right)
            for left, right in zip(standard_positions, controlled_positions, strict=True)
        ),
        "accounting_exact": standard["accounting"] == controlled["accounting"],
        "policy_exact": standard["policy"] == controlled["policy"],
        "significance_exponent_exact": standard["significance_exponent"]
        == controlled["significance_exponent"],
    }
    result = {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": "REAL_HALO_TRUE_COUNT_CALLBACK_IDENTITY",
        "case_id": case_id,
        "model_id": config["model"]["core_id"],
        "checkpoint_integrity": checkpoint,
        "config_sha256": sha256_file(config_path),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "checks": checks,
        "standard_resident_positions_sha256": _positions_sha256(standard_positions),
        "controlled_resident_positions_sha256": _positions_sha256(controlled_positions),
        "boundary_count": len(controller.summaries),
        "claim_limit": "One real-Halo case verifies callback identity; it is not a quality result.",
    }
    atomic_write_json(output_path, result)
    if result["status"] != "PASS":
        raise RuntimeError(f"real Halo count-control identity failed: {checks}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_count_causal_controls.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "results" / "halo_count_control_identity.json",
    )
    args = parser.parse_args()
    result = verify(args.config, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
