#!/usr/bin/env python3
"""Run one published Rineya Qwen3-0.6B core on answer-blind Showcase-64 arms."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SCRIPTS = LAB_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_published_core_eval import (  # noqa: E402
    Condition,
    _atomic_write_json,
    _atomic_write_jsonl,
    _check_geometry,
    _create_output_dir,
    _load_mapping,
    _prepare_case,
    _repo_path,
    _write_resolved_config,
    execute_condition,
)
from run_showcase64_eval import read_showcase_rows  # noqa: E402

from rineya_significance_lab.manifest import (  # noqa: E402
    build_manifest,
    sha256_file,
    utc_now,
)

CHECKPOINT_FILES = {
    "weights": "model.safetensors",
    "tokenizer": "tokenizer.json",
    "config": "config.json",
    "core": "CORE.json",
}


def _condition_map(config: dict[str, Any]) -> dict[str, Condition]:
    values: dict[str, Condition] = {}
    for row in config["conditions"]:
        condition_id = str(row["id"])
        if condition_id in values:
            raise ValueError(f"duplicate condition id: {condition_id}")
        exponent = row.get("exponent")
        values[condition_id] = Condition(
            condition_id=condition_id,
            policy=str(row["policy"]),
            exponent=None if exponent is None else float(exponent),
            query_conditioned=True,
        )
    return values


def _verify_core(core_key: str, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    cores = config.get("cores")
    if not isinstance(cores, dict) or core_key not in cores:
        raise ValueError(f"unknown published-family core: {core_key}")
    entry = dict(cores[core_key])
    directory = (REPO_ROOT / str(entry["directory"])).resolve(strict=True)
    expected = dict(entry["expected_hashes"])
    observed: dict[str, str] = {}
    for name, filename in CHECKPOINT_FILES.items():
        path = directory / filename
        observed[name] = sha256_file(path)
        if observed[name] != str(expected[name]).lower():
            raise RuntimeError(
                f"{core_key}: {filename} hash {observed[name]} != {expected[name]}"
            )
    core = json.loads((directory / "CORE.json").read_text(encoding="utf-8"))
    if str(core.get("core_id")) != str(entry["core_id"]):
        raise RuntimeError(f"{core_key}: CORE.json identity mismatch")
    return directory, {
        "core_key": core_key,
        "core_id": str(entry["core_id"]),
        "directory": _repo_path(directory),
        "hashes": observed,
    }


def run(
    config_path: Path,
    *,
    core_key: str,
    condition_ids: tuple[str, ...],
    output: Path,
) -> dict[str, Any]:
    """Execute one core without reading or hashing the answer key."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = utc_now()
    config_path = config_path.expanduser().resolve(strict=True)
    config = _load_mapping(config_path)
    condition_map = _condition_map(config)
    if not condition_ids:
        raise ValueError("at least one condition is required")
    unknown = [value for value in condition_ids if value not in condition_map]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition ids must be unique")

    model_path, provenance = _verify_core(core_key, config)
    prompt_path = (REPO_ROOT / str(config["paths"]["prompts"])).resolve(strict=True)
    rows = read_showcase_rows(prompt_path)
    expected_cases = int(config["benchmark"]["expected_case_count"])
    if len(rows) != expected_cases:
        raise RuntimeError(f"expected {expected_cases} showcase rows, found {len(rows)}")

    output = output.expanduser().resolve()
    _create_output_dir(output)
    resolved = copy.deepcopy(config)
    resolved["_meta"] = {
        "source_config": _repo_path(config_path),
        "selected_core": core_key,
        "selected_conditions": list(condition_ids),
        "output": _repo_path(output),
    }
    _write_resolved_config(output / "resolved_config.yaml", resolved)

    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=str(config["runtime"]["attention_implementation"]),
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        .to("cpu")
        .eval()
    )
    _check_geometry(model.config, config["model_geometry"])
    prepared = [_prepare_case(tokenizer, row, config["prompt"], config["query"]) for row in rows]
    eos = tokenizer.eos_token_id
    eos_ids = (
        {int(value) for value in eos}
        if isinstance(eos, list)
        else ({int(eos)} if eos is not None else set())
    )

    selected = tuple(condition_map[value] for value in condition_ids)
    unscored: list[dict[str, Any]] = []
    partial = output / "predictions.partial.jsonl"
    total = len(prepared) * len(selected)
    started = time.perf_counter()
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for case in prepared:
            for condition in selected:
                result = execute_condition(
                    model,
                    case,
                    condition,
                    cache_config=config["cache"],
                    runtime_config=config["runtime"],
                    eos_token_ids=eos_ids,
                )
                prediction = tokenizer.decode(
                    result.pop("generated_token_ids"), skip_special_tokens=True
                ).strip()
                record = {
                    "schema_version": "1.0",
                    "status": "OK",
                    "core": core_key,
                    "core_id": provenance["core_id"],
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "condition": condition.condition_id,
                    "prediction": prediction,
                    "input_tokens": int(case.input_ids.shape[-1]),
                    "resident_after_prefill": result["accounting"][
                        "resident_entries_after_prefill"
                    ],
                    "kv_only_bytes": result["efficiency"]["kv_only_bytes"],
                    "total_persistent_bytes": result["efficiency"][
                        "architecture_total_persistent_bytes"
                    ],
                    "inference_wall_seconds": result["timing"]["inference_wall_seconds"],
                }
                unscored.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                done = len(unscored)
                if done % 32 == 0 or done == total:
                    rate = (time.perf_counter() - started) / done
                    print(
                        f"{core_key}: {done}/{total} eta={(total - done) * rate / 60:.1f} min",
                        flush=True,
                    )
                gc.collect()

    expected_pairs = {
        (str(row["id"]), condition_id)
        for row in rows
        for condition_id in condition_ids
    }
    observed_pairs = {(str(row["case_id"]), str(row["condition"])) for row in unscored}
    if len(unscored) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise RuntimeError("answer-blind segment coverage mismatch")
    _atomic_write_jsonl(output / "predictions.unscored.jsonl", unscored)
    summary = {
        "schema_version": "1.0",
        "status": "INFERENCE_COMPLETE_UNSCORED",
        "classification": "BOUNDED_EXPLORATORY_FAMILY_SEGMENT",
        "core": core_key,
        "conditions": list(condition_ids),
        "cases_per_condition": len(rows),
        "total_cells": len(unscored),
        "answer_key_accessed": False,
        "wall_seconds_sum": sum(float(row["inference_wall_seconds"]) for row in unscored),
        "claim_limit": "Answer-blind inference segment only; no quality result.",
    }
    _atomic_write_json(output / "summary.json", summary)

    manifest = build_manifest(
        repo=REPO_ROOT,
        config=resolved,
        run_id=output.name,
        execution_mode="CLOSED_LOOP_MODEL_EXECUTION",
        model={
            "status": "VERIFIED",
            "id": provenance["core_id"],
            "path": provenance["directory"],
            "architecture": str(config["model_geometry"]["expected_architecture"]),
            "hashes": {
                "weights": provenance["hashes"]["weights"],
                "tokenizer": provenance["hashes"]["tokenizer"],
                "config": provenance["hashes"]["config"],
                "chat_template": None,
                "core": provenance["hashes"]["core"],
            },
        },
        benchmark={"name": str(config["benchmark"]["name"])},
        architecture_variant=f"published-family {','.join(condition_ids)}",
        seed=seed,
        compute_tier="B",
        status="COMPLETE",
        resolved_config_path="resolved_config.yaml",
    )
    manifest["created_at"] = started_at
    manifest["completed_at"] = utc_now()
    manifest["input_provenance"] = {
        "prompt_sha256": sha256_file(prompt_path),
        "config_sha256": sha256_file(config_path),
        "answer_key_accessed": False,
    }
    manifest["artifacts"] = {
        "resolved_config": "resolved_config.yaml",
        "predictions_unscored": "predictions.unscored.jsonl",
        "summary": "summary.json",
    }
    manifest["notes"] = [
        "Exactly one published Qwen3-0.6B family core is loaded in this segment.",
        "No answer-key path is opened or hashed by this runner.",
        "Rineya/R2 and Aegis are excluded from the configuration.",
    ]
    _atomic_write_json(output / "manifest.json", manifest)
    del model, prepared, tokenizer
    gc.collect()
    print(json.dumps({"output": str(output), "cells": len(unscored)}, indent=2))
    return {"output": str(output), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "published_family_showcase64_v2.yaml",
    )
    parser.add_argument("--core", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(
        args.config,
        core_key=str(args.core),
        condition_ids=tuple(str(value) for value in args.conditions),
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
