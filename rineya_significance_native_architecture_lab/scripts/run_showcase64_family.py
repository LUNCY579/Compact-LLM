#!/usr/bin/env python3
"""Showcase-64 evaluation across the whole published Qwen3-0.6B core family.

Closes ledger row ``N-009``, which has stood at ``NOT_RUN``: no core other than
Halo has ever decoded an answer under this lab's runtime.  The four secondary
cores have only one-case, prompt-only, 192-token compatibility smokes that
explicitly recorded ``answer_generation_performed: False``.

This runs the same protocol Halo received -- complete 64-case showcase split,
B0 full cache and B3 observation-normalized compression, identical prompts,
chunking, query access, and manual greedy decoding -- on Halo Uncensored, Lyra,
Nimbus, and Pulse.

Two things it establishes, both needed before any training work:

1. a full-cache capability baseline per core, which says which cores have enough
   headroom to be worth finetuning at all;
2. whether the Halo compression gap replicates across the family, or is specific
   to one checkpoint.

Each core keeps a separate identity.  Results are never pooled into a
family-wide efficacy claim.

Exploratory and unregistered.  The answer key is opened only after every core
has finished inference and the answer-blind artifact is durably written.
"""

from __future__ import annotations

import argparse
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
    _normalize_answer,
    _prepare_case,
    _repo_path,
    _write_resolved_config,
    execute_condition,
)
from run_showcase64_eval import read_answer_key, read_showcase_rows  # noqa: E402

from rineya_significance_lab.manifest import sha256_file, utc_now  # noqa: E402

CONDITIONS = (
    Condition(condition_id="B0", policy="full_cache", exponent=None, query_conditioned=True),
    Condition(condition_id="B3", policy="significance", exponent=1.0, query_conditioned=True),
)
# Halo already has this exact run recorded; secondary cores are the new evidence.
DEFAULT_CORES = ("halo_uncensored", "lyra", "nimbus", "pulse")


def load_family(path: Path) -> dict[str, Any]:
    family = _load_mapping(path)
    cores = family.get("cores")
    if not isinstance(cores, dict) or not cores:
        raise ValueError("published_core_family.yaml has no cores mapping")
    return family


def verify_core(core_key: str, entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Verify declared weight hash and CORE.json identity before loading."""

    directory = REPO_ROOT / str(entry["directory"])
    if not directory.is_dir():
        raise FileNotFoundError(f"{core_key}: missing core directory {directory}")
    core_path = directory / "CORE.json"
    if not core_path.is_file():
        raise FileNotFoundError(f"{core_key}: missing CORE.json")
    core = json.loads(core_path.read_text(encoding="utf-8"))
    declared = str(entry["core_id"])
    if str(core.get("core_id")) != declared:
        raise RuntimeError(
            f"{core_key}: CORE.json core_id {core.get('core_id')!r} != declared {declared!r}"
        )
    weights = directory / "model.safetensors"
    observed = sha256_file(weights)
    expected = str(entry["weight_sha256"]).lower()
    if observed != expected:
        raise RuntimeError(f"{core_key}: weight hash {observed} != declared {expected}")
    return directory, {
        "core_id": declared,
        "directory": _repo_path(directory),
        "weights_sha256": observed,
        "tokenizer_sha256": sha256_file(directory / "tokenizer.json"),
        "config_sha256": sha256_file(directory / "config.json"),
        "declared_status_before_this_run": str(entry.get("status", "")),
    }


def run(config_path: Path, cores: tuple[str, ...], output_override: Path | None = None) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = utc_now()
    config = _load_mapping(config_path.expanduser().resolve())
    config["_source_config_path"] = _repo_path(config_path)
    family = load_family(LAB_ROOT / "configs" / "published_core_family.yaml")

    prompt_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_prompts.jsonl"
    answer_key_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_answer_key.jsonl"
    output = (
        output_override.expanduser().resolve()
        if output_override
        else LAB_ROOT / "results" / "runs" / "halo_family_showcase64_v1"
    )
    _create_output_dir(output)
    _write_resolved_config(output / "resolved_config.yaml", config)

    rows = read_showcase_rows(prompt_path)
    case_ids = tuple(str(r["id"]) for r in rows)

    unknown = [c for c in cores if c not in family["cores"]]
    if unknown:
        raise ValueError(f"unknown core keys: {unknown}")

    provenance: dict[str, Any] = {}
    unscored: list[dict[str, Any]] = []
    partial = output / "predictions.partial.jsonl"
    total = len(cores) * len(rows) * len(CONDITIONS)
    done = 0
    started = time.perf_counter()

    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for core_key in cores:
            entry = family["cores"][core_key]
            directory, prov = verify_core(core_key, entry)
            provenance[core_key] = prov
            print(f"[{core_key}] verified {prov['weights_sha256'][:16]}...", flush=True)

            torch.manual_seed(int(config["runtime"]["seed"]))
            torch.use_deterministic_algorithms(True, warn_only=True)
            tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
            model = (
                AutoModelForCausalLM.from_pretrained(
                    directory,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                )
                .to("cpu")
                .eval()
            )
            _check_geometry(model.config, config["model"])
            prepared = [_prepare_case(tokenizer, r, config["prompt"], config["query"]) for r in rows]
            eos = tokenizer.eos_token_id
            eos_ids = (
                {int(v) for v in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())
            )

            for case in prepared:
                for condition in CONDITIONS:
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
                    unscored.append(
                        {
                            "core": core_key,
                            "core_id": prov["core_id"],
                            "case_id": case.case_id,
                            "scenario_id": case.scenario_id,
                            "condition": condition.condition_id,
                            "prediction": prediction,
                            "input_tokens": int(case.input_ids.shape[-1]),
                            "resident_after_prefill": result["accounting"]["resident_entries_after_prefill"],
                            "total_persistent_bytes": result["efficiency"]["architecture_total_persistent_bytes"],
                            "inference_wall_seconds": result["timing"]["inference_wall_seconds"],
                        }
                    )
                    handle.write(json.dumps(unscored[-1], ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    done += 1
                    if done % 32 == 0 or done == total:
                        rate = (time.perf_counter() - started) / done
                        print(
                            f"  {done}/{total} core={core_key} eta={(total-done)*rate/60:.1f} min",
                            flush=True,
                        )
            del model, prepared, tokenizer
            gc.collect()

    _atomic_write_jsonl(output / "predictions.unscored.jsonl", unscored)

    # First answer-key access of the run.
    answer_key = read_answer_key(answer_key_path, case_ids)
    for row in unscored:
        key_row = answer_key[row["case_id"]]
        accepted = {_normalize_answer(v) for v in key_row["accepted_answers"]}
        row["correct"] = _normalize_answer(str(row["prediction"])) in accepted
        row["variant"] = key_row.get("variant")
        row["needle_profile"] = key_row.get("needle_profile")
        row["reference_answer"] = key_row["answer"]
    _atomic_write_jsonl(output / "cases.jsonl", unscored)

    by_core: dict[str, Any] = {}
    for core_key in cores:
        per_condition = {}
        for condition in CONDITIONS:
            sel = [
                r for r in unscored
                if r["core"] == core_key and r["condition"] == condition.condition_id
            ]
            correct = sum(bool(r["correct"]) for r in sel)
            needle = {}
            for profile in sorted({str(r["needle_profile"]) for r in sel}):
                grp = [r for r in sel if str(r["needle_profile"]) == profile]
                needle[profile] = {"cases": len(grp), "correct_cases": sum(bool(r["correct"]) for r in grp)}
            per_condition[condition.condition_id] = {
                "cases": len(sel),
                "correct_cases": correct,
                "exact_accuracy": correct / len(sel) if sel else None,
                "by_needle_profile": needle,
            }
        b0 = {r["case_id"] for r in unscored if r["core"] == core_key and r["condition"] == "B0" and r["correct"]}
        b3 = {r["case_id"] for r in unscored if r["core"] == core_key and r["condition"] == "B3" and r["correct"]}
        per_condition["compression_gap"] = {
            "b0_correct": len(b0),
            "b3_correct": len(b3),
            "lost_to_compression": len(b0 - b3),
            "b0_conditioned_retention": (len(b0 & b3) / len(b0)) if b0 else None,
        }
        per_condition["provenance"] = provenance[core_key]
        by_core[core_key] = per_condition

    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "BOUNDED_EXPLORATORY",
        "registration": "NONE_EXPLORATORY_UNREGISTERED",
        "ledger_row_addressed": "N-009",
        "benchmark_scope": "complete 64-case showcase split",
        "cores_evaluated": list(cores),
        "halo_reference_same_protocol": {"B0": "30/64", "B3": "19/64"},
        "answer_key_opened_only_after_all_inference": True,
        "by_core": by_core,
        "inferential_statistics": None,
        "interpretation_limits": [
            "Each core keeps a separate identity; results are never pooled into a family-wide claim.",
            "Exploratory and unregistered; no frozen endpoint and no advance gate.",
            "Scenario packs are correlated; 64 rows are not 64 independent observations.",
            "Halo's numbers come from a separate run under the same protocol and are compared descriptively.",
            "CPU bf16 local accounting; no deployment or GPU claim.",
        ],
    }
    _atomic_write_json(output / "summary.json", summary)
    _atomic_write_json(
        output / "manifest.json",
        {
            "schema_version": "1.0",
            "run_id": output.name,
            "created_at": started_at,
            "completed_at": utc_now(),
            "classification": "BOUNDED_EXPLORATORY",
            "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
            "seed": int(config["runtime"]["seed"]),
            "cores": provenance,
            "conditions": [c.condition_id for c in CONDITIONS],
            "benchmark": {
                "split": "showcase",
                "case_count": len(case_ids),
                "prompt_sha256": sha256_file(prompt_path),
                "answer_key_sha256_after_inference": sha256_file(answer_key_path),
                "answer_key_join_order": "opened_after_unscored_predictions_were_written",
            },
            "notes": [
                "First answer-level evidence for any published core other than Halo.",
                "Every core's weight hash and CORE.json identity verified before load.",
                "Compression semantics imported unchanged from run_published_core_eval.execute_condition.",
            ],
        },
    )
    print(json.dumps({"output": str(output)}, indent=2))
    return {"output": str(output), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_showcase64_eval.yaml")
    parser.add_argument("--cores", nargs="+", default=list(DEFAULT_CORES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run(args.config, tuple(args.cores), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
