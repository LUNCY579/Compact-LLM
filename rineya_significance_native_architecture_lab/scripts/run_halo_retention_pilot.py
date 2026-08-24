#!/usr/bin/env python3
"""Run exploratory Halo B2/B3/B4 on B0-eligible calibration clusters."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.benchmarks import (  # noqa: E402
    FrozenEvidenceAnnotation,
    score_posthoc_evidence_retention,
)
from rineya_significance_lab.evaluation import (  # noqa: E402
    clustered_paired_analysis,
    compressed_retention_failure_labels,
    score_evidence_retention_endpoint,
)
from rineya_significance_lab.manifest import atomic_write_json, sha256_file, utc_now  # noqa: E402


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner helper: {path}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _load(path: Path) -> dict[str, object]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("retention pilot config must be a mapping")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(config_path: Path) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mini = _module("halo_mini_helpers", LAB_ROOT / "scripts" / "run_published_core_eval.py")
    capability = _module("halo_capability_helpers", LAB_ROOT / "scripts" / "run_halo_capability.py")
    started_at = utc_now()
    config = _load(config_path)
    registration: dict[str, object] | None = None
    if "registration" in config:
        registration_id = str(dict(config["registration"])["id"])
        if registration_id != "halo-endpoint-validation-v2-registered":
            raise RuntimeError(f"unsupported registration ID: {registration_id}")
        registration_path = (
            LAB_ROOT
            / "results"
            / "registrations"
            / "halo_endpoint_validation_v2.json"
        )
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        if registration["registration_id"] != registration_id:
            raise RuntimeError("registration artifact ID mismatch")
        if registration["registered_config_sha256"] != sha256_file(config_path):
            raise RuntimeError("registered config hash mismatch")
        if list(registration["selected_case_ids"]) != list(
            dict(config["selection"])["selected_case_ids"]
        ):
            raise RuntimeError("registered selected case IDs changed")
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    resolve = lambda key: (base / str(paths[key])).resolve()  # noqa: E731
    if registration is not None:
        source_hashes = dict(registration["source_hashes"])
        current_sources = {
            "dataset_manifest": sha256_file(resolve("inference").parent / "manifest.json"),
            "inference": sha256_file(resolve("inference")),
            "evaluation": sha256_file(resolve("evaluation")),
            "capability_manifest": sha256_file(resolve("capability_cases").parent / "manifest.json"),
            "capability_summary": sha256_file(resolve("capability_cases").parent / "summary.json"),
            "capability_cases": sha256_file(resolve("capability_cases")),
        }
        if current_sources != source_hashes:
            raise RuntimeError("registered source artifact hash mismatch")
    output = resolve("output_root")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty pilot output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_path = resolve("model")
    expected = dict(dict(config["model"])["expected_hashes"])
    observed = {
        "weights": sha256_file(model_path / "model.safetensors"),
        "tokenizer": sha256_file(model_path / "tokenizer.json"),
        "config": sha256_file(model_path / "config.json"),
    }
    if observed != expected:
        raise RuntimeError("Halo checkpoint hash mismatch")
    capability_rows = _jsonl(resolve("capability_cases"))
    selection_config = dict(config["selection"])
    wanted_families = set(selection_config["families"])
    capability_by_id = {str(row["case_id"]): row for row in capability_rows}
    registered_ids = selection_config.get("selected_case_ids")
    if registered_ids is not None:
        selected_ids = {str(value) for value in registered_ids}
        if len(selected_ids) != len(list(registered_ids)):
            raise RuntimeError("registered selected_case_ids contain duplicates")
        for case_id in selected_ids:
            row = capability_by_id.get(case_id)
            if row is None or not bool(row.get("correct")):
                raise RuntimeError(f"registered case is not B0-correct: {case_id}")
            if str(row.get("task_family")) not in wanted_families:
                raise RuntimeError(f"registered case has an undeclared family: {case_id}")
    else:
        maximum = int(selection_config["maximum_eligible_clusters_per_family"])
        eligible_by_family: defaultdict[str, list[str]] = defaultdict(list)
        for row in capability_rows:
            if row.get("correct") and row.get("task_family") in wanted_families:
                eligible_by_family[str(row["task_family"])].append(str(row["case_id"]))
        selected_ids = {
            case_id
            for family in sorted(wanted_families)
            for case_id in sorted(eligible_by_family[family])[:maximum]
        }
    minimum_clusters = int(dict(config.get("analysis", {})).get("minimum_independent_clusters", 8))
    if len(selected_ids) < minimum_clusters:
        raise RuntimeError(
            f"retention run requires at least {minimum_clusters} B0-eligible clusters"
        )
    inference = {str(row["case_id"]): row for row in _jsonl(resolve("inference")) if row["case_id"] in selected_ids}
    if set(inference) != selected_ids:
        raise RuntimeError("inference view is missing one or more registered cases")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    torch.manual_seed(int(config["runtime"]["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    conditions = [
        mini.Condition(str(row["id"]), str(row["policy"]), float(row["exponent"]), True)
        for row in config["conditions"]
    ]
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    positions_dir = output / "resident_positions"
    positions_dir.mkdir()
    unscored: list[dict[str, object]] = []
    for case_id in sorted(selected_ids):
        row = inference[case_id]
        input_ids = tokenizer(str(row["prompt"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
        query_ids = tokenizer(str(row["question"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
        if capability._token_hash(input_ids) != str(row["token_ids_sha256"]):
            raise RuntimeError(f"frozen prompt token hash mismatch for {case_id}")
        if int(query_ids.shape[-1]) > int(config["query"]["max_tokens"]):
            raise RuntimeError(f"query exceeds registered token limit for {case_id}")
        prepared = mini.PreparedCase(
            case_id=case_id,
            scenario_id=str(row["cluster_id"]),
            metadata={"family": row["family"]},
            input_ids=input_ids,
            context_ids=input_ids,
            query_ids=query_ids,
            prompt_text_sha256=str(row["prompt_sha256"]),
            input_token_sha256=str(row["token_ids_sha256"]),
            context_token_sha256=str(row["token_ids_sha256"]),
            query_token_sha256=mini._token_sha256(query_ids),
        )
        for condition in conditions:
            result = mini.execute_condition(
                model,
                prepared,
                condition,
                cache_config=config["cache"],
                runtime_config={
                    "max_new_tokens": config["runtime"]["max_new_tokens"],
                    "verify_query_rollback": True,
                },
                eos_token_ids=eos_ids,
                return_resident_positions=True,
            )
            resident = result.pop("_resident_positions")
            position_name = f"{case_id}__{condition.condition_id}.npz"
            np.savez_compressed(
                positions_dir / position_name,
                **{f"layer_{index:03d}": value for index, value in enumerate(resident)},
            )
            prediction = tokenizer.decode(result.pop("generated_token_ids"), skip_special_tokens=True).strip()
            unscored.append(
                {
                    "case_id": case_id,
                    "cluster_id": row["cluster_id"],
                    "family": row["family"],
                    "condition": condition.condition_id,
                    "prediction": prediction,
                    "resident_positions": f"resident_positions/{position_name}",
                    **result,
                }
            )
    unscored_path = output / "predictions.unscored.jsonl"
    unscored_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in unscored),
        encoding="utf-8",
    )
    # Evaluation view is opened only after all predictions and resident states are durable.
    evaluation = {str(row["case_id"]): row for row in _jsonl(resolve("evaluation"))}
    scored: list[dict[str, object]] = []
    for row in unscored:
        key = evaluation[str(row["case_id"])]
        annotation = FrozenEvidenceAnnotation.from_dict(key["evidence_annotation"])
        with np.load(output / str(row["resident_positions"]), allow_pickle=False) as arrays:
            resident = [arrays[name].tolist() for name in sorted(arrays.files)]
        retention = score_posthoc_evidence_retention(annotation, resident)
        evidence = retention.to_dict()
        endpoint = score_evidence_retention_endpoint(retention).to_dict()
        correct = capability._task_correct(
            str(row["family"]),
            str(row["prediction"]),
            [str(value) for value in key["accepted_answers"]],
        )
        scored.append(
            {
                "schema_version": "1.0",
                "run_id": output.name,
                "case_id": row["case_id"],
                "scenario_id": row["cluster_id"],
                "condition": row["condition"],
                "task_family": row["family"],
                "generator_seed": None,
                "prompt_sha256": annotation.prompt_sha256,
                "context_tokens": annotation.token_count,
                "evidence_depth": 0.5,
                "chunk_size": config["cache"]["compression_chunk_tokens"],
                "status": "OK",
                "prediction": row["prediction"],
                "correct": correct,
                "full_cache_correct": True,
                "compression_conditioned_eligible": True,
                "evidence": {
                    "required_count": annotation.required_token_count,
                    "surviving_count_final": round(
                        evidence["minimum_per_layer_required_token_availability_fraction"]
                        * annotation.required_token_count
                    ),
                    "all_required_survived": evidence[
                        "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer"
                    ],
                    **evidence,
                    **endpoint,
                },
                "efficiency": row["efficiency"],
                "failure_labels": compressed_retention_failure_labels(
                    correct=correct,
                    evidence=evidence,
                ),
                "failure_label_scope": "descriptive_final_retained_state_not_causal_attribution",
                "trace_ids": [row["resident_positions"]],
                "error": None,
                "accepted_answers": key["accepted_answers"],
            }
        )
    cases_path = output / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored),
        encoding="utf-8",
    )
    by_condition: dict[str, dict[str, object]] = {}
    condition_ids = [condition.condition_id for condition in conditions]
    for condition in condition_ids:
        rows = [row for row in scored if row["condition"] == condition]
        by_condition[condition] = {
            "clusters": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "strict_evidence_survival": sum(bool(row["evidence"]["all_required_survived"]) for row in rows),
            "mean_layer_fact_exact_completion_rate": sum(
                float(row["evidence"]["layer_fact_exact_completion_rate"])
                for row in rows
            )
            / len(rows),
            "mean_persistent_bytes": sum(row["efficiency"]["architecture_total_persistent_bytes"] for row in rows) / len(rows),
        }
    paired = {}
    for left, right in (("B2", "B3"), ("B4", "B3")):
        if left not in condition_ids or right not in condition_ids:
            continue
        left_rows = {row["case_id"]: row for row in scored if row["condition"] == left}
        right_rows = {row["case_id"]: row for row in scored if row["condition"] == right}
        paired[f"{right}_minus_{left}"] = {
            "answer_wins": sum(not left_rows[key]["correct"] and right_rows[key]["correct"] for key in left_rows),
            "answer_losses": sum(left_rows[key]["correct"] and not right_rows[key]["correct"] for key in left_rows),
            "evidence_wins": sum(
                not left_rows[key]["evidence"]["all_required_survived"]
                and right_rows[key]["evidence"]["all_required_survived"]
                for key in left_rows
            ),
            "evidence_losses": sum(
                left_rows[key]["evidence"]["all_required_survived"]
                and not right_rows[key]["evidence"]["all_required_survived"]
                for key in left_rows
            ),
        }
    analysis_config = dict(config.get("analysis", {}))
    statistics: dict[str, object] = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": (
            "PROSPECTIVE_REGISTERED_ENDPOINT_VALIDATION"
            if "registration" in config
            else "EXPLORATORY_POSTHOC"
        ),
        "independence_unit": "cluster_id",
        "primary_endpoint": "layer_fact_exact_completion_rate",
        "comparisons": {},
    }
    if "B2" in condition_ids and "B3" in condition_ids:
        left_rows = {
            str(row["case_id"]): row for row in scored if row["condition"] == "B2"
        }
        right_rows = {
            str(row["case_id"]): row for row in scored if row["condition"] == "B3"
        }
        cluster_ids = sorted(left_rows)
        primary_result = clustered_paired_analysis(
            cluster_ids,
            [
                float(left_rows[key]["evidence"]["layer_fact_exact_completion_rate"])
                for key in cluster_ids
            ],
            [
                float(right_rows[key]["evidence"]["layer_fact_exact_completion_rate"])
                for key in cluster_ids
            ],
            minimum_clusters=int(analysis_config.get("minimum_independent_clusters", 8)),
            confidence=float(analysis_config.get("confidence", 0.95)),
            bootstrap_samples=int(analysis_config.get("bootstrap_samples", 10000)),
            randomization_samples=int(analysis_config.get("randomization_samples", 100000)),
            seed=int(config["runtime"]["seed"]),
        )
        answer_result = clustered_paired_analysis(
            cluster_ids,
            [float(bool(left_rows[key]["correct"])) for key in cluster_ids],
            [float(bool(right_rows[key]["correct"])) for key in cluster_ids],
            minimum_clusters=int(analysis_config.get("minimum_independent_clusters", 8)),
            confidence=float(analysis_config.get("confidence", 0.95)),
            bootstrap_samples=int(analysis_config.get("bootstrap_samples", 10000)),
            randomization_samples=int(analysis_config.get("randomization_samples", 100000)),
            seed=int(config["runtime"]["seed"]) + 1,
        )
        statistics["comparisons"] = {
            "B3_minus_B2_primary_endpoint": asdict(primary_result),
            "B3_minus_B2_answer_correct": asdict(answer_result),
        }
    atomic_write_json(output / "statistics.json", statistics)
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": (
            "PROSPECTIVE_REGISTERED_ENDPOINT_VALIDATION"
            if "registration" in config
            else "EXPLORATORY_RETENTION_PILOT"
        ),
        "selected_B0_eligible_clusters": len(selected_ids),
        "families": sorted(wanted_families),
        "by_condition": by_condition,
        "paired_descriptive": paired,
        "inferential_statistics": statistics["comparisons"],
        "claim_limit": (
            "Locally registered prospective validation; modest 12-cluster scope, not publication preregistration."
            if "registration" in config
            else "B0-selected calibration pilot; not confirmatory or population evidence."
        ),
    }
    atomic_write_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "status": "COMPLETE",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
        "created_at": started_at,
        "completed_at": utc_now(),
        "repositories": {
            "chatbubble": {"remote": "https://github.com/LUNCY579/ChatBubble.git", "commit": "03777703dd8587f1248d9a72115eafb9356d50b8", "dirty": True},
            "source": {
                "remote": "https://github.com/LUNCY579/RineyaBeta.git",
                "implementation": {"ref": "refs/heads/main", "commit": "9e00be74098878290dc2b0d54cbffee4652c6088", "path": "scripts/kv/significance.py", "blob": "30ee29724dce52de5a288f6421123fb718240541"},
                "paper": {"ref": "refs/heads/r2-healing-harness", "commit": "1664f54de0bbdfde45e311b75bbf3038dbfffe42", "path": "paper/significance_matrix/main.tex", "blob": "b83c7f37d871f1ddb59a00ccf521c89d84b1c3e3"},
            },
        },
        "experiment": {"name": config["experiment"]["name"], "compute_tier": "B", "architecture_variant": "B2/B3/B4 exact-evidence validation", "benchmark": Path(str(paths["inference"])).parent.name, "seed": config["runtime"]["seed"], "query_conditioned": True, "registration_id": dict(config.get("registration", {})).get("id")},
        "config": {"resolved_path": f"configs/{config_path.name}", "sha256": sha256_file(config_path)},
        "model": {"status": "VERIFIED", "id": "rineya.core.halo.v1.2.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-v2-1e4", "path": "Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze", "architecture": "Qwen3ForCausalLM", "hashes": {**observed, "chat_template": None}},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": __import__("transformers").__version__, "device": "cpu", "precision": "bfloat16", "hardware": None},
        "budget": {"retained_history_entries": config["cache"]["retained_history_budget"], "active_chunk_entries": config["cache"]["compression_chunk_tokens"], "matched_across_conditions": True},
        "artifacts": {"predictions": "predictions.unscored.jsonl", "cases": "cases.jsonl", "summary": "summary.json", "statistics": "statistics.json", "resident_positions": "resident_positions/"},
        "notes": (["Locally registered prospective endpoint validation; registration/config hashes are recorded separately."] if "registration" in config else ["Exploratory B0-eligible pilot."]),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_retention_pilot.yaml")
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
