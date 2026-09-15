"""Freeze TRAIN-only integration candidates and honest tokenizer/review records.

No pretrained model, optimization, inference, evaluation prediction or approval.
The original supplied releases are read-only. Output directories are create-only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import importlib.metadata
import os
from pathlib import Path
import sys

from experiments.model_quality_gpu_v1.integrity import canonical, digest, new_dir, read_json, write_new
from .serialization import compile_row
from .sources import import_forge_v1_validator, import_forge_v2, stable_hash, verify_release
from .views import scope_only

QUOTAS = {
    "ledger_replay": {"ANSWER": 4}, "register_machine": {"ANSWER": 4},
    "temporal_route": {"ANSWER": 2, "IMPOSSIBLE": 2},
    "prerequisite_plan": {"ANSWER": 2, "IMPOSSIBLE": 2},
    "deadline_schedule": {"ANSWER": 2, "IMPOSSIBLE": 2},
    "budget_selection": {"ANSWER": 2, "UNKNOWN": 2},
    "evidence_chain": {"ANSWER": 2, "UNKNOWN": 2},
    "continuous_replan": {"ANSWER": 2, "IMPOSSIBLE": 2},
}
PRIMITIVE_QUOTAS = {family: {"EXACT_PRIMITIVE": 4} for family in (
    "deduction", "relational_join", "document_reconciliation", "signed_ledger",
    "program_trace", "dialogue_memory", "source_boundaries", "story_state")}


def read_rows(path: Path):
    from experiments.model_quality_gpu_v1.integrity import decode
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 2**24:
                raise ValueError("source row exceeds bounded reader")
            yield decode(line)


def select_microfit(candidates: list[dict], seed: int = 17, quotas=None) -> list[dict]:
    """Predeclared family/status quotas and length eligibility; no model outcomes."""
    if len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("duplicate candidate world")
    if any(r["source_split"] != "train" for r in candidates):
        raise ValueError("only original TRAIN worlds may enter microfit")
    selected = []
    for family, classes in (QUOTAS if quotas is None else quotas).items():
        for answer_class, count in classes.items():
            eligible = [r for r in candidates if r["family"] == family and r["answer_class"] == answer_class
                        and r["direct_eligible"] and r["gold_verified"]]
            eligible.sort(key=lambda r: digest(["RRP-001-microfit", seed, r["id"]]))
            if len(eligible) < count:
                raise ValueError(f"insufficient eligible TRAIN worlds: {family}/{answer_class}: {len(eligible)}/{count}")
            selected.extend(eligible[:count])
    return sorted(selected, key=lambda r: digest(["RRP-001-order", seed, r["id"]]))


def prepare(v1_archive: Path, v1_root: Path, v2_archive: Path, v2_root: Path,
            source: Path, out: Path, view="full", microfit_source="forge_v2") -> dict:
    directory = new_dir(out)
    try:
        v1 = verify_release(v1_archive, v1_root, "forge_v1")
        v2 = verify_release(v2_archive, v2_root, "forge_v2")
        write_new(directory / "SOURCE_BINDINGS.json", {"forge_v1": v1, "forge_v2": v2})
        modules = import_forge_v2(v2)
        verified_files = modules["validate"].verify_manifest(v2_root, "RELEASE_MANIFEST.json")
        # Tokenizer-only process: avoid importing Torch into a RAM-constrained host.
        if "torch" in sys.modules or "transformers" in sys.modules:
            raise ValueError("run preparation in a fresh tokenizer-only process")
        os.environ.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        if importlib.metadata.version("transformers") != "4.57.1":
            raise ValueError("preparation requires pinned Transformers 4.57.1")
        lock = read_json(Path(__file__).parents[1] / "model_quality_gpu_v1/source_lock.json")
        assets = {name: stable_hash(source / name) for name in lock["assets"] if name != "model.safetensors"}
        if any(sha != lock["assets"][name] for name, sha in assets.items()):
            raise ValueError("tokenizer/config differs from exact Supa source")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
        if not tokenizer.is_fast or tokenizer.eos_token_id != 151645 or "torch" in sys.modules:
            raise ValueError("unqualified tokenizer identity or unwanted model runtime import")
        pilots = list(read_rows(v2_root / "pilot/train_256.jsonl"))
        wanted = {r["id"] for r in pilots}
        metadata = {r["id"]: r for r in read_rows(v2_root / "metadata/train.jsonl.gz") if r["id"] in wanted}
        if len(pilots) != 256 or set(metadata) != wanted:
            raise ValueError("pilot/parent inventory mismatch")
        summaries, compiled = [], {}
        for row in pilots:
            meta = metadata[row["id"]]
            if row["split"] != "train" or meta["split"] != "train" or row["group_id"] != row["id"]:
                raise ValueError("derived or non-TRAIN world is not a microfit parent")
            world = modules["text"].parse_prompt(row["prompt"])
            if digest(row["prompt"]) != meta["prompt_sha256"] or digest(modules["text"].canonical_case(world)) != meta["layout"]["case_sha256"]:
                raise ValueError("visible prompt/world identity mismatch")
            certificate = modules["oracle"].audit_answer(world)
            if certificate != meta["certificate"] or not modules["oracle"].verify_solution(world, meta["reference"], certificate)["correct"]:
                raise ValueError("independent visible-world gold check failed")
            if not modules["oracle"].verify_trace(world, meta["reference"], meta["trace_steps"]):
                raise ValueError("released structured trace failed visible-state replay")
            selected_prompt = row["prompt"]
            view_info = {"variant": "full", "canonical_world_unchanged": True}
            if view == "scope_only":
                selected_prompt, view_info = scope_only(row["prompt"], modules["text"].parse_prompt,
                                                       modules["text"].canonical_case)
            elif view != "full":
                raise ValueError("unregistered input view")
            summary = {"id": row["id"], "parent_id": row["group_id"], "source_split": "train",
                       "family": row["family"], "answer_class": meta["reference"].get("status", "ANSWER"),
                       "case_sha256": meta["layout"]["case_sha256"], "source_prompt_sha256": meta["prompt_sha256"],
                       "prompt_sha256": digest(selected_prompt), "input_view": view_info,
                       "native_source_assets": digest(assets), "gold_verified": True,
                       "human_review": "NOT_PERFORMED", "layout": meta["layout"]["layout"],
                       "source_words": meta["layout"]["input_words"], "trace_steps": len(meta["trace_steps"])}
            for mode in ("direct", "reasoning"):
                prompt = selected_prompt
                if mode == "reasoning":
                    prompt = [{"role": "system", "content": modules["common"].SYSTEM_REASON}, prompt[1]]
                # Measurement limit is explicitly not a qualified model context window.
                measured = compile_row(prompt, meta["reference_text"], tokenizer, mode=mode,
                                       trace=meta["teacher_trace"].strip() if mode == "reasoning" else None,
                                       max_sequence=262144, completion_reserve=128)
                summary[mode + "_tokens"] = {k: measured[k] for k in ("processed_tokens", "supervised_tokens", "reasoning_content_tokens", "final_content_tokens")}
                summary[mode + "_prompt_tokens"] = len(measured["prompt_ids"])
                summary[mode + "_eligible"] = measured["processed_tokens"] <= 2048 and len(measured["prompt_ids"]) + 128 <= 2048
                if mode == "direct":
                    compiled[row["id"]] = {**measured, "max_sequence": 2048, "id": row["id"], "parent_id": row["group_id"],
                                           "source_split": "train", "family": row["family"], "prompt": selected_prompt,
                                           "reference": meta["reference_text"], "world": world, "certificate": certificate}
            summaries.append(summary)
        v1_summaries, v1_compiled = [], {}
        v1_solver = import_forge_v1_validator(v1)
        v1_meta = {r["id"]: r for r in read_rows(v1_root / "pilot/train_512.metadata.jsonl.gz")}
        for row in read_rows(v1_root / "pilot/train_512.jsonl"):
            meta = v1_meta[row["id"]]
            if meta["split"] != "train":
                raise ValueError("v1 pilot is not TRAIN")
            v1_solver.validate_row(row, meta)
            measured = compile_row(row["messages"][:-1], row["messages"][-1]["content"].strip(), tokenizer,
                                   max_sequence=262144, completion_reserve=128)
            item = {"id": row["id"], "parent_id": meta["group_id"], "source_split": meta["split"],
                                 "family": meta["family"], "case_sha256": meta["case_sha256"],
                                 "prompt_tokens": len(measured["prompt_ids"]), "full_tokens": measured["processed_tokens"],
                                 "supervised_tokens": measured["supervised_tokens"], "human_review": "NOT_PERFORMED",
                                 "answer_class": "EXACT_PRIMITIVE" if meta["answer_category"] == "exact" else "PROSE",
                                 "target_status": "UNKNOWN" if meta["reference_answer"] == "UNKNOWN" else "TASK_SPECIFIC_ANSWER",
                                 "gold_verified": True, "source_release": "forge_v1", "input_view": {"variant": "full"},
                                 "direct_eligible": measured["processed_tokens"] <= 2048 and len(measured["prompt_ids"]) + 128 <= 2048,
                                 "direct_tokens": {k: measured[k] for k in ("processed_tokens", "supervised_tokens", "reasoning_content_tokens", "final_content_tokens")},
                                 "gold_validation": "FRESH_INDEPENDENT_VISIBLE_TEXT_SOLVER"}
            v1_summaries.append(item)
            v1_compiled[row["id"]] = {**measured, "max_sequence": 2048, "id": row["id"], "parent_id": meta["group_id"],
                "source_split": "train", "family": meta["family"], "prompt": row["messages"][:-1],
                "reference": meta["reference_answer"], "verifier": "forge_v1_exact_primitive",
                "source_release": "forge_v1", "lineage": item}
        write_new(directory / "TOKENIZER_SURVEY.json", {"scope": "REAL_TOKENIZER_ONLY_NO_NATIVE_MODEL_QUALIFICATION",
                  "tokenizer_assets": assets, "transformers": importlib.metadata.version("transformers"),
                  "tokenizers": importlib.metadata.version("tokenizers"), "torch_imported": "torch" in sys.modules,
                  "forge_v2": summaries, "forge_v1": v1_summaries})
        if microfit_source == "forge_v1_primitives":
            selected = select_microfit(v1_summaries, quotas=PRIMITIVE_QUOTAS)
            selected_compiled, quotas = v1_compiled, PRIMITIVE_QUOTAS
        elif microfit_source == "forge_v2":
            selected = select_microfit(summaries)
            selected_compiled, quotas = compiled, QUOTAS
        else:
            raise ValueError("unregistered microfit source")
        with (directory / "microfit-train.jsonl").open("xb") as stream:
            for row in selected:
                stream.write(canonical({**selected_compiled[row["id"]], "lineage": row}) + b"\n")
        # Privileged world/gold fields above are verifier data. A policy only receives prompt_ids.
        reviews = []
        by_stratum = defaultdict(list)
        for row in summaries:
            by_stratum[(row["family"], row["answer_class"], row["layout"])].append(row)
        for key, group in sorted(by_stratum.items()):
            reviews.append(sorted(group, key=lambda r: digest(["review", r["id"]]))[0])
        review_ids = {r["id"] for r in reviews} | {r["id"] for r in selected}
        with (directory / "REVIEW_PACKET.jsonl").open("xb") as stream:
            if microfit_source == "forge_v1_primitives":
                for row in selected:
                    item = selected_compiled[row["id"]]
                    stream.write(canonical({"id": row["id"], "parent_id": row["parent_id"], "source_split": "train",
                        "family": row["family"], "native_prompt": item["prompt"], "reference": item["reference"],
                        "native_measurement": row, "reviewer": None, "decision": "PENDING", "unresolved_issues": []}) + b"\n")
            for row in pilots:
                if row["id"] in review_ids:
                    meta = metadata[row["id"]]
                    stream.write(canonical({"id": row["id"], "parent_id": row["group_id"], "source_split": "train",
                        "family": row["family"], "source_prompt": row["prompt"],
                        "native_prompt": compiled[row["id"]]["prompt"], "reference": meta["reference"],
                        "trace": meta["teacher_trace"], "native_measurement": next(s for s in summaries if s["id"] == row["id"]),
                        "reviewer": None, "decision": "PENDING", "unresolved_issues": []}) + b"\n")
        result = {"experiment": "RRP-001", "status": "PREPARED_REVIEW_AND_EXECUTION_BUDGET_PENDING",
                  "preparation_code": {p.relative_to(Path(__file__).parent).as_posix(): stable_hash(p)
                      for p in sorted(Path(__file__).parent.rglob("*.py"))},
                  "python_runtime": sys.version,
                  "evaluators": {"forge_v2": v2["files"]["forge_rl/rewards.py"]["sha256"],
                                 "forge_v1_visible_text_solver": v1["files"]["scripts/verify.py"]["sha256"]},
                  "source_releases": {"v1": v1["archive_sha256"], "v2": v2["archive_sha256"]},
                  "v2_release_manifest_files_verified": verified_files, "v2_train_worlds_gold_and_trace_checked": len(summaries),
                  "v1_train_worlds_gold_checked": len(v1_summaries), "microfit_source": microfit_source,
                  "selected_worlds": len(selected), "selection_seed": 17, "order_seed": 17, "quotas": quotas,
                  "input_view": "full" if microfit_source == "forge_v1_primitives" else view,
                  "v2_survey_view": view, "independent_world_count": len(selected),
                  "policy_input_fields": ["prompt_ids"], "teacher_forced_training_fields": ["ids", "labels"],
                  "verifier_only_fields": ["world", "certificate", "reference", "lineage"],
                  "selected_ids": [r["id"] for r in selected],
                  "processed_tokens_per_exposure": sum(r["direct_tokens"]["processed_tokens"] for r in selected),
                  "supervised_tokens_per_exposure": sum(r["direct_tokens"]["supervised_tokens"] for r in selected),
                  "review_packet_worlds": len(review_ids), "review_scope": "All selected worlds plus pilot family/status/layout strata; longer tiers require later packets",
                  "files": {n: stable_hash(directory / n) for n in ("SOURCE_BINDINGS.json", "TOKENIZER_SURVEY.json", "microfit-train.jsonl", "REVIEW_PACKET.jsonl")},
                  "source_archive_unchanged": True, "model_calls": 0, "optimizer_steps": 0,
                  "human_review": "NOT_PERFORMED", "native_model_qualification": "NOT_RUN",
                  "confirmation": "NOT_ESTABLISHED", "training_permission": False}
        write_new(directory / "DATA_CONTRACT.json", result)
        return result
    except Exception as exc:
        write_new(directory / "failure.json", {"type": type(exc).__name__, "message": str(exc), "model_calls": 0,
                  "training_permission": False, "partial_outputs_preserved": True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v1-archive", "v1-root", "v2-archive", "v2-root", "source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--view", choices=("full", "scope_only"), default="full")
    parser.add_argument("--microfit-source", choices=("forge_v2", "forge_v1_primitives"), default="forge_v2")
    args = parser.parse_args(argv)
    result = prepare(**vars(args))
    print(canonical(result).decode())


if __name__ == "__main__":
    main()
