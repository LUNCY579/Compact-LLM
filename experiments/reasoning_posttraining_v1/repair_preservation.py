"""Create a corrected DEV derivative and evidence-removal probes, preserving old records.

No prediction is consulted during admission. Procedural golds are solved from the
displayed input; unresolved/disagreeing golds are quarantined, never quietly relabeled.
Nonprocedural golds retain their original, explicitly limited verification status.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
from pathlib import Path
import sys

from experiments.model_quality_gpu_v1.integrity import digest, write_new
from . import grounded_replay as gr, preserve, quarantine
from .prepare import read_rows
from .sources import stable_hash
from . import budget

EXPERIMENT = "RRP-005-PRESERVATION-BASELINE"
LIMITS = {
    "total_owned_work_wall_seconds": 4500,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 3600,
    "processed_training_tokens_all_attempts": 0,
    "evaluation_and_qualification_model_calls": 1000,
    "generated_tokens_all_candidates": 65536,
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 256 * 2**20,
    "adaptive_model_selection_rounds": 0,
    "model_worker_launches": 2,
    "retries": 1,
    "cloud_or_teacher_spend": 0,
}
EXCLUDES = ("broad_SFT", "RL", "larger_teacher", "cloud", "release", "architectural_change", "training")
budget.register(EXPERIMENT, ceiling=LIMITS, phases=("baseline",), excludes=EXCLUDES)

# Quantitative amendment after baseline-001 terminated with a resource refusal.
# Old journal remains immutable: 423 s / 554 calls / 35,072 reserved tokens spent.
# Only call/token capacity and one launch are added; wall-time is the OLD remainder.
AMENDED_EXPERIMENT = "RRP-005-PRESERVATION-BASELINE-A2"
AMENDED_LIMITS = {
    **LIMITS, "total_owned_work_wall_seconds": 4077,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 3177,
    "evaluation_and_qualification_model_calls": 1446,
    "generated_tokens_all_candidates": 96000,
    "new_disk_bytes": 264241152,
    "model_worker_launches": 2, "retries": 1,
}
budget.register(AMENDED_EXPERIMENT, ceiling=AMENDED_LIMITS, phases=("baseline",), excludes=EXCLUDES)

DELTA_EXPERIMENT = "RRP-005-PRESERVATION-DELTA-A3"
DELTA_LIMITS = {**AMENDED_LIMITS,
    "total_owned_work_wall_seconds": 3490,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 2591,
    "evaluation_and_qualification_model_calls": 1131,
    "generated_tokens_all_candidates": 76608,
    "new_disk_bytes": 255007150,
    "model_worker_launches": 2, "retries": 1,
}
WAIT_POLICY = {"total_seconds": 240, "episode_seconds": 90, "poll_seconds": 2}
budget.register(DELTA_EXPERIMENT, ceiling=DELTA_LIMITS, phases=("baseline",), excludes=EXCLUDES)


def validate_scope(args, plan):
    if plan.get("experiment") == EXPERIMENT:
        raise ValueError("the nonstreamed RRP-005 attempt is retired; use its separately bound A2 successor")
    if (args.command != "baseline" or not getattr(args, "bank", None)
            or getattr(args, "heldout_bank", None)):
        raise ValueError("RRP-005 is one registered baseline measurement, not training")
    wanted = plan.get("baseline_protocol", {})
    if (stable_hash(Path(args.bank)) != wanted.get("bank_sha256") or
            getattr(args, "max_sequence", None) != wanted.get("max_sequence") or
            getattr(args, "max_new_tokens", None) != wanted.get("max_new_tokens")):
        raise ValueError("baseline bank or inference protocol differs from frozen RRP-005 plan")
    if plan.get("experiment") in (AMENDED_EXPERIMENT, DELTA_EXPERIMENT):
        if wanted.get("prediction_storage") != "create_only_hash_chained_rows_v1" or wanted.get("cache_policy") != "release_unused_between_cases":
            raise ValueError("amended baseline requires durable rows and the frozen cache policy")
    if plan.get("experiment") == DELTA_EXPERIMENT and wanted.get("resource_wait") != WAIT_POLICY:
        raise ValueError("delta baseline requires the exact bounded resource-wait policy")


def corrected_rows(old_bank, sources):
    admitted, quarantined, probes = [], [], []
    stats = collections.Counter()
    for original in old_bank:
        source = sources[original["sub_bank"]][original["source_id"]]
        if source["messages"][-1]["content"] != original["reference_text"]:
            raise ValueError("source target changed since the previous bank")
        row = copy.deepcopy(original)
        if row["sub_bank"] == "procedural_rag":
            visible = "\n".join(m["content"] for m in original["prompt"])
            stats["procedural_rows"] += 1
            stats["procedural_rule_omitted"] += int(all(d["text"] not in visible for d in source["procedures"]))
            stats["procedural_evidence_documents_omitted"] += sum(d["text"] not in visible for d in source["evidence"])
            try:
                audit = gr.audit_source(source)
            except ValueError as exc:
                quarantined.append({"id": row["id"], "source_id": row["source_id"],
                                    "reason": str(exc), "action": "EXCLUDE_NOT_RELABEL"})
                continue
            if not audit["gold_matches"] or not audit["decision_matches"]:
                quarantined.append({"id": row["id"], "source_id": row["source_id"],
                                    "reason": "declared gold/decision differs from visible-result solver",
                                    "audit": {k: v for k, v in audit.items() if k != "prompt"},
                                    "action": "EXCLUDE_NOT_RELABEL"})
                continue
            row["prompt"] = audit["prompt"]
            row["gold_verification"] = "INDEPENDENT_VISIBLE_PROCEDURE_SOLVER"
            row["visible_solver"] = audit["solver"]
            # Match question, procedure, record locator and all phrasing exactly.
            # Only the one evidence document is removed in the negative sibling.
            if row["answerable"] and len(source["evidence"]) == 1:
                pair = []
                for missing in (False, True):
                    derived = copy.deepcopy(row)
                    derived["id"] = "RRP5-probe:" + row["source_id"] + (":missing" if missing else ":present")
                    derived["pair_id"] = "RRP5-pair:" + source["topic_id"] + ":" + row["source_id"]
                    derived["group_id"] = derived["pair_id"]
                    derived["parent_topic_id"] = source["topic_id"]
                    if missing:
                        without = copy.deepcopy(source)
                        without["evidence"] = []
                        derived["prompt"] = gr.render_procedure(without)
                        derived["reference_text"] = gr.MISSING
                        derived["answerable"] = False
                        derived["decision"] = "abstain"
                        derived["required_response_kind"] = "NO_EVIDENCE"
                        derived["abstention_cause"] = "decisive_evidence_removed"
                        derived["reference_aliases"] = []
                    if gr.solve_prompt(derived["prompt"])["answer"] != derived["reference_text"]:
                        raise ValueError("derived decisive-evidence probe failed its solver")
                    derived["cluster_id"] = digest(["RRP5", source["topic_id"]])[:16]
                    derived["instrument_role"] = "DIAGNOSTIC_NOT_TRAIN_OR_SELECTION"
                    pair.append(derived)
                probes.extend(pair)
        else:
            row["prompt"] = gr.conversation_prompt(source)
            row["gold_verification"] = "SOURCE_TARGET_PRESERVED_NOT_INDEPENDENTLY_SOLVED"
            if row["prompt"] != original["prompt"]:
                stats["conversation_histories_restored"] += 1
        row["serialization"] = "dialogue_final_response_v1"
        row["source_bank"] = "RRP2-PRES-DEV"
        row["instrument_role"] = "DEV_ONLY_NOT_CONFIRMATION"
        # Preserve the old conservative clustering, despite the context expansion.
        # Changing row rendering is not a license to count correlated templates anew.
        admitted.append(row)
    return admitted, quarantined, probes, dict(stats)


def audit_replay_sources(root):
    path = root / preserve.V23 / "procedural_rag/splits/discovery.reviewed.jsonl"
    counts, bad = collections.Counter(), []
    for source in read_rows(path):
        if source["intended_split"] != "discovery":
            raise ValueError("unexpected split in optimization source")
        try:
            audit = gr.audit_source(source)
            good = audit["gold_matches"] and audit["decision_matches"]
            reason = "same-result conflict target" if not good else "verified"
        except ValueError as exc:
            good, reason = False, str(exc)
        counts["verified" if good else "quarantined"] += 1
        if not good:
            bad.append({"id": source["id"], "parent_topic_id": source["topic_id"],
                        "family": source["procedure_family"], "reason": reason,
                        "action": "EXCLUDE_NOT_RELABEL"})
    return {"source": str(path), "sha256": stable_hash(path), "counts": dict(counts),
            "quarantine": bad, "used_for_optimization": False}


def tokenizer_only(source):
    if "torch" in sys.modules:
        raise ValueError("preparation must be a fresh tokenizer-only process")
    os.environ.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import importlib.metadata
    if importlib.metadata.version("transformers") != "4.57.1":
        raise ValueError("pinned tokenizer runtime required")
    lock = json.loads((Path(__file__).parents[1] / "model_quality_gpu_v1/source_lock.json").read_text())
    for name, expected in lock["assets"].items():
        if name != "model.safetensors" and stable_hash(source / name) != expected:
            raise ValueError("tokenizer source mismatch")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
    if "torch" in sys.modules or not tok.is_fast or tok.eos_token_id != 151645:
        raise ValueError("tokenizer identity or runtime changed")
    return tok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("correction evidence is create-only")
    records = root / "research/reasoning_posttraining_v1"
    registry = quarantine.load(records / "QUARANTINE.json")
    old_path = records / "evidence/inspection-20260915/RRP2_PRES_DEV_BANK.jsonl"
    build_path = records / "evidence/inspection-20260915/RRP2_PRES_DEV_BUILD.json"
    original_report = json.loads(build_path.read_text())
    old = list(read_rows(old_path))
    sources = {}
    for sub_bank, info in original_report["sources"].items():
        quarantine.assert_readable(registry, info["path"])
        path = root / info["path"]
        if stable_hash(path) != info["sha256"]:
            raise ValueError("source changed since the bank was built")
        wanted = {r["source_id"] for r in old if r["sub_bank"] == sub_bank}
        sources[sub_bank] = {r["id"]: r for r in read_rows(path) if r["id"] in wanted}
        if set(sources[sub_bank]) != wanted:
            raise ValueError("missing source row")
    bank, bad, probes, omitted = corrected_rows(old, sources)
    tok = tokenizer_only(args.source)
    from .instrument import compile_bank
    compilation = {}
    for name, rows in (("RRP5_PRES_DEV", bank), ("RRP5_EVIDENCE_PAIRS", probes)):
        compiled, rejected = compile_bank(rows, tok, max_sequence=2048, completion_reserve=128,
            reference_field="reference_text", family_field="sub_bank", stratum_field="decision",
            serialization="dialogue_final_response_v1")
        compilation[name] = {"rows": len(rows), "native_compiled": len(compiled),
                              "rejected": rejected, "max_processed_tokens": max(len(r["ids"]) for r in compiled),
                              "max_supervised_tokens": max(r["supervised_tokens"] for r in compiled)}
    replay = audit_replay_sources(root)
    negative_probes = [p for p in probes if not p["answerable"]]
    # Each positive probe is already evaluated as a main-bank row. Reuse it only
    # after exact prompt+target equality is proven; never count it twice as evidence.
    by_source = {r["source_id"]: r for r in bank}
    aliases = {}
    for p in probes:
        if p["answerable"]:
            original = by_source[p["source_id"]]
            if (p["prompt"], p["reference_text"]) != (original["prompt"], original["reference_text"]):
                raise ValueError("positive probe cannot reuse a different model input")
            aliases[p["id"]] = original["id"]
    inputs = bank + negative_probes
    report = {"record": "RRP005_CONTEXT_AND_GOLD_CORRECTION", "model_calls": 0,
              "invalidated_bank_sha256": stable_hash(old_path), "original_rows": len(old),
              "corrected_rows": len(bank), "quarantined_rows": bad, "omissions": omitted,
              "corrected_composition": dict(collections.Counter(r["sub_bank"] for r in bank)),
              "compilation": compilation, "source_files": original_report["sources"],
              "human_review": "NOT_PERFORMED", "confirmation": "NOT_ESTABLISHED",
              "positive_control_limit": "Only procedural rows/probes have independent visible-input solver verification. Other targets remain source-reported; gold self-scoring verifies the scorer, not question answerability.",
              "prior_baseline_reusable": False, "quality_claim": False,
              "baseline_input_rows": len(inputs), "positive_probe_prediction_aliases": aliases,
              "analysis_partitions": "Main bank and paired diagnostics MUST be reported separately; pooled input accuracy has no interpretation."}
    out.mkdir(parents=True)
    for name, rows in (("RRP5_PRES_DEV.jsonl", bank), ("RRP5_EVIDENCE_PAIRS.jsonl", probes),
                       ("RRP5_BASELINE_INPUTS.jsonl", inputs)):
        with (out / name).open("x", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    report["bank_files"] = {name: stable_hash(out / name) for name in
                            ("RRP5_PRES_DEV.jsonl", "RRP5_EVIDENCE_PAIRS.jsonl", "RRP5_BASELINE_INPUTS.jsonl")}
    write_new(out / "CORRECTION.json", report)
    write_new(out / "REPLAY_ADMISSION_AUDIT.json", replay)
    print(json.dumps({"corrected_rows": len(bank), "quarantined_rows": len(bad),
                      "paired_probe_rows": len(probes), "omissions": omitted,
                      "compilation": compilation, "replay_audit": replay["counts"]}, indent=2))


if __name__ == "__main__":
    main()
