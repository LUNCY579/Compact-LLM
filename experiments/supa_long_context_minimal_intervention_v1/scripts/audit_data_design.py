#!/usr/bin/env python3
"""CPU-only, model-free audit of cached Phase 0 data and Phase 1 curriculum.

Run from repository root with ../Chat-Bubble/.venv/Scripts/python.exe.
Never trains, loads a model, or changes canonical files. Scores are recomputed
from raw predictions in per_example.jsonl, with one-based evidence line numbers.
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
sys.path.insert(0, str(LAB / "src"))
from transformers import AutoTokenizer
import transformers
from slcmi.context_selection import anchor_text, assemble, chunk_context
from slcmi.evaluation import normalize, score
from slcmi.preservation import PRESERVATION_CLASSES, score_preservation
from slcmi.provenance import CANONICAL_CORE, CORES, DEMO_SUITE, checkpoint_files, sha256
from slcmi.suites import SUITES
from slcmi.training_data import LCR_SFT, load_packs, sample_curriculum, short_target, tokenize_example

OUT = LAB / "results" / "audit_20260905" / "data_design.json"


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [{**json.loads(line), "_source_line": n} for n, line in enumerate(handle, 1) if line.strip()]


def loc(path):
    return str(Path(path).resolve()).replace("\\", "/")


def proportion(hits, total):
    return {"n": total, "count": hits, "percent": 100 * hits / total if total else None}


def overlaps(left, right):
    return {field: len({r[field] for r in left} & {r[field] for r in right})
            for field in ("context", "question", "title", "scenario_id", "id")}


def dist(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def evidence(rows):
    return [{"example_id": r.get("example_id", r.get("id")), "line": r["_source_line"]} for r in rows]


def constant_values(path):
    values = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    return values


def main():
    before = {name: checkpoint_files(core["path"]) for name, core in CORES.items()}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
              "transformers": transformers.__version__, "model_loads": 0, "gpu_used": False,
              "command": "../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/audit_data_design.py"}
    lcr = read(LCR_SFT)
    demo = read(DEMO_SUITE)
    lcr_by_id = {r["id"]: r for r in lcr}
    dev = [r for r in lcr if r["split"] == "dev"]
    train = [r for r in lcr if r["split"] == "train"]
    answerable = [r for r in lcr if r["answerable"]]
    unique_counts = Counter(len([m for m in r["must_include"] if m.lower() not in r["question"].lower()]) for r in answerable)
    targets = [short_target(r) for r in answerable]
    report["short_target"] = {"status": "CONFIRMED" if unique_counts == {1: 6000} else "REFUTED",
                              "source": loc(LCR_SFT), "answerable_n": len(answerable),
                              "residual_span_count_distribution": unique_counts,
                              "targets_present_in_original_answer": sum(t.lower() in r["answer"].lower() for t, r in zip(targets, answerable))}
    report["disjointness"] = {"lcr_all_vs_demo": overlaps(lcr, demo), "lcr_train_vs_lcr_dev": overlaps(train, dev),
                              "intended_use_demo": dict(Counter(r["intended_use"] for r in demo)),
                              "intended_use_lcr": dict(Counter(r["intended_use"] for r in lcr)),
                              "n_lcr": len(lcr), "n_train": len(train), "n_dev": len(dev), "n_demo": len(demo)}
    tokenizer = AutoTokenizer.from_pretrained(CANONICAL_CORE, trust_remote_code=True, local_files_only=True)
    count = lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)
    report["oracle"] = {}
    report["context_format"] = {}
    records_by_suite = {}
    for suite_key, stage in (("demo", "phase0_eval"), ("lcr_dev", "phase0_eval_lcr_dev")):
        source = LAB / "results" / stage / "per_example.jsonl"
        records = read(source)
        records_by_suite[suite_key] = records
        raw_rows = SUITES[suite_key].loader(False)
        by_id = {r["id"]: r for r in raw_rows}
        eligible_ids = {r["example_id"] for r in records if r["condition"] == "D_oracle"}
        subsets = {"evaluated_answerable": [r for r in raw_rows if r["answerable"]],
                   "oracle_eligible": [r for r in raw_rows if r["id"] in eligible_ids]}
        if suite_key == "lcr_dev":
            subsets["full_dev_answerable"] = [r for r in dev if r["answerable"]]
        oracle_stats = {"per_example_source": loc(source)}
        for name, rows in subsets.items():
            contains = lambda r: any((short_target(r) if suite_key == "lcr_dev" else r["answer"]).lower() in f.lower() for f in r["supporting_facts"])
            facts = [(r, f) for r in rows for f in r["supporting_facts"]]
            oracle_stats[name] = {"gold_span_inside_a_fact_case_insensitive": proportion(sum(contains(r) for r in rows), len(rows)),
                                  "facts_verbatim_in_context": proportion(sum(f in r["context"] for r, f in facts), len(facts))}
        all_fact_rows = dev if suite_key == "lcr_dev" else demo
        all_facts = [(r, f) for r in all_fact_rows for f in r["supporting_facts"]]
        oracle_stats["all_raw_rows_fact_verbatim"] = proportion(sum(f in r["context"] for r, f in all_facts), len(all_facts))
        reconstructed = {}
        for row in raw_rows:
            chunks = chunk_context(row["context"], count)
            reconstructed[row["id"]] = (chunks, anchor_text(row["context"]))
        oracle_survival = []
        token_mismatch = []
        format_only_changed = []
        for row in raw_rows:
            chunks, anchor = reconstructed[row["id"]]
            if assemble(chunks, range(len(chunks)), anchor) != row["context"]:
                format_only_changed.append(row["id"])
        for rec in records:
            row = by_id[rec["example_id"]]
            chunks, anchor = reconstructed[rec["example_id"]]
            selected = row["context"] if rec["condition"] == "A_full" else assemble(chunks, rec["selected_chunk_ids"], anchor)
            measured = count(selected)
            if measured != rec["selected_context_tokens"]:
                token_mismatch.append({"id": rec["example_id"], "condition": rec["condition"], "cached": rec["selected_context_tokens"], "measured": measured})
            if rec["condition"] == "D_oracle":
                oracle_survival.append({"example_id": rec["example_id"], "line": rec["_source_line"], "n_facts": len(row["supporting_facts"]),
                                        "facts_verbatim_in_selected_text": sum(f in selected for f in row["supporting_facts"]),
                                        "localisation_confidences": rec["selector_diagnostics"]["localisation_confidence"]})
        oracle_stats["selected_fact_survival"] = {"all_facts_preserved_rows": sum(r["n_facts"] == r["facts_verbatim_in_selected_text"] for r in oracle_survival),
                                                 "n": len(oracle_survival), "rows": oracle_survival}
        report["oracle"][suite_key] = oracle_stats
        format_stats = {"source": loc(source), "all_chunks_reassembled_differ_from_raw": len(format_only_changed),
                        "n_raw": len(raw_rows), "cached_token_count_mismatches": token_mismatch,
                        "conditions": {}}
        for condition in sorted({r["condition"] for r in records}):
            sub = [r for r in records if r["condition"] == condition]
            format_stats["conditions"][condition] = {"tokens": dist([r["selected_context_tokens"] for r in sub]),
                                                       "mean_retention_ratio_recomputed": statistics.mean(r["selected_context_tokens"] / r["original_context_tokens"] for r in sub),
                                                       "source_lines": evidence(sub)}
        report["context_format"][suite_key] = format_stats
    source = LAB / "results" / "phase0_eval_lcr_dev" / "per_example.jsonl"
    scores = {}
    for condition in ("A_full", "B_query", "C_random", "D_oracle"):
        rows = [r for r in records_by_suite["lcr_dev"] if r["condition"] == condition and r["answerable"]]
        strict = [r for r in rows if score(r["raw_prediction"], [short_target(lcr_by_id[r["example_id"]])])]
        lenient = [r for r in rows if normalize(short_target(lcr_by_id[r["example_id"]])) in normalize(r["raw_prediction"])]
        raw_lenient = [r for r in rows if short_target(lcr_by_id[r["example_id"]]).lower() in r["raw_prediction"].lower()]
        strict_ids = {r["example_id"] for r in strict}
        extras = [r for r in lenient if r["example_id"] not in strict_ids]
        scores[condition] = {"strict": proportion(len(strict), len(rows)), "lenient_normalized_substring": proportion(len(lenient), len(rows)),
                             "lenient_raw_case_insensitive_substring": proportion(len(raw_lenient), len(rows)),
                             "cached_correct_disagreements": [r["example_id"] for r in rows if bool(r["correct"]) != (r["example_id"] in strict_ids)],
                             "lenient_only_rows": [{**e, "short_target": short_target(lcr_by_id[e["example_id"]]), "raw_prediction": e["raw_prediction"]} for e in extras],
                             "strict_correct_source_lines": evidence(strict), "lenient_correct_source_lines": evidence(lenient)}
    report["scoring_sensitivity"] = {"source": loc(source), "conditions_answerable": scores,
                                      "D_minus_A_strict_points": scores["D_oracle"]["strict"]["percent"] - scores["A_full"]["strict"]["percent"],
                                      "D_minus_A_lenient_points": scores["D_oracle"]["lenient_normalized_substring"]["percent"] - scores["A_full"]["lenient_normalized_substring"]["percent"]}
    constants = constant_values(LAB / "scripts" / "phase1_run.py")
    packs = load_packs()
    report["curriculum"] = {"runner_constants": {k: constants[k] for k in ("MAX_LENGTH", "TRAIN_PACKS", "DEV_PACKS", "SMOKE_TRAIN_PACKS", "SMOKE_DEV_PACKS")}, "sets": {}}
    for name, split, pack_count in (("bounded_train", "train", constants["TRAIN_PACKS"]), ("bounded_dev", "dev", constants["DEV_PACKS"]),
                                   ("smoke_train", "train", constants["SMOKE_TRAIN_PACKS"]), ("smoke_dev", "dev", constants["SMOKE_DEV_PACKS"])):
        rows = sample_curriculum(packs, split, pack_count)
        lengths = [(r, len(tokenize_example(tokenizer, r, 10**9)["input_ids"])) for r in rows]
        ids = [r["example_id"] for r in rows]
        report["curriculum"]["sets"][name] = {"n_packs": len({r["scenario_id"] for r in rows}), "token_lengths": dist([n for r, n in lengths]),
                                              "sequence_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                                              "kept_at_2048": proportion(sum(n <= 2048 for r, n in lengths), len(rows)),
                                              "kept_at_2560": proportion(sum(n <= 2560 for r, n in lengths), len(rows)),
                                              "dropped_at_2560_by_task": dict(Counter(r["task_type"] for r, n in lengths if n > 2560)),
                                              "dropped_at_2560_by_variant": dict(Counter(r["variant"] for r, n in lengths if n > 2560)),
                                              "rows": [{"example_id": r["example_id"], "tokens": n, "task_type": r["task_type"], "variant": r["variant"], "retained_2560": n <= 2560} for r, n in lengths]}
    preservation_source = LAB / "results" / "audit_20260905" / "prior_phase1_smoke" / "arm_A_preservation.jsonl"
    preservation = read(preservation_source)
    report["preservation"] = {"actual_classes": list(PRESERVATION_CLASSES), "smoke_source": loc(preservation_source),
                               "by_class": {cls: {"n": sum(r["task_family"] == cls for r in preservation),
                                                   "matched_recomputed": sum(score_preservation(r["raw_prediction"], {"reference": r["reference"], "reference_numbers": __import__("re").findall(r"-?\d+(?:\.\d+)?", r["reference"])})["matched"] for r in preservation if r["task_family"] == cls)} for cls in PRESERVATION_CLASSES},
                               "probe_receipts_found_in_lab_results": [loc(p) for p in (LAB / "results").rglob("*probe*")]}
    report["selector_history"] = {"git_log": subprocess.run(["git", "log", "--format=%H %s", "--", str(LAB.relative_to(REPO))], cwd=REPO, capture_output=True, text=True).stdout,
                                  "recorded_stage_selector_hashes": {p.parent.name: json.loads(p.read_text(encoding="utf-8")).get("source_sha256", {}).get("experiments/supa_long_context_minimal_intervention_v1/src/slcmi/context_selection.py") for p in (LAB / "results").glob("phase0_*/run_manifest.json")}}
    after = {name: checkpoint_files(core["path"]) for name, core in CORES.items()}
    report["checkpoint_sha256_before"] = before
    report["checkpoint_sha256_after"] = after
    report["canonical_checkpoint_unchanged"] = before == after
    assert report["canonical_checkpoint_unchanged"]
    sources = [Path(__file__), LCR_SFT, DEMO_SUITE, *list((LAB / "src" / "slcmi").glob("*.py")), LAB / "scripts" / "phase1_run.py",
               LAB / "results" / "phase0_eval" / "per_example.jsonl", source, preservation_source]
    report["source_sha256"] = {loc(p): sha256(p) for p in sources}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": loc(OUT), "short_target": report["short_target"], "disjointness": report["disjointness"],
                      "scoring": {k: {f: v[f] for f in ("strict", "lenient_normalized_substring", "lenient_raw_case_insensitive_substring")} for k, v in scores.items()},
                      "curriculum": {k: {f: v[f] for f in ("token_lengths", "kept_at_2048", "kept_at_2560")} for k, v in report["curriculum"]["sets"].items()},
                      "canonical_checkpoint_unchanged": report["canonical_checkpoint_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
