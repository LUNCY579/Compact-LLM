#!/usr/bin/env python3
"""Descriptive Phase 1 target frames from complete per-arm raw observations.

Runs on CPU without importing model libraries. Lives outside frozen experiment
source. Recomputes correctness and abstention; never chooses an arm or applies
an advancement gate. Incomplete stages are explicitly labeled PARTIAL.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
AUDIT = Path(__file__).resolve().parent
LAB = AUDIT.parents[1]
REPO = LAB.parents[1]
ARMS = ("A", "B", "C")
STAGES = {"phase1": ("a11", False), "phase1_a1": ("a1", False),
          "phase1_smoke": ("a11", True), "phase1_smoke_a1": ("a1", True)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_jsonl_bytes(raw):
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def reference_rules():
    scorer = REPO / "Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py"
    spec = importlib.util.spec_from_file_location("phase1_frame_official_scorer", scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    abstention_source = LAB / "scripts/phase0_analyse.py"
    tree = ast.parse(abstention_source.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if
                (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "UNKNOWN_FORMS" for t in node.targets))
                or (isinstance(node, ast.FunctionDef) and node.name == "is_abstention")]
    if len(selected) != 2:
        raise ValueError("could not isolate the existing Phase 0 abstention rule")
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(abstention_source), "exec"), namespace)
    return module.normalize, namespace["is_abstention"], {
        "official_normalizer": str(scorer.relative_to(REPO)),
        "official_normalizer_sha256": sha(scorer.read_bytes()),
        "abstention_source": str(abstention_source.relative_to(REPO)),
        "abstention_source_sha256": sha(abstention_source.read_bytes()),
        "normalized_abstention_forms": sorted(namespace["UNKNOWN_FORMS"]),
    }


def frame(ids, records):
    out = {"n": len(ids), "arms": {}, "matched_deltas_vs_A": {}}
    for arm, by_id in records.items():
        correct = sum(by_id[i]["correct"] for i in ids)
        abstentions = sum(by_id[i]["abstention"] for i in ids)
        out["arms"][arm] = {
            "correct": correct, "accuracy_points": round(100 * correct / len(ids), 6),
            "abstentions": abstentions, "abstention_points": round(100 * abstentions / len(ids), 6),
        }
    if "A" in records:
        for arm in ("B", "C"):
            if arm not in records:
                continue
            baseline, adapted = records["A"], records[arm]
            correct_delta = sum(int(adapted[i]["correct"]) - int(baseline[i]["correct"]) for i in ids)
            abstention_delta = sum(int(adapted[i]["abstention"]) - int(baseline[i]["abstention"]) for i in ids)
            out["matched_deltas_vs_A"][arm] = {
                "paired_n": len(ids), "net_correct": correct_delta,
                "accuracy_delta_points": round(100 * correct_delta / len(ids), 6),
                "net_abstentions": abstention_delta,
                "abstention_delta_points": round(100 * abstention_delta / len(ids), 6),
                "A_only_correct": sum(baseline[i]["correct"] and not adapted[i]["correct"] for i in ids),
                "arm_only_correct": sum(adapted[i]["correct"] and not baseline[i]["correct"] for i in ids),
                "new_abstentions": sum(not baseline[i]["abstention"] and adapted[i]["abstention"] for i in ids),
                "removed_abstentions": sum(baseline[i]["abstention"] and not adapted[i]["abstention"] for i in ids),
            }
    return out


def arm_records(stage, arm, core, expected, normalize, is_abstention, rules):
    directory = LAB / "results" / stage / f"arm_{arm}"
    manifest_path = directory / "run_manifest.json"
    path = directory / "per_example.jsonl"
    info = {"status": "INCOMPLETE", "source": str(path.relative_to(REPO))}
    if not manifest_path.is_file() or not path.is_file():
        info["reason"] = "completed manifest or raw observations absent"
        return info, None
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError:
        info["reason"] = "arm manifest write is in progress; rerun after completion"
        return info, None
    raw = path.read_bytes()
    info.update(raw_sha256=sha(raw), manifest_sha256=sha(manifest_bytes),
                manifest_status=manifest.get("status"))
    if manifest.get("status") != "COMPLETE":
        info["reason"] = "arm manifest is not COMPLETE; observations are not analyzed"
        return info, None
    config = manifest.get("config", {})
    errors = []
    if config.get("core") != core or config.get("arm") != arm:
        errors.append("arm/core metadata differs from requested stage")
    if config.get("external_scorer_sha256") != rules["official_normalizer_sha256"]:
        errors.append("current official scorer differs from arm scorer receipt")
    all_rows = read_jsonl_bytes(raw)
    rows = [r for r in all_rows if r.get("evaluation") == "target"]
    by_id = {}
    mismatch_counts = {"correct": 0, "normalized_prediction": 0, "target_metadata": 0}
    for row in rows:
        row_id = row["example_id"]
        if row_id in by_id:
            errors.append(f"duplicate target example: {row_id}")
        if row_id not in expected:
            errors.append(f"unexpected target example: {row_id}")
            continue
        gold = expected[row_id]
        normalized = normalize(row["raw_prediction"])
        correct = normalized in {normalize(a) for a in gold["accepted_answers"]}
        mismatch_counts["correct"] += row.get("correct") != correct
        mismatch_counts["normalized_prediction"] += row.get("normalized_prediction") != normalized
        metadata_ok = row.get("arm") == arm and row.get("condition") == "A_full" and all(
            row.get(field) == gold[field] for field in ("answerable", "variant", "task_type", "split", "question"))
        mismatch_counts["target_metadata"] += not metadata_ok
        by_id[row_id] = {"correct": correct, "raw_prediction": row["raw_prediction"],
                         "abstention": is_abstention({"normalized_prediction": normalized})}
    if set(by_id) != set(expected) or len(rows) != len(expected):
        errors.append("target ID set or count differs from the complete expected suite")
    if any(mismatch_counts.values()):
        errors.append("stored scores or metadata disagree with independent rescoring")
    stable = manifest_path.read_bytes() == manifest_bytes and path.read_bytes() == raw
    if not stable:
        errors.append("arm files changed while being read; rerun after completion")
    info.update(status="INVALID" if errors else "COMPLETE", target_rows=len(rows),
                snapshot_stable=stable, verification_discrepancies=mismatch_counts, errors=errors[:5])
    return info, None if errors else by_id


def analyse_stage(stage, source_rows, normalize, is_abstention, rules):
    core, smoke = STAGES[stage]
    expected = {r["id"]: r for r in (source_rows[:32] if smoke else source_rows)}
    directory = LAB / "results" / stage
    root_manifest_path = directory / "run_manifest.json"
    root_raw_path = directory / "per_example.jsonl"
    root_manifest_bytes = root_manifest_path.read_bytes() if root_manifest_path.is_file() else None
    root_raw_bytes = root_raw_path.read_bytes() if root_raw_path.is_file() else None
    try:
        root_manifest = json.loads(root_manifest_bytes) if root_manifest_bytes else {}
    except json.JSONDecodeError:
        root_manifest = {"status": "READ_IN_PROGRESS"}
    arms, records = {}, {}
    for arm in ARMS:
        arms[arm], values = arm_records(stage, arm, core, expected, normalize, is_abstention, rules)
        if values is not None:
            records[arm] = values
    root_stable = (root_manifest_path.read_bytes() if root_manifest_path.is_file() else None) == root_manifest_bytes and (
        root_raw_path.read_bytes() if root_raw_path.is_file() else None) == root_raw_bytes
    expected_root_targets = {(arm, row_id): row["raw_prediction"] for arm, by_id in records.items() for row_id, row in by_id.items()}
    try:
        root_target_rows = [r for r in read_jsonl_bytes(root_raw_bytes) if r.get("evaluation") == "target"] if root_raw_bytes is not None else []
        actual_root_targets = {(r.get("arm"), r.get("example_id")): r.get("raw_prediction") for r in root_target_rows}
        root_matches_arms = (root_raw_bytes is not None and len(root_target_rows) == len(expected_root_targets)
                             and actual_root_targets == expected_root_targets)
    except json.JSONDecodeError:
        root_matches_arms = False
    complete = (set(records) == set(ARMS) and root_manifest.get("status") == "COMPLETE"
                and root_stable and root_matches_arms)
    status = "INVALID" if any(a["status"] == "INVALID" for a in arms.values()) or root_manifest.get("status") == "INVALID" else "COMPLETE" if complete else "PARTIAL"
    ids = sorted(expected)
    groups = {
        "answerability": {"answerable": [i for i in ids if expected[i]["answerable"]],
                          "unanswerable": [i for i in ids if not expected[i]["answerable"]]},
        **{field: {str(value): [i for i in ids if expected[i][field] == value]
                   for value in sorted({expected[i][field] for i in ids})}
           for field in ("variant", "task_type", "split")},
    }
    output = {
        "core": core, "smoke": smoke, "status": status, "preliminary": status != "COMPLETE",
        "completed_analyzed_arms": sorted(records), "arms": arms,
        "raw_stage_source": str(root_raw_path.relative_to(REPO)),
        "raw_stage_sha256": sha(root_raw_bytes) if root_raw_bytes is not None else None,
        "root_manifest_status": root_manifest.get("status", "ABSENT"),
        "root_manifest_sha256": sha(root_manifest_bytes) if root_manifest_bytes is not None else None,
        "root_snapshot_stable": root_stable,
        "root_target_predictions_match_analyzed_arms": root_matches_arms,
        "all": frame(ids, records),
        "groups": {name: {label: frame(selected, records) for label, selected in values.items() if selected}
                   for name, values in groups.items()},
        "descriptive_composite_decomposition_vs_A": {},
    }
    for arm in ("B", "C"):
        if "A" not in records or arm not in records:
            continue
        answerable = output["groups"]["answerability"]["answerable"]["matched_deltas_vs_A"][arm]
        unanswerable = output["groups"]["answerability"]["unanswerable"]["matched_deltas_vs_A"][arm]
        total = output["all"]["matched_deltas_vs_A"][arm]["net_correct"]
        assert total == answerable["net_correct"] + unanswerable["net_correct"]
        output["descriptive_composite_decomposition_vs_A"][arm] = {
            "all_net_correct": total,
            "answerable_net_correct": answerable["net_correct"],
            "unanswerable_net_correct": unanswerable["net_correct"],
            "answerable_contribution_to_all_accuracy_points": round(100 * answerable["net_correct"] / len(ids), 6),
            "unanswerable_contribution_to_all_accuracy_points": round(100 * unanswerable["net_correct"] / len(ids), 6),
            "answerable_net_abstentions": answerable["net_abstentions"],
            "unanswerable_net_abstentions": unanswerable["net_abstentions"],
        }
    return output


def markdown(report):
    lines = ["# Descriptive Phase 1 target frames", "",
             "Scores and abstentions are recomputed from raw predictions. Only completed, verified arms appear.",
             "PARTIAL stages are preliminary. This analysis makes no selection or advancement decision.",
             "Composite changes may combine answerable performance and abstention changes; these observations do not establish intrinsic reasoning improvement.", ""]
    for stage, result in report["stages"].items():
        lines += [f"## {stage} — {result['status']}", "",
                  f"Analyzed completed arms: {', '.join(result['completed_analyzed_arms']) or 'none'}.", ""]
        frames = [("all", result["all"])] + [(f"{name}/{label}", values) for name, group in result["groups"].items() for label, values in group.items()]
        for metric, value_key, points_key, delta_key in (
            ("Accuracy", "correct", "accuracy_points", "accuracy_delta_points"),
            ("Abstention", "abstentions", "abstention_points", "abstention_delta_points")):
            lines += [f"### {metric}", "", "| Frame | n | A count (%) | B count (%) | C count (%) | B−A points | C−A points |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for name, values in frames:
                cells = [f"{values['arms'][a][value_key]} ({values['arms'][a][points_key]:.2f})" if a in values["arms"] else "—" for a in ARMS]
                deltas = [f"{values['matched_deltas_vs_A'][a][delta_key]:+.2f}" if a in values["matched_deltas_vs_A"] else "—" for a in ("B", "C")]
                lines.append(f"| {name} | {values['n']} | " + " | ".join(cells + deltas) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), default=["phase1", "phase1_a1"])
    parser.add_argument("--output-stem", default="phase1_frames", help="Filename stem inside this audit directory")
    args = parser.parse_args()
    if Path(args.output_stem).name != args.output_stem or args.output_stem in ("", ".", ".."):
        raise SystemExit("--output-stem must be a simple filename stem")
    normalize, is_abstention, rules = reference_rules()
    target_source = REPO / "Rineya_Long_Context_Demo_v1/data/rineya_long_context_demo_full.jsonl"
    target_bytes = target_source.read_bytes()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Independent raw-output exact scoring; existing Phase 0 abstention function; paired descriptive count differences",
        "helper_sha256": sha(Path(__file__).read_bytes()),
        "target_source": str(target_source.relative_to(REPO)), "target_source_sha256": sha(target_bytes),
        "rules": rules,
        "stages": {stage: analyse_stage(stage, read_jsonl_bytes(target_bytes), normalize, is_abstention, rules) for stage in args.stages},
        "limitations": ["No statistical inference or causal attribution is performed.",
                        "PARTIAL or INVALID stages have no final experimental interpretation.",
                        "Use the separate mechanical and gate audits before interpreting any adaptation result."]}
    json_path = AUDIT / f"{args.output_stem}.json"
    markdown_path = AUDIT / f"{args.output_stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    compact = {}
    for stage, result in report["stages"].items():
        brief = lambda values: {"n": values["n"],
            "arms": {arm: {"correct": metrics["correct"], "abstentions": metrics["abstentions"]} for arm, metrics in values["arms"].items()},
            "deltas_vs_A": {arm: {"accuracy_points": metrics["accuracy_delta_points"], "abstention_points": metrics["abstention_delta_points"]}
                            for arm, metrics in values["matched_deltas_vs_A"].items()}}
        compact[stage] = {"status": result["status"], "completed_analyzed_arms": result["completed_analyzed_arms"],
                          "all": brief(result["all"]),
                          "answerability": {label: brief(values) for label, values in result["groups"]["answerability"].items()}}
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "stages": compact}, indent=2))


if __name__ == "__main__":
    main()
