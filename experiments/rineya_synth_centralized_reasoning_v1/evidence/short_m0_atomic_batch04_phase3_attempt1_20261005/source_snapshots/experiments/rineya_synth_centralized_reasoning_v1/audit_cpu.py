"""Audit bounded CPU fixtures; no model/GPU/network/train/corpus write capability."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest
import io

from rs_contract import VERSION as CONTRACT_VERSION, digest, format_answer, require, validate_spec
from rs_solver import VERSION as SOLVER_VERSION, solve
from rs_verifier import VERSION as VERIFIER_VERSION, solve_visible, verify
from task_compiler import VERSION as COMPILER_VERSION, compile_records, export_training, fixture_catalog

PACKAGE = Path(__file__).resolve().parent


def audit(seed: int = 1003, parent_count: int = 48) -> dict:
    start = time.perf_counter()
    catalog = fixture_catalog(seed, parent_count)
    records = compile_records(catalog)
    counts = Counter()
    attempts = []
    examples = []
    for row in records:
        outcome = solve(row["spec"])
        other = solve_visible(row["messages"])
        require(outcome["status"] == other["status"] and outcome["solutions"] == other["solutions"],
                "reference disagreement")
        counts["independent_path_agreements"] += 1
        base = catalog["parents"][row["parent_id"]]["spec"]
        original = format_answer(solve(base)["solutions"][0])
        require(verify(row["messages"], original)["accepted"] ==
                (row["variant"]["answer_effect"] == "preserve"), "condition witness expectation failed")
        counts["answer_change_witnesses" if row["variant"]["answer_effect"] == "change"
               else "answer_preservation_witnesses"] += 1
        if row["skill"] == "R":
            wrong = [j for j in range(len(row["spec"]["initial"])) if j not in outcome["solutions"][0]][:1]
        else:
            wrong = list(range(row["spec"]["selection"]["k"] + 1))
        candidates = [("wrong_indices_or_cardinality", format_answer(wrong)),
                      ("false_unknown", format_answer("UNKNOWN")),
                      ("boolean_index", '{"answer":[true]}'),
                      ("extra_intermediate_state", '{"answer":[0],"steps":["incorrect"]}')]
        for kind, candidate in candidates:
            verdict = verify(row["messages"], candidate)
            require(not verdict["accepted"], "deliberately incorrect candidate accepted")
            attempt = {"row_id": row["row_id"], "kind": kind, "candidate": candidate,
                       "accepted": False, "reason": verdict["reason"]}
            attempts.append(attempt)
            counts["rejected_negative_controls"] += 1
            if row["arm"] == "M2" and row["variant"]["answer_effect"] == "change" and not any(
                    e["skill"] == row["skill"] and e["kind"] == kind for e in examples):
                examples.append({**attempt, "skill": row["skill"], "messages": row["messages"]})
    exports = {arm: export_training(records, catalog, arm) for arm in ("M1", "M2")}
    pools = {arm: Counter((r["parent_id"], r["skill"]) for r in records
                         if r["split"] == "train" and r["arm"] == arm) for arm in ("M1", "M2")}
    require(pools["M1"] == pools["M2"], "arm parent/exposure/skill mismatch")
    require(all(validate_spec(row["spec"]) in ("R", "S") for row in records), "fixture signature leak")
    # Exports are transient CPU fixtures. Counts/byte sizes are not tokenizer budgets.
    export_summary = {arm: {"rows": len(rows), "sha256": digest(rows),
                           "utf8_bytes": len(json.dumps(rows, ensure_ascii=True).encode("utf-8")),
                           "tokenizer_measured_nonpadding_tokens": None,
                           "materialized_training_corpus": False} for arm, rows in exports.items()}
    return {"status": "CPU_FIXTURE_AUDIT_PASSED", "experiment_id": "SYNTH-CRS-001",
            "scope": "bounded_cpu_compiler_and_verifier_mechanics_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "seed": seed, "parent_count": parent_count,
            "versions": {"contract": CONTRACT_VERSION, "primary_solver": SOLVER_VERSION,
                         "independent_verifier": VERIFIER_VERSION, "compiler": COMPILER_VERSION},
            "catalog_sha256": catalog["sha256"], "records_sha256": digest(records),
            "attempts_sha256": digest(attempts), "records": len(records),
            "dependency_components": len({a["component_id"] for a in catalog["assignments"].values()}),
            "parents_by_split": dict(sorted(Counter(a["split"] for a in catalog["assignments"].values()).items())),
            "records_by_split": dict(sorted(Counter(r["split"] for r in records).items())),
            "records_by_arm_skill": dict(sorted(Counter(r["arm"] + ":" + r["skill"] for r in records).items())),
            "checks": dict(counts), "negative_control_examples": examples,
            "negative_control_policy": "four predeclared wrong candidates per row; all attempts hashed; no selective replacement",
            "condition_preserve_definition": "the original certificate remains valid; additional S solutions may appear",
            "training_fixture_exports": export_summary,
            "matching_established": ["source_parents", "per_parent_row_exposures", "skill_allocation", "final_answer_format", "acceptance_policy"],
            "matching_not_established": ["nonpadding_training_tokens", "supervised_tokens", "optimizer_steps", "difficulty_distribution"],
            "native_inference_performed": False, "training_performed": False,
            "protected_banks_accessed": False, "model_quality_evidence": False,
            "transfer_or_retention_measured": False, "launch_authorized": False,
            "wall_seconds": round(time.perf_counter() - start, 6)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1003)
    parser.add_argument("--parents", type=int, default=48)
    parser.add_argument("--output", type=Path, help="new JSON receipt inside this package's evidence directory")
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve() if args.output else None
        if output:
            require(output.parent == PACKAGE / "evidence" and output.suffix == ".json" and
                    output.name not in {"local_validation.json"} and not output.exists(),
                    "receipt must be a new evidence/*.json file; preserve earlier receipts")
        report = audit(args.seed, args.parents)
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(PACKAGE / "tests"))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report["unittest"] = {"run": result.testsRun, "failures": len(result.failures),
                              "errors": len(result.errors), "skipped": len(result.skipped),
                              "output": stream.getvalue()}
        require(result.wasSuccessful() and not result.skipped, "CPU tests failed or skipped")
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, text=True,
                             capture_output=True, check=True)
        report["source_head"] = git.stdout.strip()
        report["source_identity"] = "HEAD plus current working-tree file hashes; no commit created by this audit"
        paths = [p for p in PACKAGE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and "evidence" not in p.relative_to(PACKAGE).parts]
        report["file_sha256"] = {p.relative_to(PACKAGE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(paths)}
        report["founding_manifest_sha256"] = hashlib.sha256((PACKAGE / "manifest.json").read_bytes()).hexdigest()
        decisions = json.loads((PACKAGE / "PILOT_DECISIONS.json").read_text(encoding="utf-8"))
        report["pilot_decisions"] = {"path": "PILOT_DECISIONS.json",
                                     "sha256": hashlib.sha256((PACKAGE / "PILOT_DECISIONS.json").read_bytes()).hexdigest(),
                                     "state": decisions["state"],
                                     "trainable_parameters": decisions["adaptation"]["trainable_parameters"],
                                     "nonpadding_tokens_ceiling_per_trained_arm": decisions["budget"]["ceiling_per_trained_arm_per_seed"],
                                     "decisions_are_native_qualification": False}
        report["run_requirements_unset"] = ["current_runtime_native_binding_and_qualification",
                                           "actual_token_matching_schedule_and_replacement_policy", "native_qualification",
                                           "fresh_confirmation_families_and_RS_transfer_set", "prior_correct_guard_enrollment",
                                           "effect_and_retention_margins", "statistics_and_power", "host_resource_caps", "launch_approval"]
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if output:
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        print(json.dumps({k: report[k] for k in ("status", "records", "checks", "launch_authorized")}
                         | {"unittest": {k: v for k, v in report["unittest"].items() if k != "output"},
                            "receipt": str(output) if output else None}, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, RecursionError) as exc:
        print(json.dumps({"status": "CPU_AUDIT_FAILED", "reason": str(exc), "launch_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
