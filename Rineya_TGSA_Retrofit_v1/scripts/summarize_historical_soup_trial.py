#!/usr/bin/env python3
"""Run split evaluations and apply the frozen A1.1/Baya transfer gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
CONFIG = LAB / "configs" / "historical_soup_a11_baya_midpoint_trial.json"
EVIDENCE = LAB / "artifacts" / "historical_soup_a11_baya_midpoint_v1" / "evidence"
EVALUATOR = REPO / "Rineya_Long_Context_Demo_v1" / "scripts" / "evaluate_predictions.py"
DATA = REPO / "Rineya_Long_Context_Demo_v1" / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def key_ids(path: Path) -> set[str]:
    return {str(row["id"]) for row in load_jsonl(path)}


def filter_predictions(source: Path, ids: set[str], target: Path) -> None:
    rows = load_jsonl(source)
    selected = [row for row in rows if str(row["id"]) in ids]
    if len(selected) != len(ids) or {str(row["id"]) for row in selected} != ids:
        raise ValueError(f"prediction/key identity mismatch for {target.name}")
    write_jsonl(target, selected)


def run_evaluator(
    key: Path, predictions: Path, stem: str, *, evidence_dir: Path = EVIDENCE
) -> Path:
    report = evidence_dir / f"{stem}_report.json"
    markdown = evidence_dir / f"{stem}_report.md"
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--key",
            str(key),
            "--predictions",
            str(predictions),
            "--json-report",
            str(report),
            "--markdown-report",
            str(markdown),
        ],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return report


def report_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    items = int(run["key_items"])
    packs = int(run["pack_metrics"]["complete_scenario_packs_in_key"])
    variants = {
        name: round(float(value["strict_exact"]) * int(value["items"]))
        for name, value in run["by_variant"].items()
    }
    return {
        "report": str(path),
        "report_sha256": sha256_file(path),
        "items": items,
        "prediction_rows": int(run["prediction_rows"]),
        "covered_items": int(run["covered_items"]),
        "coverage": float(run["coverage"]),
        "strict_exact_hits": round(float(run["strict_exact"]) * items),
        "strict_exact": float(run["strict_exact"]),
        "short_response_rate_le_12_words": float(run["short_response_rate_le_12_words"]),
        "mean_response_words": float(run["mean_response_words"]),
        "all_four_packs": round(
            float(run["pack_metrics"]["all_four_variants_correct"]) * packs
        ),
        "base_paraphrase_both_correct_packs": round(
            float(run["pack_metrics"]["base_paraphrase_both_correct"]) * packs
        ),
        "counterfactual_pair_packs": round(
            float(run["pack_metrics"]["counterfactual_pair_both_correct_and_changed"])
            * packs
        ),
        "unanswerable_hits": round(
            float(run["pack_metrics"]["unanswerable_accuracy"]) * packs
        ),
        "variant_hits": variants,
        "duplicate_prediction_ids": run["duplicate_prediction_ids"],
        "unexpected_prediction_ids": run["unexpected_prediction_ids"],
    }


def generation_health(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    errors = [str(row["id"]) for row in rows if row.get("status") != "ok"]
    empty = [str(row["id"]) for row in rows if not str(row.get("prediction", "")).strip()]
    repeated = []
    for row in rows:
        words = str(row.get("prediction", "")).lower().split()
        if len(words) >= 4 and any(
            words[index : index + 4] == [words[index]] * 4
            for index in range(len(words) - 3)
        ):
            repeated.append(str(row["id"]))
    return {
        "rows": len(rows),
        "status_errors": errors,
        "empty_predictions": empty,
        "fourfold_word_repetition": repeated,
        "passed": not errors and not empty and not repeated,
    }


def equivalence_result() -> dict[str, Any]:
    original = load_jsonl(
        EVIDENCE / "invalidated_pre_amendment" / "parent_all256_partial_unscored.jsonl"
    )[:4]
    portable = load_jsonl(EVIDENCE / "amendment_equivalence_parent_portable.jsonl")
    fields = ("id", "status", "prediction", "input_tokens", "generated_tokens")
    left = [{key: row.get(key) for key in fields} for row in original]
    right = [{key: row.get(key) for key in fields} for row in portable]
    encoded = json.dumps(left, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    encoded_right = json.dumps(
        right, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return {
        "rows": len(left),
        "matched": left == right,
        "original_sha256": hashlib.sha256(encoded).hexdigest(),
        "portable_sha256": hashlib.sha256(encoded_right).hexdigest(),
    }


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_all = EVIDENCE / "parent_all256.jsonl"
    candidate_all = EVIDENCE / "candidate_all256.jsonl"
    for path in (parent_all, candidate_all):
        if not path.is_file():
            raise FileNotFoundError(path)

    splits = {
        "showcase": DATA / "showcase_answer_key.jsonl",
        "stability": DATA / "stability_answer_key.jsonl",
    }
    reports: dict[str, dict[str, dict[str, Any]]] = {"parent": {}, "candidate": {}}
    for split, key in splits.items():
        ids = key_ids(key)
        for label, source in (("parent", parent_all), ("candidate", candidate_all)):
            filtered = EVIDENCE / f"{label}_{split}.jsonl"
            filter_predictions(source, ids, filtered)
            report = run_evaluator(key, filtered, f"{label}_{split}")
            reports[label][split] = report_metrics(report)
    reports["parent"]["all256"] = report_metrics(EVIDENCE / "parent_all256_report.json")
    reports["candidate"]["all256"] = report_metrics(EVIDENCE / "candidate_all256_report.json")

    parent = reports["parent"]
    candidate = reports["candidate"]
    checks = {
        "strict_all256_gain": (
            candidate["all256"]["strict_exact_hits"]
            >= parent["all256"]["strict_exact_hits"] + 1
        ),
        "stability_preserved": (
            candidate["stability"]["strict_exact_hits"]
            >= parent["stability"]["strict_exact_hits"]
        ),
        "all_four_packs_preserved": (
            candidate["all256"]["all_four_packs"] >= parent["all256"]["all_four_packs"]
        ),
        "coverage_and_identity": all(
            value[split]["coverage"] == 1.0
            and not value[split]["duplicate_prediction_ids"]
            and not value[split]["unexpected_prediction_ids"]
            for value in (parent, candidate)
            for split in ("showcase", "stability", "all256")
        ),
        "candidate_generation_health": generation_health(candidate_all)["passed"],
        "portable_policy_equivalence": equivalence_result()["matched"],
        "fresh_process_reload": json.loads(
            (EVIDENCE / "fresh_process_smoke.json").read_text(encoding="utf-8")
        ).get("status")
        == "PASS",
        "artifact_provenance_complete": all(
            path.is_file()
            for path in (
                EVIDENCE / "preflight.json",
                EVIDENCE / "build_result.json",
                LAB
                / "artifacts"
                / "historical_soup_a11_baya_midpoint_v1"
                / "candidate"
                / "TRIAL_ARTIFACT.json",
            )
        ),
    }
    passed = all(checks.values())
    deltas = {
        split: {
            key: candidate[split][key] - parent[split][key]
            for key in (
                "strict_exact_hits",
                "all_four_packs",
                "base_paraphrase_both_correct_packs",
                "counterfactual_pair_packs",
                "unanswerable_hits",
            )
        }
        for split in ("showcase", "stability", "all256")
    }
    outcome = {
        "schema_version": "rineya.historical-transfer-outcome.v1",
        "trial_id": config["trial_id"],
        "registration_sha256": sha256_file(CONFIG),
        "runner_sha256": sha256_file(REPO / config["evaluation"]["runner"]["path"]),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "equivalence": equivalence_result(),
        "reports": reports,
        "deltas": deltas,
        "parent_generation_health": generation_health(parent_all),
        "candidate_generation_health": generation_health(candidate_all),
        "transfer_gate": checks,
        "transfer_result": "TRANSFER_POSITIVE" if passed else "TRANSFER_FAIL",
        "production_preservation_warning": (
            "Aggregate gain is driven by unanswerable behavior; answerable base/paraphrase "
            "regressions require the separate no-blocking-regression shipping gate."
        ),
    }
    output = EVIDENCE / "transfer_outcome.json"
    output.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# Historical soup transfer outcome",
        "",
        f"Result: **{outcome['transfer_result']}**",
        "",
        "| Split | Parent hits | Candidate hits | Delta | Parent all-four | Candidate all-four |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("showcase", "stability", "all256"):
        markdown.append(
            f"| {split} | {parent[split]['strict_exact_hits']} | "
            f"{candidate[split]['strict_exact_hits']} | "
            f"{deltas[split]['strict_exact_hits']:+d} | "
            f"{parent[split]['all_four_packs']} | {candidate[split]['all_four_packs']} |"
        )
    markdown.extend(["", "## Frozen transfer checks", ""])
    markdown.extend(f"- {name}: **{'PASS' if value else 'FAIL'}**" for name, value in checks.items())
    markdown.extend(
        [
            "",
            "## Production warning",
            "",
            outcome["production_preservation_warning"],
            "",
        ]
    )
    (EVIDENCE / "transfer_outcome.md").write_text(
        "\n".join(markdown), encoding="utf-8", newline="\n"
    )
    print(json.dumps({"result": outcome["transfer_result"], "checks": checks, "deltas": deltas}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
