from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from consensus_lab.io import read_jsonl, sha256_file, write_json_atomic  # noqa: E402


RAW = LAB_ROOT / "results/sealed/v22_frozen_finalists.jsonl"
OUTPUT = LAB_ROOT / "results/phase2_mechanism/semantic_audit_selection.json"
REVIEW = LAB_ROOT / "reports/PHASE2_SEMANTIC_AUDIT_REVIEW.md"
PER_METHOD = 50
METHODS = ("hard_plurality", "probability_mean")
SELECTION_VERSION = "phase2-semantic-audit-v1"


def stable_key(method: str, example_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_VERSION}:{method}:{example_id}".encode()).hexdigest()


def user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def balanced_select(rows: list[dict[str, Any]], method: str, limit: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("decision")), str(row.get("reasoning_family")))].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: stable_key(method, str(row["example_id"])))
    selected: list[dict[str, Any]] = []
    depth = 0
    ordered_buckets = sorted(buckets)
    while len(selected) < limit:
        added = False
        for bucket in ordered_buckets:
            values = buckets[bucket]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != limit:
        raise RuntimeError(f"Only selected {len(selected)} of {limit} rows for {method}")
    return selected


def main() -> None:
    raw_rows = read_jsonl(RAW)
    benchmark_paths = {str(row["benchmark_path"]) for row in raw_rows}
    if len(benchmark_paths) != 1:
        raise RuntimeError(f"Expected one benchmark path, found {sorted(benchmark_paths)}")
    benchmark = Path(next(iter(benchmark_paths)))
    dataset_by_id = {str(row["id"]): row for row in read_jsonl(benchmark)}

    cases: list[dict[str, Any]] = []
    for method in METHODS:
        candidates = [
            row
            for row in raw_rows
            if row.get("committee") == "full_k7"
            and row.get("method") == method
            and not bool(row.get("correct"))
        ]
        for row in balanced_select(candidates, method, PER_METHOD):
            example_id = str(row["example_id"])
            dataset_row = dataset_by_id[example_id]
            cases.append(
                {
                    "audit_id": f"{method}:{example_id}",
                    "selection_sha256": stable_key(method, example_id),
                    "method": method,
                    "example_id": example_id,
                    "domain": row.get("domain"),
                    "reasoning_family": row.get("reasoning_family"),
                    "decision": row.get("decision"),
                    "prompt": user_prompt(dataset_row),
                    "gold": row.get("gold"),
                    "output": row.get("output"),
                    "official_exact_correct": False,
                    "semantic_label": None,
                    "error_type": None,
                    "adjudication_note": None,
                }
            )

    packet = {
        "schema_version": "1.0",
        "selection_version": SELECTION_VERSION,
        "raw_path": str(RAW.resolve()),
        "raw_sha256": sha256_file(RAW),
        "benchmark_path": str(benchmark.resolve()),
        "benchmark_sha256": sha256_file(benchmark),
        "population": "full_k7 official exact-match failures on v2.2 TEST",
        "sample_size": len(cases),
        "per_method": PER_METHOD,
        "selection_rule": (
            "For each method, group by (decision, reasoning_family), sort each group by "
            "SHA256(selection_version:method:example_id), then round-robin groups to 50."
        ),
        "label_definitions": {
            "semantically_correct": "Output conveys the gold answer despite a surface-form mismatch.",
            "semantically_incorrect": "Output conveys a different answer, abstains incorrectly, fails to abstain, or is uninterpretable.",
            "ambiguous": "Prompt/output does not support a reliable binary semantic judgment.",
        },
        "cases": cases,
    }
    write_json_atomic(OUTPUT, packet)

    lines = [
        "# Phase 2 Semantic Audit Review Sheet",
        "",
        f"- Population: full K=7 exact-match failures on v2.2 TEST",
        f"- Controlled sample: {len(cases)} responses ({PER_METHOD} per method)",
        f"- Raw SHA256: `{packet['raw_sha256']}`",
        "- Primary metric remains the official exact score.",
        "- Review status: **PENDING BLINDED ADJUDICATION**",
        "",
        "Each row was fixed before semantic labels were viewed. Adjudicators should use",
        "`semantically_correct`, `semantically_incorrect`, or `ambiguous`, plus one error type:",
        "`format_or_paraphrase`, `wrong_content`, `false_abstention`, `failure_to_abstain`,",
        "`malformed`, or `ambiguous`.",
        "",
        "| # | Method | Example | Decision | Family | Gold | Output | Label | Error type |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    for index, case in enumerate(cases, start=1):
        gold = str(case["gold"]).replace("|", "\\|").replace("\n", " ")
        output = str(case["output"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {index} | {case['method']} | `{case['example_id']}` | {case['decision']} | "
            f"{case['reasoning_family']} | {gold} | {output} | PENDING | PENDING |"
        )
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "review": str(REVIEW), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
