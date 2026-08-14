from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/runs/v22_negative_controls_smoke.jsonl"
CONFIG = ROOT / "configs/negative_controls.yaml"
OUTPUT = ROOT / "results/phase2_mechanism/negative_controls_bounded_dev.json"
REPORT = ROOT / "reports/PHASE2_NEGATIVE_CONTROLS_DEV.md"
METHODS = ("hard_plurality", "probability_mean", "logit_mean")
COMMITTEES = (
    "general_k1",
    "duplicate_general_k3",
    "core_k2",
    "weak_injection_k3",
    "overconfident_k3",
    "fixed_random_proxy_k3",
)
PAIRS = (
    ("general_k1", "duplicate_general_k3", "duplicate-expert invariance"),
    ("core_k2", "weak_injection_k3", "weak-expert injection"),
    ("weak_injection_k3", "overconfident_k3", "same social adapter, logit scale 1 versus 5"),
    ("core_k2", "overconfident_k3", "overconfident-expert injection"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in RUN.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = {(row["committee"], row["method"], row["example_id"]) for row in rows}
    if len(rows) != 144 or len(keys) != 144:
        raise RuntimeError(f"Expected 144 unique rows, observed {len(rows)} rows / {len(keys)} keys")
    config_hashes = {row["config_sha256"] for row in rows}
    if len(config_hashes) != 1:
        raise RuntimeError(f"Expected one config hash, observed {sorted(config_hashes)}")
    counts = Counter((row["committee"], row["method"]) for row in rows)
    expected = {(committee, method): 8 for committee in COMMITTEES for method in METHODS}
    if dict(counts) != expected:
        raise RuntimeError("Negative-control condition grid is incomplete")

    indexed = {
        (row["committee"], row["method"], row["example_id"]): row for row in rows
    }
    summaries: list[dict[str, Any]] = []
    for committee in COMMITTEES:
        for method in METHODS:
            group = [row for row in rows if row["committee"] == committee and row["method"] == method]
            summaries.append(
                {
                    "committee": committee,
                    "committee_role": group[0]["committee_role"],
                    "method": method,
                    "n": len(group),
                    "correct": sum(bool(row["correct"]) for row in group),
                    "accuracy": sum(bool(row["correct"]) for row in group) / len(group),
                    "mean_generated_tokens": sum(row["generated_tokens"] for row in group) / len(group),
                }
            )

    comparisons: list[dict[str, Any]] = []
    for left, right, interpretation in PAIRS:
        for method in METHODS:
            ids = sorted(
                {row["example_id"] for row in rows if row["committee"] == left and row["method"] == method}
                & {row["example_id"] for row in rows if row["committee"] == right and row["method"] == method}
            )
            left_rows = [indexed[(left, method, example_id)] for example_id in ids]
            right_rows = [indexed[(right, method, example_id)] for example_id in ids]
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "comparison_role": interpretation,
                    "method": method,
                    "paired_n": len(ids),
                    "exact_token_sequence_matches": sum(
                        a["output_token_ids"] == b["output_token_ids"]
                        for a, b in zip(left_rows, right_rows, strict=True)
                    ),
                    "left_correct": sum(bool(row["correct"]) for row in left_rows),
                    "right_correct": sum(bool(row["correct"]) for row in right_rows),
                }
            )
    for item in comparisons:
        item["exact_token_sequence_match_rate"] = (
            item["exact_token_sequence_matches"] / item["paired_n"]
        )

    duplicate = [row for row in comparisons if row["comparison_role"] == "duplicate-expert invariance"]
    if not all(row["exact_token_sequence_match_rate"] == 1.0 for row in duplicate):
        raise RuntimeError("Duplicate-expert invariance did not reproduce")

    packet = {
        "schema_version": 1,
        "status": "BOUNDED_DEV_NEGATIVE_CONTROLS_AUDITED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV BOUNDED PILOT / NEGATIVE CONTROL; NOT FINAL INFERENCE",
        "provenance": {
            "run": str(RUN.relative_to(ROOT)).replace("\\", "/"),
            "run_sha256": sha256(RUN),
            "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
            "config_file_sha256": sha256(CONFIG),
            "embedded_config_sha256": next(iter(config_hashes)),
            "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
        },
        "row_count": len(rows),
        "unique_condition_example_keys": len(keys),
        "condition_summaries": summaries,
        "paired_sequence_comparisons": comparisons,
        "conclusions": [
            "Three exact copies of General reproduce the General K=1 token sequence for every one of 8 examples under all three aggregation methods.",
            "Adding the weak Social adapter changes sequences relative to the K=2 core; this tiny packet cannot identify a general accuracy effect.",
            "Scaling the same Social adapter's logits by 5 changes soft-fusion sequences and can also change hard free-running sequences through confidence-based tie resolution and subsequent path divergence; hard top-1 proposals remain scale invariant away from ties.",
            "The fixed-pool proxy is not a matched random-seed control and cannot isolate semantic specialization.",
        ],
        "limitations": [
            "Only 8 DEV examples per condition were run; accuracy differences are deliberately not tested or generalized.",
            "The fixed random-pool proxy reuses trained adapters and is not a competence-matched random-seed family.",
            "The duplicate control tests algebraic replication, not ensemble diversity.",
            "Autoregressive sequence differences conflate the first changed decision with later path effects.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")

    summary_by = {(row["committee"], row["method"]): row for row in summaries}
    lines = [
        "# Phase 2 bounded DEV negative controls",
        "",
        "Status: **BOUNDED_DEV_NEGATIVE_CONTROLS_AUDITED**",
        "",
        "This is an audit of the pre-existing 8-example-per-condition DEV smoke packet. It is negative-control evidence, not a powered accuracy experiment.",
        "",
        "## Condition summaries",
        "",
        "| Committee | Method | Correct / 8 |",
        "|---|---|---:|",
    ]
    for committee in COMMITTEES:
        for method in METHODS:
            item = summary_by[(committee, method)]
            lines.append(f"| {committee} | {method} | {item['correct']} / 8 |")
    lines.extend(
        [
            "",
            "## Paired sequence controls",
            "",
            "| Left | Right | Method | Exact sequence matches |",
            "|---|---|---|---:|",
        ]
    )
    for item in comparisons:
        lines.append(
            f"| {item['left']} | {item['right']} | {item['method']} | "
            f"{item['exact_token_sequence_matches']} / {item['paired_n']} |"
        )
    lines.extend(
        [
            "",
            "Duplicate General copies are exactly invariant on all 24 method-example comparisons. Weak-adapter injection changes some trajectories. Increasing the same Social adapter's logit scale from 1 to 5 changes all hard sequences in this packet because confidence-based tie resolution is scale-sensitive at tied vote counts; this does not contradict top-1 proposal invariance away from ties.",
            "",
            "The fixed-pool proxy is not a matched random-seed or competence-matched control. With only eight examples, no accuracy difference here supports a causal specialization claim.",
            "",
            f"Raw SHA-256: `{packet['provenance']['run_sha256']}`.",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": packet["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
