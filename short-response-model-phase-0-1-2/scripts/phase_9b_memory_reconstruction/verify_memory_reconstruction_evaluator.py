from __future__ import annotations

"""Calibrate the deterministic Phase 9B reconstruction diagnostic on curated candidates."""

import argparse
from collections import defaultdict
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9b_memory_reconstruction.phase9b_common import reconstruction_candidate_score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 9B reconstruction evaluator ordering.")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=ROOT / "data" / "phase_9b_memory_reconstruction" / "reconstruction_corpus_v2_large",
    )
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="dev")
    args = parser.parse_args()
    name = "candidate_reconstructions.all.jsonl" if args.split == "all" else f"{args.split}.candidate_reconstructions.jsonl"
    rows = read_jsonl(args.corpus_root / "evaluator" / name)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scores_by_failure: dict[str, list[float]] = defaultdict(list)
    violations: list[str] = []
    for row in rows:
        packet = reconstruction_candidate_score(
            row["approved_reconstruction"],
            row["candidate_reconstruction"],
            memories=row.get("memories") or [],
        )
        failure = row.get("expected_failure_type") or "approved"
        row_packet = {**row, "diagnostic_score": packet["score"]}
        by_source[str(row["source_record_id"])].append(row_packet)
        scores_by_failure[failure].append(packet["score"])
    for source_id, candidates in by_source.items():
        approved = [row for row in candidates if row.get("expected_failure_type") is None]
        if len(approved) != 1:
            violations.append(f"{source_id}: expected exactly one approved evaluator candidate")
            continue
        approved_score = approved[0]["diagnostic_score"]
        for candidate in candidates:
            if candidate is approved[0]:
                continue
            if approved_score <= candidate["diagnostic_score"]:
                violations.append(
                    f"{source_id}: approved score {approved_score:.3f} <= negative {candidate['diagnostic_score']:.3f}"
                )
    means = {key: statistics.fmean(values) for key, values in scores_by_failure.items() if values}
    if means.get("approved", 0.0) < 0.95:
        violations.append("approved mean diagnostic score < 0.95")
    for failure in ("verbatim_replay_without_state_check", "over_rejection", "wrong_source_or_unsupported_reconstruction"):
        if failure in means and means[failure] >= means.get("approved", 0.0):
            violations.append(f"{failure} mean is not below approved mean")
    packet = {
        "status": "PASS" if not violations else "FAIL",
        "split": args.split,
        "rows": len(rows),
        "source_records": len(by_source),
        "mean_scores": means,
        "violations": violations,
    }
    print(json.dumps(packet, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
