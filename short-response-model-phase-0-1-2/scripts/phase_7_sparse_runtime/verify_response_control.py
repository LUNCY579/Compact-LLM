from __future__ import annotations

"""Run deterministic contract cases for the strict one-sentence controller."""

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import write_json
from scripts.common.response_control import finalize_response, first_sentence_boundary


CASES: tuple[dict[str, Any], ...] = (
    {"id": "simple", "text": "Sounds good.", "boundary": True},
    {"id": "decimal", "text": "The value is 3.14", "boundary": False},
    {"id": "version", "text": "Use v2.1", "boundary": False},
    {"id": "abbreviation", "text": "Ask Dr. Smith", "boundary": False},
    {"id": "domain", "text": "Visit example.com", "boundary": False},
    {"id": "quoted", "text": 'He said "Stop."', "boundary": True},
    {"id": "unclosed_quote", "text": 'He said "Stop.', "boundary": False},
    {"id": "markdown_code", "text": "Run `print('ok.')`", "boundary": False},
    {"id": "ordered_list", "text": "1. Do this", "boundary": False},
    {"id": "internal_ellipsis", "text": "Well... okay", "boundary": False},
    {"id": "chinese", "text": "可以。", "boundary": True},
    {"id": "chinese_attached_next_sentence", "text": "可以。再见。", "boundary": True},
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the response-control boundary contract.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/phase_7_router/response_control_verification.json",
    )
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for case in CASES:
        observed = first_sentence_boundary(str(case["text"])).found
        results.append({**case, "observed_boundary": observed, "passed": observed == case["boundary"]})

    finalized = finalize_response("First sentence. Second sentence.", "one_sentence")
    truncation_passed = finalized.text == "First sentence." and finalized.trimmed
    payload = {
        "schema_version": 1,
        "case_count": len(results),
        "passed_case_count": sum(1 for row in results if row["passed"]),
        "all_boundary_cases_passed": all(row["passed"] for row in results),
        "strict_finalization_passed": truncation_passed,
        "all_passed": all(row["passed"] for row in results) and truncation_passed,
        "cases": results,
        "strict_finalization": {
            "input": "First sentence. Second sentence.",
            "output": finalized.text,
            "trimmed": finalized.trimmed,
        },
    }
    write_json(args.out, payload)
    print(args.out)
    if not payload["all_passed"]:
        raise SystemExit("Response-control contract verification failed.")


if __name__ == "__main__":
    main()
