from __future__ import annotations

"""Deterministic semantic audit for procedure examples used by v2.1.

The first implemented check targets a concrete v1.7.5 issue found during the
Phase 9 postmortem: latency rows whose two values tie even though the rule says
"choose the smaller latency", while the gold names one relay.  Historical
artifacts are not silently rewritten; v2.1 excludes such rows from preservation
training and generates new golds with explicit no-tie invariants.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

LATENCY_RE = re.compile(r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+latency is\s+(-?\d+(?:\.\d+)?)\s*ms", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checked = 0
    for row in rows:
        if str(row.get("mode") or "").casefold() != "procedure" or str(row.get("decision") or "").casefold() != "answer":
            continue
        procedure_text = " ".join(str(x.get("text") or "") for x in row.get("procedures") or []).casefold()
        if "smaller latency" not in procedure_text and "lower-latency" not in procedure_text and "lower latency" not in procedure_text:
            continue
        support_ids = {str(x) for x in row.get("support_document_ids") or []}
        target_items = [
            item for item in row.get("evidence") or []
            if not support_ids or str(item.get("document_id") or item.get("record_id") or "") in support_ids
        ]
        matches = []
        for item in target_items:
            local = LATENCY_RE.findall(str(item.get("text") or ""))
            if len(local) >= 2:
                matches.extend(local)
        if len(matches) < 2:
            continue
        checked += 1
        values = [float(value) for _label, value in matches]
        answer = str(row.get("approved_answer") or row.get("answer") or "").strip()
        if len(set(values)) < len(values) and answer:
            violations.append({
                "id": row.get("id"),
                "topic_id": row.get("topic_id"),
                "answer": answer,
                "latencies": matches,
                "reason": "equal target-record latency has no unique smaller option but gold selects a label",
            })
    return {"checked_latency_answer_rows": checked, "violation_count": len(violations), "violations": violations}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--require-clean", action="store_true")
    args = p.parse_args()
    packet = {"input": str(args.input), **audit_rows(read_jsonl(args.input))}
    print(json.dumps(packet, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.require_clean and packet["violation_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
