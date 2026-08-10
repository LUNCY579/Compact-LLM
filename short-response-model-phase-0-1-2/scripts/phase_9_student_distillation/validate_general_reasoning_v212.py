from __future__ import annotations

"""Static validator for the v2.1.2 general-reasoning corpus."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ABSTENTION_TEXT = "I don't know based on the available information."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--corpus-root", type=Path, required=True)
    p.add_argument("--require-rows", type=int, default=10000)
    args=p.parse_args()
    root=args.corpus_root
    all_rows=read_jsonl(root/"prompts.all.jsonl")
    problems=[]
    ids=[]
    by_split=Counter(); by_family=Counter(); by_decision=Counter()
    for r in all_rows:
        rid=str(r.get("id") or "")
        ids.append(rid)
        by_split[str(r.get("intended_split") or "")]+=1
        by_family[str(r.get("reasoning_family") or "")]+=1
        by_decision[str(r.get("decision") or "")]+=1
        msgs=r.get("messages") or []
        if len(msgs)!=3 or [m.get("role") for m in msgs] != ["system","user","assistant"]:
            problems.append(f"{rid}:invalid_messages")
            continue
        answer=str(r.get("approved_answer") or "")
        if not answer or str(msgs[-1].get("content") or "") != answer:
            problems.append(f"{rid}:assistant_target_mismatch")
        if answer != answer.strip() or len(answer.split()) > 12:
            problems.append(f"{rid}:target_not_compact")
        if r.get("decision") == "abstain" and answer != ABSTENTION_TEXT:
            problems.append(f"{rid}:bad_abstention_target")
        if not isinstance(r.get("verification"), dict) or "result" not in r["verification"]:
            problems.append(f"{rid}:missing_verification")
        if r.get("question") is not None:
            problems.append(f"{rid}:general_reasoning_row_should_not_trigger_grounded_prompt_rebuild")
    if len(ids)!=len(set(ids)): problems.append("duplicate_ids")
    if len(all_rows)<args.require_rows: problems.append(f"too_few_rows:{len(all_rows)}")
    for split in ("train","dev","test"):
        split_rows=read_jsonl(root/"reviewed_splits"/f"{split}.reviewed.jsonl")
        if any(r.get("intended_split") != split for r in split_rows): problems.append(f"{split}:split_marker_mismatch")
        if len(split_rows)!=by_split[split]: problems.append(f"{split}:split_count_mismatch")
    if min(by_family.values() or [0]) <= 0: problems.append("empty_family")
    packet={"status":"PASS" if not problems else "FAIL","canonical_rows":len(all_rows),"split_counts":dict(by_split),"family_count":len(by_family),"decision_counts":dict(by_decision),"problems":problems}
    print(json.dumps(packet, indent=2, sort_keys=True))
    if problems: raise SystemExit(2)

if __name__ == "__main__": main()
